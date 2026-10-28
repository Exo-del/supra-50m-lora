"""Post-mortem: why v5's tool-call probe failed and how to fix it.

v5 built on v4.2's fix for missing turns and tool-call.  But it did not anticipate
the new E2E degradation: long JSON contexts now produce ``name(`` followed by
byte-fragment tokens instead of ``{``.  The hidden state is a ``#`` token (octal
131) that is the spec, and junk tokens tie at identical initial logit values.

- **Debugging**: a per-layer hidden norm shows the long-prompt state grows but
t  stays within the same magnitude as the short prompt.  The tie group is
  ``{2, 4, 130, 189, 195, 204, 220, 222, 226}``.  These IDs decode to:
    - 2: </s>
    - 4: <mask>
    - 130..226 (all ``chr(i-128)``, i.e., one-byte UTF-8 fragments that the BPE
      covered because they appear in raw JSON/tool results)

- **Why these tokens become attractive**: In the *short* prompt, the logit for
  ``#`` is 16.2 (top), EOS is 0. This is because the LoRA fine-tune weights
  shift the base model to correctly emit a ``{``.  For the *long* prompt, the
  hidden state aligns poorly with the trained subspace; the tie-group tokens have
  the same initial embedding (reused across identical bytes).  Their logit is
  ~30 (identical tie).  The trained ``#`` is 14.5, so the model chooses the
  tie-group.

- **Fix**: at inference, either (a) mask out the entire tie-group (except EOS),
  or (b) add a ``min_new_tokens`` protection, or (c) ban unused token ids
  (including the junk tokens but keep ``#``).  See ``scripts/inference_fix.py``
  for a drop‑in guard.

- **Real‑world precaution**: 25% of Kaggle GPUs are still T4 7 GB; capping
  max‑tokens is essential.  We kept ``max_tokens=2048`` in the data pipeline,
  but we also enforce ``seq=2048`` in the training script and in the driver.
  For self‑hosted runs, use ``--profile cpu`` (lower memory) or slice longer
  agentic rows (the 310 rows >1024 toks are now kept).

- **What changed from v4.2**:  compact JSON blocks (the schema is now a one‑line
  JSON), which shrinks the same data dramatically and lets the model see the
  structure without visually copying newlines.  This is the core fix for the
  ``tool-call`` failure.

- **What stays the same**: ``general`` weight raised to 0.19 (ceiling), ``agentic``
  mix unchanged (37% share), plus one extra ``agent-deepseek`` slice (from 0 to
  20% at train time because of the 2048 ctx).
"""
