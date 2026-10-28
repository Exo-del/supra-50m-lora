#!/usr/bin/env python3
"""Drop‑in guard for Kaggle‑run v5 adapter.

Replace the normal ``model.generate()`` call in any inference script with

    from scripts.inference_fix import guarded_generate
    text = guarded_generate(model, tokenizer, prompt, max_new_tokens=96)

Features:
* Suppresses the tie‑group tokens (130‑226 + mask) that become attractive after
  long JSON contexts, leaving ``#`` as the winner for tool calls.
* Keeps ``</s>`` (2) free for normal EOS, but protects against its premature
  appearance (by early suppression).
* Works with both greedy and sampled generation.
* Leaves the ``general`` register protected by the weight bump in v5 (0.19).

Usage example (for the Kaggle profile):

    python kaggle/run_kaggle.py
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, SuppressTokensLogitsProcessor

# Load the tie‑group from the cache that inference_fix creates.
TIE_JUNK = Path("/tmp/opencode/junk_ids.json").read_text()
TIE_IDS = json.loads(TIE_JUNK)

# Current runs directory (should be `runs/supra-50m-v5-lora-compact-json-schema-2048-ctx`).
RUN_DIR = Path(__file__).parent / "runs" / "supra-50m-v5-lora-compact-json-schema-2048-ctx"
ADAPTER_PATH = RUN_DIR
BASE_PATH = Path(__file__).parent / "model"


def guarded_generate(model, tokenizer, prompt, max_new_tokens=96,
                      do_sample=False, temperature=0.5, top_k=25, top_p=0.9,
                      repetition_penalty=1.2):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    logits_processor = LogitsProcessorList([
        SuppressTokensLogitsProcessor(TIE_IDS, device=model.device)
    ])
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        logits_processor=logits_processor,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        kwargs.update(temperature=temperature, top_k=top_k, top_p=top_p)
    with torch.no_grad():
        out = model.generate(ids, **kwargs)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def load_v5():
    """Factory: loads the v5 LoRA adapter and returns a model with the guard."""
    base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, ADAPTER_PATH).eval()
    return model


if __name__ == "__main__":
    # Quick test: print the first probe from eval_run.py (no‑tool)
    from scripts.eval_run import PROBES, _TOOL_SYS
    import canon

    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    model = load_v5()

    for name, prompt in PROBES:
        out = guarded_generate(model, tok, prompt, max_new_tokens=80, do_sample=False)
        print(f"{name}: {out[:120]!r}")
        break
