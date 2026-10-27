# v4 run: what the numbers said

Kernel `alouhmiwaslost/supra-50m-v4-lora-english-agentic-reasoning`, T4, 2026-10-27.
Log: `runs/supra-50m-v4-lora-english-agentic-reasoning/kernel.log`.

## Training

| | |
|---|---|
| unique tokens built | 3,278,510 |
| rows | 16,543 |
| epochs | 3 |
| optimizer steps | 1,740 (bs 4 x accum 8) |
| tokens seen | 9,336,023 |
| wall clock | 39.0 min @ 3,994 tok/s |
| LoRA | r16 / alpha 32, 1.79M trainable of 53.6M (3.35%) |
| val loss | 1.7177 (ep3) |

Loss fell from 3.12 to 1.59 across the run with no divergence, so the optimizer
was healthy. The problem was **what** it was training on.

## The mix did not match the weights

| source | intended budget | built | realised share |
|---|---|---|---|
| `agent-deepseek` | 1,560,000 | 173 rows / **58,099 tok** | **2%** |
| `agent-glaive` | 1,080,000 | 2,614 rows / 1,080,072 tok | **33%** |
| `math-gsm8k` | 780,000 | 5,362 rows / 780,020 tok | 24% |
| `math-instruct` | 660,000 | 2,910 rows / 660,014 tok | 20% |
| `think-code` | 960,000 | 11 rows / **7,240 tok** | **0%** |
| `general` | 840,000 | 8,450 rows / 691,670 tok | 21% |

Two sources collapsed, and the report only says "kept 173 rows" — it does not
say *why*. Both causes are structural, not noise:

- **`agent-deepseek`: rows are conversation prefixes.** Every row with the same
  `id` is the same conversation cut at a different assistant turn (3, 5, 7 ...
  messages); 2,510 rows are 284 conversations. The builder deduped by `id` and
  kept whichever prefix it saw first, i.e. the one-turn prefix. Result: one
  single-turn example per conversation, and the multi-turn objective — the
  entire point of the project — was 2% of the data.
- **`think-code`: the traces do not fit.** Median assistant trace is 24,357
  chars, p10 9,500, and **0 of 120 sampled rows** fit a 3,000-char budget. On a
  1024-token context the source can only be used by clipping, which §5 of
  `v4-data-audit.md` rules out. 11 rows squeaked through by luck.

## Per-source loss reduction (base -> adapter, 30 rows/source)

| source | base | +LoRA | reduction | reading |
|---|---|---|---|---|
| `agent-glaive` | 2.885 | 0.250 | **+91%** | memorised: 33% of the mix, 3 epochs |
| `agent-deepseek` | 2.982 | 1.067 | +64% | on only 173 rows |
| `math-gsm8k` | 2.983 | 1.090 | +63% | |
| `math-instruct` | 2.815 | 1.330 | +53% | |
| `seed` | 4.040 | 2.527 | +37% | 24 rows, held-out-ish |
| `general` | 3.633 | 2.314 | +36% | |
| `think-code` | 2.868 | 2.239 | +22% | 11 rows, no signal |

A 91% reduction on the largest slice is not a capability gain, it is recall of
2,614 training rows. The honest signal is that every slice moved, and the
weakest are exactly the two that were starved.

## Probes: 2/4

```
no-tool    2/2   (answers without calling a tool -> negative class works)
tool-call  0/2   (emitted ### Response: again instead of a call)
math       0/2
code       0/2
```

Two separate failures there:

1. **Sampling artefact.** The adapter does not emit `</s>` reliably, so greedy
   decoding runs past the answer and opens a second `### Response:` block. The
   probes scored that rambling as the answer. `eval_run.py` now cuts at the next
   section marker, the way a deployment stop-sequence would.
2. **Genuine weakness.** Even the first block was wrong for math/code. A 50M
   model with r16 LoRA on 3.3M tokens learns the *shape* `### Thinking:` then
   `### Response:` and the tool-call syntax, but not arithmetic it cannot do.

## Fixes carried into v4.2

- take the **longest prefix** per conversation id, then slice by turn:
  892 rows / 621,344 tokens, 908 with tool calls, 736 with tool results
  (from 173 rows / 58,099)
- `code-magicoder` replaces `think-code`; rows that do not fit are dropped
- rebalance: glaive 33% -> 18%, deepseek 2% -> 13%, code 21%, math 33%
- eval stops at the next `### ` marker instead of scoring sampler rambling
