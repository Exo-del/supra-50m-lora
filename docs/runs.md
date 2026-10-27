# Run log: v4 → v4.2

Three runs on the same base, T4, 6M-token target, 3 epochs, bs 4 x accum 8,
seq 1024, lr 5e-4, r16 / alpha 32. Each one fixed something the previous log
exposed. Logs in `runs/<slug>/kernel.log`, adapters in `runs/<slug>/*.zip`.

| run | kernel slug | unique tokens | wall | val loss |
|---|---|---|---|---|
| v4 | `supra-50m-v4-lora-english-agentic-reasoning` | 3.28M | 39.0 min | 1.718 |
| v4.1 | `supra-50m-v4-1-lora-english-agentic-reasoning` | 4.11M | 45.3 min | 1.698 |
| v4.2 | `supra-50m-v4-2-lora-agent-prefixes-reasoning-mix` | 4.67M | 51.5 min | 1.728 |

Throughput is ~4.3k tok/s at bs 4 x 1024 on a T4, so a run costs under an hour
of the 30 h weekly GPU quota. Iterating is cheap; shipping a broken mix is not.

## Realised mix per run

The weights are intent. What the builder actually kept is the truth, and it
differed from the weights in every run:

| source | v4 | v4.1 | v4.2 |
|---|---|---|---|
| `agent-deepseek` (multi-turn + reasoning) | **2%** | 1% | **13%** |
| `agent-glaive` (tool call + result) | 33% | 26% | 18% |
| `code-magicoder` | — | 20% | 21% |
| `think-code` (OpenThoughts) | **0%** | — | — |
| `math-gsm8k` | 24% | 19% | 18% |
| `math-instruct` | 20% | 16% | 15% |
| `general` | 21% | 17% | 15% |

## v4: two sources silently collapsed

| source | intended budget | built | share |
|---|---|---|---|
| `agent-deepseek` | 1,560,000 | 173 rows / 58,099 tok | 2% |
| `agent-glaive` | 1,080,000 | 2,614 rows / 1,080,072 tok | 33% |
| `think-code` | 960,000 | 11 rows / 7,240 tok | 0% |

The report only says "kept 173 rows". The reasons were structural:

- **`agent-deepseek` rows are conversation prefixes.** Every row with the same
  `id` is the same conversation cut at a different assistant turn (3, 5, 7 ...
  messages): 2,510 rows are 284 conversations. The builder deduped by `id` and
  kept whichever prefix came first — the one-turn prefix. So the multi-turn
  objective, the whole point of the project, was 2% of the data.
- **`think-code` never fit.** Median assistant trace 24,357 chars, p10 9,500,
  and 0 of 120 sampled rows under a 3,000-char budget. At 1024 tokens the only
  way in is clipping the chain of thought, which is what section 5 of
  `v4-data-audit.md` rules out. Eleven rows squeaked through by luck.

### Per-source loss reduction (30 rows/source)

| source | base | +LoRA | reduction | reading |
|---|---|---|---|---|
| `agent-glaive` | 2.885 | 0.250 | **+91%** | memorised: 33% of mix, 3 epochs |
| `agent-deepseek` | 2.982 | 1.067 | +64% | from 173 rows |
| `math-gsm8k` | 2.983 | 1.090 | +63% | |
| `math-instruct` | 2.815 | 1.330 | +53% | |
| `seed` | 4.040 | 2.527 | +37% | |
| `general` | 3.633 | 2.314 | +36% | |
| `think-code` | 2.868 | 2.239 | +22% | 11 rows, no signal |

A 91% reduction on the largest slice is recall of 2,614 training rows, not a
capability. The slices that moved least are exactly the two the builder starved.

Probes: `no-tool 2/2`, `tool-call 0/2`, `math 0/2`, `code 0/2`.

## v4.1: the code slot starts working

`code-magicoder` replaced OpenThoughts, nothing else changed — so this run is a
clean A/B on the reasoning slot:

| source | base | +LoRA | reduction |
|---|---|---|---|
| `agent-glaive` | 2.770 | 0.301 | +89% (still memorising) |
| `agent-deepseek` | 2.930 | 0.999 | +66% (still 173 rows) |
| `math-gsm8k` | 3.037 | 1.244 | +59% |
| `code-magicoder` | 2.781 | 1.211 | **+56%** |
| `math-instruct` | 2.969 | 1.471 | +50% |
| `seed` | 4.064 | 2.565 | +37% |
| `general` | 3.440 | 2.223 | +35% |

`+56%` on 1,709 rows of real code versus `+22%` on 11 rows of nothing. Val loss
1.698, the best of the three. Probes unchanged: `no-tool 2/2`, rest `0/2`.

## v4.2: the agentic slice exists

Longest prefix per conversation id, then slice by turn:

| | v4 | v4.2 |
|---|---|---|
| rows | 173 | 892 |
| tokens | 58,099 | 621,344 |
| with tool calls | — | 908 |
| with tool results | — | 736 |
| with per-turn reasoning | — | 892 |

Mix moved from glaive-dominated to code/math/glaive roughly balanced with 13%
deepseek. Val loss 1.728 (slightly up: more multi-turn rows, longer contexts,
less repetition of the easy glaive slice).

### Per-source loss reduction (12 rows/source, base -> adapter)

| source | base | +LoRA | reduction |
|---|---|---|---|
| `math-gsm8k` | 3.204 | 1.114 | +65% |
| `code-magicoder` | 2.810 | 1.252 | +55% |
| `math-instruct` | 3.238 | 1.463 | +55% |
| `agent-deepseek` | 3.207 | 1.551 | **+52%** |
| `agent-glaive` | 2.986 | 0.373 | +88% (recall) |
| `general` | 3.523 | 2.234 | +37% |
| `seed` | 4.224 | 2.769 | +34% |

Read the deepseek number carefully. In v4 it was +64% on **one-turn** rows the
builder had effectively seen 70 times each; here it is +52% on **multi-turn**
rows (2-4 assistant turns, tool calls, tool results) drawn from 892 examples
that make up 13% of the mix. Slightly smaller percentage, much harder task.
`agent-glaive` is still at 0.37: 18% of the mix and simple single-tool shapes,
so it is still being recalled rather than learned.

### Probes

```
no-tool      1/1   PASS  answers without calling a tool
tool-follow  1/1   PASS  used the tool result, did not re-call the tool
tool-call    0/1   fail
math         0/1   fail
code         0/1   fail
```

`tool-follow` is the one that matters: given a rendered
`system -> user -> tool call -> tool result` history, the adapter answered from
the payload ("Sky: clear") and did not loop on the tool. That is the first
multi-turn behaviour any of these adapters has demonstrated, and it is new with
the prefix fix. It then drifts ("-sky: cloudy" afterwards) — 50M parameters and
60 greedy tokens, not a deployment.

`tool-call` still failing while `tool-follow` passes is informative: the model
will *use* a tool result but will not *emit* a call from a cold prompt. In the
mix, calls always follow a system prompt that lists a tool schema (glaive,
deepseek); the probe's system line is terse. Worth retesting with a
glaive-shaped system block before concluding anything about the adapter.

## What the probes mean

All three adapters pass `no-tool` and fail the other single-turn probes. Two
separable causes:

1. **Sampler artefact.** A 50M adapter does not emit `</s>` reliably, so greedy
   decoding runs past the answer and opens a second `### Response:` block. The
   probes were scoring that rambling. `eval_run.py` now cuts at the next
   section marker, exactly like a deployment stop-sequence.
2. **Real weakness.** Even the first block is wrong for arithmetic and code. A
   50M model with r16 LoRA on ~4M tokens learns the shape `### Thinking:` then
   `### Response:` and the tool-call syntax; it does not learn to do arithmetic
   it cannot do.

The `tool-follow` probe exists for the capability that actually matters: after a
tool result, does the model use the payload, or call the same tool again? That
question was untested until v4.2 — every other probe is single-turn.

## Bugs the runs exposed, in the order they appeared

1. `think-code` = 0 rows (measured before launch, too late: the kernel had
   already cloned).
2. `agent-deepseek` = 2% because prefixes were treated as duplicates.
3. `eval_run.py` scored sampler rambling as the model's answer.
4. `U+2028` inside one GSM8K question: legal JSON, but `str.splitlines()`
   splits on it, so the mix file broke every line-oriented reader and killed
   the v4.2 eval with `Unterminated string`. Fixed in `canon.sanitize`, with a
   regression test.
