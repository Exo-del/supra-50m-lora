# v4 data audit: why the mix was rebuilt

Context: v1–v3 of this adapter targeted "multi-turn agentic + structured
reasoning". The data did not match that target in three separate ways. This
note records what was measured, so the next rebuild does not repeat it.

## 1. Language (the reported problem)

`Davizig10jojo/Kimi-K3-And-DeepSeek-V4-Pro-0813-Distillation-in-PT-BR` is
Portuguese. Row domains confirm it: `cultura_br`, `conversa_ptbr`,
`atualidades_br`. In `mix_colab.jsonl` it was 64/811 rows (8% of rows, 3% of
tokens) and in `mix_v3.jsonl` 22/2632 rows.

Measured effect on T4 (v3 recipe, 891 rows, 98 steps):

| source | rows | base loss | +LoRA | reduction |
|---|---|---|---|---|
| agent-tool (en) | 699 | 3.04 | 1.37 | 55% |
| seed (en) | 48 | 4.06 | 2.64 | 35% |
| pt-br-reasoning | 64 | 4.07 | 3.34 | 18% |

The Portuguese slice moved **3x less** than the English agentic slice. That is
the expected result: 58 source rows at ~0.01% of the 20B pretraining budget
cannot install a new language, and the adapter is meant to be English anyway.
It was removed in v4.

## 2. The reasoning objective was under-covered

`mix_v3.jsonl` composition by rows:

```
general (dolly)   2000
agent-tool         562
seed                48
pt-br-reasoning     22
```

So the "structured reasoning" half of the goal was 22 rows of a language we do
not want. `### Thinking:` was effectively untrained. v4 budgets **tokens per
source** rather than rows, and reserves ~40% for reasoning (math + code +
OpenThoughts) and ~44% for agentic.

## 3. The agent dataset is 8.8x duplicated

`zake7749/deepseek-v4-pro-agent-tool-calling-trajectory`:

- 2510 rows, but `id` has only **284 unique values**
- 5 domains only: movie 892, bank 843, ecommerce 448, medicine 309, basketball 18
- rendered-row hashing threw away the clones and with them most of the budget

v4 dedupes by `id` first, then slices each surviving trajectory into up to 6
turn-level examples, each ending on a complete assistant turn.

## 4. Two silent data bugs found while rebuilding

**GSM8K calculator markers.** The source writes `3 + 4 = <<3+4=7>>7 apples`.
Substituting the annotated value produced `3 + 4 = 77 apples`. The annotation
must be *deleted*, leaving the value that already follows it.

**glaive function calls.** Calls look like
`<functioncall> {"name": ..., "arguments": '{"country": "US"}'}` — the arguments
are a single-quoted JSON string, the closing tag is often absent, and a
non-greedy regex stops inside the wrong brace. Result: every call was dropped
and the raw tag leaked into the assistant text. Fixed with a quote-aware brace
scan plus a repair path (see `tests/test_parsers.py`).

**OpenThoughts template markers.** Splitting Thought/Solution after
`canon.sanitize()` is too late: sanitize strips `<|end_of_thought|>`, so the
whole trace landed in the answer and the model never saw a thinking block.
The split now happens on the raw string.

## 5. Truncation vs filtering

Reasoning rows are *filtered* to those whose whole trace fits the budget
(`max_chars=3000` for OpenThoughts, `max_tokens` overall). Truncating a chain
of thought mid-way teaches the model to ramble and never conclude. Agentic
rows are different: they are multi-turn, so they are sliced at turn
boundaries and every slice still ends on a complete assistant turn.

## 6. What v4 trains on

| source | role | token share |
|---|---|---|
| `agent-deepseek` | multi-turn tool calling with per-turn reasoning | 26% |
| `agent-glaive` | tool calls with real tool responses | 18% |
| `think-code` | R1-style Thought/Solution, short only | 16% |
| `math-gsm8k` | short arithmetic CoT | 13% |
| `math-instruct` | short CoT / program-of-thought | 11% |
| `general` | no-tool answers (negative class) | 14% |
| `seed` | hand-written English anchors | 2% |

Every row passes `canon.is_english()`; the build prints the reject count per
source, and `data/mix_v4.report.json` records rows, tokens, share and rejects.

Expected shortfall: `agent-deepseek` caps out around 1.5k slices (284
trajectories) and `think-code` short traces are rare, so the realised total
lands under the 6M token target. `agent-glaive` and `math-gsm8k` fill the rest;
the report records the truth, the weights are only intent.
