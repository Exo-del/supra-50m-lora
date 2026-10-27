# supra-50m-lora

LoRA fine-tuning of [`SupraLabs/Supra-50M-Base`](https://huggingface.co/SupraLabs/Supra-50M-Base)
(51.8M params, Llama-style, 32k custom BPE, tied embeddings).

Target behaviour: **multi-turn agentic tool use** with **structured reasoning**
(a `Thinking` block before the answer, DeepSeek-style) on an **English-only**
SFT mix, plus a logic floor from math and code traces.

## What changed in v4 (the language correction)

v1–v3 mixed in `Davizig10jojo/Kimi-K3-And-DeepSeek-V4-Pro-0813-Distillation-in-PT-BR`.
That set is **Portuguese**, and it was 3% of the mix while consuming ~18% of the
training signal on a capability we never wanted. Measured on T4:

| batch | rows | share of tokens | val loss (base -> LoRA) |
|---|---|---|---|
| pt-br-reasoning | 64 | 3% | 4.07 -> 3.34 (-18%) |
| agent-tool | 699 | 78% | 3.04 -> 1.37 (-55%) |
| seed (en) | 48 | 19% | 4.06 -> 2.64 (-35%) |

The Portuguese slice moved **3x less** than the English agentic slice for 3% of
the budget, and it is the one thing the model cannot use. v4 drops it entirely
and spends those tokens on English reasoning data.

Two other defects found while re-reading the mixes:

1. **The reasoning objective was under-covered, not just mislanguaged.** v3 had
   2000 dolly rows vs 562 agent rows and only 22 thin PT "thinking" rows, so the
   model was mostly learning plain Q&A. v4 budgets by *tokens per source*, with
   agentic 44% and reasoning/math/code 40%.
2. **The agent dataset is 8.8x duplicated.** `zake7749/...` has 2510 rows but
   only **284 unique `id`s** across 5 domains. Hashing rendered rows throws away
   75% of the set; v4 dedupes by `id` first, then slices turns, so each unique
   trajectory contributes several complete examples instead of clones.

## The v4 mix

| source | dataset | role |
|---|---|---|
| `agent-deepseek` | `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory` | multi-turn tool calling **with per-turn reasoning** (dedupe by id, slice by turn) |
| `agent-glaive` | `glaiveai/glaive-function-calling-v2` | high-volume tool calls **with real tool responses**, so the model sees cause and effect |
| `math-gsm8k` | `openai/gsm8k` | short clean CoT, `<<..=..>>` markers stripped |
| `math-instruct` | `TIGER-Lab/MathInstruct` | short CoT / program-of-thought across maths |
| `code-magicoder` | `ise-uiuc/Magicoder-OSS-Instruct-75K` | short English programming tasks; prose comments are hoisted into `### Thinking:` and the answer keeps clean code |
| `general` | `databricks/databricks-dolly-15k` | no-tool answers: the negative class |
| `seed` | `data/seed.jsonl` | hand-written English anchors |

Every row passes `canon.is_english()` (accented-letter density + stopword
profile) and is dropped if it fails, so the mix is English by construction and
the build prints the reject count per source.

### Why reasoning rows are filtered, not clipped

Truncating a chain-of-thought mid-way teaches the model to ramble and never
conclude. Reasoning sources keep only rows whose **whole** trace fits the
sequence budget (`max_tokens` overall, plus per-field char caps), and a solution
that does not fit is dropped rather than cut mid-function.
Agentic rows are the opposite: they are multi-turn, so they are sliced at turn
boundaries and every slice ends on a complete assistant turn.

This is why `open-thoughts/OpenThoughts-114k` is not in the mix: its assistant
traces have a median of 24.4k chars (p10 9.5k) and **0 of 120 sampled rows fit a
3k-char budget**, so at 1024 tokens it can only be used by clipping the chain of
thought — the exact failure mode above.

## Canonical markup instead of new special tokens

The tokenizer is a fixed 32k byte-level BPE with `tie_word_embeddings=True`.
Adding `<|think|>`-style tokens forces `resize_token_embeddings`, which creates
random embedding rows that **LoRA cannot train** — they would stay noise. So
both sources are normalised to plain-text markers the existing vocab already
encodes, and every foreign control token is stripped:

```
### System:   ### User:   ### Thinking:   ### Response:
### Tool call:   ### Tool result:
```

Only assistant spans are supervised; prompts and tool results are masked
with `-100`.

## Files

```
canon.py            markup, sanitizer, English gate, tool-call serialisation
prepare_data.py     token-budgeted mix -> data/mix_v4.{jsonl,train,val}
train.py            LoRA trainer (CPU or CUDA, fp16/bf16 auto, resume, snapshots)
eval_run.py         per-source loss delta + 4 capability probes
kaggle/run_kaggle.py  one-shot Kaggle GPU driver (clone -> data -> train -> eval -> zip)
data/seed.jsonl     24 short English anchors
```

## Local (CPU)

```bash
python prepare_data.py --profile cpu --target-tokens 400000 --only agent-deepseek,math-gsm8k,general
python train.py --data data/mix.train.jsonl --max-seq 384 --epochs 3
```

## Kaggle (T4)

`kaggle/run_kaggle.py` is pushed as a script kernel (GPU T4, internet on). It
clones this repo, so the kernel always runs committed code:

```bash
git clone https://github.com/Exo-del/supra-50m-lora.git
python kaggle/run_kaggle.py          # 6M unique tokens, 3 epochs, 1024 ctx
```

Config: `--target-tokens 6000000 --slice-turns 6 --epochs 3 --batch-size 4
--grad-accum 8 --max-seq 1024 --r 16 --alpha 32 --save-steps 500`.

## Measured results

CPU, 42 rows, 36 steps, val 3.42 -> 2.97 (LoRA r=16, 1.79M trainable / 53.58M),
v1-v3 recipe, kept for reference.

T4, 891 rows, 1024 ctx, batch 4 x accum 8, 98 steps at 2.8 s/step (~5 min),
val 2.16 -> 1.79 monotonic — that run still contained the Portuguese slice:

| Source | base loss | +LoRA | reduction |
|---|---|---|---|
| agent-tool | 3.04 | 1.37 | 55% |
| seed (en) | 4.06 | 2.64 | 35% |
| pt-br-reasoning | 4.07 | 3.34 | 18% |

Qualitatively the agent slice produced a genuinely new capability — the adapter
emits a function-call shape that the base never produces. That is why v4 keeps
agentic data as the largest block.

### v4 → v4.2 (English-only)

Three T4 runs, full detail in [`docs/runs.md`](docs/runs.md):

| run | unique tokens | wall | val | agentic share | probes |
|---|---|---|---|---|---|
| v4 | 3.28M | 39 min | 1.718 | deepseek **2%** | 2/4 |
| v4.1 | 4.11M | 45 min | 1.698 | deepseek 1%, code 20% | 2/4 |
| v4.2 | 4.67M | 52 min | 1.728 | deepseek **13%**, code 21% | 3/5 |

v4.2 per-source loss reduction (12 rows/source):

| source | base | +LoRA | reduction |
|---|---|---|---|
| `math-gsm8k` | 3.204 | 1.114 | 65% |
| `code-magicoder` | 2.810 | 1.252 | 55% |
| `math-instruct` | 3.238 | 1.463 | 55% |
| `agent-deepseek` (multi-turn) | 3.207 | 1.551 | 52% |
| `agent-glaive` | 2.986 | 0.373 | 88% (recall, not capability) |
| `general` | 3.523 | 2.234 | 37% |

Probes: `no-tool` and **`tool-follow`** pass; `tool-call`, `math`, `code` fail.
`tool-follow` is the first multi-turn behaviour demonstrated: given a tool
result, the adapter answers from the payload instead of re-calling the tool.

The three runs exist because each log exposed a data bug the previous one hid:
OpenThoughts traces never fit the context, the agent rows are conversation
prefixes (dedupe-by-id kept only the one-turn prefix), and one GSM8K question
contained a `U+2028` that broke every line-oriented reader. All three are
regression-tested in `tests/test_parsers.py`.

## License notes

Base weights Apache-2.0 (SupraLabs). Agent set see its card, Magicoder MIT,
GSM8K MIT, MathInstruct MIT, Dolly MIT, Glaive Apache-2.0.
Code here is provided as-is.
