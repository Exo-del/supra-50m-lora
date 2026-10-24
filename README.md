# supra-50m-lora

LoRA fine-tuning of [`SupraLabs/Supra-50M-Base`](https://huggingface.co/SupraLabs/Supra-50M-Base)
(51.8M params, Llama-style, 32k custom BPE, tied embeddings) on a mixed
SFT set built from two public datasets.

## Why the data pipeline looks like this

| Source | Rows | Problem |
|---|---|---|
| `Davizig10jojo/Kimi-K3-And-DeepSeek-V4-Pro-0813-Distillation-in-PT-BR` | 58 | thinking median 1157 chars, answers 1649 |
| `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory` | 2510 | median 11 turns / 16k chars per row, ~75% duplicate content |

Raw rows are far too long for a 50M model, so `prepare_data.py` clips on
sentence boundaries, filters by score, and slices multi-turn trajectories
into turn-level examples (`--slice-turns`). Dedupe hashes the **whole**
rendered row — hashing a prefix merged distinct PT rows that share a
system prompt.

## Canonical markup instead of new special tokens

The tokenizer is a fixed 32k byte-level BPE with `tie_word_embeddings=True`.
Adding `<|think|>`-style tokens forces `resize_token_embeddings`, which
creates random embedding rows that **LoRA cannot train** — they would stay
noise. So both sources are normalised to plain-text markers the existing
vocab already encodes, and every foreign control token is stripped:

```
### System:   ### User:   ### Thinking:   ### Response:
### Tool call:   ### Tool result:
```

Only assistant spans are supervised; prompts and tool results are masked
with `-100`.

## Files

```
canon.py         markup, sanitizer, tool-call serialisation, renderer
prepare_data.py  sampling, clipping, slicing, mixing -> data/mix.jsonl
train.py         LoRA trainer (CPU or CUDA, fp16/bf16 auto)
colab_setup.py   one-shot Colab/T4 driver
data/seed.jsonl  24 short English anchors
```

## Local (CPU)

```bash
python prepare_data.py --profile cpu --pt 4 --agent 12
python train.py --epochs 6                       # ~4 min, adapter only
```

## Colab (T4)

```bash
pip install -q -U peft datasets accelerate
python colab_setup.py     # 1024 ctx, turn slicing, ~4k example-passes
```

Checkpoints go to `/content/lora_out`, copied to Drive if mounted.

## Measured results

CPU, 42 rows, 36 steps, val 3.42 -> 2.97 (LoRA r=16, 1.79M trainable / 53.58M):
English prose style transfers; PT-BR and tool-call grammar need orders of
magnitude more tokens than 6 or 12 rows.

T4, 891 rows, 1024 ctx, batch 4 x accum 8, 98 steps at 2.8 s/step (~5 min),
val 2.16 -> 1.79 monotonic:

| Source | base loss | +LoRA | reduction |
|---|---|---|---|
| agent-tool | 3.07 | 1.15 | 62% |
| seed (en) | 3.84 | 0.88 | 77% |
| pt-br-reasoning | 4.29 | 3.53 | 18% |

The per-source split is the useful signal: aggregate loss hides the fact
that PT-BR moves ~4x less than the other two. Its text keeps the markup
and some Portuguese function words but no content, which is what 58 source
rows can buy. `eval_colab.py` reproduces this table.

## Reproduce the diagnostic

```bash
python eval_colab.py --adapter /content/lora_out --data data/mix_colab.jsonl
```


## License notes

Base weights Apache-2.0 (SupraLabs), PT-BR set MIT, agent set see its card.
Code here is provided as-is.
