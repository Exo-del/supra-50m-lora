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

| Prompt | Base | +LoRA |
|---|---|---|
| "What is a language model?" | incoherent list fragments | coherent definition |
| PT-BR question | English junk | still junk |
| "Update my shipping address…" | shell noise | still junk |

English prose style transfers; PT-BR and tool-call grammar need orders of
magnitude more tokens than 6 or 12 rows. That is the point of the Colab run.

## License notes

Base weights Apache-2.0 (SupraLabs), PT-BR set MIT, agent set see its card.
Code here is provided as-is.
