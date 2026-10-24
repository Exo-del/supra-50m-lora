"""Post-training diagnostic: per-source loss, base vs LoRA, plus samples.

    !python eval_colab.py                       # default adapter /content/lora_out
    !python eval_colab.py --adapter lora_out --data data/mix_colab.jsonl

Why per-source: an aggregate val loss can improve purely from the agent
rows while the PT-BR slice learns nothing. Splitting the loss by source
is the only cheap way to see which capability actually moved.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import canon

BASE = Path("model")


def load(path, dtype, device):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)


def encode(msgs, tok, max_len=1024):
    ids, labels = [], []
    for seg in canon.render(msgs):
        seg_ids = tok(seg["text"], add_special_tokens=False).input_ids
        if seg["eos"]:
            seg_ids = seg_ids + [tok.eos_token_id]
        ids += seg_ids
        labels += seg_ids if seg["train"] else [-100] * len(seg_ids)
    if len(ids) > max_len or all(l == -100 for l in labels):
        return None
    return ids, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="/content/lora_out")
    ap.add_argument("--data", default="data/mix_colab.jsonl")
    ap.add_argument("--per-source", type=int, default=40)
    ap.add_argument("--max-seq", type=int, default=1024)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"

    # Group examples by source.
    buckets: dict[str, list] = defaultdict(list)
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = canon.normalize_record(row)
        enc = msgs and encode(msgs, tok, args.max_seq)
        if enc:
            buckets[row.get("source", "?")].append((row, enc))

    rng = random.Random(0)
    sample = {}
    for src, items in buckets.items():
        rng.shuffle(items)
        sample[src] = items[: args.per_source]

    def measure(model, tag):
        model.eval()
        out = {}
        for src, items in sample.items():
            total, n = 0.0, 0
            bs = 4
            for i in range(0, len(items), bs):
                chunk = items[i : i + bs]
                width = max(len(e[0]) for _, e in chunk)
                ids, labels = [], []
                for _row, (e_ids, e_lab) in chunk:
                    pad = width - len(e_ids)
                    ids.append(e_ids + [tok.pad_token_id] * pad)
                    labels.append(e_lab + [-100] * pad)
                ids = torch.tensor(ids, device=device)
                labels = torch.tensor(labels, device=device)
                with torch.no_grad():
                    total += model(input_ids=ids, labels=labels).loss.item()
                n += 1
            out[src] = total / max(1, n)
        print(f"\n[{tag}] loss by source")
        for src, v in sorted(out.items()):
            print(f"    {src:<20} {v:6.3f}")
        return out

    base = load(BASE, dtype, device)
    before = measure(base, "base")
    del base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    from peft import PeftModel

    model = PeftModel.from_pretrained(load(BASE, dtype, device), args.adapter)
    after = measure(model, f"lora {args.adapter}")

    print("\n[delta] loss reduction")
    for src in sorted(before):
        d = before[src] - after[src]
        pct = 100 * d / before[src]
        print(f"    {src:<20} {before[src]:6.3f} -> {after[src]:6.3f}   {d:+.3f} ({pct:+.0f}%)")

    # --- samples
    probes = [
        ("en-question", canon.M["user"] + "What is a language model?\n" + canon.M["response"]),
        ("pt-question", canon.M["user"] + "Explique o que é inflação em português.\n" + canon.M["response"]),
        (
            "tool-call",
            canon.M["system"]
            + "You are a customer service agent.\n"
            + canon.M["user"]
            + "Update my shipping address to 42 Main St.\n"
            + canon.M["response"],
        ),
    ]

    def gen(model, prompt, n=50):
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids,
                max_new_tokens=n,
                do_sample=True,
                temperature=0.5,
                top_k=25,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip().replace("\n", " ")

    print("\n[samples] adapter only (base comparison needs a reload)")
    for name, prompt in probes:
        print(f"\n  {name}: {gen(model, prompt)[:220]}")


if __name__ == "__main__":
    main()
