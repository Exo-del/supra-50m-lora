"""Post-training diagnostic: per-source loss (base vs LoRA) + generation probes.

    python eval_run.py --adapter lora_out_v4 --data data/mix_v4.jsonl

Why per-source: an aggregate val loss can improve purely from one slice.
Splitting the loss by source shows which capability actually moved --
agentic tool use, math/logic, code reasoning, or plain answers.

Probes answer the four questions the mix is supposed to teach:
    no-tool   a plain question must NOT produce a tool call (negative class)
    tool-call must produce "### Tool call:" + a parseable fn({...}) block
    math      must produce "### Thinking:" steps then "### Response:"
    code      same shape, explaining code rather than calling a tool
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import canon

BASE = Path("model")
_CALL_LINE = re.compile(r"^\s*([A-Za-z_][\w.]*)\((\{.*\})\)\s*$", re.M)

PROBES = [
    ("no-tool", canon.M["user"] + "What is a language model?\n" + canon.M["response"]),
    (
        "tool-call",
        canon.M["system"]
        + "You are a customer service agent with access to update_shipping_location(location: str).\n"
        + canon.M["user"]
        + "Please update my shipping address to 42 Main St, Casablanca.\n"
        + canon.M["response"],
    ),
    (
        "math",
        canon.M["user"] + "A shop sells 3 pens for 7 dollars. How much do 12 pens cost?\n" + canon.M["response"],
    ),
    (
        "code",
        canon.M["user"] + "Write a Python function that returns the reverse of a string.\n" + canon.M["response"],
    ),
]


def load(path, dtype, device):
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).to(device)


def encode(msgs, tok, max_len):
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="lora_out_v4")
    ap.add_argument("--data", default="data/mix_v4.jsonl")
    ap.add_argument("--per-source", type=int, default=30)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--skip-base", action="store_true", help="only score the adapter")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "right"

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
        # The LM head dominates memory at vocab 32000: one 1024-token row is
        # 32k x 1024 fp32 logits (~131 MB) before the softmax. On a 7 GB CPU
        # box a batch of 4 overflows, so scale the batch with the device.
        bs = 4 if device.type == "cuda" else 1
        for src, items in sample.items():
            total, n = 0.0, 0
            for i in range(0, len(items), bs):
                chunk = items[i: i + bs]
                width = max(len(e[0]) for _, e in chunk)
                ids, labels = [], []
                for _row, (e_ids, e_lab) in chunk:
                    pad = width - len(e_ids)
                    ids.append(e_ids + [tok.pad_token_id] * pad)
                    labels.append(e_lab + [-100] * pad)
                ids_t = torch.tensor(ids, device=device)
                lab_t = torch.tensor(labels, device=device)
                with torch.no_grad():
                    total += model(input_ids=ids_t, labels=lab_t).loss.item()
                n += 1
            out[src] = total / max(1, n)
        print(f"\n[{tag}] loss by source")
        for src, v in sorted(out.items()):
            print(f"    {src:<18} {v:6.3f}  ({len(sample[src])} rows)")
        return out

    before = {}
    if not args.skip_base:
        base = load(BASE, dtype, device)
        before = measure(base, "base")
        del base
        import gc

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    from peft import PeftModel

    model = PeftModel.from_pretrained(load(BASE, dtype, device), args.adapter)
    after = measure(model, f"lora {args.adapter}")

    if before:
        print("\n[delta] loss reduction")
        for src in sorted(before):
            d = before[src] - after[src]
            print(f"    {src:<18} {before[src]:6.3f} -> {after[src]:6.3f}   {d:+.3f} ({100 * d / before[src]:+.0f}%)")

    def gen(prompt, n):
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        kw = dict(
            max_new_tokens=n,
            do_sample=not args.greedy,
            repetition_penalty=1.2,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if not args.greedy:
            kw.update(temperature=0.5, top_k=25, top_p=0.9)
        with torch.no_grad():
            out = model.generate(ids, **kw)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    print(f"\n[samples] {args.samples}x per probe, up to {args.max_new_tokens} new tokens")
    checks = {}
    for name, prompt in PROBES:
        print(f"\n{'=' * 70}\n{name}")
        hits = 0
        for k in range(args.samples):
            text = gen(prompt, args.max_new_tokens)
            called = canon.M["tool_call"].strip() in text
            if name == "tool-call":
                parsed = bool(_CALL_LINE.search(text))
                ok = called and parsed
                flag = f"tool_call={called} parsed={parsed}"
            elif name == "no-tool":
                ok = not called
                flag = f"tool_call={called}"
            else:
                has_think = canon.M["thinking"].strip() in text
                has_resp = canon.M["response"].strip() in text
                ok = has_resp or has_think
                flag = f"thinking={has_think} response={has_resp}"
            hits += ok
            print(f"  [{k + 1}] {'PASS' if ok else 'fail'} ({flag})")
            for line in text.splitlines()[:6]:
                print(f"      {line[:150]}")
        checks[name] = f"{hits}/{args.samples}"
        print(f"  -> {name}: {hits}/{args.samples}")

    print(f"\n[summary] {checks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
