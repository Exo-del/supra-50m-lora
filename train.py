"""LoRA fine-tune of SupraLabs/Supra-50M-Base on a mixed SFT set.

Pipeline:
    python prepare_data.py      # build data/mix.jsonl from the HF sets
    python train.py             # LoRA train (CPU, adapter only)

Chat format is canonicalised in canon.py: ### System / ### User /
### Thinking / ### Response / ### Tool call / ### Tool result.
Prompt-side segments are masked (-100); only assistant thinking,
tool calls and answers are supervised.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import canon

REPO_ID = "SupraLabs/Supra-50M-Base"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "data" / "mix.jsonl"
OUT_DIR = ROOT / "lora_out"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_base(dtype):
    """Load the base model, tolerating old `torch_dtype` vs new `dtype` kwarg."""
    from transformers import AutoModelForCausalLM

    kw = dict(low_cpu_mem_usage=True)
    try:
        return AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, **kw)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=dtype, **kw)


def pick_device(requested: str = "auto"):
    """Return (device, dtype, needs_scaler) for CUDA / CPU."""
    import torch

    if requested != "auto":
        if requested == "cpu":
            return torch.device("cpu"), torch.float32, False
        return torch.device(requested), (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16), True

    if not torch.cuda.is_available():
        return torch.device("cpu"), torch.float32, False
    if torch.cuda.is_bf16_supported():
        return torch.device("cuda"), torch.bfloat16, False
    return torch.device("cuda"), torch.float16, True


# ---------------------------------------------------------------- pull
def pull(force: bool = False) -> Path:
    """Fetch the base weights into ./model (no-op when already present)."""
    from huggingface_hub import snapshot_download

    if MODEL_DIR.exists() and any(MODEL_DIR.glob("*.safetensors")) and not force:
        print(f"[*] Using local weights: {MODEL_DIR}")
        return MODEL_DIR

    print(f"[*] Pulling {REPO_ID} -> {MODEL_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=MODEL_DIR,
        force_download=force,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
    )
    return MODEL_DIR


# ---------------------------------------------------------------- data
def load_records(path: Path) -> list[dict]:
    """Read JSONL rows and return canonical message lists."""
    if not path.exists():
        sys.exit(f"[!] Missing dataset: {path}\n    Run: python prepare_data.py")

    rows = []
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"[!] {path}:{ln}: invalid JSON ({exc})")
            msgs = canon.normalize_record(raw)
            if msgs:
                rows.append(msgs)

    if len(rows) < 2:
        sys.exit(f"[!] Need at least 2 usable rows, got {len(rows)}")
    return rows


def encode(rows: list[dict], tok, max_len: int) -> tuple[list[dict], int]:
    """Render canonical messages to ids/labels; mask non-assistant spans."""
    out, dropped = [], 0
    for msgs in rows:
        ids, labels = [], []
        for seg in canon.render(msgs):
            seg_ids = tok(seg["text"], add_special_tokens=False).input_ids
            if seg["eos"]:
                seg_ids = seg_ids + [tok.eos_token_id]
            ids += seg_ids
            labels += seg_ids if seg["train"] else [-100] * len(seg_ids)

        if len(ids) > max_len:
            dropped += 1
            continue
        if all(lab == -100 for lab in labels):
            dropped += 1
            continue
        out.append({"input_ids": ids, "labels": labels})
    return out, dropped


def collate(batch: list[dict], pad_id: int):
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long),
    )


# ---------------------------------------------------------------- train
def train(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device, dtype, needs_scaler = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) - 1)))
    print(f"[*] device: {device} ({str(dtype).replace('torch.', '')})")

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "right"
    model = load_base(dtype)
    model.config.use_cache = False
    model.to(device)

    rows = load_records(args.data)
    random.shuffle(rows)
    split = max(1, int(len(rows) * args.val_split))
    val_rows, train_rows = rows[:split], rows[split:]

    train_set, d_tr = encode(train_rows, tok, args.max_seq)
    val_set, d_va = encode(val_rows, tok, args.max_seq)
    med = sorted(len(x["input_ids"]) for x in train_set)[len(train_set) // 2] if train_set else 0
    print(f"[*] rows: {len(train_set)} train / {len(val_set)} val (dropped {d_tr + d_va} overlong)")
    print(f"[*] median train length: {med} tokens (max_seq {args.max_seq})")
    if not train_set:
        sys.exit("[!] Nothing survived tokenization — raise --max-seq")

    cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[*] LoRA trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = None
    if device.type == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=needs_scaler)
        except TypeError:  # torch < 2.3
            scaler = torch.cuda.amp.GradScaler(enabled=needs_scaler)

    micro_per_epoch = max(1, len(train_set) // args.batch_size)
    total_steps = max(1, micro_per_epoch * args.epochs // args.grad_accum)
    warmup = max(1, int(total_steps * args.warmup))
    print(f"[*] schedule: {total_steps} optimizer steps, {warmup} warmup")

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    order = list(range(len(train_set)))
    step, micro, t0 = 0, 0, time.time()
    model.train()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        running, seen = 0.0, 0

        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch = [train_set[j] for j in order[i : i + args.batch_size]]
            ids, labels, mask = [t.to(device, non_blocking=True) for t in collate(batch, tok.pad_token_id)]

            loss = model(input_ids=ids, labels=labels, attention_mask=mask).loss / args.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += loss.item() * args.grad_accum
            seen += 1
            micro += 1

            if micro % args.grad_accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                if scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1

                if step % args.log_every == 0 or step == 1:
                    speed = (time.time() - t0) / max(1, step)
                    print(
                        f"  ep{epoch} step {step}/{total_steps} "
                        f"loss {running / max(1, seen):.4f} lr {lr_at(step):.2e} "
                        f"{speed:.1f}s/step"
                    )
                    running, seen = 0.0, 0

        model.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for i in range(0, len(val_set), args.batch_size):
                ids, labels, mask = [t.to(device) for t in collate(val_set[i : i + args.batch_size], tok.pad_token_id)]
                vl += model(input_ids=ids, labels=labels, attention_mask=mask).loss.item()
                vn += 1
        model.train()
        print(f"[=] epoch {epoch} | val loss {vl / max(1, vn):.4f}")
        if args.ckpt_every and epoch % args.ckpt_every == 0:
            ckpt = args.out / f"epoch{epoch}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            print(f"[+] checkpoint -> {ckpt}")

    if micro % args.grad_accum:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[+] Adapter saved -> {out_dir}")

    if args.merge:
        merged = model.merge_and_unload()
        merged_dir = out_dir.parent / (out_dir.name + "_merged")
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"[+] Merged model saved -> {merged_dir}")


# ---------------------------------------------------------------- demo
def sample(prompt: str, adapter: Path | None = None, max_new_tokens: int = 80) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device, dtype, _ = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = load_base(dtype).to(device)

    if adapter and adapter.exists():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    ids = tok(canon.M["user"] + prompt + "\n" + canon.M["response"], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_k=25,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0][ids.shape[1] :], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA SFT for Supra-50M-Base")
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-seq", type=int, default=384)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--val-split", type=float, default=0.12)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="adapter output directory")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--ckpt-every", type=int, default=1, help="save an adapter checkpoint every N epochs")
    ap.add_argument("--merge", action="store_true", help="also save merged fp32 weights")
    ap.add_argument("--demo", action="store_true", help="generate a sample after training")
    ap.add_argument("--force", action="store_true", help="re-download base weights")
    args = ap.parse_args()

    pull(force=args.force)
    train(args)

    if args.demo:
        print("\n[*] Demo generation")
        print("-" * 60)
        print(sample("Explain what a language model is.", args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
