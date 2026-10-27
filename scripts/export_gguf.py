"""Merge a LoRA adapter into Supra-50M and export GGUF.

    python scripts/export_gguf.py --adapter runs/v4/lora_out_v4 --quant q8_0

Why q8_0 by default: llama.cpp's convert_hf_to_gguf.py accepts f32/f16/bf16/
q8_0/auto for --outtype; k-quants (q4_k_m) need a separately built
llama-quantize binary. q8_0 is the zero-build fallback and is exact enough
for a 50M model (~52 MB).

Requires: peft, transformers, and llama.cpp checked out with
convert_hf_to_gguf.py available at $LLAMA_CPP or ~/.unsloth/llama.cpp.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def sh(cmd: str, cwd: pathlib.Path | None = None) -> None:
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(f"[!] failed ({r.returncode}): {cmd}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=pathlib.Path)
    ap.add_argument("--base", type=pathlib.Path, default=ROOT / "model")
    ap.add_argument("--merged", type=pathlib.Path, default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--quant", default="q8_0", choices=["f16", "bf16", "q8_0", "f32"])
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP", str(pathlib.Path.home() / ".unsloth/llama.cpp")))
    args = ap.parse_args()

    merged = args.merged or args.adapter.parent / (args.adapter.name + "-merged")
    gguf = args.out or args.adapter.parent / "supra50m-v4-q8_0.gguf"

    if not (merged / "config.json").exists():
        print(f"[*] merging {args.adapter} into {args.base} -> {merged}")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained(str(args.base), torch_dtype="auto")
        tok = AutoTokenizer.from_pretrained(str(args.base))
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model = model.merge_and_unload()
        merged.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(merged, safe_serialization=True)
        tok.save_pretrained(merged)
        print(f"[+] merged -> {merged}")
    else:
        print(f"[*] merged model already present: {merged}")

    conv = pathlib.Path(args.llama_cpp) / "convert_hf_to_gguf.py"
    if not conv.exists():
        raise SystemExit(f"[!] {conv} not found; set LLAMA_CPP=/path/to/llama.cpp")

    sh(f'python "{conv}" "{merged}" --outfile "{gguf}" --outtype {args.quant}')
    print(f"[+] {gguf} ({gguf.stat().st_size / 1e6:.1f} MB)")

    # sanity: the chat template must be embedded, the loader needs it
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf))
        template = reader.get_field("tokenizer.chat_template")
        print(f"[*] chat template embedded: {bool(template)}")
        if not template:
            print("[!] no chat template in GGUF: pass --chat-template at inference")
    except Exception as exc:
        print(f"[i] gguf python module unavailable ({exc}); skipped verification")

    print("\nRun:")
    print(f"  llama-cli -m {gguf} --chat-template chatml -p '### User:\\nWhat is 2+2?\\n### Response:\\n'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
