"""Kaggle driver: build the v4 English mix, LoRA-train Supra-50M, evaluate.

This file is pushed to Kaggle as a *script* kernel (GPU T4, internet on).
It clones the repo, so the code that runs is exactly the code in main.

    Kaggle -> code -> this file (kernelType script, enableGpu true)

Hardware notes
    Kaggle T4: 16 GB, fp16 only (cc 7.5) -> train.py auto-selects fp16 + scaler.
    The LM head (vocab 32000) is the memory driver, not the 50M model, so
    batch 4 x 1024 is the safe default and fit_batch_size() backs off on OOM.
    Script kernels get 12 h; the run below is ~3-4 h at 6M unique tokens
    x 3 epochs, so checkpoints exist (--save-steps 500) but a resume should
    not be needed.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = "https://github.com/Exo-del/supra-50m-lora.git"
BASE_REPO = "SupraLabs/Supra-50M-Base"
WORK = pathlib.Path("/kaggle/working")
SRC = WORK / "repo"
VERSION = "v42"
OUT = WORK / f"lora_out_{VERSION}"
DATA = SRC / "data" / f"mix_{VERSION}.jsonl"

TARGET_TOKENS = 6_000_000
EPOCHS = 3
BATCH = 4
ACCUM = 8
SEQ = 1024
LR = 5e-4
SAVE_STEPS = 500


def sh(cmd: str, cwd: pathlib.Path = SRC, check: bool = True) -> int:
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=str(cwd))
    print(f"  -> rc={r.returncode} in {time.time() - t0:.1f}s", flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"[!] command failed ({r.returncode}): {cmd}")
    return r.returncode


def banner(msg: str) -> None:
    print(f"\n{'=' * 78}\n== {msg}\n{'=' * 78}", flush=True)


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)

    banner("code")
    if SRC.exists():
        shutil.rmtree(SRC, ignore_errors=True)
    sh(f"git clone --depth 1 {REPO} {SRC}", cwd=WORK)
    print(subprocess.run("git log --oneline -1", shell=True, cwd=str(SRC),
                         capture_output=True, text=True).stdout)

    banner("deps")
    sh("pip install -q -U peft datasets accelerate")
    sh("pip uninstall -y -q torchao", check=False)  # PEFT aborts on torchao < 0.16

    import torch

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu {p.name} | {p.total_memory / 1e9:.1f} GB | "
              f"cc {torch.cuda.get_device_capability(0)} | bf16 {torch.cuda.is_bf16_supported()}")

    banner("base weights")
    if not any((SRC / "model").glob("*.safetensors")):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=BASE_REPO,
            local_dir=str(SRC / "model"),
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
        )
    print(sorted(p.name for p in (SRC / "model").iterdir()))

    banner("data: english agentic + reasoning mix")
    sh(
        "python prepare_data.py --profile kaggle "
        f"--target-tokens {TARGET_TOKENS} "
        "--slice-turns 6 --cap-per-source 20000 --val-frac 0.05 "
        "--out data/mix_v42.jsonl"
    )

    banner("train")
    sh(
        f"python train.py --data {DATA.with_suffix('.train.jsonl')} "
        f"--val-data {DATA.with_suffix('.val.jsonl')} "
        f"--epochs {EPOCHS} --batch-size {BATCH} --grad-accum {ACCUM} "
        f"--lr {LR} --max-seq {SEQ} --r 16 --alpha 32 --grad-checkpoint "
        f"--save-steps {SAVE_STEPS} --ckpt-every 1 --out {OUT}"
    )

    banner("eval")
    sh(
        f"python eval_run.py --adapter {OUT} --data {DATA} "
        f"--per-source 30 --samples 2 --max-new-tokens 120",
        check=False,
    )

    banner("package")
    zip_path = WORK / ("supra50m-" + VERSION + "-lora.zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=OUT)
    size = zip_path.stat().st_size / 1e6
    print(f"adapter zip: {zip_path} ({size:.1f} MB)")
    for f in sorted(WORK.glob("*")):
        print(f"  {f.name:<40} {f.stat().st_size / 1e6:.1f} MB")
    print("\nDONE_" + VERSION.upper())
    return 0


if __name__ == "__main__":
    sys.exit(main())
