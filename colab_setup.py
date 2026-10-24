"""One-shot Colab/T4 setup for the Supra-50M LoRA run.

Run this in a Colab cell:
    !python colab_setup.py

It installs deps, pulls the base model, builds the mixed SFT set at the
colab profile (1024 ctx, light clipping, turn slicing) and trains a LoRA
adapter. Checkpoints land in /content/lora_out; if Drive is mounted they
are copied there too so a disconnect does not lose the run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# Must be set before CUDA initialises: reduces allocator fragmentation when
# the LM-head logits (vocab x seq x batch) fragment the 16 GB T4.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
OUT = Path("/content/lora_out")
DRIVE = Path("/content/drive/MyDrive/supra50m_lora")

EPOCHS = 4
BATCH = 4       # LM head at vocab 32000 is the memory driver, not the model
ACCUM = 8       # keeps the effective batch at 32
SEQ = 1024
AGENT_ROWS = 1500
PT_ROWS = 58
SLICE = 4


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}")
    r = subprocess.run(cmd, shell=True, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"command failed ({r.returncode}): {cmd}")


def main() -> None:
    import torch

    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"gpu: {name} | {vram:.1f} GB | bf16 {torch.cuda.is_bf16_supported()}")

    sh("pip install -q -U peft datasets accelerate")

    # PEFT 0.21 probes torchao for every Linear target and *raises* if the
    # installed torchao is older than 0.16 (Colab ships 0.10). Removing it
    # makes PEFT take the plain-LoRA path instead of aborting.
    import importlib.util

    if importlib.util.find_spec("torchao") is not None:
        sh("pip uninstall -y -q torchao")

    if not (ROOT / "model" / "model.safetensors").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="SupraLabs/Supra-50M-Base",
            local_dir=ROOT / "model",
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*"],
        )

    sh(
        f"python prepare_data.py --profile colab --slice-turns {SLICE} "
        f"--agent {AGENT_ROWS} --pt {PT_ROWS} --pt-repeat 2 --seed-repeat 2 "
        f"--out data/mix_colab.jsonl"
    )

    sh(
        f"python train.py --data data/mix_colab.jsonl "
        f"--epochs {EPOCHS} --batch-size {BATCH} --grad-accum {ACCUM} "
        f"--lr 5e-4 --max-seq {SEQ} --r 16 --alpha 32 --grad-checkpoint "
        f"--ckpt-every 1 --out {OUT}"
    )

    if DRIVE.parent.exists():
        sh(f"mkdir -p {DRIVE} && cp -r {OUT}/* {DRIVE}/")
        print(f"[+] copied adapter to {DRIVE}")
    else:
        print("[i] Drive not mounted; adapter stays in /content/lora_out")

    print("\nDone. Load with:")
    print("  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base, '{OUT}')")


if __name__ == "__main__":
    main()
