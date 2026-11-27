"""The copy of models/ that goes into the full installer:

    python installer\\slim_models.py <models> <out> [--fp16]

- Every model is in it once. The Hugging Face cache keeps each file in blobs/ and a link to it in snapshots/;
  the installer would pack both, so the copy keeps only real files in snapshots/ (all the app reads).
- With --fp16 (build-installer.bat -Max) the F5-TTS checkpoints keep only their EMA weights, in fp16. That is
  half the size (ESpeech also drops its training state); the app widens the weights back to fp32 on loading,
  so the voice is the same, and fine-tuning starts from them just as well.

Files that don't change are hard-linked, not copied: the copy costs no disk space for them.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

F5_REPOS = ("models--Misha24-10--F5-TTS_RUSSIAN", "models--ESpeech--ESpeech-TTS-1_RL-V2")
SKIP_DIRS = {"blobs", "xet", ".locks", "checkpoints"}  # duplicates, download leftovers, torch.hub's own checkpoints


def _half(state: dict) -> dict:
    import torch

    return {k: v.half() if torch.is_floating_point(v) else v for k, v in state.items()}


def slim_checkpoint(src: Path, dst: Path) -> None:
    """fp16 EMA weights only, in the same format (.pt or .safetensors) the app and f5_tts read."""
    if dst.suffix == ".safetensors":  # by the name: a cache blob behind a link has no extension
        from safetensors.torch import load_file, save_file

        save_file(_half(load_file(str(src))), str(dst))
        return
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({"ema_model_state_dict": _half(ckpt.get("ema_model_state_dict", ckpt))}, dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # another drive, or no hard links on this file system
        shutil.copy2(src, dst)


def build_copy(models: Path, out: Path, fp16: bool) -> None:
    if out.exists():
        shutil.rmtree(out)
    total = saved = 0
    for root, dirs, files in os.walk(models):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        rel = Path(root).relative_to(models)
        (out / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = (Path(root) / name).resolve()  # a snapshot entry may be a link into blobs/
            dst = out / rel / name
            if fp16 and name.endswith((".pt", ".safetensors")) and any(repo in rel.parts for repo in F5_REPOS):
                size = src.stat().st_size
                print(f"fp16: {name} {size / 2**30:.2f} GB", end=" ", flush=True)
                slim_checkpoint(src, dst)
                saved += size - dst.stat().st_size
                print(f"-> {dst.stat().st_size / 2**30:.2f} GB", flush=True)
            else:
                _link_or_copy(src, dst)
            total += dst.stat().st_size
    print(f"Models for the installer: {total / 2**30:.2f} GB" + (f" (fp16 saved {saved / 2**30:.2f} GB)" if fp16 else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="the models/ copy for the installer")
    parser.add_argument("models", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fp16", action="store_true", help="F5-TTS weights in fp16 (the -Max installer)")
    args = parser.parse_args()
    build_copy(args.models, args.out, args.fp16)
