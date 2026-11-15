"""Fine-tune a Russian F5-TTS checkpoint on one voice. Runs as a separate process:

    python -m ai_reader.voices.f5_train --voice-dir <dir> --base misha24-v2 --epochs 60

Checkpoints go to <voice>/train/ (training resumes from there if it already has model_last.pt).
At the end, or with --export-only, the EMA weights are saved as <voice>/model.pt (fp16) and voice.json is updated.
Progress lines for the GUI start with "@@".
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

TARGET_SAMPLE_RATE = 24_000
HOP_LENGTH = 256
MEL_KWARGS = dict(
    n_fft=1024,
    hop_length=HOP_LENGTH,
    win_length=1024,
    n_mel_channels=100,
    target_sample_rate=TARGET_SAMPLE_RATE,
    mel_spec_type="vocos",
)
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)  # F5TTS_v1_Base
TINY_CFG = dict(dim=64, depth=2, heads=2, ff_mult=2, text_dim=32, conv_layers=1)  # --debug-tiny smoke test only


def say(*parts) -> None:
    print("@@", *parts, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI Reader: дообучение F5-TTS на своём голосе")
    p.add_argument("--voice-dir", required=True)
    p.add_argument("--base", default="misha24-v2", help="id базовой модели из F5Engine.MODELS")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--batch-frames", type=int, default=3200, help="кадров мел-спектра на батч (3200 ≈ 34 с аудио)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--bnb", action="store_true", help="8-битный Adam (bitsandbytes): меньше видеопамяти")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--export-only", action="store_true", help="только сохранить голос из последней контрольной точки")
    p.add_argument("--debug-tiny", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def base_files(base: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download

    from ai_reader.tts.f5 import F5Engine

    repo, ckpt, vocab, label = F5Engine.MODELS[base]
    say("status", f"Загрузка базовой модели «{label}» (если её ещё нет в кэше)…")
    return hf_hub_download(repo, ckpt), hf_hub_download(repo, vocab)


def link_or_copy(src: str, dst: Path) -> None:
    if dst.exists():
        return
    src = os.path.realpath(src)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_dataset(voice):
    from datasets import Dataset as HFDataset
    from f5_tts.model.dataset import CustomDataset
    from f5_tts.model.utils import convert_char_to_pinyin

    clips = [c for c in voice.enabled_clips() if 0.5 <= c.duration <= 30]
    if len(clips) < 5:
        raise SystemExit(f"Слишком мало фрагментов для обучения: {len(clips)} (нужно хотя бы 5, лучше от 100)")
    texts = [c.text.strip() for c in clips]
    data = {
        "audio_path": [str(voice.path(c.file)) for c in clips],
        "text": convert_char_to_pinyin(texts, polyphone=True),
        "duration": [float(c.duration) for c in clips],
    }
    hf = HFDataset.from_dict(data)
    return CustomDataset(hf, durations=data["duration"], **MEL_KWARGS), clips


def export_model(voice, ckpt_dir: Path) -> bool:
    import torch

    last = ckpt_dir / "model_last.pt"
    if not last.is_file():
        say("status", "Нет контрольной точки model_last.pt — экспортировать нечего.")
        return False
    say("status", "Сохранение голоса…")
    ckpt = torch.load(last, map_location="cpu", weights_only=True)
    ema = {
        k: (v.half() if torch.is_floating_point(v) else v)
        for k, v in ckpt["ema_model_state_dict"].items()
    }
    tmp = voice.dir / "model.pt.tmp"
    torch.save({"ema_model_state_dict": ema}, tmp)
    os.replace(tmp, voice.dir / "model.pt")
    voice.model = "model.pt"
    voice.trained_updates = int(ckpt.get("update", 0))
    voice.trained_seconds = round(voice.total_seconds(), 1)
    voice.save()
    say("exported", voice.trained_updates)
    return True


def main(argv=None) -> int:
    args = parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    from ai_reader.paths import use_models_dir
    from ai_reader.voices.library import Voice

    use_models_dir()  # same caches as the app when run by hand
    voice = Voice.load(Path(args.voice_dir))
    ckpt_dir = voice.train_dir / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        return 0 if export_model(voice, ckpt_dir) else 1

    import torch
    from f5_tts.model import CFM, DiT, Trainer
    from f5_tts.model.utils import get_tokenizer

    if args.debug_tiny:
        vocab_path = str(voice.train_dir / "vocab.txt")
        chars = sorted({ch for c in voice.enabled_clips() for ch in c.text} | {" "})
        Path(vocab_path).write_text("\n".join(chars) + "\n", encoding="utf-8")
        cfg = TINY_CFG
    else:
        marker = voice.train_dir / "base_model.txt"
        if (ckpt_dir / "model_last.pt").exists() and marker.is_file() and marker.read_text().strip() != args.base:
            raise SystemExit(
                f"Обучение этого голоса начато с базовой модели «{marker.read_text().strip()}». "
                "Выберите её же или нажмите «Удалить файлы обучения», чтобы начать заново."
            )
        marker.write_text(args.base)
        voice.base_model = args.base
        voice.save()
        ckpt_path, vocab_path = base_files(args.base)
        if not (ckpt_dir / "model_last.pt").exists():
            link_or_copy(ckpt_path, ckpt_dir / ("pretrained_" + os.path.basename(ckpt_path)))
        cfg = MODEL_CFG

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    dataset, clips = build_dataset(voice)
    seconds = sum(c.duration for c in clips)

    total_frames = sum(c.duration for c in clips) * TARGET_SAMPLE_RATE / HOP_LENGTH
    batches = max(1, math.ceil(total_frames / args.batch_frames))
    updates = math.ceil(batches / args.grad_accum) * args.epochs
    warmup = max(10, min(int(updates * 0.05), 1000))
    save_every = max(20, updates // 20)
    say("plan", f"clips={len(clips)}", f"minutes={seconds / 60:.1f}", f"updates={updates}", f"epochs={args.epochs}")

    mixed = "no"
    if torch.cuda.is_available():
        major = torch.cuda.get_device_properties(0).major
        mixed = "bf16" if torch.cuda.is_bf16_supported() and major >= 8 else ("fp16" if major >= 7 else "no")
        say("status", f"GPU: {torch.cuda.get_device_name(0)}, точность: {mixed}")
    else:
        say("status", "Видеокарта не найдена — обучение на CPU будет очень долгим.")

    model = CFM(
        transformer=DiT(**cfg, text_num_embeds=vocab_size, mel_dim=MEL_KWARGS["n_mel_channels"]),
        mel_spec_kwargs=MEL_KWARGS,
        vocab_char_map=vocab_char_map,
    )
    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=warmup,
        save_per_updates=10**9,
        keep_last_n_checkpoints=0,
        last_per_updates=save_every,
        checkpoint_path=str(ckpt_dir),
        batch_size_per_gpu=args.batch_frames,
        batch_size_type="frame",
        max_samples=64,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        logger=None,
        bnb_optimizer=args.bnb,
        accelerate_kwargs={"mixed_precision": mixed},
    )
    started = time.time()
    trainer.train(dataset, num_workers=max(1, args.workers), resumable_with_seed=666)  # saves model_last.pt at the end
    say("status", f"Обучение завершено за {(time.time() - started) / 60:.0f} мин.")
    del trainer, model
    return 0 if export_model(voice, ckpt_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
