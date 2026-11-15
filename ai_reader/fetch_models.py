"""Download models into the project's models/ folder ahead of time, e.g. before building the installer.

    python -m ai_reader.fetch_models lite    Silero v4 only (~40 MB)
    python -m ai_reader.fetch_models base    Silero, RUAccent, Piper voices (~1 GB)
    python -m ai_reader.fetch_models clone   base + F5-TTS and Whisper for own voices (~4.5 GB)
    python -m ai_reader.fetch_models all     clone + XTTS-v2 (~6.5 GB)
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable

SILERO_MODELS = ("v4_ru", "v5_6_ru")
PIPER_LANGUAGE = "ru"
F5_MODEL = "misha24-v2"
VOCOS_REPO = "charactr/vocos-mel-24khz"
WHISPER_PATTERNS = ["*.json", "*.txt", "*.safetensors"]
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def _silero(models: tuple[str, ...] = SILERO_MODELS) -> None:
    import torch

    for model_id in models:
        torch.hub.load("snakers4/silero-models", "silero_tts", language="ru", speaker=model_id, trust_repo=True)


def _silero_default() -> None:
    from .tts.silero import DEFAULT_MODEL

    _silero((DEFAULT_MODEL,))


def _ruaccent() -> None:
    from .stress import stresser

    stresser.load(print)


def _piper() -> None:
    from huggingface_hub import hf_hub_download

    from .tts.piper import REPO, PiperEngine

    for voice, _label in PiperEngine.VOICES[PIPER_LANGUAGE]:
        path = PiperEngine._repo_path(voice)
        for suffix in (".onnx", ".onnx.json"):
            hf_hub_download(REPO, path + suffix)


def _f5() -> None:
    from huggingface_hub import hf_hub_download, snapshot_download

    from .tts.f5 import F5Engine

    repo, checkpoint, vocab, _label = F5Engine.MODELS[F5_MODEL]
    hf_hub_download(repo, checkpoint)
    hf_hub_download(repo, vocab)
    snapshot_download(VOCOS_REPO)


def _whisper() -> None:
    from huggingface_hub import snapshot_download

    from .voices.dataset import ASR_MODEL

    snapshot_download(ASR_MODEL, allow_patterns=WHISPER_PATTERNS)


def _xtts() -> None:
    os.environ["COQUI_TOS_AGREED"] = "1"  # CPML, non-commercial: the same notice the app shows before first use
    from TTS.utils.manage import ModelManager

    ModelManager(progress_bar=True).download_model(XTTS_MODEL)


STEPS: dict[str, Callable[[], None]] = {
    "silero-default": _silero_default, "silero": _silero, "ruaccent": _ruaccent, "piper": _piper, "f5": _f5, "whisper": _whisper, "xtts": _xtts,
}
SETS = {
    "lite": ("silero-default",),
    "base": ("silero", "ruaccent", "piper"),
    "clone": ("silero", "ruaccent", "piper", "f5", "whisper"),
    "all": ("silero", "ruaccent", "piper", "f5", "whisper", "xtts"),
}


def main(argv: list[str] | None = None) -> int:
    from .paths import use_models_dir

    args = sys.argv[1:] if argv is None else argv
    chosen = args[0] if args else "base"
    if chosen not in SETS:
        print(__doc__)
        return 2
    print(f"Models folder: {use_models_dir()}")
    failed = []
    for name in SETS[chosen]:
        print(f"== {name}")
        try:
            STEPS[name]()
        except Exception as error:  # noqa: BLE001 — one missing engine must not stop the other downloads
            failed.append(name)
            print(f"!! {name}: {error}")
    print("Done." if not failed else f"Done, but failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
