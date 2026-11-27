"""Download the models into the models/ folder ahead of time, so the app never downloads anything while you use it.

    python -m ai_reader.fetch_models          everything (~6 GB): what the full installer ships
    python -m ai_reader.fetch_models lite     Silero v4 only (~40 MB): what the lite installer ships

Or double-click download-models.bat in the project folder.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

VOCOS_REPO = "charactr/vocos-mel-24khz"  # the vocoder F5-TTS turns spectrograms into sound with
WHISPER_PATTERNS = ["*.json", "*.txt", "*.safetensors"]


def _silero(models: tuple[str, ...] | None = None) -> None:
    """Silero v4, v5 and v5 CIS (~30 voices with accents); loading through the engine also saves the CIS voice list."""
    from .tts.silero import MODELS, SileroEngine

    engine = SileroEngine()
    engine.device = "cpu"
    for model_id in models or tuple(MODELS):
        engine._ensure_model(model_id, print)


def _silero_default() -> None:
    from .tts.silero import DEFAULT_MODEL

    _silero((DEFAULT_MODEL,))


def _ruaccent() -> None:
    from .stress import stresser

    stresser.load(print)


def _f5() -> None:
    from huggingface_hub import snapshot_download

    from .tts.f5 import F5Engine, base_checkpoint

    for model_id in F5Engine.MODELS:
        base_checkpoint(model_id, print)
    snapshot_download(VOCOS_REPO)


def _whisper() -> None:
    from huggingface_hub import snapshot_download

    from .voices.dataset import ASR_MODEL

    snapshot_download(ASR_MODEL, allow_patterns=WHISPER_PATTERNS)


SETS: dict[str, tuple[Callable[[], None], ...]] = {
    "lite": (_silero_default,),
    "all": (_silero, _ruaccent, _f5, _whisper),
}


def main(argv: list[str] | None = None) -> int:
    from .paths import use_models_dir

    args = sys.argv[1:] if argv is None else argv
    chosen = args[0] if args else "all"
    if chosen not in SETS:
        print(__doc__)
        return 2
    print(f"Папка моделей: {use_models_dir()}")
    failed = []
    for step in SETS[chosen]:
        name = step.__name__.strip("_")
        print(f"== {name}")
        try:
            step()
        except Exception as error:  # noqa: BLE001 — one missing engine must not stop the other downloads
            failed.append(name)
            print(f"!! {name}: {error}")
    print("Готово." if not failed else f"Готово, но не скачались: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
