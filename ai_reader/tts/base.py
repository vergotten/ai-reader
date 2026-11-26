from __future__ import annotations

import gc
import importlib.util
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np

Log = Callable[[str], None]


def resolve_device(preference: str) -> str:
    """'auto' | 'cpu' | 'cuda' -> 'cpu' | 'cuda'."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if preference == "cuda":
            raise RuntimeError("PyTorch не установлен — CUDA недоступна") from None
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError("CUDA недоступна: PyTorch не видит видеокарту")
    return "cpu"


def free_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@dataclass(frozen=True)
class Style:
    """How to speak. Each engine uses the controls it lists in `style_controls` and ignores the rest."""

    pitch: int = 0  # percent, about -30…+30 (Silero)
    expressiveness: float = 0.5  # 0 = even and stable, 1 = lively but less predictable (Piper, XTTS)
    quality: int = 32  # diffusion steps: more = cleaner and slower (F5-TTS)


class TTSEngine(ABC):
    id: str = ""
    name: str = ""
    description: str = ""
    languages: tuple[str, ...] = ("ru",)
    supports_cloning: bool = False
    needs_reference_text: bool = False  # cloning also needs the transcript of the reference clip
    stress_marks: bool = False  # understands "+" before a stressed vowel; otherwise the marks are removed
    builtin_stress: bool = False  # puts stresses itself with RUAccent (no need to do it in the pipeline)
    style_controls: tuple[str, ...] = ()  # names of the Style fields this engine reacts to
    max_chars: int = 400
    min_chunk_chars: int = 0  # shorter sentences are joined to the next one (cloning models garble «Да.»)
    requirements: tuple[str, ...] = ()  # importable module names
    extra: str = ""  # pyproject extra that installs the requirements
    license_notice: str | None = None  # shown once; user must accept before first load

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loaded = False
        self.device: str | None = None
        self.sample_rate = 24_000

    @classmethod
    def missing_requirements(cls) -> list[str]:
        return [m for m in cls.requirements if importlib.util.find_spec(m) is None]

    @classmethod
    def install_hint(cls) -> str:
        return f"uv sync --extra {cls.extra}" if cls.extra else ""

    def target_device(self, preference: str) -> str:
        return resolve_device(preference)

    @abstractmethod
    def voices(self, language: str) -> list[tuple[str, str]]:
        """List of (voice_id, label) for a text language."""

    @abstractmethod
    def _load(self, device: str, log: Log) -> None: ...

    def _unload(self) -> None:
        pass

    def prepare(self, voice: str, log: Log) -> None:
        """Called before synthesis with a given voice (e.g. to download a per-voice model)."""

    def requires_stress(self, voice: str) -> bool:
        """True if this voice only sounds right with a stress mark in every word (so RUAccent must run)."""
        return False

    def forget_voices(self) -> None:
        """Drop per-voice caches after the user's voice library changed (e.g. a voice was retrained)."""

    @abstractmethod
    def _synthesize(
        self,
        text: str,
        voice: str,
        speed: float,
        language: str,
        speaker_wav: str | None,
        speaker_text: str | None,
        style: Style,
    ) -> np.ndarray:
        """Return mono float32 audio at self.sample_rate."""

    def load(self, device: str, log: Log) -> None:
        with self._lock:
            self._load(device, log)
            self.loaded = True
            self.device = device

    def unload(self) -> None:
        with self._lock:
            self._unload()
            self.loaded = False
            self.device = None
        free_torch_memory()

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        language: str = "ru",
        speaker_wav: str | None = None,
        speaker_text: str | None = None,
        style: Style | None = None,
    ) -> np.ndarray:
        with self._lock:
            audio = self._synthesize(text, voice, speed, language, speaker_wav, speaker_text, style or Style())
        return np.asarray(audio, dtype=np.float32).reshape(-1)


def to_numpy(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)
