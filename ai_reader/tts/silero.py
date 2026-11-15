from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from ..paths import models_dir, silero_voices_dir
from .base import Log, Style, TTSEngine, to_numpy

DEFAULT_MODEL = "v4_ru"
CIS_MODEL = "v5_cis_base"
RANDOM_PREFIX = "rnd:"  # a generated voice saved in silero_voices_dir()
_RU_SPEAKERS = [
    ("xenia", "Ксения (жен.)"),
    ("baya", "Бая (жен.)"),
    ("kseniya", "Ксения 2 (жен.)"),
    ("aidar", "Айдар (муж.)"),
    ("eugene", "Евгений (муж.)"),
]


@dataclass(frozen=True)
class _Model:
    label: str
    auto_stress: bool  # the model puts stresses and «ё» itself
    speakers: tuple[tuple[str, str], ...] | None  # None: read from the model after it is loaded


MODELS = {
    "v4_ru": _Model("v4", True, tuple(_RU_SPEAKERS)),
    "v5_6_ru": _Model("v5, новее", True, tuple(_RU_SPEAKERS)),
    # ~30 speakers of CIS languages reading Russian ("ru_" prefix); needs a stress mark in every word
    CIS_MODEL: _Model("v5 CIS", False, None),
}


def _parse_voice(voice: str) -> tuple[str, str, str | None]:
    """voice id -> (model id, speaker, path of a saved random voice)."""
    if voice.startswith(RANDOM_PREFIX):
        return DEFAULT_MODEL, "random", str(silero_voices_dir() / f"{voice[len(RANDOM_PREFIX):]}.pt")
    if ":" in voice:
        model_id, speaker = voice.split(":", 1)
        if model_id in MODELS:
            return model_id, speaker, None
    return DEFAULT_MODEL, voice, None  # ids saved by older versions are plain v4 speakers


class SileroEngine(TTSEngine):
    id = "silero"
    name = "Silero TTS — русский (v4 и v5)"
    description = (
        "Лучшее качество для русского: автоматически ставит ударения и букву «ё», быстро работает даже на CPU. "
        "Модели (~40–100 МБ каждая) скачиваются с GitHub snakers4/silero-models: v4, более новая v5 и "
        "v5 CIS (десятки голосов с разными акцентами, для неё ударения расставляет RUAccent). "
        "Кнопка «🎲» создаёт новый случайный голос. Высота голоса и скорость настраиваются в «Стиле речи». "
        "Ударение вручную — «+» перед гласной: «з+амок». Лицензия v4/v5 — CC BY-NC-SA (некоммерческая), CIS — MIT."
    )
    languages = ("ru",)
    stress_marks = True
    style_controls = ("pitch",)
    max_chars = 500
    requirements = ("torch", "torchaudio", "omegaconf")
    extra = "silero"

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 48_000
        self._model = None
        self._model_id: str | None = None
        self._symbols: set[str] | None = None
        self._percent_prosody = True  # v4 accepts rate="115%"; set to False if a model rejects it

    # ------------------------------------------------------------------ voices

    def voices(self, language: str) -> list[tuple[str, str]]:
        out = list(MODELS[DEFAULT_MODEL].speakers)
        out += [(RANDOM_PREFIX + p.stem, f"🎲 {p.stem}") for p in sorted(silero_voices_dir().glob("*.pt"))]
        for model_id, spec in MODELS.items():
            if model_id == DEFAULT_MODEL:
                continue
            speakers = spec.speakers or self._cached_speakers(model_id)
            if speakers:
                out += [(f"{model_id}:{sid}", f"{label} — {spec.label}") for sid, label in speakers]
            else:
                out.append((f"{model_id}:", f"Ещё ~30 голосов ({spec.label}) — появятся после первой загрузки"))
        return out

    def requires_stress(self, voice: str) -> bool:
        return not MODELS[_parse_voice(voice)[0]].auto_stress

    @staticmethod
    def _speakers_file(model_id: str) -> Path:
        return models_dir() / f"silero_{model_id}_speakers.json"

    def _cached_speakers(self, model_id: str) -> list[tuple[str, str]]:
        try:
            names = json.loads(self._speakers_file(model_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return [(n, n.removeprefix("ru_").replace("_", " ").title()) for n in names]

    def create_random_voice(self, name: str, log: Log) -> str:
        """Generate a new synthetic speaker (v4 only), save it under `name` and return its voice id."""
        import torch

        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .") or "голос"
        path = silero_voices_dir() / f"{stem}.pt"
        with self._lock:
            self._ensure_model(DEFAULT_MODEL, log)
            with torch.no_grad():  # with speaker="random" and no voice_path the model invents a voice
                self._model.apply_tts(text="Новый голос.", speaker="random", sample_rate=self.sample_rate)
            self._model.save_random_voice(str(path))
        return RANDOM_PREFIX + stem

    @staticmethod
    def delete_random_voice(voice: str) -> None:
        if voice.startswith(RANDOM_PREFIX):
            Path(_parse_voice(voice)[2]).unlink(missing_ok=True)

    # ------------------------------------------------------------------ loading

    def _load(self, device: str, log: Log) -> None:
        import torch  # noqa: F401 — fail early if missing; the model itself is loaded per voice in prepare()

    def _unload(self) -> None:
        self._model = None
        self._model_id = None

    def prepare(self, voice: str, log: Log) -> None:
        with self._lock:
            self._ensure_model(_parse_voice(voice)[0], log)

    def _ensure_model(self, model_id: str, log: Log) -> None:
        if model_id == self._model_id:
            return
        import torch

        self._model = None
        self._model_id = None
        log(f"Silero: загрузка модели {model_id} через torch.hub (в первый раз скачивается)…")
        model, _ = torch.hub.load(
            "snakers4/silero-models", "silero_tts", language="ru", speaker=model_id, trust_repo=True
        )
        model.to(torch.device(self.device))
        symbols = getattr(model, "symbols", None)
        self._symbols = set(symbols) if symbols else None
        self._model, self._model_id = model, model_id
        self._percent_prosody = True
        if MODELS[model_id].speakers is None:
            names = [s for s in getattr(model, "speakers", []) if s.startswith("ru_")]
            self._speakers_file(model_id).write_text(json.dumps(names), encoding="utf-8")

    # ------------------------------------------------------------------ synthesis

    def _clean(self, text: str) -> str:
        text = text.replace("—", "–").replace("...", "…")
        if self._symbols:
            text = "".join(c for c in text if c.lower() in self._symbols or c == "+")
        return " ".join(text.split())

    def _synthesize(self, text, voice, speed, language, speaker_wav, speaker_text, style: Style) -> np.ndarray:
        import torch

        model_id, speaker, voice_path = _parse_voice(voice)
        if model_id != self._model_id:
            raise RuntimeError("Модель Silero для этого голоса не загружена")
        text = self._clean(text)
        if not any(c.isalpha() for c in text):
            return np.zeros(0, dtype=np.float32)
        if not speaker:
            speaker = next((sid for sid, _label in self._cached_speakers(model_id)), "")
        kwargs: dict = dict(speaker=speaker, sample_rate=self.sample_rate)
        if voice_path:
            if not Path(voice_path).is_file():
                raise RuntimeError("Файл случайного голоса удалён — выберите другой голос")
            kwargs["voice_path"] = voice_path
        if MODELS[model_id].auto_stress:
            kwargs.update(put_accent=True, put_yo=True)
        # JIT optimization re-profiles for the first several calls (~20 s on GPU) and gains nothing here
        with torch.no_grad(), torch.jit.optimized_execution(False):
            return to_numpy(self._apply(text, speed, style.pitch, kwargs))

    def _apply(self, text: str, speed: float, pitch: int, kwargs: dict):
        attributes = _prosody(speed, pitch, self._percent_prosody)
        if not attributes:
            return self._model.apply_tts(text=text, **kwargs)
        ssml = f"<speak><prosody {attributes}>{escape(text)}</prosody></speak>"
        try:
            return self._model.apply_tts(ssml_text=ssml, **kwargs)
        except AssertionError:
            if not self._percent_prosody:
                raise
            self._percent_prosody = False  # this model only knows the named levels ("fast", "high")
            return self._apply(text, speed, pitch, kwargs)


_RATE_LEVELS = ((0.8, "x-slow"), (0.95, "slow"), (1.05, ""), (1.3, "fast"))
_PITCH_LEVELS = ((-25, "x-low"), (-8, "low"), (8, ""), (25, "high"))


def _level(value: float, levels, top: str) -> str:
    return next((name for limit, name in levels if value < limit or (value == limit and not name)), top)


def _prosody(speed: float, pitch: int, percent: bool) -> str:
    """SSML <prosody> attributes for a speed factor and a pitch shift in percent; '' when both are neutral."""
    if percent:
        rate = f"{round(speed * 100)}%" if abs(speed - 1.0) > 0.02 else ""
        tone = f"{pitch:+d}%" if pitch else ""
    else:
        rate = _level(speed, _RATE_LEVELS, "x-fast")
        tone = _level(pitch, _PITCH_LEVELS, "x-high")
    return " ".join(f'{key}="{value}"' for key, value in (("rate", rate), ("pitch", tone)) if value)
