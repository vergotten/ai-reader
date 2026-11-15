from __future__ import annotations

import numpy as np

from .base import Log, TTSEngine, to_numpy


class MMSEngine(TTSEngine):
    id = "mms"
    name = "Meta MMS-TTS — facebook/mms-tts-rus (HF)"
    description = (
        "Модели Massively Multilingual Speech через 🤗 transformers (~150 МБ на язык). "
        "Один голос на язык, звучит проще Silero, зато полностью с Hugging Face. Лицензия CC BY-NC 4.0."
    )
    languages = ("ru", "uk", "en")
    max_chars = 300
    requirements = ("torch", "transformers")
    extra = "mms"

    MODELS = {
        "ru": ("facebook/mms-tts-rus", "MMS русский"),
        "uk": ("facebook/mms-tts-ukr", "MMS українська"),
        "en": ("facebook/mms-tts-eng", "MMS English"),
    }

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 16_000
        self._models: dict[str, tuple[object, object]] = {}
        self._uroman = None

    def voices(self, language: str) -> list[tuple[str, str]]:
        return [self.MODELS[language]] if language in self.MODELS else []

    def _load(self, device: str, log: Log) -> None:
        import transformers  # noqa: F401

    def _unload(self) -> None:
        self._models.clear()

    def prepare(self, voice: str, log: Log) -> None:
        with self._lock:
            if voice in self._models:
                return
            from transformers import AutoTokenizer, VitsModel

            log(f"MMS: загрузка {voice} с Hugging Face…")
            tokenizer = AutoTokenizer.from_pretrained(voice)
            model = VitsModel.from_pretrained(voice).to(self.device).eval()
            self._models[voice] = (tokenizer, model)
            self.sample_rate = model.config.sampling_rate

    def _romanize(self, text: str) -> str:
        if self._uroman is None:
            try:
                import uroman
            except ImportError as e:
                raise RuntimeError("Для этой модели нужен пакет uroman: uv sync --extra mms") from e
            self._uroman = uroman.Uroman()
        return self._uroman.romanize_string(text)

    def _synthesize(self, text, voice, speed, language, speaker_wav, speaker_text, style) -> np.ndarray:
        import torch

        tokenizer, model = self._models[voice]
        if getattr(tokenizer, "is_uroman", False):
            text = self._romanize(text)
        inputs = tokenizer(text, return_tensors="pt").to(self.device)
        if inputs["input_ids"].shape[-1] == 0:
            return np.zeros(0, dtype=np.float32)
        model.speaking_rate = speed
        with torch.no_grad():
            waveform = model(**inputs).waveform[0]
        return to_numpy(waveform)
