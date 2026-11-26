from __future__ import annotations

import os

import numpy as np

from ..voices.library import LIB_PREFIX, VoiceLibrary
from .base import Log, TTSEngine, to_numpy

REFERENCE_VOICE = "__reference__"


class XTTSEngine(TTSEngine):
    id = "xtts"
    name = "XTTS-v2 — coqui/XTTS-v2 (HF), клонирование голоса"
    description = (
        "Многоязычная модель (~1,8 ГБ), умеет говорить голосом из короткого образца (6–30 с WAV). "
        "Лучше всего работает на видеокарте. Медленнее Silero, иногда ошибается в ударениях."
    )
    languages = ("ru", "en")
    supports_cloning = True
    style_controls = ("expressiveness",)
    max_chars = 180  # XTTS limit for Russian is ~182 characters per call
    min_chunk_chars = 25
    requirements = ("TTS", "torch")
    extra = "xtts"
    license_notice = (
        "Модель XTTS-v2 распространяется под лицензией Coqui Public Model License (CPML): "
        "только некоммерческое использование. https://coqui.ai/cpml\n\n"
        "Принять условия лицензии и скачать модель?"
    )

    DEFAULT_SPEAKERS = ["Ana Florence", "Claribel Dervla", "Daisy Studious", "Viktor Eka", "Damien Black"]

    def __init__(self) -> None:
        super().__init__()
        self._tts = None
        self._speakers: list[str] = []
        self._latents: dict[tuple, tuple[object, object]] = {}

    def voices(self, language: str) -> list[tuple[str, str]]:
        speakers = self._speakers or self.DEFAULT_SPEAKERS
        mine = [(v.lib_id, f"★ {v.name} (клон по записям)") for v in VoiceLibrary().list() if v.enabled_clips()]
        return mine + [(REFERENCE_VOICE, "Свой голос (эталонный WAV)")] + [(s, s) for s in speakers]

    def _load(self, device: str, log: Log) -> None:
        os.environ["COQUI_TOS_AGREED"] = "1"  # user accepted the CPML in the GUI
        from TTS.api import TTS

        log("XTTS-v2: загрузка модели (первый раз скачивается ~1,8 ГБ)…")
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
        self._speakers = list(self._tts.speakers or [])
        self.sample_rate = self._tts.synthesizer.output_sample_rate

    def forget_voices(self) -> None:
        with self._lock:
            self._latents.clear()

    def _unload(self) -> None:
        self._tts = None
        self._latents.clear()

    def _synthesize(self, text, voice, speed, language, speaker_wav, speaker_text, style) -> np.ndarray:
        import torch

        with torch.no_grad():
            refs = None
            if voice.startswith(LIB_PREFIX):
                v = VoiceLibrary().get(voice)
                refs = v.xtts_references() if v else []
                if not refs:
                    raise RuntimeError("В голосе нет годных фрагментов записи (окно «Мои голоса»)")
            elif voice == REFERENCE_VOICE:
                if not speaker_wav or not os.path.isfile(speaker_wav):
                    raise RuntimeError("Выберите WAV-файл с образцом голоса")
                refs = [speaker_wav]
            model = self._tts.synthesizer.tts_model
            if refs is not None:
                key = tuple((r, os.path.getmtime(r)) for r in refs)
                if key not in self._latents:
                    self._latents[key] = model.get_conditioning_latents(audio_path=refs)
                gpt_latent, speaker_embedding = self._latents[key]
            else:
                if voice not in model.speaker_manager.speakers:
                    raise RuntimeError(f"В XTTS-v2 нет голоса «{voice}» — выберите другой из списка")
                gpt_latent, speaker_embedding = model.speaker_manager.speakers[voice].values()
            temperature = 0.35 + 0.8 * style.expressiveness  # 0.75 (XTTS default) at 0.5
            out = model.inference(
                text, language, gpt_latent, speaker_embedding, speed=speed, temperature=temperature
            )
        return to_numpy(out["wav"])
