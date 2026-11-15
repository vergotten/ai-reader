from __future__ import annotations

import numpy as np

from .base import Log, TTSEngine

REPO = "rhasspy/piper-voices"


class PiperEngine(TTSEngine):
    id = "piper"
    name = "Piper — rhasspy/piper-voices (HF)"
    description = (
        "Лёгкие VITS-голоса в формате ONNX с Hugging Face. Очень быстро работает на CPU, "
        "каждый голос ~60 МБ. Можно вписать имя любого голоса из rhasspy/piper-voices, "
        "например ru_RU-irina-medium. Лицензия MIT/зависит от голоса."
    )
    languages = ("ru", "uk", "en")
    style_controls = ("expressiveness",)
    max_chars = 800
    requirements = ("piper", "huggingface_hub", "onnxruntime")
    extra = "piper"

    VOICES = {
        "ru": [
            ("ru_RU-irina-medium", "Ирина (жен.)"),
            ("ru_RU-denis-medium", "Денис (муж.)"),
            ("ru_RU-dmitri-medium", "Дмитрий (муж.)"),
            ("ru_RU-ruslan-medium", "Руслан (муж.)"),
        ],
        "uk": [("uk_UA-ukrainian_tts-medium", "Ukrainian TTS")],
        "en": [
            ("en_US-lessac-medium", "Lessac (US)"),
            ("en_US-ryan-high", "Ryan (US)"),
            ("en_GB-alan-medium", "Alan (GB)"),
        ],
    }

    def __init__(self) -> None:
        super().__init__()
        self.sample_rate = 22_050
        self._voices: dict[str, object] = {}

    def voices(self, language: str) -> list[tuple[str, str]]:
        return list(self.VOICES.get(language, []))

    def target_device(self, preference: str) -> str:
        if preference == "cpu":
            return "cpu"
        try:
            import onnxruntime

            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                return "cuda"
        except ImportError:
            pass
        return "cpu"  # Piper is fast enough on CPU; GPU needs onnxruntime-gpu

    def _load(self, device: str, log: Log) -> None:
        import piper  # noqa: F401  — fail early if missing

    def _unload(self) -> None:
        self._voices.clear()

    @staticmethod
    def _repo_path(voice: str) -> str:
        # ru_RU-irina-medium -> ru/ru_RU/irina/medium/ru_RU-irina-medium
        lang_full, rest = voice.split("-", 1)
        name, quality = rest.rsplit("-", 1)
        return f"{lang_full.split('_')[0]}/{lang_full}/{name}/{quality}/{voice}"

    def prepare(self, voice: str, log: Log) -> None:
        with self._lock:
            if voice in self._voices:
                self.sample_rate = self._voices[voice].config.sample_rate
                return
            from huggingface_hub import hf_hub_download
            from piper import PiperVoice

            path = self._repo_path(voice)
            log(f"Piper: загрузка голоса {voice} с Hugging Face…")
            model = hf_hub_download(REPO, path + ".onnx")
            config = hf_hub_download(REPO, path + ".onnx.json")
            loaded = PiperVoice.load(model, config_path=config, use_cuda=self.device == "cuda")
            self._voices[voice] = loaded
            self.sample_rate = loaded.config.sample_rate

    def _synthesize(self, text, voice, speed, language, speaker_wav, speaker_text, style) -> np.ndarray:
        pv = self._voices[voice]
        length_scale = 1.0 / max(speed, 0.1)
        try:
            from piper import SynthesisConfig  # piper-tts >= 1.3
        except ImportError:
            raw = b"".join(pv.synthesize_stream_raw(text, length_scale=length_scale))
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        parts = []
        config = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=0.3 + 0.74 * style.expressiveness,  # 0.667 (Piper's default) at 0.5
            noise_w_scale=0.4 + 0.8 * style.expressiveness,  # 0.8 at 0.5
        )
        for chunk in pv.synthesize(text, syn_config=config):
            audio = getattr(chunk, "audio_float_array", None)
            if audio is None:
                audio = chunk.audio_int16_array.astype(np.float32) / 32768.0
            parts.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
