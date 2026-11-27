from __future__ import annotations

from .base import Log, TTSEngine, free_torch_memory
from .f5 import F5Engine
from .silero import SileroEngine

# both read '+' stress marks: Silero is the light one (fast, no GPU needed), F5-TTS clones and learns voices
ENGINE_CLASSES: list[type[TTSEngine]] = [SileroEngine, F5Engine]


class EngineManager:
    """Owns engine instances; keeps at most one model in memory."""

    def __init__(self) -> None:
        self._engines = {cls.id: cls() for cls in ENGINE_CLASSES}

    def all(self) -> list[TTSEngine]:
        return list(self._engines.values())

    def get(self, engine_id: str) -> TTSEngine:
        return self._engines[engine_id]

    def ensure_loaded(self, engine: TTSEngine, device_preference: str, voice: str, log: Log) -> None:
        missing = engine.missing_requirements()
        if missing:
            raise RuntimeError(
                f"Для «{engine.name}» не установлены пакеты: {', '.join(missing)}.\n"
                f"Установите: {engine.install_hint()}"
            )
        device = engine.target_device(device_preference)
        if not (engine.loaded and engine.device == device):
            for other in self._engines.values():
                if other.loaded:
                    other.unload()
            log(f"Загрузка «{engine.name}» ({device})…")
            engine.load(device, log)
            log("Модель загружена.")
        engine.prepare(voice, log)


__all__ = ["ENGINE_CLASSES", "EngineManager", "TTSEngine", "free_torch_memory"]
