"""Where the app keeps user data (voices, pronunciation dictionary)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    """%LOCALAPPDATA%\\ai-reader on Windows, ~/.local/share/ai-reader elsewhere. Override with AI_READER_HOME."""
    override = os.environ.get("AI_READER_HOME")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "ai-reader"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "ai-reader"
    base.mkdir(parents=True, exist_ok=True)
    return base


def models_dir() -> Path:
    """Downloaded models: <project>/models, shared by the Windows and WSL installs. Override with AI_READER_MODELS."""
    override = os.environ.get("AI_READER_MODELS")
    path = Path(override) if override else Path(__file__).resolve().parents[1] / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def use_models_dir() -> Path:
    """Point every model cache at models_dir(). Call before huggingface_hub, torch or TTS are imported.

    Hugging Face (Piper, MMS, F5-TTS, Whisper, RUAccent, XTTS weights) -> models/huggingface,
    Silero (torch.hub) -> models/torch, coqui XTTS -> models/tts. Caches the user set explicitly win.
    """
    root = models_dir()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    os.environ.setdefault("TTS_HOME", str(root / "tts"))
    return root


def voices_dir() -> Path:
    path = data_dir() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def silero_voices_dir() -> Path:
    """Random Silero voices the user generated and kept."""
    path = data_dir() / "silero_voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dictionary_path() -> Path:
    return data_dir() / "pronunciation.json"
