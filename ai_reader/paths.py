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

    Hugging Face (F5-TTS, Whisper, RUAccent) -> models/huggingface,
    Silero (torch.hub) -> models/torch. Caches the user set explicitly win.
    """
    root = models_dir()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
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


def hf_cached(repo: str) -> bool:
    """True when a Hugging Face model repo is already in the local cache (nothing to download)."""
    from huggingface_hub import scan_cache_dir

    try:
        return any(r.repo_id == repo and r.size_on_disk > 0 for r in scan_cache_dir().repos)
    except Exception:
        return False


def download_note(repo: str, size: str) -> str:
    """For a log line: tells that a model will be downloaded, but only when it really isn't on disk yet."""
    return "" if hf_cached(repo) else f", скачивается {size}"
