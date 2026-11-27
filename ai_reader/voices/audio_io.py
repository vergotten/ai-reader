"""Decode any voice recording (Telegram/WhatsApp voice messages, phone memos, MP3...) to mono float32."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".mp4", ".webm", ".wma", ".amr", ".3gp")
AUDIO_FILTER = "Аудио (" + " ".join(f"*{e}" for e in AUDIO_EXTENSIONS) + ");;Все файлы (*)"


def collect_audio(paths: list[str]) -> list[str]:
    """Recordings among the given paths; folders are searched with all their subfolders."""
    found: list[str] = []
    for path in map(Path, paths):
        if path.is_dir():
            found += sorted(str(p) for p in path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)
        elif path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            found.append(str(path))
    return list(dict.fromkeys(found))


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Mono float32 at sample_rate. Tries libsndfile (WAV/FLAC/MP3/OGG/Opus), then PyAV, then ffmpeg."""
    errors = []
    for decoder in (_load_soundfile, _load_pyav, _load_ffmpeg):
        try:
            audio, sr = decoder(path, sample_rate)
        except _Unavailable:
            continue
        except Exception as e:  # noqa: BLE001 — try the next decoder
            errors.append(f"{decoder.__name__[6:]}: {e}")
            continue
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            import soxr

            audio = soxr.resample(audio, sr, sample_rate).astype(np.float32)
        return audio
    hint = "" if _has_pyav() or shutil.which("ffmpeg") else " Для M4A/AAC/AMR установите пакет av (uv sync --extra train) или ffmpeg."
    raise RuntimeError(f"Не удалось прочитать {path}: {'; '.join(errors) or 'формат не поддерживается'}.{hint}")


class _Unavailable(Exception):
    pass


def _load_soundfile(path: str, sample_rate: int):
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    return audio, sr


def _has_pyav() -> bool:
    import importlib.util

    return importlib.util.find_spec("av") is not None


def _load_pyav(path: str, sample_rate: int):
    if not _has_pyav():
        raise _Unavailable
    import av

    chunks = []
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("в файле нет звуковой дорожки")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError("пустая запись")
    return np.concatenate(chunks), sample_rate


def _load_ffmpeg(path: str, sample_rate: int):
    exe = shutil.which("ffmpeg")
    if not exe:
        raise _Unavailable
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(
        [exe, "-v", "error", "-i", path, "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True,
        creationflags=flags,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()[:300])
    return np.frombuffer(proc.stdout, dtype=np.float32), sample_rate
