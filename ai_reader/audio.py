"""Streaming audio writers: MP3 via LAME (no ffmpeg needed), WAV/FLAC/OGG/Opus via libsndfile."""

from __future__ import annotations

from pathlib import Path

import numpy as np

FORMATS = {
    "mp3": "MP3",
    "ogg": "OGG Vorbis",
    "opus": "Opus (OGG)",
    "flac": "FLAC (без потерь)",
    "wav": "WAV (без сжатия)",
}
BITRATES = [64, 96, 128, 160, 192, 256, 320]
_SNDFILE = {"wav": ("WAV", "PCM_16"), "flac": ("FLAC", "PCM_16"), "ogg": ("OGG", "VORBIS"), "opus": ("OGG", "OPUS")}
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(max(0, int(seconds * sample_rate)), dtype=np.float32)


class AudioWriter:
    """Write mono float32 audio incrementally so whole books never sit in memory."""

    def __init__(self, path: str | Path, sample_rate: int, fmt: str = "mp3", bitrate: int = 128) -> None:
        self.path = Path(path)
        self.fmt = fmt
        self.rate = sample_rate
        self.frames = 0
        self._resampler = None
        self._enc = self._fh = self._sf = None

        if fmt == "opus" and sample_rate not in _OPUS_RATES:
            import soxr

            self.rate = 48_000
            self._resampler = soxr.ResampleStream(sample_rate, self.rate, 1, dtype="float32")

        if fmt == "mp3":
            import lameenc

            enc = lameenc.Encoder()
            enc.set_bit_rate(bitrate)
            enc.set_in_sample_rate(self.rate)
            enc.set_channels(1)
            enc.set_quality(2)
            self._enc = enc
            self._fh = open(self.path, "wb")
        elif fmt in _SNDFILE:
            import soundfile as sf

            container, subtype = _SNDFILE[fmt]
            self._sf = sf.SoundFile(self.path, "w", samplerate=self.rate, channels=1, format=container, subtype=subtype)
        else:
            raise ValueError(f"Неизвестный формат: {fmt}")

    def write(self, samples: np.ndarray, last: bool = False) -> None:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._resampler is not None:
            x = self._resampler.resample_chunk(x, last=last)
        if x.size == 0:
            return
        x = np.clip(x, -1.0, 1.0)
        self.frames += x.size
        if self._enc is not None:
            self._fh.write(self._enc.encode((x * 32767).astype(np.int16).tobytes()))
        else:
            self._sf.write(x)

    @property
    def duration(self) -> float:
        return self.frames / self.rate

    def close(self) -> None:
        if self._resampler is not None:
            self.write(np.zeros(0, dtype=np.float32), last=True)
            self._resampler = None
        if self._enc is not None:
            self._fh.write(self._enc.flush())
            self._fh.close()
            self._enc = None
        if self._sf is not None:
            self._sf.close()
            self._sf = None

    def __enter__(self) -> "AudioWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_tags(path: Path, title: str, artist: str = "", album: str = "", track: int | None = None, total: int | None = None) -> None:
    try:
        import mutagen
    except ImportError:
        return
    try:
        f = mutagen.File(path, easy=True)
        if f is None:
            return
        if f.tags is None:
            f.add_tags()
        f["title"] = title
        if artist:
            f["artist"] = artist
        if album:
            f["album"] = album
        f["genre"] = "Audiobook"
        if track is not None:
            f["tracknumber"] = f"{track}/{total}" if total else str(track)
        f.save()
    except Exception:
        pass  # tags are nice-to-have
