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


def trim_silence(audio: np.ndarray, sample_rate: int, threshold_db: float = -42.0, keep: float = 0.05) -> np.ndarray:
    """Cut near-silence at both ends (relative to the fragment's peak), keeping `keep` seconds of it.

    Engines add their own silence of random length around every sentence, which made the pause settings
    unpredictable; after trimming, the pauses heard are the ones chosen in the settings.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    frame = max(1, sample_rate // 100)  # 10 ms
    n = len(x) // frame
    if n < 5:
        return x
    rms = np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame)), axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms / (rms.max() + 1e-12)) > threshold_db)[0]
    if loud.size == 0:
        return x
    pad = int(keep * sample_rate)
    start = max(0, loud[0] * frame - pad)
    end = min(len(x), (loud[-1] + 1) * frame + pad)
    out = x[start:end].copy()
    fade = min(len(out) // 4, sample_rate // 200)  # 5 ms fades: no clicks at the cut
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return out


def chapter_chime(sample_rate: int, volume: float = 0.18) -> np.ndarray:
    """A short soft two-note bell (≈1.6 s with its tail) that marks the start of a chapter. Made on the fly:
    no sound files to ship, and it is quiet enough not to startle someone listening at night."""
    def bell(freq: float, seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        tone = sum(a * np.sin(2 * np.pi * freq * k * t) for k, a in ((1, 1.0), (2.0, 0.35), (3.01, 0.12), (4.2, 0.05)))
        attack = np.minimum(1.0, t / 0.008)
        return (tone * attack * np.exp(-t * 3.2)).astype(np.float32)

    first, second = bell(659.25, 1.3), bell(987.77, 1.3)  # E5 -> B5, a calm rising fifth
    offset = int(0.28 * sample_rate)
    out = np.zeros(offset + len(second) + int(0.35 * sample_rate), dtype=np.float32)
    out[: len(first)] += first
    out[offset : offset + len(second)] += 0.8 * second
    return (out / (np.max(np.abs(out)) + 1e-9) * volume).astype(np.float32)


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


def _timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    sec, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"


def write_chapters(path: Path, chapters: list[tuple[str, float]], total: float) -> bool:
    """Chapter markers in a one-file book, so phone players (Smart AudioBook Player, Listen, VLC, foobar2000…)
    can jump between chapters and still remember the position in the single file.

    MP3: ID3v2 CHAP/CTOC frames. OGG/Opus/FLAC: CHAPTERxxx Vorbis comments. WAV has no place for them.
    """
    try:
        import mutagen
    except ImportError:
        return False
    try:
        if path.suffix.lower() == ".mp3":
            from mutagen.id3 import CHAP, CTOC, ID3, TIT2, CTOCFlags

            tags = ID3(path)
            tags.delall("CHAP")
            tags.delall("CTOC")
            ids = []
            for n, (title, start) in enumerate(chapters):
                end = chapters[n + 1][1] if n + 1 < len(chapters) else total
                cid = f"ch{n:03d}"
                ids.append(cid)
                tags.add(CHAP(element_id=cid, start_time=int(start * 1000), end_time=int(end * 1000),
                              sub_frames=[TIT2(text=[title])]))
            tags.add(CTOC(element_id="toc", flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED, child_element_ids=ids,
                          sub_frames=[TIT2(text=["Главы"])]))
            tags.save(path)
            return True
        f = mutagen.File(path)
        if f is None or path.suffix.lower() == ".wav":
            return False
        if f.tags is None:
            f.add_tags()
        for n, (title, start) in enumerate(chapters, 1):
            f.tags[f"CHAPTER{n:03d}"] = _timestamp(start)
            f.tags[f"CHAPTER{n:03d}NAME"] = title
        f.save()
        return True
    except Exception:
        return False  # markers are nice-to-have
