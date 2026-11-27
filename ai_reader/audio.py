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


# The chapter tune: the same short phrase before every chapter, like the signature tune of an audiobook series.
# (beat, MIDI note, length in beats) — D major, a rising phrase that settles on the tonic.
_MELODY = (
    (0.0, 74, 1.0), (1.0, 78, 0.5), (1.5, 81, 0.5), (2.0, 86, 1.5), (3.5, 85, 0.5),
    (4.0, 81, 1.0), (5.0, 83, 0.5), (5.5, 81, 0.5), (6.0, 78, 1.0), (7.0, 74, 2.0),
)
_MELODY_CHORDS = ((0.0, (50, 57, 62, 66)), (4.0, (47, 55, 59, 62)), (6.0, (45, 52, 57, 61)), (7.0, (50, 57, 62, 66)))
_MELODY_BEAT = 0.36  # seconds: an unhurried tempo


def _midi(note: int) -> float:
    return 440.0 * 2 ** ((note - 69) / 12)


def chapter_melody(sample_rate: int, volume: float = 0.3) -> np.ndarray:
    """The chapter tune (≈4.5 s): a soft music-box phrase over warm chords, always the same, so a new chapter is
    recognised at once. Synthesized here: no sound files to ship, quiet enough for listening at night."""
    beat = _MELODY_BEAT
    total = 9.0 * beat + 1.8
    out = np.zeros(int(total * sample_rate), dtype=np.float32)

    def place(start: float, sound: np.ndarray) -> None:
        i = int(start * sample_rate)
        n = min(len(sound), len(out) - i)
        out[i : i + n] += sound[:n]

    for at, note, length in _MELODY:  # music box: bright attack, ringing decay
        seconds = length * beat + 1.2
        t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        f = _midi(note)
        tone = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) + 0.08 * np.sin(2 * np.pi * 3.01 * f * t)
        place(at * beat, 0.5 * tone * np.minimum(1.0, t / 0.004) * np.exp(-t * 2.6))

    for k, (at, notes) in enumerate(_MELODY_CHORDS):  # soft pad chords under the tune
        end = _MELODY_CHORDS[k + 1][0] if k + 1 < len(_MELODY_CHORDS) else 9.0
        seconds = (end - at) * beat + (1.6 if k + 1 == len(_MELODY_CHORDS) else 0.25)
        t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
        pad = sum(np.sin(2 * np.pi * _midi(n) * t) + 0.15 * np.sin(2 * np.pi * 2 * _midi(n) * t) for n in notes)
        envelope = np.minimum(1.0, t / 0.25) * np.clip((seconds - t) / 0.6, 0.0, 1.0)
        place(at * beat, 0.09 * pad * envelope)

    delay = int(0.27 * sample_rate)  # a little room around the music box
    out[delay:] += 0.25 * out[:-delay].copy()
    fade = int(0.4 * sample_rate)
    out[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
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

    def __enter__(self) -> AudioWriter:
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
    ms = round(seconds * 1000)
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
