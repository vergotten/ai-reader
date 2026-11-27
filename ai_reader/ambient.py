"""Quiet background music that follows the mood of the text.

The mood of every part of a chapter is guessed from its words (fear, sadness, joy, mystery, tenderness or calm),
and a soft ambient bed for that mood plays under the narration: it dips while someone speaks and swells a
little in the pauses. Every chapter opens with a few seconds of music and fades out at the end.

The music is synthesized on the fly (slow chords, a few bell-like notes, a light echo), so there are no sound
files to ship or license. Own tracks can be put into data_dir()/music/<mood>/ (or straight into music/ for any
mood): MP3, OGG, FLAC or WAV; they are looped and used instead of the synthesized music.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .paths import data_dir

MOODS = {
    "calm": "спокойное",
    "joy": "светлое, вдохновляющее",
    "sad": "грустное",
    "tense": "напряжённое",
    "mystery": "загадочное",
    "warm": "тёплое, нежное",
}
DEFAULT_MOOD = "calm"

# word beginnings (ё -> е) that hint at a mood; short ones are matched only as whole words
_LEXICON: dict[str, tuple[str, ...]] = {
    "tense": (
        "страх", "страш", "ужас", "испуг", "пуга", "кров", "убий", "убит", "убил", "смерт", "крик", "крич", "вопл",
        "погон", "выстрел", "стрел", "оруж", "пистолет", "нож", "драк", "удар", "опасн", "тревог", "паник",
        "взрыв", "враг", "битв", "сражен", "атак", "гнев", "ярост", "злоб", "угроз", "дрож", "бежал", "бросил",
        "схватил", "рана", "ранен", "погиб", "война", "войн", "огонь", "пожар", "беда", "бой", "боя", "кошмар",
    ),
    "sad": (
        "груст", "печал", "слез", "слёз", "плак", "плач", "тоск", "одинок", "потер", "утрат", "горе", "горьк",
        "умер", "умерл", "похорон", "прощ", "жаль", "жалел", "боль", "разочаров", "пуст", "уныл", "скорб",
        "отчаян", "сожал", "покинул", "ушел", "ушла", "больше никогда", "вдов", "сирот", "болезн",
    ),
    "joy": (
        "радост", "радова", "счаст", "смех", "смея", "смеял", "улыб", "весел", "праздн", "побед", "успех",
        "успеш", "восторг", "ура", "солнц", "мечт", "достиж", "достиг", "богат", "доход", "возможност",
        "свобод", "вдохнов", "уверен", "сила", "энерги", "лучш", "прекрасн", "отличн", "блестящ", "цель",
    ),
    "mystery": (
        "тайн", "таин", "загад", "странн", "тень", "тени", "тьм", "туман", "мрак", "ночь", "ночн", "шепот",
        "шёпот", "шорох", "призрак", "маги", "волшеб", "колдов", "древн", "неизвест", "секрет", "легенд",
        "луна", "лунн", "сумрак", "заклин", "пророч", "чудовищ", "подвал", "замок", "пещер", "свеч",
    ),
    "warm": (
        "любов", "любил", "люблю", "любим", "нежн", "поцел", "обнял", "обним", "объят", "ласк", "милы",
        "милая", "дорог", "тепл", "уют", "свидан", "влюб", "красив", "сердц", "душ", "семь", "мама", "мамы",
        "отец", "дети", "детей", "дом", "родн", "забот",
    ),
    "calm": (
        "природ", "лес", "поле", "полям", "море", "тиш", "тих", "спокой", "мирн", "утро", "утрен", "рассвет",
        "ветер", "ветерок", "сад", "река", "реки", "небо", "облак", "медленн", "отдых", "путь",
        "думал", "размышл", "вспомин", "чай", "кофе",
    ),
}
_SHORT = 4  # stems shorter than this must match the whole word, so «бой» doesn't hit «бойкий»
_WORD = re.compile(r"[а-яё]+", re.IGNORECASE)


def mood_scores(text: str) -> dict[str, float]:
    """Hits per 100 words for every mood."""
    words = [w.lower().replace("ё", "е") for w in _WORD.findall(text)]
    scores = {mood: 0.0 for mood in MOODS}
    if not words:
        return scores
    vocab: dict[str, int] = {}
    for w in words:
        vocab[w] = vocab.get(w, 0) + 1
    for mood, stems in _LEXICON.items():
        hits = 0
        for stem in stems:
            stem = stem.replace("ё", "е")
            if " " in stem:
                hits += text.lower().replace("ё", "е").count(stem)
            elif len(stem) < _SHORT:
                hits += vocab.get(stem, 0)
            else:
                hits += sum(n for w, n in vocab.items() if w.startswith(stem))
        scores[mood] = 100.0 * hits / len(words)
    scores["calm"] *= 0.6  # calm words are everywhere; a scene has to be really quiet to count as calm
    return scores


def detect_mood(text: str, prior: dict[str, float] | None = None, threshold: float = 0.6) -> str:
    """The strongest mood of a text, or DEFAULT_MOOD when nothing stands out."""
    scores = mood_scores(text)
    if prior:
        scores = {m: scores[m] + 0.5 * prior.get(m, 0.0) for m in scores}
    mood, best = max(scores.items(), key=lambda kv: kv[1])
    return mood if best >= threshold else DEFAULT_MOOD


def section_moods(texts: Sequence[str], section_chars: int = 1500) -> list[str]:
    """A mood for every text piece (chunk) of a chapter: pieces are grouped into sections of ~section_chars,
    and the chapter as a whole is a prior, so the music doesn't change with every paragraph."""
    prior = mood_scores(" ".join(texts))
    moods: list[str] = []
    group: list[str] = []
    size = 0
    for text in texts:
        group.append(text)
        size += len(text)
        if size >= section_chars:
            moods += [detect_mood(" ".join(group), prior)] * len(group)
            group, size = [], 0
    if group:
        # a short tail joins the previous section's mood instead of flipping for a few sentences
        tail = detect_mood(" ".join(group), prior) if size >= section_chars / 3 or not moods else moods[-1]
        moods += [tail] * len(group)
    return moods


# ------------------------------------------------------------------ synthesized ambient


# root frequency, chords (semitones from the root), seconds per chord, brightness (overtone level),
# bell notes per chord, the scale the bell notes are taken from, heartbeat pulse
_STYLES: dict[str, dict] = {
    "calm": {"root": 146.83, "chords": [(0, 7, 12, 16), (-3, 4, 9, 12), (-7, 5, 9, 12), (-5, 7, 11, 14)],
                 "length": 9.0, "bright": 0.35, "bells": 1.2, "scale": (0, 2, 4, 7, 9), "pulse": False},
    "joy": {"root": 164.81, "chords": [(0, 4, 7, 12), (-7, 5, 9, 12), (-3, 4, 9, 12), (-5, 7, 11, 14)],
                "length": 6.5, "bright": 0.45, "bells": 1.8, "scale": (0, 2, 4, 7, 9), "pulse": False},
    "sad": {"root": 130.81, "chords": [(0, 3, 7, 12), (-4, 3, 8, 12), (-7, 3, 8, 12), (-5, 2, 7, 10)],
                "length": 10.0, "bright": 0.25, "bells": 0.8, "scale": (0, 3, 5, 7, 10), "pulse": False},
    "tense": {"root": 110.0, "chords": [(0, 7, 13), (0, 6, 15), (-1, 6, 11), (0, 3, 13)],
                  "length": 7.0, "bright": 0.3, "bells": 0.4, "scale": (0, 1, 3, 6, 7), "pulse": True},
    "mystery": {"root": 123.47, "chords": [(0, 7, 14, 15), (-4, 3, 10, 14), (-2, 5, 10, 14), (0, 6, 11, 15)],
                    "length": 11.0, "bright": 0.3, "bells": 1.6, "scale": (0, 2, 3, 6, 7, 10), "pulse": False},
    "warm": {"root": 138.59, "chords": [(0, 4, 7, 11), (-7, 5, 9, 12), (2, 5, 9, 12), (-5, 7, 11, 14)],
                 "length": 8.0, "bright": 0.4, "bells": 1.2, "scale": (0, 2, 4, 7, 9, 11), "pulse": False},
}
_ECHO = ((0.43, 0.28), (0.71, 0.17), (1.13, 0.09))  # (delay in seconds, level): a light, roomy tail
_BED_GAIN = 0.55  # brings the synthesized bed to about the loudness of a normalized own track


class SynthBed:
    """An endless ambient stream for one mood: overlapping slow chords plus a few bell notes."""

    def __init__(self, mood: str, sample_rate: int, seed: int = 0) -> None:
        self.sr = sample_rate
        self.style = _STYLES.get(mood, _STYLES[DEFAULT_MOOD])
        self.rng = random.Random(f"{mood}:{seed}")
        self._chord = self.rng.randrange(len(self.style["chords"]))
        self._buf = np.zeros(0, dtype=np.float32)
        self._next = 0  # where in _buf the next chord starts; everything before it is final
        longest = max(d for d, _ in _ECHO)
        self._history = np.zeros(int(longest * sample_rate) + 1, dtype=np.float32)

    def _tone(self, t: np.ndarray, freq: float, partials: Sequence[tuple[float, float]]) -> np.ndarray:
        out = np.zeros_like(t)
        for ratio, level in partials:
            out += level * np.sin(2 * np.pi * freq * ratio * t + self.rng.uniform(0, 2 * np.pi))
        return out

    def _render_chord(self) -> np.ndarray:
        st, sr, rng = self.style, self.sr, self.rng
        length = st["length"] * rng.uniform(0.9, 1.1)
        release = 4.0
        n = int((length + release) * sr)
        t = np.arange(n, dtype=np.float32) / sr
        chord = st["chords"][self._chord]
        self._chord = (self._chord + 1) % len(st["chords"])

        partials = [(1.0, 1.0)] + [(k, st["bright"] ** (k - 1)) for k in (2, 3, 4)]
        norm = sum(level for _, level in partials)
        pad = np.zeros(n, dtype=np.float32)
        for semis in chord:
            f = st["root"] * 2 ** (semis / 12)
            for detune in (0.9975, 1.0025):  # two slightly detuned voices: a slow, soft chorus
                pad += self._tone(t, f * detune, partials) / norm
        pad *= 0.5 / len(chord)
        pad += 0.12 * np.sin(2 * np.pi * st["root"] / 2 * t)  # a low drone holds the chords together
        swell = 1.0 + 0.18 * np.sin(2 * np.pi * t / (length + release) + rng.uniform(0, 2 * np.pi))
        attack = np.clip(t / 2.5, 0, 1)
        fade = np.clip((length + release - t) / release, 0, 1)
        out = pad * swell * np.sin(attack * np.pi / 2) ** 2 * np.sin(fade * np.pi / 2) ** 2

        for _ in range(self._poisson(st["bells"])):  # bell-like notes up high, now and then
            start = rng.uniform(0.5, length)
            degree = rng.choice(st["scale"]) + 12 * rng.choice((1, 2))
            f = st["root"] * 2 ** (degree / 12)
            i0 = int(start * sr)
            tb = t[: n - i0]
            bell = self._tone(tb, f, ((1, 1.0), (2.76, 0.25), (5.4, 0.06)))
            out[i0:] += rng.uniform(0.08, 0.16) * bell * np.exp(-tb * 1.6) * np.minimum(1.0, tb / 0.01)

        if st["pulse"]:  # a slow, low heartbeat under tense scenes
            beat = 1.15
            for k in range(int(length / beat)):
                for offset, level in ((0.0, 0.22), (0.26, 0.14)):
                    i0 = int((k * beat + offset) * sr)
                    tb = t[: min(n - i0, int(0.5 * sr))]
                    out[i0 : i0 + len(tb)] += level * np.sin(2 * np.pi * 55.0 * tb) * np.exp(-tb * 9.0)
        self._advance = int(length * sr)
        return out.astype(np.float32)

    def _poisson(self, mean: float) -> int:
        limit, k, p = math.exp(-mean), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= limit:
                return k
            k += 1

    def read(self, n: int) -> np.ndarray:
        while self._next < n:
            seg = self._render_chord()
            need = self._next + len(seg)
            if len(self._buf) < need:
                self._buf = np.concatenate([self._buf, np.zeros(need - len(self._buf), dtype=np.float32)])
            self._buf[self._next : need] += seg
            self._next += self._advance
        dry, self._buf = self._buf[:n], self._buf[n:]
        self._next -= n
        # echo: delayed copies of what was played before, kept across calls
        full = np.concatenate([self._history, dry])
        wet = dry.copy()
        for delay, level in _ECHO:
            d = int(delay * self.sr)
            start = len(self._history) - d
            wet += level * full[start : start + n]
        self._history = full[-len(self._history):]
        return (np.tanh(wet * _BED_GAIN * 1.6) / 1.6).astype(np.float32)


class FileBed:
    """An own music file, looped with a soft crossfade at the seam."""

    def __init__(self, path: Path, sample_rate: int) -> None:
        import soundfile as sf

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        if rate != sample_rate:
            import soxr

            audio = soxr.resample(audio, rate, sample_rate).astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
        audio = audio * (0.18 / rms)  # every track about as loud as the synthesized bed
        np.clip(audio, -1.0, 1.0, out=audio)
        if len(audio) < sample_rate:
            raise ValueError("слишком короткая запись")
        xf = min(len(audio) // 4, 2 * sample_rate)
        if xf > 0:
            ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            head = audio[:xf] * ramp + audio[-xf:] * ramp[::-1]
            audio = np.concatenate([head, audio[xf:-xf]])
        self._loop = audio.astype(np.float32)
        self._pos = 0

    def read(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            take = min(n - filled, len(self._loop) - self._pos)
            out[filled : filled + take] = self._loop[self._pos : self._pos + take]
            filled += take
            self._pos = (self._pos + take) % len(self._loop)
        return out


MUSIC_EXTENSIONS = (".mp3", ".ogg", ".oga", ".opus", ".flac", ".wav")
_README = """Своя фоновая музыка для AI Reader

Положите музыку (MP3, OGG, FLAC, WAV) в папку настроения — она будет играть тихим фоном вместо
встроенной музыки, когда текст подходит под это настроение:

  calm     — спокойное (природа, размышления)
  joy      — светлое, вдохновляющее (успех, радость, мотивация)
  sad      — грустное (потери, расставания)
  tense    — напряжённое (опасность, погони, драки)
  mystery  — загадочное (тайны, ночь, магия)
  warm     — тёплое, нежное (любовь, семья, дом)

Файлы прямо в этой папке (не в подпапках) подходят для любого настроения.
Лучше всего — спокойный эмбиент без слов. Если папка настроения пуста, играет встроенная музыка.
Берите музыку, которую вам можно использовать (своя, свободная лицензия, CC0 и т. п.).
"""


def music_dir(create: bool = False) -> Path:
    path = data_dir() / "music"
    if create:
        for mood in MOODS:
            (path / mood).mkdir(parents=True, exist_ok=True)
        readme = path / "ПРОЧТИ.txt"
        if not readme.exists():
            readme.write_text(_README, encoding="utf-8")
    return path


def own_tracks(mood: str) -> list[Path]:
    root = music_dir()
    found: list[Path] = []
    for folder in (root / mood, root):
        if folder.is_dir():
            found += sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MUSIC_EXTENSIONS)
    return found


# ------------------------------------------------------------------ mixing under the narration


class AmbientMixer:
    """Mixes the mood music under the speech, quieter while someone speaks and a bit louder in pauses.

    volume 0..1 (0.5 is the default): music under speech then sits about 15 dB below the voice.
    """

    FRAME = 0.02  # s: the speech detector's resolution
    XFADE = 5.0  # s: crossfade when the mood changes inside a chapter

    def __init__(self, sample_rate: int, volume: float = 0.5, seed: int = 0, log=None) -> None:
        self.sr = sample_rate
        self.volume = max(0.0, min(1.0, volume))
        self.seed = seed
        self.log = log or (lambda _m: None)
        self.mood: str | None = None
        self._bed = None
        self._old = None  # the previous bed while crossfading
        self._xpos = 0
        self.gain = 0.0
        self._attack = math.exp(-self.FRAME / 0.12)  # quick dip when speech starts
        self._release = math.exp(-self.FRAME / 0.9)  # slow swell in pauses

    # levels of the music: under speech, in pauses, and alone at a chapter's start
    @property
    def speech_level(self) -> float:
        return 0.32 * self.volume

    @property
    def pause_level(self) -> float:
        return 0.55 * self.volume

    @property
    def intro_level(self) -> float:
        return 1.3 * self.volume

    def _make_bed(self, mood: str):
        self.seed += 1
        tracks = own_tracks(mood)
        if tracks:
            track = random.Random(self.seed).choice(tracks)
            try:
                bed = FileBed(track, self.sr)
                self.log(f"Фоновая музыка: {track.name}")
                return bed
            except Exception as e:
                self.log(f"Не удалось открыть музыку {track.name} ({e}), играет встроенная")
        return SynthBed(mood, self.sr, self.seed)

    def start(self, mood: str) -> None:
        """A fresh bed for a new chapter (the gain is expected to be at or near zero)."""
        self.mood = mood
        self._bed = self._make_bed(mood)
        self._old = None

    def set_mood(self, mood: str) -> None:
        if self._bed is None:
            self.start(mood)
        elif mood != self.mood:
            self._old, self._xpos = self._bed, 0
            self._bed, self.mood = self._make_bed(mood), mood

    def _music(self, n: int) -> np.ndarray:
        if self._bed is None:
            self.start(DEFAULT_MOOD)
        new = self._bed.read(n)
        if self._old is None:
            return new
        total = int(self.XFADE * self.sr)
        pos = self._xpos + np.arange(n)
        mix = np.clip(pos / total, 0.0, 1.0).astype(np.float32)
        out = new * np.sin(mix * np.pi / 2) + self._old.read(n) * np.cos(mix * np.pi / 2)
        self._xpos += n
        if self._xpos >= total:
            self._old = None
        return out.astype(np.float32)

    def _ramp(self, n: int, start: float, end: float) -> np.ndarray:
        x = np.linspace(0.0, 1.0, n, dtype=np.float32)
        return start + (end - start) * (0.5 - 0.5 * np.cos(np.pi * x))

    def intro(self, seconds: float = 6.0) -> np.ndarray:
        """Music alone before a chapter: fades in, holds, and settles down to the level under speech."""
        n = int(seconds * self.sr)
        up, down = int(1.5 * self.sr), int(2.5 * self.sr)
        hold = max(0, n - up - down)
        gain = np.concatenate([
            self._ramp(up, self.gain, self.intro_level),
            np.full(hold, self.intro_level, dtype=np.float32),
            self._ramp(down, self.intro_level, self.speech_level),
        ])
        self.gain = self.speech_level
        return (self._music(len(gain)) * gain).astype(np.float32)

    def outro(self, seconds: float = 3.0) -> np.ndarray:
        """The music fades out after a chapter."""
        n = int(seconds * self.sr)
        gain = self._ramp(n, self.gain, 0.0)
        self.gain = 0.0
        return (self._music(n) * gain).astype(np.float32)

    def mix(self, speech: np.ndarray) -> np.ndarray:
        """speech (mono float32) with the music under it."""
        speech = np.asarray(speech, dtype=np.float32).reshape(-1)
        n = len(speech)
        if n == 0 or self.volume <= 0:
            return speech
        frame = max(1, int(self.FRAME * self.sr))
        frames = (n + frame - 1) // frame
        padded = np.zeros(frames * frame, dtype=np.float32)
        padded[:n] = speech
        rms = np.sqrt(np.mean(np.square(padded.reshape(frames, frame)), axis=1))
        voiced = rms > 0.01
        # look ahead ~0.2 s: the music starts to dip just before the voice comes in
        ahead = max(1, int(0.2 / self.FRAME))
        soon = np.array([voiced[i : i + ahead].any() for i in range(frames)])
        gains = np.empty(frames, dtype=np.float32)
        g = self.gain
        for i in range(frames):
            target = self.speech_level if soon[i] else self.pause_level
            coef = self._attack if target < g else self._release
            g = target + (g - target) * coef
            gains[i] = g
        points = np.concatenate([[self.gain], gains])
        where = np.concatenate([[0], (np.arange(frames) + 1) * frame])
        per_sample = np.interp(np.arange(n), where, points).astype(np.float32)
        self.gain = g
        out = speech + self._music(n) * per_sample
        return np.clip(out, -1.0, 1.0).astype(np.float32)
