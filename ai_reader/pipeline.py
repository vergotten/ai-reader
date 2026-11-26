"""Book -> audio: chunking, synthesis, export to files and live read-aloud playback."""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import AudioWriter, chapter_chime, silence, trim_silence, write_chapters, write_tags
from .dialogue import SPEECH, segment
from .books import Book
from .stress import PronunciationDictionary, strip_stress, stresser
from .text import Chunk, chunk_text, is_speakable, normalize
from .tts import TTSEngine
from .tts.base import Style

Log = Callable[[str], None]


class Cancelled(Exception):
    pass


@dataclass
class SynthSettings:
    voice: str
    speed: float = 1.0
    language: str = "ru"
    speaker_wav: str | None = None
    speaker_text: str | None = None  # transcript of speaker_wav (F5-TTS)
    expand_numbers: bool = True
    sentence_pause: float = 0.15
    paragraph_pause: float = 0.5
    auto_stress: bool = False  # run RUAccent for engines that read "+" but don't stress text themselves (Silero)
    style: Style = field(default_factory=Style)
    dictionary: PronunciationDictionary | None = None
    # dialogue lines read by other voices of the same engine ("" = the narrator's voice)
    dialogue_voices: bool = False
    male_voice: str = ""
    female_voice: str = ""
    trim: bool = True  # cut the engine's own silence at chunk edges so the pauses above are exact

    def all_voices(self) -> list[str]:
        voices = [self.voice]
        if self.dialogue_voices:
            voices += [v for v in (self.male_voice, self.female_voice) if v and v not in voices]
        return voices

    def voice_for(self, gender: str | None) -> str:
        if gender == "f" and self.female_voice:
            return self.female_voice
        return self.male_voice or self.voice


def needs_stresser(engine: TTSEngine, s: SynthSettings) -> bool:
    return s.language == "ru" and (engine.builtin_stress or any(_stress_in_pipeline(engine, s, v) for v in s.all_voices()))


def _stress_in_pipeline(engine: TTSEngine, s: SynthSettings, voice: str | None = None) -> bool:
    wanted = s.auto_stress or engine.requires_stress(voice or s.voice)
    return s.language == "ru" and engine.stress_marks and wanted and not engine.builtin_stress


def prepare_text(engine: TTSEngine, text: str, s: SynthSettings, voice: str | None = None) -> str:
    """Book text -> what the engine is given: user dictionary, numbers as words, Latin words, stress marks."""
    if s.dictionary is not None:
        text = s.dictionary.apply(text)
    text = normalize(text, s.language, s.expand_numbers)
    if _stress_in_pipeline(engine, s, voice):
        text = stresser.process(text)
    if not engine.stress_marks:
        text = strip_stress(text)
    return text


def make_chunks(engine: TTSEngine, text: str, s: SynthSettings, start: int = 0, end: int | None = None) -> list[Chunk]:
    # leave headroom: number expansion and Latin transliteration make text longer
    max_chars = max(60, int(engine.max_chars * 0.85))
    min_chars = engine.min_chunk_chars
    if not (s.dialogue_voices and s.language == "ru" and len(s.all_voices()) > 1):
        return chunk_text(text, max_chars, s.sentence_pause, s.paragraph_pause, start, end, min_chars)

    chunks: list[Chunk] = []
    segments = segment(text, start, end)
    for i, seg in enumerate(segments):
        pieces = chunk_text(text, max_chars, s.sentence_pause, s.paragraph_pause, seg.start, seg.end, min_chars)
        if not pieces:
            continue
        line_goes_on = i + 1 < len(segments) and "\n" not in text[seg.end : segments[i + 1].start]
        if line_goes_on:  # speech -> «сказала она» in the same paragraph: a short breath, not a paragraph pause
            last = pieces[-1]
            pieces[-1] = replace(last, pause=min(s.sentence_pause, 0.25) if last.text.rstrip().endswith(",") else s.sentence_pause)
        if seg.role == SPEECH:
            voice = s.voice_for(seg.gender)
            pieces = [replace(c, voice=voice) if voice != s.voice else c for c in pieces]
        chunks.extend(pieces)
    return chunks


def synth_chunk(engine: TTSEngine, chunk: Chunk, s: SynthSettings) -> np.ndarray | None:
    voice = chunk.voice or s.voice
    text = prepare_text(engine, chunk.text, s, voice)
    if not is_speakable(text):
        return None
    if s.dialogue_voices:
        # several voices: make sure this one's model is the active one (a no-op when it already is)
        engine.prepare(voice, lambda _m: None)
    audio = engine.synthesize(text, voice, s.speed, s.language, s.speaker_wav, s.speaker_text, s.style)
    if s.trim:
        audio = trim_silence(audio, engine.sample_rate)
    return audio if audio.size else None


def prepare_voices(engine: TTSEngine, s: SynthSettings, log: Log) -> None:
    """Download/load whatever the extra dialogue voices need before reading starts; the narrator goes last."""
    for voice in reversed(s.all_voices()):
        engine.prepare(voice, log)


def safe_filename(name: str, limit: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name).strip(" .")
    return (name[:limit].rstrip(" .") or "untitled")


PARTIAL_SUFFIX = ".part"


@dataclass
class ExportJob:
    book: Book
    chapter_indices: list[int]
    out_dir: Path
    fmt: str = "mp3"
    bitrate: int = 128
    single_file: bool = False
    read_titles: bool = True
    chapter_pause: float = 1.5
    chime: bool = False  # a short soft bell before each chapter, so a new chapter is heard at once
    skip_existing: bool = False  # per-chapter export: keep chapter files that are already there


def run_export(
    job: ExportJob,
    engine: TTSEngine,
    settings: SynthSettings,
    progress: Callable[[int, int, str], None],
    log: Log,
    cancelled: Callable[[], bool],
) -> list[Path]:
    book = job.book
    plan = []
    for idx in job.chapter_indices:
        ch = book.chapters[idx]
        text = f"{ch.title}\n\n{ch.text}" if job.read_titles else ch.text
        plan.append((ch, make_chunks(engine, text, settings)))
    total = sum(len(c.text) for _, chunks in plan for c in chunks) or 1
    done = 0
    errors = 0

    book_dir = job.out_dir / safe_filename(book.title)
    book_dir.mkdir(parents=True, exist_ok=True)
    sr = engine.sample_rate
    files: list[Path] = []
    writer: AudioWriter | None = None
    marks: list[tuple[str, float]] = []  # chapter markers of a one-file book: (title, start in seconds)
    chime = chapter_chime(sr) if job.chime else None

    def open_writer(path: Path) -> AudioWriter:
        log(f"Запись: {path.name}")
        # written under a temporary name: a file with the final name is always a complete chapter
        return AudioWriter(path.with_name(path.name + PARTIAL_SUFFIX), sr, job.fmt, job.bitrate)

    def finish(writer: AudioWriter, *tags) -> Path:
        writer.close()
        final = writer.path.with_name(writer.path.name[: -len(PARTIAL_SUFFIX)])
        writer.path.replace(final)
        write_tags(final, *tags)
        return final

    try:
        if job.single_file:
            writer = open_writer(book_dir / f"{safe_filename(book.title)}.{job.fmt}")
        for n, (ch, chunks) in enumerate(plan, 1):
            if not job.single_file:
                target = book_dir / f"{n:03d} - {safe_filename(ch.title)}.{job.fmt}"
                if job.skip_existing and target.is_file() and target.stat().st_size > 0:
                    log(f"Уже готово, пропуск: {target.name}")
                    files.append(target)
                    done += sum(len(c.text) for c in chunks)
                    progress(done, total, ch.title)
                    continue
                writer = open_writer(target)
            if job.single_file:
                marks.append((ch.title, writer.duration))
            if chime is not None:
                writer.write(chime)
            for chunk in chunks:
                if cancelled():
                    raise Cancelled
                try:
                    audio = synth_chunk(engine, chunk, settings)
                except Exception as e:  # keep going: one bad sentence shouldn't kill a 10-hour book
                    errors += 1
                    log(f"Ошибка синтеза ({e}): «{chunk.text[:80]}»")
                    audio = None
                if audio is not None:
                    writer.write(audio)
                    writer.write(silence(chunk.pause, sr))
                done += len(chunk.text)
                progress(done, total, ch.title)
            if job.single_file:
                writer.write(silence(job.chapter_pause, sr))
            else:
                files.append(finish(writer, ch.title, book.author, book.title, n, len(plan)))
                writer = None
        if job.single_file and writer is not None:
            total_seconds = writer.duration
            path = finish(writer, book.title, book.author, book.title)
            writer = None
            if len(marks) > 1 and write_chapters(path, marks, total_seconds):
                log(f"В файл записаны метки глав: {len(marks)} — в плеере можно переходить по главам.")
            files.append(path)
    except BaseException:
        if writer is not None:
            writer.close()
            writer.path.unlink(missing_ok=True)
        raise

    if errors:
        log(f"Готово, но с ошибками в {errors} фрагментах (см. журнал выше).")
    return files


def _open_output(sd, sample_rate: int):
    try:
        return sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32")
    except sd.PortAudioError as e:
        raise RuntimeError(
            f"Нет звукового устройства для воспроизведения ({e}).\n"
            "В WSL: выполните «bash setup-wsl.sh» (настроит звук через WSLg) и перезапустите программу.\n"
            "Экспорт в файлы работает и без звука."
        ) from e


class ReadAloud:
    """Synthesizes ahead in one thread and plays through the sound card in another, reporting the chunk being spoken."""

    def __init__(
        self,
        engine: TTSEngine,
        settings: SynthSettings,
        chunks: list[Chunk],
        on_chunk: Callable[[int, int], None],
        log: Log,
        intro: np.ndarray | None = None,
    ) -> None:
        self.intro = intro
        self.engine = engine
        self.settings = settings
        self.chunks = chunks
        self.on_chunk = on_chunk
        self.log = log
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._stop = threading.Event()
        self._resume = threading.Event()
        self._resume.set()
        self.completed = False  # True when played to the end without stop()

    @property
    def paused(self) -> bool:
        return not self._resume.is_set()

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def stop(self) -> None:
        self._stop.set()
        self._resume.set()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self) -> None:
        try:
            for chunk in self.chunks:
                if self._stop.is_set():
                    return
                try:
                    audio = synth_chunk(self.engine, chunk, self.settings)
                except Exception as e:
                    self.log(f"Ошибка синтеза ({e}): «{chunk.text[:80]}»")
                    audio = None
                if audio is None:
                    audio = np.zeros(0, dtype=np.float32)
                if not self._put((chunk, audio)):
                    return
        finally:
            self._put(None)

    def run(self) -> None:
        """Blocking: call from a worker thread after the engine is loaded."""
        import sounddevice as sd

        sr = self.engine.sample_rate
        producer = threading.Thread(target=self._produce, daemon=True)
        producer.start()
        block = sr // 10
        stream = _open_output(sd, sr)
        stream.start()
        if self.intro is not None and self.intro.size:
            stream.write(self.intro.reshape(-1, 1).astype(np.float32))
        try:
            while not self._stop.is_set():
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    self.completed = True
                    break
                chunk, audio = item
                self.on_chunk(chunk.start, chunk.end)
                data = np.concatenate([audio, silence(chunk.pause, sr)]).reshape(-1, 1)
                for i in range(0, len(data), block):
                    while not self._resume.wait(0.1):
                        if self._stop.is_set():
                            break
                    if self._stop.is_set():
                        break
                    stream.write(data[i : i + block])
        finally:
            if self._stop.is_set():
                stream.abort(ignore_errors=True)
            else:
                stream.stop(ignore_errors=True)  # waits for the buffered tail to play
            stream.close(ignore_errors=True)
            self._stop.set()
            producer.join(timeout=5)
