"""Book -> audio: chunking, synthesis, export to files and live read-aloud playback."""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .audio import AudioWriter, silence, write_tags
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


def needs_stresser(engine: TTSEngine, s: SynthSettings) -> bool:
    return s.language == "ru" and (engine.builtin_stress or _stress_in_pipeline(engine, s))


def _stress_in_pipeline(engine: TTSEngine, s: SynthSettings) -> bool:
    wanted = s.auto_stress or engine.requires_stress(s.voice)
    return s.language == "ru" and engine.stress_marks and wanted and not engine.builtin_stress


def prepare_text(engine: TTSEngine, text: str, s: SynthSettings) -> str:
    """Book text -> what the engine is given: user dictionary, numbers as words, stress marks."""
    if s.dictionary is not None:
        text = s.dictionary.apply(text)
    text = normalize(text, s.language, s.expand_numbers)
    if _stress_in_pipeline(engine, s):
        text = stresser.process(text)
    if not engine.stress_marks:
        text = strip_stress(text)
    return text


def make_chunks(engine: TTSEngine, text: str, s: SynthSettings, start: int = 0, end: int | None = None) -> list[Chunk]:
    # leave headroom: number expansion makes text longer
    max_chars = max(60, int(engine.max_chars * 0.85))
    return chunk_text(text, max_chars, s.sentence_pause, s.paragraph_pause, start, end)


def synth_chunk(engine: TTSEngine, chunk: Chunk, s: SynthSettings) -> np.ndarray | None:
    text = prepare_text(engine, chunk.text, s)
    if not is_speakable(text):
        return None
    audio = engine.synthesize(text, s.voice, s.speed, s.language, s.speaker_wav, s.speaker_text, s.style)
    return audio if audio.size else None


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
            files.append(finish(writer, book.title, book.author, book.title))
            writer = None
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
    ) -> None:
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
