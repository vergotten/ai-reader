"""Stress marks for the whole book right after it is opened, in the background.

The chapter on screen goes first, then the rest in order. Reading and export work meanwhile: the stress model is
shared and taken in small portions, so a read-aloud never waits long for it.
"""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal

from ..books import Book
from ..stress import stresser

LINES_PER_STEP = 20  # one portion: short enough that reading aloud never waits long for the shared model


class BookStresser(QObject):
    """Emits chapter_done(index, original, stressed) for every chapter; the window decides whether to apply it."""

    chapter_done = Signal(int, str, str)
    log = Signal(str)
    progress = Signal(int, int)  # chapters done, total
    failed = Signal(str)
    finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, book: Book, first: int) -> None:
        self.cancel()
        self._cancel = threading.Event()
        order = [first] + [i for i in range(len(book.chapters)) if i != first]
        texts = [(i, book.chapters[i].text) for i in order]
        self._thread = threading.Thread(
            target=self._run, args=(texts, self._cancel), daemon=True, name="ai-reader-book-stress"
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def _run(self, texts: list[tuple[int, str]], cancel: threading.Event) -> None:
        try:
            stresser.load(self.log.emit)
            for done, (index, original) in enumerate(texts, 1):
                lines = original.split("\n")
                out = []
                for i in range(0, len(lines), LINES_PER_STEP):
                    if cancel.is_set():
                        return
                    out.append(stresser.process("\n".join(lines[i : i + LINES_PER_STEP])))
                if cancel.is_set():
                    return
                self.chapter_done.emit(index, original, "\n".join(out))
                self.progress.emit(done, len(texts))
        except Exception as error:
            self.failed.emit(str(error) or error.__class__.__name__)
        finally:
            if not cancel.is_set():
                self.finished.emit()
