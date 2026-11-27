from __future__ import annotations

import itertools
import re
from pathlib import Path

from .model import Book, Chapter, join_paragraphs, split_long

_HEADING = re.compile(
    r"^\s*(?:"
    r"(?:глава|часть|книга|пролог|эпилог|предисловие|послесловие|введение|заключение"
    r"|розділ|chapter|part|book|prologue|epilogue|preface|introduction)\b.{0,80}"
    r"|[IVXLC]{1,8}\.?"
    r")\s*$",
    re.IGNORECASE,
)
_TERMINATED = re.compile(r"[.!?…:;»\"”)\]]\s*$")


def decode_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except ImportError:
        pass
    return raw.decode("cp1251", errors="replace")


def _paragraphs(lines: list[str]) -> list[str]:
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return []
    unterminated = sum(1 for line in nonempty if not _TERMINATED.search(line)) / len(nonempty)
    if unterminated < 0.4:
        # one paragraph per line
        return [line.strip() for line in nonempty]

    # hard-wrapped text: paragraphs are separated by blank lines or by indentation
    blank_separated = sum(1 for line in lines if not line.strip()) > len(nonempty) * 0.1
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            text = current[0]
            for nxt in current[1:]:
                if text.endswith("-") and nxt[:1].islower():
                    text = text[:-1] + nxt
                else:
                    text += " " + nxt
            paragraphs.append(text)
            current.clear()

    for line in lines:
        if not line.strip():
            flush()
            continue
        if not blank_separated and line[:1] in (" ", "\t", "　"):
            flush()
        current.append(line.strip())
    flush()
    return paragraphs


def text_to_book(text: str, title: str) -> Book:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heads = [i for i, line in enumerate(lines) if _HEADING.match(line)]

    chapters: list[Chapter] = []
    if len(heads) >= 2:
        if heads[0] > 0:
            intro = join_paragraphs(_paragraphs(lines[: heads[0]]))
            if intro:
                chapters.append(Chapter("Начало", intro))
        bounds = heads + [len(lines)]
        for start, end in itertools.pairwise(bounds):
            chapters.append(Chapter(lines[start].strip(), join_paragraphs(_paragraphs(lines[start + 1 : end]))))
    else:
        chapters.extend(split_long("Текст", join_paragraphs(_paragraphs(lines)), max_chars=30_000))
    return Book(title=title, chapters=chapters)


def load_txt(path: Path) -> Book:
    book = text_to_book(decode_bytes(path.read_bytes()), path.stem)
    book.path = path
    return book
