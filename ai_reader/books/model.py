from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chapter:
    title: str
    text: str  # paragraphs separated by blank lines


@dataclass
class Book:
    title: str
    author: str = ""
    language: str | None = None
    path: Path | None = None
    chapters: list[Chapter] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(c.text) for c in self.chapters)


def clean_paragraphs(paragraphs: Iterable[str]) -> list[str]:
    out = []
    for p in paragraphs:
        p = " ".join(p.split())
        if p:
            out.append(p)
    return out


def join_paragraphs(paragraphs: Iterable[str]) -> str:
    return "\n\n".join(clean_paragraphs(paragraphs))


def _key(s: str) -> str:
    return re.sub(r"[\W_]+", "", s.casefold())


def split_long(title: str, text: str, max_chars: int = 40_000) -> list[Chapter]:
    """Split an oversized chapter into parts at paragraph boundaries."""
    if len(text) <= max_chars:
        return [Chapter(title, text)]
    parts: list[list[str]] = [[]]
    size = 0
    for para in text.split("\n\n"):
        if size and size + len(para) > max_chars:
            parts.append([])
            size = 0
        parts[-1].append(para)
        size += len(para) + 2
    return [Chapter(f"{title} (часть {i})", "\n\n".join(p)) for i, p in enumerate(parts, 1)]


def finalize(book: Book) -> Book:
    """Drop empty chapters, remove a leading paragraph that repeats the title, fill in missing titles."""
    chapters = []
    for ch in book.chapters:
        text = ch.text.strip()
        title = " ".join(ch.title.split())
        if title and text:
            first, _, rest = text.partition("\n\n")
            if _key(first) == _key(title):
                text = rest.strip()
        if not text:
            continue
        if not title:
            title = f"Глава {len(chapters) + 1}"
        chapters.append(Chapter(title, text))
    if not book.title and book.path is not None:
        book.title = book.path.stem
    book.chapters = chapters
    return book
