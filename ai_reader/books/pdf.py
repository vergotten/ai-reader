from __future__ import annotations

import itertools
import re
from pathlib import Path

from .model import Book, Chapter, join_paragraphs

_PAGE_NUMBER = re.compile(r"[\d\s\-–—./]+")
_HYPHEN_BREAK = re.compile(r"[-\xad]\n(?=[a-zа-яёіїєґ])")
PAGES_PER_PART = 10


def _page_paragraphs(page) -> list[str]:
    paragraphs = []
    for block in page.get_text("blocks", sort=True):
        if block[6] != 0:  # not a text block
            continue
        # PyMuPDF often reports real hyphens as U+00AD; join line-break hyphenation, keep the rest
        text = " ".join(_HYPHEN_BREAK.sub("", block[4]).replace("\xad", "-").split())
        if text and not _PAGE_NUMBER.fullmatch(text):
            paragraphs.append(text)
    return paragraphs


def load_pdf(path: Path) -> Book:
    try:
        import pymupdf
    except ImportError:
        import fitz as pymupdf  # older PyMuPDF

    with pymupdf.open(path) as doc:
        meta = doc.metadata or {}
        pages = [_page_paragraphs(p) for p in doc]
        toc = [(title, page - 1) for level, title, page in doc.get_toc(simple=True) if level == 1 and page >= 1]

    chapters: list[Chapter] = []
    toc = sorted({p: t for t, p in reversed(toc)}.items())  # one entry per start page
    if len(toc) >= 2:
        if toc[0][0] > 0:
            chapters.append(Chapter("Начало", join_paragraphs(p for pg in pages[: toc[0][0]] for p in pg)))
        bounds = toc + [(len(pages), "")]
        for (start, title), (end, _) in itertools.pairwise(bounds):
            chapters.append(Chapter(title, join_paragraphs(p for pg in pages[start:end] for p in pg)))
    else:
        for start in range(0, len(pages), PAGES_PER_PART):
            end = min(start + PAGES_PER_PART, len(pages))
            chapters.append(
                Chapter(f"Страницы {start + 1}–{end}", join_paragraphs(p for pg in pages[start:end] for p in pg))
            )

    return Book(
        title=(meta.get("title") or "").strip() or path.stem,
        author=(meta.get("author") or "").strip(),
        path=path,
        chapters=chapters,
    )
