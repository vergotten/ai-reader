from __future__ import annotations

import zipfile
from pathlib import Path

from .headings import split_oversized
from .model import Book, Chapter, finalize, join_paragraphs, split_long

FILE_FILTER = "Книги (*.txt *.fb2 *.fb2.zip *.zip *.epub *.pdf *.html *.htm *.xhtml);;Все файлы (*)"


def load_book(path: str | Path) -> Book:
    path = Path(path)
    name = path.name.lower()

    if name.endswith((".fb2", ".fb2.zip")):
        from .fb2 import load_fb2

        book = load_fb2(path)
    elif name.endswith(".epub"):
        from .epub import load_epub

        book = load_epub(path)
    elif name.endswith(".pdf"):
        from .pdf import load_pdf

        book = load_pdf(path)
    elif name.endswith((".html", ".htm", ".xhtml")):
        from .htmltext import html_to_paragraphs

        paragraphs, heading = html_to_paragraphs(path.read_bytes())
        book = Book(title=heading or path.stem, path=path, chapters=split_long("Текст", join_paragraphs(paragraphs)))
    elif name.endswith(".zip") and zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            if not any(n.lower().endswith(".fb2") for n in z.namelist()):
                raise ValueError("Поддерживаются только ZIP-архивы с FB2")
        from .fb2 import load_fb2

        book = load_fb2(path)
    else:
        from .txt import load_txt

        book = load_txt(path)

    book.chapters = [part for chapter in book.chapters for part in split_oversized(chapter)]
    book = finalize(book)
    if not book.chapters:
        raise ValueError("В книге не найден текст (возможно, это скан без текстового слоя)")
    return book


__all__ = ["FILE_FILTER", "Book", "Chapter", "load_book"]
