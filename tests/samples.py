"""Small books in every supported format, built on the fly so the tests need no files from outside."""

from __future__ import annotations

import zipfile
from pathlib import Path

CHAPTERS = [
    ("Глава 1. Начало", ["Жил-был на свете кот.", "Он любил слушать книги вслух."]),
    ("Глава 2. Продолжение", ["Однажды кот нашёл старую книгу.", "И прочитал её до конца."]),
]
PHRASE = "кот нашёл старую книгу"
LATIN_CHAPTERS = [
    ("Chapter 1", ["Once upon a time there was a cat.", "It liked audiobooks."]),
    ("Chapter 2", ["One day the cat found an old book.", "It read the book to the end."]),
]
LATIN_PHRASE = "the cat found an old book"


def _plain_text() -> str:
    return "\n\n".join(f"{title}\n\n" + "\n\n".join(paragraphs) for title, paragraphs in CHAPTERS) + "\n"


def make_txt(folder: Path, encoding: str = "utf-8") -> Path:
    path = folder / f"sample-{encoding}.txt"
    path.write_bytes(_plain_text().encode(encoding))
    return path


def _fb2_xml() -> str:
    sections = "".join(
        f"<section><title><p>{title}</p></title>{''.join(f'<p>{p}</p>' for p in paragraphs)}</section>"
        for title, paragraphs in CHAPTERS
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0"><description><title-info>'
        "<author><first-name>Иван</first-name><last-name>Тестов</last-name></author>"
        "<book-title>Пробная книга</book-title><lang>ru</lang></title-info></description>"
        f"<body>{sections}</body></FictionBook>"
    )


def make_fb2(folder: Path) -> Path:
    path = folder / "sample.fb2"
    path.write_text(_fb2_xml(), encoding="utf-8")
    return path


def make_fb2_zip(folder: Path) -> Path:
    path = folder / "sample.fb2.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample.fb2", _fb2_xml())
    return path


def make_parallel_fb2(folder: Path) -> Path:
    """A bilingual edition: original and translation side by side in a table, no chapter markup."""
    rows = "<tr><td>PART I</td><td>* ЧАСТЬ ПЕРВАЯ *</td></tr>"
    for number in ("I", "II"):
        rows += f"<tr><td>CHAPTER {number}</td><td>{number}</td></tr>"
        rows += "<tr><td>The book was old.</td><td>Книга была старая, подержанная.</td></tr>" * 1500
    path = folder / "parallel.fb2"
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0">'
        "<description><title-info><book-title>Двуязычная</book-title><lang>ru</lang></title-info></description>"
        f"<body><section><table>{rows}</table></section></body></FictionBook>",
        encoding="utf-8",
    )
    return path


def make_epub(folder: Path) -> Path:
    path = folder / "sample.epub"
    items = "".join(f'<item id="c{i}" href="c{i}.xhtml" media-type="application/xhtml+xml"/>' for i in (1, 2))
    refs = "".join(f'<itemref idref="c{i}"/>' for i in (1, 2))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
            '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
            "</rootfiles></container>",
        )
        archive.writestr(
            "OEBPS/content.opf",
            '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Пробная книга</dc:title>'
            "<dc:creator>Иван Тестов</dc:creator><dc:language>ru</dc:language></metadata>"
            f"<manifest>{items}</manifest><spine>{refs}</spine></package>",
        )
        for number, (title, paragraphs) in enumerate(CHAPTERS, 1):
            body = f"<h1>{title}</h1>" + "".join(f"<p>{p}</p>" for p in paragraphs)
            archive.writestr(
                f"OEBPS/c{number}.xhtml",
                f'<?xml version="1.0" encoding="UTF-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>{title}</title></head><body>{body}</body></html>",
            )
    return path


def make_html(folder: Path) -> Path:
    path = folder / "sample.html"
    body = "".join(f"<h2>{t}</h2>" + "".join(f"<p>{p}</p>" for p in ps) for t, ps in CHAPTERS)
    path.write_text(
        f'<html><head><meta charset="utf-8"><title>Пробная книга</title><style>p{{color:red}}</style></head>'
        f"<body><script>var x = 1;</script>{body}</body></html>",
        encoding="utf-8",
    )
    return path


def make_pdf(folder: Path) -> Path:
    import pymupdf

    path = folder / "sample.pdf"
    document = pymupdf.open()
    for title, paragraphs in LATIN_CHAPTERS:  # the built-in PDF fonts have no Cyrillic
        page = document.new_page()
        page.insert_text((72, 90), title, fontsize=18)
        for line, paragraph in enumerate(paragraphs):
            page.insert_text((72, 140 + 30 * line), paragraph, fontsize=12)
    document.set_toc([[1, title, number] for number, (title, _p) in enumerate(LATIN_CHAPTERS, 1)])
    document.save(path)
    document.close()
    return path
