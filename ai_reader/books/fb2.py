from __future__ import annotations

import re
import zipfile
from pathlib import Path

from lxml import etree

from .model import Book, Chapter, join_paragraphs

_TEXT_TAGS = {"p", "subtitle", "text-author", "v", "td", "th"}
_CONTAINER_TAGS = {"poem", "stanza", "cite", "epigraph", "table", "tr"}
_SKIP_BODIES = {"notes", "comments", "footnotes"}


_CYRILLIC = re.compile(r"[а-яёіїєґ]", re.IGNORECASE)
_LATIN = re.compile(r"[a-z]", re.IGNORECASE)
_SCRIPT_SHARE = 0.8  # a column or paragraph "is in" a script when this share of its letters belongs to it


def _script(text: str) -> str | None:
    """'cyrillic' | 'latin' | None for mixed or letterless text."""
    cyrillic, latin = len(_CYRILLIC.findall(text)), len(_LATIN.findall(text))
    total = cyrillic + latin
    if not total:
        return None
    if cyrillic >= total * _SCRIPT_SHARE:
        return "cyrillic"
    if latin >= total * _SCRIPT_SHARE:
        return "latin"
    return None


def _drop_translation(root, lang: str | None) -> bool:
    """Parallel editions keep the original and a translation in two table columns: keep the book's language.

    Returns True if such tables were found; then stray paragraphs in the other script are removed as well.
    """
    keep = "latin" if (lang or "ru").lower()[:2] in ("en", "de", "fr", "es", "it", "pl") else "cyrillic"
    found = False
    for table in list(root.iter("table")):
        rows = [[c for c in tr if c.tag in ("td", "th")] for tr in table.iter("tr")]
        if not rows or any(len(cells) != 2 for cells in rows):
            continue
        columns = [_script(" ".join(_text(cells[i]) for cells in rows)) for i in (0, 1)]
        if None in columns or columns[0] == columns[1]:
            continue
        drop = columns.index("latin" if keep == "cyrillic" else "cyrillic")
        for cells in rows:
            cells[drop].getparent().remove(cells[drop])
        found = True
    if found:
        for p in list(root.iter("p")):
            script = _script(_text(p))
            if script and script != keep and p.getparent() is not None and p.getparent().tag in ("section", "body"):
                p.getparent().remove(p)
    return found


def _read_fb2_bytes(path: Path) -> bytes:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".fb2")]
            if not names:
                raise ValueError("В архиве нет файла .fb2")
            return z.read(names[0])
    return path.read_bytes()


def _text(el) -> str:
    return " ".join("".join(el.itertext()).split())


def _title(section) -> str:
    title = section.find("title")
    if title is None:
        return ""
    parts = [_text(p) for p in title if isinstance(p.tag, str)]
    return ". ".join(p for p in parts if p) or _text(title)


def _collect(el, out: list[str]) -> None:
    for child in el:
        tag = child.tag
        if not isinstance(tag, str) or tag in ("section", "title", "image", "empty-line", "annotation"):
            continue
        if tag in _TEXT_TAGS:
            out.append(_text(child))
        elif tag in _CONTAINER_TAGS:
            if tag == "poem" and child.find("title") is not None:
                out.append(_title(child))
            _collect(child, out)


def _walk(section, parents: list[str], chapters: list[Chapter]) -> None:
    own = _title(section)
    titles = parents + [own] if own else parents
    paragraphs: list[str] = []
    _collect(section, paragraphs)
    if paragraphs:
        name = " — ".join(titles[-2:]) if own and len(titles) > 1 else (own or (titles[-1] if titles else ""))
        chapters.append(Chapter(name, join_paragraphs(paragraphs)))
    for sub in section.findall("section"):
        _walk(sub, titles, chapters)


def _remove_note_links(root) -> None:
    for a in list(root.iter("a")):
        if a.get("type") != "note":
            continue
        parent = a.getparent()
        if parent is None:
            continue
        tail = a.tail or ""
        prev = a.getprevious()
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
        parent.remove(a)


def load_fb2(path: Path) -> Book:
    parser = etree.XMLParser(recover=True, huge_tree=True, remove_comments=True, resolve_entities=False, no_network=True)
    root = etree.fromstring(_read_fb2_bytes(path), parser)
    if root is None:
        raise ValueError("Не удалось разобрать FB2")

    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    for binary in root.findall("binary"):
        root.remove(binary)
    _remove_note_links(root)

    info = root.find("description/title-info")
    title, author, lang = "", "", None
    if info is not None:
        title = _text(info.find("book-title")) if info.find("book-title") is not None else ""
        authors = []
        for a in info.findall("author"):
            name = " ".join(
                _text(a.find(tag)) for tag in ("first-name", "middle-name", "last-name") if a.find(tag) is not None
            ).strip()
            if not name and a.find("nickname") is not None:
                name = _text(a.find("nickname"))
            if name:
                authors.append(name)
        author = ", ".join(authors)
        if info.find("lang") is not None:
            lang = _text(info.find("lang")) or None

    _drop_translation(root, lang)
    chapters: list[Chapter] = []
    for body in root.findall("body"):
        if (body.get("name") or "").lower() in _SKIP_BODIES:
            continue
        loose: list[str] = []
        _collect(body, loose)
        if loose:
            chapters.append(Chapter(_title(body) or title, join_paragraphs(loose)))
        for section in body.findall("section"):
            _walk(section, [], chapters)

    return Book(title=title or path.stem, author=author, language=lang, path=path, chapters=chapters)
