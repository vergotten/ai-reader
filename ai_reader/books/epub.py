from __future__ import annotations

import posixpath
import zipfile
from pathlib import Path
from urllib.parse import unquote

from lxml import etree

from .htmltext import html_to_paragraphs
from .model import Book, Chapter, join_paragraphs

_OPS_TYPE = "{http://www.idpf.org/2007/ops}type"
_PARSER = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)


def _resolve(base: str, href: str) -> str:
    href = unquote(href.split("#", 1)[0])
    return posixpath.normpath(posixpath.join(base, href)) if href else ""


def _xml(z: zipfile.ZipFile, name: str):
    return etree.fromstring(z.read(name), _PARSER)


def _first_text(parent, tag: str) -> str:
    el = parent.find(tag) if parent is not None else None
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def _nav_titles(z: zipfile.ZipFile, nav_path: str) -> dict[str, str]:
    root = _xml(z, nav_path)
    if root is None:
        return {}
    navs = list(root.iter("{*}nav"))
    toc = next((n for n in navs if n.get(_OPS_TYPE) == "toc"), navs[0] if navs else root)
    base = posixpath.dirname(nav_path)
    titles: dict[str, str] = {}
    for a in toc.iter("{*}a"):
        href, text = a.get("href"), " ".join("".join(a.itertext()).split())
        if href and text:
            titles.setdefault(_resolve(base, href), text)
    return titles


def _ncx_titles(z: zipfile.ZipFile, ncx_path: str) -> dict[str, str]:
    root = _xml(z, ncx_path)
    if root is None:
        return {}
    base = posixpath.dirname(ncx_path)
    titles: dict[str, str] = {}
    for point in root.iter("{*}navPoint"):
        label = point.find("{*}navLabel/{*}text")
        content = point.find("{*}content")
        if label is None or content is None or not content.get("src"):
            continue
        text = " ".join("".join(label.itertext()).split())
        if text:
            titles.setdefault(_resolve(base, content.get("src")), text)
    return titles


def load_epub(path: Path) -> Book:
    with zipfile.ZipFile(path) as z:
        container = _xml(z, "META-INF/container.xml")
        opf_path = container.find(".//{*}rootfile").get("full-path")
        opf = _xml(z, opf_path)
        base = posixpath.dirname(opf_path)

        meta = opf.find("{*}metadata")
        title = _first_text(meta, "{*}title")
        author = ", ".join(
            " ".join("".join(c.itertext()).split()) for c in (meta.findall("{*}creator") if meta is not None else [])
        )
        lang = _first_text(meta, "{*}language") or None

        manifest: dict[str, tuple[str, str, str]] = {}
        for item in opf.iterfind("{*}manifest/{*}item"):
            if item.get("id") and item.get("href"):
                manifest[item.get("id")] = (
                    _resolve(base, item.get("href")),
                    item.get("media-type") or "",
                    item.get("properties") or "",
                )

        titles: dict[str, str] = {}
        for href, _, props in manifest.values():
            if "nav" in props.split():
                try:
                    titles.update(_nav_titles(z, href))
                except KeyError:
                    pass
        spine = opf.find("{*}spine")
        toc_id = spine.get("toc") if spine is not None else None
        if toc_id in manifest:
            try:
                for k, v in _ncx_titles(z, manifest[toc_id][0]).items():
                    titles.setdefault(k, v)
            except KeyError:
                pass

        chapters: list[Chapter] = []
        for ref in spine.iterfind("{*}itemref") if spine is not None else []:
            item = manifest.get(ref.get("idref"))
            if item is None or ref.get("linear") == "no" or "html" not in item[1]:
                continue
            try:
                content = z.read(item[0])
            except KeyError:
                continue
            paragraphs, heading = html_to_paragraphs(content)
            if not paragraphs:
                continue
            chapters.append(Chapter(titles.get(item[0]) or heading or "", join_paragraphs(paragraphs)))

    return Book(title=title or path.stem, author=author, language=lang, path=path, chapters=chapters)
