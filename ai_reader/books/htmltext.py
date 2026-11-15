from __future__ import annotations

import re

from lxml import etree
from lxml import html as lhtml

BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "dd", "dt", "tr", "td", "th", "table",
    "blockquote", "section", "article", "aside", "header", "footer", "pre",
    "figcaption", "h1", "h2", "h3", "h4", "h5", "h6", "hr",
}
_FOOTNOTE_MARK = re.compile(r"\s*[\[(]?\d{1,4}[\])]?\s*")


def html_to_paragraphs(content: bytes | str) -> tuple[list[str], str | None]:
    """Extract plain-text paragraphs and the first heading from an (X)HTML document."""
    try:
        doc = lhtml.fromstring(content)
    except (etree.ParserError, ValueError):
        return [], None

    drop = []
    for el in doc.iter():
        if not isinstance(el.tag, str):
            continue
        if el.tag in ("script", "style", "head", "title"):
            drop.append(el)
        elif el.get("epub:type") in ("noteref", "footnote", "endnote", "rearnote"):
            drop.append(el)
        elif el.tag == "sup" and _FOOTNOTE_MARK.fullmatch(el.text_content() or ""):
            drop.append(el)
    for el in drop:
        if el.getparent() is not None:
            el.drop_tree()

    heading = None
    for el in doc.iter("h1", "h2", "h3"):
        text = " ".join(el.text_content().split())
        if text:
            heading = text
            break

    for el in doc.iter():
        if isinstance(el.tag, str) and el.tag in BLOCK_TAGS:
            el.text = "\n" + (el.text or "")
            el.tail = "\n" + (el.tail or "")

    paragraphs = [" ".join(line.split()) for line in doc.text_content().split("\n")]
    return [p for p in paragraphs if p], heading
