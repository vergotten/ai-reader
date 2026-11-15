"""Find chapters inside one oversized block of text by its heading paragraphs ("Часть первая", "Глава 3", "IV")."""

from __future__ import annotations

import re

from .model import Chapter, split_long

SPLIT_ABOVE = 60_000  # characters; shorter chapters are left as they are
HARD_LIMIT = 120_000  # a chapter still longer than this without headings is cut into numbered parts
_MAX_HEADING = 90
_DECORATION = "*=-–—_~•·.: \t"

# A heading is a keyword alone or followed by its number ("Часть первая", "Глава 3. Название", "Book II").
# Anything else after the keyword is ordinary text: «Книга была старая…», «Глава семьи молчал.»
_ORDINAL = (
    r"(?:(?:двадцать|тридцать|сорок|twenty|thirty)[\s-]+)?"
    r"(?:(?:перв|втор|трет|четв[её]рт|пят|шест|седьм|восьм|девят|десят|одиннадцат|двенадцат|тринадцат|четырнадцат"
    r"|пятнадцат|шестнадцат|семнадцат|восемнадцат|девятнадцат|двадцат|тридцат|сороков)[а-яё]{1,3}"
    r"|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth"
    r"|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
)
_NUMBERED = (
    rf"(?:\s+(?:(?:\d{{1,3}}|[IVXLC]{{1,8}})(?:\s*[.:,—–-]\s*.{{1,70}}|\s+.{{1,70}})?"  # "3", "IV. Название", "3 Название"
    rf"|{_ORDINAL}(?:\s*[.:,—–-]\s*.{{1,70}})?))?"  # "первая", "первая. Название"
)
_PART = re.compile(
    rf"^(?:(?:часть|книга|том|part|book|volume){_NUMBERED}|пролог|эпилог|prologue|epilogue)$", re.IGNORECASE
)
_CHAPTER = re.compile(
    rf"^(?:(?:глава|розділ|chapter){_NUMBERED}"
    r"|предисловие|послесловие|введение|заключение|от автора|preface|introduction|afterword)$",
    re.IGNORECASE,
)
_ROMAN = re.compile(r"^(?=[IVXLC])C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\.?$")


def _heading_level(paragraph: str) -> tuple[int, str]:
    """(1 = part, 2 = chapter, 0 = plain text, cleaned heading)."""
    if len(paragraph) > _MAX_HEADING:
        return 0, ""
    text = paragraph.strip(_DECORATION)
    if not text:
        return 0, ""
    if _PART.match(text):
        return 1, _tidy(text)
    if _CHAPTER.match(text) or _ROMAN.match(text):
        return 2, _tidy(text)
    return 0, ""


def _tidy(heading: str) -> str:
    if heading.isupper() and not _ROMAN.match(heading):
        return heading.capitalize()  # "ЧАСТЬ ПЕРВАЯ" -> "Часть первая"
    return heading


def split_by_headings(chapter: Chapter) -> list[Chapter]:
    """Chapters named "Часть первая — II"; the chapter itself if it has fewer than two headings."""
    paragraphs = chapter.text.split("\n\n")
    marks = [(i, *_heading_level(p)) for i, p in enumerate(paragraphs)]
    marks = [(i, level, title) for i, level, title in marks if level]
    if len(marks) < 2:
        return [chapter]

    out: list[Chapter] = []
    intro = "\n\n".join(paragraphs[: marks[0][0]]).strip()
    if intro:
        out.append(Chapter(chapter.title or "Начало", intro))
    part = ""
    for (index, level, title), following in zip(marks, marks[1:] + [(len(paragraphs), 0, "")], strict=True):
        if level == 1:
            part = title
            name = title
        else:
            name = f"{part} — {title}" if part else title
        text = "\n\n".join(paragraphs[index + 1 : following[0]]).strip()
        if text:
            out.append(Chapter(name, text))
    return out if len(out) >= 2 else [chapter]


def split_oversized(chapter: Chapter) -> list[Chapter]:
    """Books whose file has no chapter markup arrive as one huge chapter: recover the chapters from headings."""
    if len(chapter.text) <= SPLIT_ABOVE:
        return [chapter]
    result: list[Chapter] = []
    for piece in split_by_headings(chapter):
        if len(piece.text) > HARD_LIMIT:
            result.extend(split_long(piece.title or "Текст", piece.text))
        else:
            result.append(piece)
    return result
