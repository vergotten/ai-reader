"""Splitting book text into TTS-sized chunks and normalizing it for speech."""

from __future__ import annotations

import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"[^\n]+")
_SENTENCE_END = re.compile(r"[.!?…]+[»\"”’)\]]*\s+")
_CLAUSE_END = re.compile(r"[,;:]\s+|\s+[—–-]\s+")
_SPACE = re.compile(r"\s+")
_LEVELS = (_SENTENCE_END, _CLAUSE_END, _SPACE)


@dataclass(frozen=True)
class Chunk:
    start: int  # offsets into the source text
    end: int
    text: str
    pause: float  # seconds of silence after this chunk
    voice: str | None = None  # None: the narrator's voice (dialogue lines may get another one)


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _split_by(text: str, s: int, e: int, regex: re.Pattern) -> list[tuple[int, int]]:
    pieces, pos = [], s
    for m in regex.finditer(text, s, e):
        if m.end() >= e:
            break
        pieces.append((pos, m.end()))
        pos = m.end()
    pieces.append((pos, e))
    return pieces


def _pack(pieces: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for s, e in pieces:
        if cur is None:
            cur = (s, e)
        elif e - cur[0] <= max_chars:
            cur = (cur[0], e)
        else:
            out.append(cur)
            cur = (s, e)
    if cur is not None:
        out.append(cur)
    return out


def _split_span(text: str, s: int, e: int, max_chars: int, level: int = 0) -> list[tuple[int, int]]:
    s, e = _strip(text, s, e)
    if s >= e:
        return []
    if e - s <= max_chars:
        return [(s, e)]
    if level >= len(_LEVELS):
        return [(i, min(i + max_chars, e)) for i in range(s, e, max_chars)]
    result = []
    for ps, pe in _pack(_split_by(text, s, e, _LEVELS[level]), max_chars):
        result.extend(_split_span(text, ps, pe, max_chars, level + 1))
    return result


_TRAILING_CLOSERS = "»\"”’)] \t"


def sentence_pause_for(fragment: str, base: float) -> float:
    """A narrator breathes longer after a question, an exclamation or a trailing-off «…» than after a full stop,
    and barely at all inside a sentence that had to be cut at a comma."""
    tail = fragment.rstrip(_TRAILING_CLOSERS)
    if tail.endswith(("…", "...")):
        return base * 1.5
    if tail.endswith(("?", "!")):
        return base * 1.2
    if tail.endswith("."):
        return base
    return base * 0.4


def _sentences(text: str, s: int, e: int, min_chars: int) -> list[tuple[int, int]]:
    """Sentences of text[s:e]; ones shorter than min_chars are joined to the next (some engines garble «Да.»)."""
    out: list[tuple[int, int]] = []
    for ps, pe in _split_by(text, s, e, _SENTENCE_END):
        ps, pe = _strip(text, ps, pe)
        if ps >= pe:
            continue
        if out and out[-1][1] - out[-1][0] < min_chars:
            out[-1] = (out[-1][0], pe)
        else:
            out.append((ps, pe))
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_chars:
        last = out.pop()
        out[-1] = (out[-1][0], last[1])
    return out


def chunk_text(
    text: str,
    max_chars: int = 400,
    sentence_pause: float = 0.15,
    paragraph_pause: float = 0.5,
    start: int = 0,
    end: int | None = None,
    min_chars: int = 0,
) -> list[Chunk]:
    """Split text[start:end] into chunks: one sentence each (long ones cut at clauses to fit max_chars).

    Each line is a paragraph. Every sentence ends with its own pause, so the "pause between phrases"
    setting really works between phrases (earlier, several sentences went to the engine at once and the
    model ran them together with its own tiny pause, which sounded rushed).
    Chunk offsets refer to the original text so the GUI can highlight them.
    """
    end = len(text) if end is None else end
    chunks: list[Chunk] = []
    for m in _PARAGRAPH.finditer(text, start, end):
        sentences = _sentences(text, m.start(), m.end(), min_chars)
        for si, (ss, se) in enumerate(sentences):
            spans = _split_span(text, ss, se, max_chars)
            for i, (s, e) in enumerate(spans):
                piece = text[s:e]
                if i < len(spans) - 1:
                    pause = sentence_pause_for(piece, sentence_pause)
                elif si == len(sentences) - 1:
                    pause = paragraph_pause
                else:
                    pause = sentence_pause_for(piece, sentence_pause)
                chunks.append(Chunk(s, e, piece, pause))
    return chunks


_REPLACEMENTS = str.maketrans(
    {
        "«": '"', "»": '"', "„": '"', "“": '"', "”": '"', "‘": "'", "’": "'",
        " ": " ", " ": " ", " ": " ", "­": None, "\u200b": None,
        "*": None, "#": None, "_": None, "|": None, "~": None,
    }
)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_FOOTNOTE = re.compile(r"\[\d{1,4}\]|\{\d{1,4}\}")
_LEADING_DASH = re.compile(r"^\s*[—–-]\s*")
_DASH_AFTER_PUNCT = re.compile(r"([.!?…,;:])\s+[—–]\s+")
_DASH_IN_SENTENCE = re.compile(r"\s+[—–]\s+")
_SPEAKABLE = re.compile(r"[^\W\d_]")

# --- Russian: ordinals in context (years, centuries, "5-го"), Roman numerals
_ROMAN = r"(?=[IVXLC]+\b)C{0,3}(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
_NOT_DECIMAL = r"(?<![\d.,])"
_RU_YEAR = re.compile(rf"{_NOT_DECIMAL}\b(\d{{1,4}})\s+(год|года|году|годом|годе)\b", re.IGNORECASE)
_RU_YEAR_ABBR = re.compile(rf"{_NOT_DECIMAL}\b(\d{{3,4}})\s*гг?\.(?=\s|$|[,;:)])")
_RU_CENTURY = re.compile(rf"{_NOT_DECIMAL}\b(\d{{1,2}}|{_ROMAN})\s+(век|века|веке|веку|веком|веков|веках)\b", re.IGNORECASE)
_RU_SUFFIX = re.compile(rf"{_NOT_DECIMAL}\b(\d+)\s?-\s?([а-яё]{{1,3}})\b", re.IGNORECASE)
_RU_ROMAN_ALONE = re.compile(rf"(?<![°\-\w])\b({_ROMAN})\b(?![+#\-])")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:])")
_SENTENCE_START = re.compile(r"\s*(?:$|[А-ЯЁA-Z«\"—–-])")
_RU_WORD_CASE = {
    "год": "nominative", "года": "genitive", "году": "prepositional", "годом": "instrumental", "годе": "prepositional",
    "век": "nominative", "века": "genitive", "веке": "prepositional", "веку": "dative", "веком": "instrumental",
    "веков": "genitive", "веках": "prepositional",
}
_RU_SUFFIX_FORM = [  # (endings, case, gender, plural)
    (("ого", "его", "го"), "genitive", "m", False),
    (("ому", "ему", "му"), "dative", "m", False),
    (("ым", "им"), "instrumental", "m", False),
    (("ом", "ем", "м"), "prepositional", "m", False),
    (("ый", "ий", "ой", "й"), "nominative", "m", False),
    (("ая", "яя", "я"), "nominative", "f", False),
    (("ую", "юю", "ю"), "accusative", "f", False),
    (("ое", "ее", "е"), "nominative", "n", False),
    (("ых", "их", "х"), "genitive", "m", True),
    (("ые", "ие"), "nominative", "m", True),
]


def _roman_to_int(s: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = values[a]
        total += -v if b != " " and values[b] > v else v
    return total


def _to_int(s: str) -> int:
    return int(s) if s.isdigit() else _roman_to_int(s.upper())


def _expand_ru(text: str, num2words) -> str:
    def ordinal(n: int, case: str, gender: str = "m", plural: bool = False) -> str:
        return num2words(n, lang="ru", to="ordinal", case=case, gender=gender, plural=plural)

    def year(m: re.Match) -> str:
        return f"{ordinal(int(m.group(1)), _RU_WORD_CASE[m.group(2).lower()])} {m.group(2)}"

    def year_abbr(m: re.Match) -> str:
        after_v = re.search(r"\bв\s*$", m.string[: m.start()], re.IGNORECASE)
        words = f"{ordinal(int(m.group(1)), 'prepositional')} году" if after_v else f"{ordinal(int(m.group(1)), 'genitive')} года"
        # the abbreviation dot may also end the sentence
        return words + "." if _SENTENCE_START.match(m.string, m.end()) else words

    def century(m: re.Match) -> str:
        word = m.group(2).lower()
        return f"{ordinal(_to_int(m.group(1)), _RU_WORD_CASE[word], plural=word in ('веков', 'веках'))} {m.group(2)}"

    def suffixed(m: re.Match) -> str:
        ending = m.group(2).lower()
        for endings, case, gender, plural in _RU_SUFFIX_FORM:
            if ending in endings:
                return ordinal(int(m.group(1)), case, gender, plural)
        return m.group(0)

    def roman(m: re.Match) -> str:
        if not m.group(1):
            raise ValueError("empty")
        return num2words(_roman_to_int(m.group(1)), lang="ru")

    for regex, fn in ((_RU_CENTURY, century), (_RU_YEAR, year), (_RU_YEAR_ABBR, year_abbr), (_RU_SUFFIX, suffixed)):
        text = regex.sub(lambda m, fn=fn: _safe(fn, m), text)
    return _RU_ROMAN_ALONE.sub(lambda m: _safe(roman, m), text)


def _safe(fn, m: re.Match) -> str:
    try:
        return f" {fn(m)} "
    except Exception:
        return m.group(0)


def expand_numbers(text: str, lang: str = "ru") -> str:
    try:
        from num2words import num2words
    except ImportError:
        return text

    if lang == "ru":
        text = _expand_ru(text, num2words)

    def cardinal(m: re.Match) -> str:
        s = m.group(0).replace(",", ".")
        value = float(s) if "." in s else int(s)
        return num2words(value, lang=lang)

    return _NUMBER.sub(lambda m: _safe(cardinal, m), text)


def normalize(text: str, lang: str = "ru", numbers: bool = True) -> str:
    text = _FOOTNOTE.sub("", text.translate(_REPLACEMENTS))
    text = _LEADING_DASH.sub("", text)
    text = _DASH_AFTER_PUNCT.sub(r"\1 ", text)
    text = _DASH_IN_SENTENCE.sub(", ", text)
    if numbers:
        text = expand_numbers(text, lang)
    if lang in ("ru", "uk"):
        # Russian voices skip Latin letters entirely ("рефери NFL" -> "рефери"); spell them in Cyrillic
        from .latin import transliterate

        text = transliterate(text)
    return _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(text.split()))


def is_speakable(text: str) -> bool:
    return bool(_SPEAKABLE.search(text))
