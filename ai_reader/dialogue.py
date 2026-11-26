"""Find dialogue in Russian prose so lines can be read by a male or a female voice.

Russian books mark speech with a dash at the start of the paragraph, the author's words are set off
by «, — » / «. — »:

    — Привет, — сказала Маша. — Как дела?
    Он ответил: — Хорошо.

The speaker's gender comes from the author's words ("сказала", "она", "ответил"). A line without
them gets the gender of the conversation: when two different genders were just talking, lines
alternate; otherwise the last speaker continues. It is a heuristic, not a parser: it is right on most
dialogue and harmless when wrong (the line is still read, only by the other voice).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NARRATOR, SPEECH = "narrator", "speech"

_DASH_START = re.compile(r"\s*[—–-]\s+")  # used with .match(text, pos): anchored at pos
# a dash that switches between speech and the author's words: after punctuation, spaced on both sides
_SWITCH = re.compile(r"(?<=[,.!?…:])\s+[—–]\s+")
_COLON_DASH = re.compile(r":\s+[—–]\s+")
_LINE = re.compile(r"[^\n]+")
_WORD = re.compile(r"[а-яё]+", re.IGNORECASE)
_FEMALE_PRON = {"она", "девушка", "женщина", "девочка", "мать", "мама", "бабушка", "старуха", "жена", "сестра", "дочь", "госпожа", "королева", "хозяйка", "подруга"}
_MALE_PRON = {"он", "парень", "мужчина", "мальчик", "отец", "папа", "дедушка", "старик", "муж", "брат", "сын", "господин", "король", "хозяин", "друг"}
_NOT_VERBS = {"стол", "пол", "зал", "угол", "пел", "сила", "мила", "была", "дела", "тела", "села", "земля", "пила", "стрела", "мол", "долг", "школа", "игла", "скала", "мгла", "вол", "посол", "футбол", "тепла", "светла", "мало", "тело", "дело", "било", "весла", "колокол", "ангел", "дятел", "орёл", "котёл", "козёл"}


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    role: str  # NARRATOR or SPEECH
    gender: str | None = None  # "m" / "f" for SPEECH when known or guessed


def _gender_of(words_text: str) -> str | None:
    words = [w.lower() for w in _WORD.findall(words_text)][:5]
    for w in words:
        if w in _FEMALE_PRON:
            return "f"
        if w in _MALE_PRON:
            return "m"
    for w in words:
        if w in _NOT_VERBS or len(w) < 3:
            continue
        if w.endswith(("ала", "ила", "ела", "яла", "ыла", "ула", "ёла", "лась")):
            return "f"
        if w.endswith(("ал", "ил", "ел", "ял", "ыл", "ул", "ёл", "лся")):
            return "m"
    return None


def _strip(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _split_line(text: str, s: int, e: int) -> list[tuple[int, int, str]]:
    """Spans of one paragraph with their role, or [] when the paragraph has no dialogue."""
    m = _DASH_START.match(text, s, e)
    lead: list[tuple[int, int, str]] = []
    if m:
        pos = m.end()
    else:
        c = _COLON_DASH.search(text, s, e)  # «Он ответил: — Хорошо.»
        if not c:
            return []
        lead = [(s, c.start() + 1, NARRATOR)]
        pos = c.end()
    out = list(lead)
    role = SPEECH
    for sw in _SWITCH.finditer(text, pos, e):
        out.append((pos, sw.start(), role))
        pos = sw.end()
        role = NARRATOR if role == SPEECH else SPEECH
    out.append((pos, e, role))
    return [(a, b, r) for a, b, r in (( *_strip(text, a, b), r) for a, b, r in out) if a < b]


def segment(text: str, start: int = 0, end: int | None = None) -> list[Segment]:
    """Paragraph-by-paragraph segments of text[start:end]; narration paragraphs come back whole."""
    end = len(text) if end is None else end
    out: list[Segment] = []
    history: list[str] = []  # genders of recent dialogue paragraphs
    prev_dialogue = False
    for m in _LINE.finditer(text, start, end):
        spans = _split_line(text, m.start(), m.end())
        if not spans:
            s, e = _strip(text, m.start(), m.end())
            if s < e:
                out.append(Segment(s, e, NARRATOR))
            prev_dialogue = False
            continue
        author = " ".join(text[a:b] for a, b, r in spans if r == NARRATOR)
        gender = _gender_of(author)
        if gender is None and prev_dialogue and history:
            mixed = len(history) >= 2 and history[-1] != history[-2]
            gender = ("m" if history[-1] == "f" else "f") if mixed else history[-1]
        if gender is not None:
            history = (history + [gender])[-4:]
        prev_dialogue = True
        out.extend(Segment(a, b, r, gender if r == SPEECH else None) for a, b, r in spans)
    return out
