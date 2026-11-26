"""Latin words inside Russian text, spelled out in Cyrillic so Russian voices read them.

Russian TTS models (Silero above all) silently drop every character outside their alphabet, so
"бывший рефери NFL" came out as "бывший рефери". Acronyms are read letter by letter ("эн эф эл"),
words get an English-like transliteration ("Google" -> "Гугл", "Smith" -> "Смит"). It is a heuristic:
for a word it gets wrong, add it to the pronunciation dictionary (the dictionary is applied first).
"""

from __future__ import annotations

import re

LETTER_NAMES = {
    "A": "эй", "B": "би", "C": "си", "D": "ди", "E": "и", "F": "эф", "G": "джи", "H": "эйч", "I": "ай",
    "J": "джей", "K": "кей", "L": "эл", "M": "эм", "N": "эн", "O": "оу", "P": "пи", "Q": "кью", "R": "ар",
    "S": "эс", "T": "ти", "U": "ю", "V": "ви", "W": "дабл-ю", "X": "экс", "Y": "уай", "Z": "зед",
}

# acronyms that are said as words, and frequent words that the rules below get wrong
WORDS = {
    "nasa": "наса", "nato": "нато", "unesco": "юнеско", "unicef": "юнисеф", "fifa": "фифа", "uefa": "уефа",
    "ikea": "икеа", "covid": "ковид", "laser": "лазер", "radar": "радар", "opec": "опек", "ok": "окей",
    "okay": "окей", "the": "зе", "and": "энд", "of": "оф", "a": "э", "an": "эн", "to": "ту", "in": "ин",
    "is": "из", "it": "ит", "for": "фор", "you": "ю", "your": "ёр", "my": "май", "we": "ви", "me": "ми",
    "new": "нью", "york": "йорк", "news": "ньюс", "one": "уан", "two": "ту", "three": "фри", "love": "лав",
    "google": "гугл", "apple": "эпл", "microsoft": "майкрософт", "windows": "виндоус", "iphone": "айфон",
    "ipad": "айпэд", "youtube": "ютуб", "facebook": "фейсбук", "instagram": "инстаграм", "twitter": "твиттер",
    "telegram": "телеграм", "whatsapp": "вотсап", "amazon": "амазон", "netflix": "нетфликс", "tesla": "тесла",
    "email": "имейл", "e-mail": "имейл", "online": "онлайн", "offline": "офлайн", "internet": "интернет",
    "wi-fi": "вай-фай", "wifi": "вайфай", "bluetooth": "блютус", "smartphone": "смартфон", "manager": "менеджер",
    "business": "бизнес", "coach": "коуч", "fitness": "фитнес", "club": "клаб", "team": "тим", "start": "старт",
    "startup": "стартап", "marketing": "маркетинг", "brand": "бренд", "money": "мани", "time": "тайм",
    "life": "лайф", "mind": "майнд", "people": "пипл", "world": "ворлд", "game": "гейм", "show": "шоу",
    "street": "стрит", "house": "хаус", "home": "хоум", "office": "офис", "rich": "рич", "dad": "дэд",
    "poor": "пур", "think": "синк", "grow": "гроу", "john": "джон", "michael": "майкл", "david": "дэвид",
    "james": "джеймс", "george": "джордж", "steve": "стив", "jobs": "джобс", "bill": "билл", "gates": "гейтс",
    "mark": "марк", "elon": "илон", "musk": "маск", "jeff": "джефф", "bezos": "безос", "warren": "уоррен",
    "buffett": "баффет", "napoleon": "наполеон", "hill": "хилл", "tony": "тони", "robbins": "роббинс",
    "harry": "гарри", "potter": "поттер", "london": "лондон", "america": "америка", "english": "инглиш",
    "mr": "мистер", "mrs": "миссис", "ms": "мисс", "dr": "доктор", "inc": "инк", "ltd": "лтд", "vs": "версус",
}

_ROMAN_ONLY = re.compile(r"^[IVXLCDM]+$")
_TOKEN = re.compile(r"[A-Za-z]+(?:[-'’][A-Za-z]+)*")

# longest patterns first; applied left to right on the lowercase word
_RULES: list[tuple[str, str]] = [
    ("tion", "шн"), ("sion", "жн"), ("ture", "чер"), ("ough", "оу"), ("augh", "о"), ("igh", "ай"),
    ("sch", "ш"), ("tch", "ч"), ("dge", "дж"), ("ck", "к"), ("sh", "ш"), ("ch", "ч"), ("th", "т"),
    ("ph", "ф"), ("wh", "у"), ("qu", "кв"), ("kn", "н"), ("wr", "р"), ("hn", "н"), ("gh", "г"), ("ng", "нг"),
    ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ie", "и"), ("ei", "эй"), ("ai", "эй"), ("ay", "эй"),
    ("ey", "эй"), ("oy", "ой"), ("oi", "ой"), ("ou", "ау"), ("ow", "оу"), ("au", "о"), ("aw", "о"),
    ("oa", "оу"), ("ue", "ю"), ("ew", "ью"), ("er", "ер"), ("ar", "ар"), ("or", "ор"), ("ir", "ер"),
    ("ur", "ер"), ("x", "кс"), ("j", "дж"), ("w", "у"), ("h", "х"),
    ("a", "а"), ("b", "б"), ("d", "д"), ("e", "е"), ("f", "ф"), ("g", "г"), ("i", "и"), ("k", "к"),
    ("l", "л"), ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("r", "р"), ("s", "с"), ("t", "т"),
    ("u", "у"), ("v", "в"), ("z", "з"),
]
_MAGIC_E = re.compile(r"^(.*?)([aiou])([bcdfgklmnprstvz])e(s?)$")  # "Mike" -> "майк", "home" -> "хоум"
_LONG = {"a": "эй", "i": "ай", "o": "оу", "u": "ю"}
_VOWELS_RU = "аеёиоуыэюя"


def _rules(word: str) -> str:
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch == "c":
            out.append("с" if word[i + 1 : i + 2] in ("e", "i", "y") else "к")
            i += 1
            continue
        if ch == "y":
            out.append("й" if i == 0 or (i + 1 < len(word) and word[i + 1] in "aeiou") else "и")
            i += 1
            continue
        if ch == "e" and (i == 0 or (out and out[-1][-1] in _VOWELS_RU)) and not word.startswith(("er", "ew", "ea", "ee", "ei", "ey"), i):
            out.append("э")
            i += 1
            continue
        if i > 0 and ch == word[i - 1] and ch not in "aeiou":  # doubled consonant
            i += 1
            continue
        for pattern, value in _RULES:
            if word.startswith(pattern, i):
                out.append(value)
                i += len(pattern)
                break
        else:
            i += 1  # apostrophes and the like
    return "".join(out)


def transliterate_word(word: str) -> str:
    """One Latin word -> Cyrillic. Case of the first letter is kept."""
    lower = word.lower().replace("’", "'")
    if lower in WORDS:
        ru = WORDS[lower]
    elif word.isupper() and len(word) > 1 and len(word) <= 6 and "-" not in word:
        ru = " ".join(LETTER_NAMES[c] for c in word)  # NFL -> эн эф эл
        return ru
    elif len(word) == 1 and word.isupper():
        return LETTER_NAMES[word]
    else:
        parts = []
        for part in re.split(r"([-'])", lower):
            if part in ("-", "'") or not part:
                parts.append("-" if part == "-" else "")
                continue
            if part in WORDS:
                parts.append(WORDS[part])
                continue
            prefix = ""
            if part.startswith("mc") and len(part) > 3:  # McDonald -> Макдоналд
                prefix, part = "мак", part[2:]
            m = _MAGIC_E.match(part) if len(part) >= 4 else None
            if m:
                head, vowel, cons, plural = m.groups()
                parts.append(prefix + _rules(head) + _LONG[vowel] + _rules(cons) + ("с" if plural else ""))
            else:
                parts.append(prefix + _rules(part))
        ru = "".join(parts)
    return ru[:1].upper() + ru[1:] if word[:1].isupper() else ru


def transliterate(text: str, keep_roman: bool = True) -> str:
    """Replace every Latin word in text. Roman numerals ("XX век") are left for the number expander."""

    def repl(m: re.Match) -> str:
        word = m.group(0)
        if keep_roman and _ROMAN_ONLY.match(word) and word not in ("I", "D", "M", "C", "L", "CD", "DC", "MD", "DM", "MC", "CM", "LCD"):
            return word
        return transliterate_word(word)

    return _TOKEN.sub(repl, text)
