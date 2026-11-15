"""Russian stress marks and the user's pronunciation dictionary.

Stress is written as "+" before the stressed vowel ("з+амок"), the convention used by Silero, RUAccent
and the Russian F5-TTS fine-tunes. Engines that don't understand "+" get the text with the marks removed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .paths import dictionary_path

Log = Callable[[str], None]

VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯaeiouyAEIOUY"
_STRESS_MARK = re.compile(rf"\+(?=[{VOWELS}])")
# words the user already stressed by hand ("з+амок"); RUAccent must leave them alone
MANUAL_STRESS = r"[^\W\d_]*\+[^\W\d_+]+"


def strip_stress(text: str) -> str:
    return _STRESS_MARK.sub("", text)


def has_stress(text: str) -> bool:
    return bool(_STRESS_MARK.search(text))


# ---------------------------------------------------------------------------------------------- RUAccent


class Stresser:
    """Thread-safe, lazily loaded RUAccent. One instance is shared by every engine (it takes ~0.5 GB)."""

    MODEL_SIZE = "turbo3.1"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accentizer = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("ruaccent") is not None

    @property
    def loaded(self) -> bool:
        return self._accentizer is not None

    def load(self, log: Log = lambda _m: None) -> None:
        with self._lock:
            if self._accentizer is not None:
                return
            if not self.available():
                raise RuntimeError("Пакет ruaccent не установлен. Установите: uv sync --extra stress")
            from huggingface_hub import constants
            from ruaccent import RUAccent

            log("RUAccent: загрузка модели ударений (первый раз скачивается ~0,5 ГБ)…")
            accentizer = RUAccent()
            workdir = os.path.join(constants.HF_HOME, "ruaccent")
            os.makedirs(workdir, exist_ok=True)
            accentizer.load(omograph_model_size=self.MODEL_SIZE, use_dictionary=True, workdir=workdir)
            self._accentizer = accentizer
            log("RUAccent загружен.")

    def unload(self) -> None:
        with self._lock:
            self._accentizer = None

    def process(self, text: str) -> str:
        """Put '+' before stressed vowels and restore «ё», keeping stresses the user typed. Line breaks are kept."""
        if self._accentizer is None:
            self.load()
        out = []
        with self._lock:
            for line in text.split("\n"):
                if line.strip() and re.search(r"[а-яА-ЯёЁ]", line):
                    lead = line[: len(line) - len(line.lstrip())]
                    trail = line[len(line.rstrip()):]
                    line = lead + self._accentizer.process_all(line.strip(), skip_regex=MANUAL_STRESS) + trail
                out.append(line)
        return "\n".join(out)


stresser = Stresser()


# ---------------------------------------------------------------------------------------------- dictionary


@dataclass
class Entry:
    word: str  # what is written in the book (case-insensitive, whole words)
    say: str  # how to read it; may contain "+" stress marks or be a respelling ("ПВО" -> "пэ вэ +о")


class PronunciationDictionary:
    """User fixes applied to every chunk before synthesis, for all engines."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or dictionary_path()
        self.entries: list[Entry] = []
        self._regex: re.Pattern | None = None
        self._map: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        self.entries = []
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.entries = [Entry(str(e["word"]), str(e["say"])) for e in data if e.get("word")]
            except (OSError, ValueError, KeyError, TypeError):
                self.entries = []
        self._compile()

    def save(self) -> None:
        data = [{"word": e.word, "say": e.say} for e in self.entries if e.word.strip()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        self._compile()

    def set_entries(self, entries: list[Entry]) -> None:
        self.entries = [Entry(strip_stress(e.word).strip(), e.say.strip()) for e in entries if e.word.strip() and e.say.strip()]
        self.save()

    def _compile(self) -> None:
        self._map = {}
        for e in self.entries:
            key = strip_stress(e.word).strip().lower()
            if key:
                self._map[key] = e.say
        if not self._map:
            self._regex = None
            return
        words = sorted(self._map, key=len, reverse=True)
        alternation = "|".join(r"\s+".join(map(re.escape, w.split())) for w in words)
        # a "+" right before the word means the user already stressed it in the text
        self._regex = re.compile(rf"(?<![\w+])(?:{alternation})(?!\w)", re.IGNORECASE)

    def apply(self, text: str) -> str:
        if self._regex is None:
            return text

        def repl(m: re.Match) -> str:
            found = m.group(0)
            say = self._map.get(" ".join(found.lower().split()))
            if say is None:
                return found
            return _match_case(found, say)

        return self._regex.sub(repl, text)


def _match_case(original: str, replacement: str) -> str:
    letters = [c for c in original if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters) and any(c.isupper() for c in replacement):
        return replacement  # an abbreviation spelled out by the user: keep their spelling
    if letters and letters[0].isupper():
        for i, c in enumerate(replacement):
            if c.isalpha():
                return replacement[:i] + c.upper() + replacement[i + 1:]
    return replacement
