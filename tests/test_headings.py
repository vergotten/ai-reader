from __future__ import annotations

import pytest

from ai_reader.books.headings import _heading_level, split_by_headings
from ai_reader.books.model import Chapter


@pytest.mark.parametrize(
    ("paragraph", "level", "title"),
    [
        ("* ЧАСТЬ ПЕРВАЯ *", 1, "Часть первая"),
        ("Книга вторая", 1, "Книга вторая"),
        ("Том 2. Возвращение", 1, "Том 2. Возвращение"),
        ("ЭПИЛОГ", 1, "Эпилог"),
        ("Глава 3", 2, "Глава 3"),
        ("Глава двадцать первая", 2, "Глава двадцать первая"),
        ("Глава IV. В дороге", 2, "Глава IV. В дороге"),
        ("Chapter Five", 2, "Chapter Five"),
        ("XII", 2, "XII"),
        ("Предисловие", 2, "Предисловие"),
    ],
)
def test_headings_are_recognised(paragraph: str, level: int, title: str) -> None:
    assert _heading_level(paragraph) == (level, title)


@pytest.mark.parametrize(
    "paragraph",
    [
        "Книга-с, вот-с",
        "Книга была старая, подержанная, в кожаном переплете",
        "Глава семьи молчал.",
        "Часть денег он отдал.",
        "Том вошёл в комнату.",
        "I am here.",
        "- Да.",
        "",
    ],
)
def test_ordinary_text_is_not_a_heading(paragraph: str) -> None:
    assert _heading_level(paragraph)[0] == 0


def test_split_names_chapters_after_their_part() -> None:
    text = "Вступление.\n\nЧАСТЬ ПЕРВАЯ\n\nI\n\nТекст один.\n\nII\n\nТекст два.\n\nЭПИЛОГ\n\nКонец."
    chapters = split_by_headings(Chapter("", text))
    assert [(c.title, c.text) for c in chapters] == [
        ("Начало", "Вступление."),
        ("Часть первая — I", "Текст один."),
        ("Часть первая — II", "Текст два."),
        ("Эпилог", "Конец."),
    ]


def test_text_without_headings_stays_whole() -> None:
    chapter = Chapter("Рассказ", "Первый абзац.\n\nВторой абзац.")
    assert split_by_headings(chapter) == [chapter]
