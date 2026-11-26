"""Text preparation and export features added after listening feedback."""

from __future__ import annotations

import numpy as np

from ai_reader.audio import chapter_chime, trim_silence
from ai_reader.books.model import Book, Chapter
from ai_reader.dialogue import NARRATOR, SPEECH, segment
from ai_reader.latin import transliterate
from ai_reader.pipeline import ExportJob, SynthSettings, make_chunks, run_export
from ai_reader.text import chunk_text, normalize
from ai_reader.tts.base import TTSEngine


def test_latin_acronyms_and_words_are_spoken():
    assert "эн эф эл" in normalize("бывший рефери NFL", "ru")
    assert normalize("Rich Dad Poor Dad", "ru") == "Рич Дэд Пур Дэд"
    assert transliterate("Mike Smith") == "Майк Смит"
    assert not any("a" <= c.lower() <= "z" for c in normalize("основатель компании Fitness Club", "ru"))


def test_roman_numerals_still_become_numbers():
    assert "двадцат" in normalize("XX век", "ru")


def test_english_text_is_left_alone():
    assert normalize("Hello world", "en") == "Hello world"


def test_every_sentence_gets_its_own_pause():
    chunks = chunk_text("Раз. Два? Три…\nЧетыре.", 400, 0.4, 1.0)
    assert [c.text for c in chunks] == ["Раз.", "Два?", "Три…", "Четыре."]
    assert chunks[0].pause == 0.4
    assert chunks[1].pause > chunks[0].pause  # longer after a question
    assert chunks[2].pause == 1.0  # end of paragraph
    assert chunks[3].pause == 1.0


def test_short_sentences_joined_for_cloning_engines():
    chunks = chunk_text("Да. Конечно, я приду завтра утром.", 400, 0.3, 0.8, min_chars=10)
    assert len(chunks) == 1


def test_dialogue_gender_from_author_words():
    text = "— Привет, — сказала Маша. — Как дела?\n— Хорошо, — ответил Петя.\n— Правда?\nОн подумал: — Странно."
    segs = segment(text)
    speech = [(text[s.start:s.end], s.gender) for s in segs if s.role == SPEECH]
    assert speech[0] == ("Привет,", "f")
    assert speech[1] == ("Как дела?", "f")
    assert speech[2] == ("Хорошо,", "m")
    assert speech[3] == ("Правда?", "f")  # unattributed: the conversation alternates
    assert speech[4] == ("Странно.", "m")
    assert any(s.role == NARRATOR and text[s.start:s.end] == "сказала Маша." for s in segs)


def test_narration_with_dash_inside_is_not_dialogue():
    segs = segment("Это было — давно.")
    assert len(segs) == 1 and segs[0].role == NARRATOR


class FakeEngine(TTSEngine):
    id = "fake"
    max_chars = 400

    def voices(self, language):
        return [("m", "Он (муж.)"), ("f", "Она (жен.)")]

    def _load(self, device, log):
        pass

    def _synthesize(self, text, voice, speed, language, speaker_wav, speaker_text, style):
        self.calls.append((voice, text))
        tone = np.sin(np.arange(2400) / 3).astype(np.float32) * 0.3
        return np.concatenate([np.zeros(7200, np.float32), tone, np.zeros(7200, np.float32)])


def _engine():
    e = FakeEngine()
    e.calls = []
    e.sample_rate = 24000
    return e


def test_dialogue_lines_use_their_voices():
    e = _engine()
    s = SynthSettings(voice="narrator", dialogue_voices=True, male_voice="m", female_voice="f")
    chunks = make_chunks(e, "Вечер.\n— Привет, — сказала она.", s)
    assert [(c.text, c.voice) for c in chunks] == [("Вечер.", None), ("Привет,", "f"), ("сказала она.", None)]


def test_trim_removes_engine_silence():
    e = _engine()
    audio = e._synthesize("x", "m", 1, "ru", None, None, None)
    assert len(trim_silence(audio, 24000)) < len(audio)


def test_single_file_export_with_chapter_marks(tmp_path):
    import mutagen

    book = Book(title="Книга", author="Автор", chapters=[Chapter("Глава 1", "Раз. Два."), Chapter("Глава 2", "Три.")])
    job = ExportJob(book, [0, 1], tmp_path, fmt="mp3", single_file=True, chime=True)
    logs = []
    files = run_export(job, _engine(), SynthSettings(voice="m"), lambda *a: None, logs.append, lambda: False)
    assert len(files) == 1 and files[0].stat().st_size > 0
    tags = mutagen.File(files[0])
    assert any(k.startswith("CHAP") for k in tags.tags)
    assert len(chapter_chime(24000)) > 24000
