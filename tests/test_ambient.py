from itertools import pairwise

import numpy as np
import pytest

from ai_reader import ambient
from ai_reader.ambient import (
    AmbientMixer,
    FileBed,
    SynthBed,
    detect_mood,
    section_moods,
)

SR = 24000


@pytest.mark.parametrize(
    "text, mood",
    [
        ("Раздался выстрел. Он бежал, задыхаясь от страха, кровь стучала в висках, враг был рядом.", "tense"),
        ("Она плакала всю ночь. Тоска и одиночество после похорон отца не отпускали, слёзы душили.", "sad"),
        ("Успех приходит к тем, кто верит в мечту. Радость победы, уверенность и свобода — ваша цель.", "joy"),
        ("В тумане за древним замком мелькнула тень. Шёпот, шорох, и снова тайна, призрак в лунном свете.", "mystery"),
        ("Он нежно обнял её и поцеловал. Любовь, тепло родного дома и уют.", "warm"),
        ("Отчёт о квартальных показателях компании представлен в таблице ниже.", "calm"),
    ],
)
def test_detect_mood(text, mood):
    assert detect_mood(text) == mood


def test_section_moods_one_per_chunk_and_stable():
    texts = ["Тихое утро у реки, спокойный ветер."] * 10 + ["Выстрел! Кровь, крик, враг атакует, страх и ужас."] * 60
    moods = section_moods(texts, section_chars=400)
    assert len(moods) == len(texts)
    assert moods[-1] == "tense"
    # no flipping back and forth: moods change in runs, not chunk by chunk
    changes = sum(a != b for a, b in pairwise(moods))
    assert changes <= 2


@pytest.mark.parametrize("mood", list(ambient.MOODS))
def test_synth_bed_is_continuous_and_bounded(mood):
    bed = SynthBed(mood, SR, seed=1)
    parts = [bed.read(n) for n in (1000, SR * 3, 7, SR * 12)]
    audio = np.concatenate(parts)
    assert audio.dtype == np.float32
    assert len(audio) == 1000 + SR * 3 + 7 + SR * 12
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() < 1.0
    rms = float(np.sqrt(np.mean(np.square(audio[SR * 3 :]))))
    assert 0.03 < rms < 0.4
    # no clicks where the reads were joined
    assert np.abs(np.diff(audio)).max() < 0.2


def test_mixer_ducks_under_speech():
    mixer = AmbientMixer(SR, volume=0.5, seed=3)
    mixer.start("calm")
    mixer.intro()
    t = np.arange(SR * 4) / SR
    speech = np.zeros_like(t, dtype=np.float32)
    speech[: SR * 2] = (0.3 * np.sin(2 * np.pi * 220 * t[: SR * 2])).astype(np.float32)  # 2 s voice, 2 s pause
    out = mixer.mix(speech)
    music = out - speech
    under_voice = np.sqrt(np.mean(np.square(music[SR // 2 : SR * 2])))
    in_pause = np.sqrt(np.mean(np.square(music[SR * 3 :])))
    assert in_pause > under_voice * 1.3
    assert under_voice < 0.05  # well below the voice
    fade = mixer.outro()
    assert abs(fade[-100:]).max() < 1e-3
    assert mixer.gain == 0.0


def test_mood_change_crossfades_without_jumps():
    mixer = AmbientMixer(SR, volume=0.6, seed=2)
    mixer.start("calm")
    mixer.intro()
    silence = np.zeros(SR, dtype=np.float32)
    first = mixer.mix(silence)
    mixer.set_mood("tense")
    second = np.concatenate([mixer.mix(silence) for _ in range(7)])
    joined = np.concatenate([first, second])
    assert np.abs(np.diff(joined)).max() < 0.1
    assert mixer._old is None  # the crossfade finished


def test_own_track_is_used(tmp_path, monkeypatch):
    import soundfile as sf

    monkeypatch.setattr(ambient, "data_dir", lambda: tmp_path)
    folder = ambient.music_dir(create=True) / "sad"
    t = np.arange(SR * 3) / SR
    sf.write(folder / "piano.wav", (0.5 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), SR)
    assert (tmp_path / "music" / "ПРОЧТИ.txt").exists()
    assert ambient.own_tracks("sad") == [folder / "piano.wav"]
    assert ambient.own_tracks("joy") == []

    mixer = AmbientMixer(SR, seed=1)
    mixer.start("sad")
    assert isinstance(mixer._bed, FileBed)
    loop = mixer._bed.read(SR * 10)  # longer than the file: loops
    assert len(loop) == SR * 10 and np.isfinite(loop).all()
