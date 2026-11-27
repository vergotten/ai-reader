from ai_reader.voices.audio_io import collect_audio
from ai_reader.voices.runner import (
    PrepareProcess,
    TrainProcess,
    TrainSettings,
    TrainStatus,
)


def test_collect_audio_takes_every_recording_from_a_folder(tmp_path):
    (tmp_path / "chat" / "old").mkdir(parents=True)
    for name in ("chat/a.ogg", "chat/b.OPUS", "chat/old/c.m4a", "chat/notes.txt", "single.mp3"):
        (tmp_path / name).write_bytes(b"x")
    found = collect_audio([str(tmp_path / "chat"), str(tmp_path / "single.mp3"), str(tmp_path / "chat" / "a.ogg")])
    names = [p.replace("\\", "/").split("/")[-1] for p in found]
    assert names == ["a.ogg", "b.OPUS", "c.m4a", "single.mp3"]  # subfolders too, no duplicates, no text files


def _bare(cls):
    """An instance without starting a process: only the output parsing is tested."""
    obj = cls.__new__(cls)
    obj.lines = []
    obj.on_line = obj.lines.append
    return obj


def test_prepare_process_turns_output_into_progress_and_error():
    proc = _bare(PrepareProcess)
    proc.progress, proc.error = [], ""
    proc.on_progress = lambda *a: proc.progress.append(a)
    assert proc.handle("@@ progress 3 12 Распознавание речи (Whisper)")
    assert proc.progress == [(3, 12, "Распознавание речи (Whisper)")]
    assert proc.handle("@@ error Не хватает памяти\\nЗакройте программы")
    assert proc.error == "Не хватает памяти\nЗакройте программы"
    assert not proc.handle("voice_01.ogg: 35 с → 9 фрагм.")  # ordinary lines go to the log


def test_train_process_reads_tqdm_progress():
    proc = _bare(TrainProcess)
    statuses = []
    proc.on_status = statuses.append
    proc._status = TrainStatus(epochs=TrainSettings().epochs)
    proc._last_progress = ""
    assert proc.handle("Epoch 2/4:  50%|#####     | 10/20 [00:30<00:30, loss=0.612, update=30]")
    st = statuses[-1]
    assert (st.epoch, st.epochs, st.update) == (2, 4, 30)
    assert abs(st.fraction - 0.375) < 1e-6 and st.loss == 0.612


def test_truncated_checkpoint_is_dropped(tmp_path):
    import zipfile

    from ai_reader.voices.f5_train import drop_broken_checkpoint

    good = tmp_path / "good"
    good.mkdir()
    with zipfile.ZipFile(good / "model_last.pt", "w") as z:
        z.writestr("data.pkl", b"x" * 1000)
    drop_broken_checkpoint(good)
    assert (good / "model_last.pt").exists()

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "model_last.pt").write_bytes((good / "model_last.pt").read_bytes()[:500])
    drop_broken_checkpoint(bad)
    assert not (bad / "model_last.pt").exists()


def test_bundled_voice_is_copied_once(tmp_path):
    from ai_reader.voices.library import install_bundled_voices

    shipped = tmp_path / "app" / "voices" / "vovan"
    shipped.mkdir(parents=True)
    (shipped / "voice.json").write_text('{"id": "vovan", "name": "Вован"}', encoding="utf-8")
    (shipped / "model.pt").write_bytes(b"shipped")
    library = tmp_path / "library"
    library.mkdir()

    assert install_bundled_voices(shipped.parent, library) == ["vovan"]
    assert (library / "vovan" / "model.pt").read_bytes() == b"shipped"

    (library / "vovan" / "model.pt").write_bytes(b"retrained by the user")
    assert install_bundled_voices(shipped.parent, library) == []
    assert (library / "vovan" / "model.pt").read_bytes() == b"retrained by the user"
