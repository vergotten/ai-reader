"""«Мои голоса»: build a voice from recordings, review transcripts, fine-tune F5-TTS on it."""

from __future__ import annotations

import math
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..voices.audio_io import AUDIO_EXTENSIONS, AUDIO_FILTER
from ..voices.library import Voice, VoiceLibrary
from ..voices.runner import TrainProcess, TrainSettings, TrainStatus

MIN_TRAIN_MINUTES = 5
GOOD_TRAIN_MINUTES = 30
TARGET_UPDATES = 3000  # enough for a single-speaker fine-tune to clearly take on the voice
FRAME_SECONDS = 256 / 24000  # one mel frame (hop 256 at 24 kHz)


class _Signals(QObject):
    log = Signal(str)
    progress = Signal(int, int, str)
    done = Signal(object, object)
    train_status = Signal(object)
    train_exit = Signal(int)


def _dropped_audio(mime) -> list[str]:
    paths = [u.toLocalFile() for u in mime.urls() if u.isLocalFile()]
    return [p for p in paths if p.lower().endswith(AUDIO_EXTENSIONS)]


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} мин"


class VoicesDialog(QDialog):
    """Non-modal window. The main window passes callbacks so we can free the GPU and know when it is busy."""

    voices_changed = Signal()
    voice_ready = Signal(str)  # library id of a voice whose recordings have just been prepared

    def __init__(
        self,
        parent: QWidget,
        device_preference: Callable[[], str],
        release_gpu: Callable[[], None],
        main_busy: Callable[[], bool],
        main_log: Callable[[str], None],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Мои голоса и дообучение")
        self.resize(1200, 760)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)
        self.library = VoiceLibrary()
        self.settings = QSettings()
        self.voice: Voice | None = None
        self._device_preference = device_preference
        self._release_gpu = release_gpu
        self._main_busy = main_busy
        self._main_log = main_log
        self._busy: str | None = None  # None | "prep" | "train"
        self._cancel: threading.Event | None = None
        self._train: TrainProcess | None = None
        self._train_voice_id: str | None = None
        self._train_started = 0.0
        self._stopping = False
        self._prep_signals: _Signals | None = None  # kept alive while a preparation thread runs
        self._filling = False
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(600)
        self._save_timer.timeout.connect(self._save_voice)

        self.sig = _Signals()
        self.sig.log.connect(self._log)
        self.sig.progress.connect(self._on_progress)
        self.sig.train_status.connect(self._on_train_status)
        self.sig.train_exit.connect(self._on_train_exit)

        self.setAcceptDrops(True)  # drop recordings to add them to the selected voice (or to make a new one)
        self._build_ui()
        self._reload_list()
        self._update_controls()

    @property
    def training(self) -> bool:
        return self._busy == "train"

    @property
    def busy(self) -> bool:
        return self._busy is not None

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("Голоса:"))
        self.voice_list = QListWidget()
        self.voice_list.currentRowChanged.connect(self._on_voice_selected)
        lv.addWidget(self.voice_list, 1)
        self.new_btn = QPushButton("Новый голос…")
        self.new_btn.clicked.connect(self._new_voice)
        self.rename_btn = QPushButton("Переименовать…")
        self.rename_btn.clicked.connect(self._rename_voice)
        self.delete_btn = QPushButton("Удалить голос")
        self.delete_btn.clicked.connect(self._delete_voice)
        folder_btn = QPushButton("Открыть папку")
        folder_btn.clicked.connect(self._open_folder)
        for b in (self.new_btn, self.rename_btn, self.delete_btn, folder_btn):
            lv.addWidget(b)
        hint = QLabel(
            "Голос сразу появляется в движках <b>F5-TTS</b> и <b>XTTS-v2</b> как «★ Имя» — "
            "это клонирование по записям. После дообучения F5-TTS звучит заметно ближе к оригиналу."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        lv.addWidget(hint)

        right = QWidget()
        rv = QVBoxLayout(right)

        # --- step 1: recordings
        rec = QGroupBox("1. Записи голоса")
        recv = QVBoxLayout(rec)
        intro = QLabel(
            "Добавьте голосовые сообщения или любые записи речи одного человека (OGG/Opus из Telegram и WhatsApp, "
            "MP3, M4A, WAV…). Программа нарежет их по паузам на фрагменты 2–12 с, распознает текст (Whisper) "
            "и расставит ударения. Проверьте тексты: ошибки в них портят обучение. "
            f"Для клонирования хватит минуты речи, для дообучения нужно от {MIN_TRAIN_MINUTES} мин, "
            f"лучше {GOOD_TRAIN_MINUTES}–60 мин чистой речи без музыки и шума."
        )
        intro.setWordWrap(True)
        recv.addWidget(intro)
        row = QHBoxLayout()
        self.add_btn = QPushButton("Добавить записи…")
        self.add_btn.clicked.connect(self._add_recordings)
        self.stress_check = QCheckBox("Ударения в текстах (RUAccent)")
        self.stress_check.setChecked(True)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self._cancel_prep)
        row.addWidget(self.add_btn)
        row.addWidget(self.stress_check)
        row.addStretch(1)
        row.addWidget(self.cancel_btn)
        recv.addLayout(row)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        recv.addWidget(self.summary)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["✓", "Фрагмент", "Сек", "Текст (ударение — «+» перед гласной)"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.cellDoubleClicked.connect(lambda r, c: self._play_row(r) if c in (1, 2) else None)
        recv.addWidget(self.table, 1)

        clip_row = QHBoxLayout()
        self.play_btn = QPushButton("▶ Прослушать")
        self.play_btn.clicked.connect(lambda: self._play_row(self.table.currentRow()))
        self.ref_btn = QPushButton("Сделать эталоном")
        self.ref_btn.setToolTip("Этот фрагмент F5-TTS использует как образец голоса при озвучке (лучше 6–10 с, чисто и ровно)")
        self.ref_btn.clicked.connect(self._make_reference)
        self.remove_btn = QPushButton("Удалить фрагменты")
        self.remove_btn.clicked.connect(self._remove_clips)
        for b in (self.play_btn, self.ref_btn, self.remove_btn):
            clip_row.addWidget(b)
        clip_row.addStretch(1)
        recv.addLayout(clip_row)

        # --- step 2: fine-tuning
        train = QGroupBox("2. Дообучение F5-TTS (нужна видеокарта NVIDIA, лучше от 8 ГБ)")
        form = QFormLayout(train)
        from ..tts.f5 import F5Engine

        self.base_combo = QComboBox()
        for model_id, spec in F5Engine.MODELS.items():
            self.base_combo.addItem(spec[3], model_id)
        form.addRow("Базовая модель:", self.base_combo)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 2000)
        self.epochs_spin.setValue(60)
        self.epochs_spin.valueChanged.connect(self._update_train_estimate)
        self.auto_epochs = QPushButton("Подобрать")
        self.auto_epochs.setToolTip("≈ 3000 шагов обучения, но не меньше 20 и не больше 300 эпох")
        self.auto_epochs.clicked.connect(self._suggest_epochs)
        er = QHBoxLayout()
        er.addWidget(self.epochs_spin, 1)
        er.addWidget(self.auto_epochs)
        form.addRow("Эпох:", er)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(7)
        self.lr_spin.setRange(1e-7, 1e-3)
        self.lr_spin.setSingleStep(1e-6)
        self.lr_spin.setValue(1e-5)
        form.addRow("Скорость обучения:", self.lr_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(400, 38400)
        self.batch_spin.setSingleStep(400)
        self.batch_spin.setValue(3200)
        self.batch_spin.setToolTip("Кадров мел-спектра в батче: 3200 ≈ 34 с аудио, ~8–10 ГБ видеопамяти. При нехватке памяти уменьшите.")
        self.batch_spin.valueChanged.connect(self._update_train_estimate)
        form.addRow("Размер батча (кадров):", self.batch_spin)

        self.bnb_check = QCheckBox("8-битный оптимизатор (меньше видеопамяти, нужен bitsandbytes)")
        form.addRow(self.bnb_check)

        self.estimate = QLabel("")
        self.estimate.setStyleSheet("color: gray;")
        form.addRow(self.estimate)

        tr = QHBoxLayout()
        self.train_btn = QPushButton("Начать / продолжить обучение")
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("Остановить и сохранить")
        self.stop_btn.clicked.connect(self._stop_training)
        self.clean_btn = QPushButton("Удалить файлы обучения")
        self.clean_btn.setToolTip("Контрольные точки занимают 5–7 ГБ. Готовый голос (model.pt) останется, но продолжить обучение будет нельзя.")
        self.clean_btn.clicked.connect(self._clean_training)
        self.export_btn = QPushButton("Сохранить голос")
        self.export_btn.setToolTip("Сохранить голос из последней контрольной точки (если обучение было прервано)")
        self.export_btn.clicked.connect(lambda: self._run_trainer(export_only=True))
        tr.addWidget(self.train_btn, 1)
        tr.addWidget(self.stop_btn)
        tr.addWidget(self.export_btn)
        tr.addWidget(self.clean_btn)
        form.addRow(tr)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("")
        self.progress.setTextVisible(True)
        form.addRow(self.progress)
        self.train_label = QLabel("")
        self.train_label.setWordWrap(True)
        form.addRow(self.train_label)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setPlaceholderText("Журнал подготовки и обучения")

        vsplit = QSplitter(Qt.Orientation.Vertical)
        vsplit.addWidget(rec)
        vsplit.addWidget(train)
        vsplit.addWidget(self.log_view)
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 0)
        vsplit.setStretchFactor(2, 1)
        rv.addWidget(vsplit)

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([260, 940])
        layout = QVBoxLayout(self)
        layout.addWidget(split)

        s = self.settings
        self.epochs_spin.setValue(int(s.value("train/epochs", 60)))
        self.lr_spin.setValue(float(s.value("train/lr", 1e-5)))
        self.batch_spin.setValue(int(s.value("train/batch", 3200)))
        self.bnb_check.setChecked(s.value("train/bnb", False, type=bool))
        self.stress_check.setChecked(s.value("train/stress", True, type=bool))

    # ------------------------------------------------------------------ voice list

    def _reload_list(self, select_id: str | None = None) -> None:
        current = select_id or (self.voice.id if self.voice else None)
        self.voice_list.blockSignals(True)
        self.voice_list.clear()
        row = -1
        for i, v in enumerate(self.library.list()):
            item = QListWidgetItem(v.label())
            item.setData(Qt.ItemDataRole.UserRole, v.id)
            self.voice_list.addItem(item)
            if v.id == current:
                row = i
        self.voice_list.blockSignals(False)
        if row < 0 and self.voice_list.count():
            row = 0
        self.voice_list.setCurrentRow(row)
        self._on_voice_selected(row)

    def _on_voice_selected(self, row: int) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_voice()
        item = self.voice_list.item(row) if row >= 0 else None
        self.voice = self.library.get(item.data(Qt.ItemDataRole.UserRole)) if item else None
        if self.voice:
            idx = self.base_combo.findData(self.voice.base_model)
            if idx >= 0:
                self.base_combo.setCurrentIndex(idx)
        self._fill_table()
        self._update_summary()
        self._update_controls()

    def _new_voice(self) -> None:
        name, ok = QInputDialog.getText(self, "Новый голос", "Название голоса (например, «Мой голос» или имя диктора):")
        if not ok or not name.strip():
            return
        voice = self.library.create(name.strip())
        self._reload_list(voice.id)
        self.voices_changed.emit()
        self._add_recordings()

    def _rename_voice(self) -> None:
        if not self.voice:
            return
        name, ok = QInputDialog.getText(self, "Переименовать", "Новое название:", text=self.voice.name)
        if ok and name.strip():
            self.voice.name = name.strip()
            self.voice.save()
            self._reload_list(self.voice.id)
            self.voices_changed.emit()

    def _delete_voice(self) -> None:
        if not self.voice or self.busy:
            return
        answer = QMessageBox.question(
            self, "Удалить голос", f"Удалить голос «{self.voice.name}» со всеми фрагментами и обученной моделью?"
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.library.delete(self.voice)
            self.voice = None
            self._reload_list()
            self.voices_changed.emit()

    def _open_folder(self) -> None:
        path = self.voice.dir if self.voice else self.library.root
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ clips table

    def _fill_table(self) -> None:
        self._filling = True
        self.table.setRowCount(0)
        clips = self.voice.clips if self.voice else []
        self.table.setRowCount(len(clips))
        for r, c in enumerate(clips):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            check.setCheckState(Qt.CheckState.Checked if c.enabled else Qt.CheckState.Unchecked)
            name = QTableWidgetItem(Path(c.file).stem + (" ★" if c.file == self.voice.reference else ""))
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            dur = QTableWidgetItem(f"{c.duration:.1f}")
            dur.setFlags(dur.flags() & ~Qt.ItemFlag.ItemIsEditable)
            text = QTableWidgetItem(c.text)
            for col, item in enumerate((check, name, dur, text)):
                self.table.setItem(r, col, item)
            self._shade_row(r, c.enabled)
        self._filling = False

    def _shade_row(self, row: int, enabled: bool) -> None:
        brush = QBrush() if enabled else QBrush(QColor(128, 128, 128, 60))
        for col in range(4):
            item = self.table.item(row, col)
            if item:
                item.setBackground(brush)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or not self.voice:
            return
        r = item.row()
        if r >= len(self.voice.clips):
            return
        clip = self.voice.clips[r]
        if item.column() == 0:
            clip.enabled = item.checkState() == Qt.CheckState.Checked
            self._filling = True
            self._shade_row(r, clip.enabled)
            self._filling = False
        elif item.column() == 3:
            clip.text = " ".join(item.text().split())
            if clip.file == self.voice.reference:
                self.voice.reference_text = clip.text
        self._save_timer.start()
        self._update_summary()

    def _save_voice(self) -> None:
        if self.voice:
            self.voice.save()

    def _selected_rows(self) -> list[int]:
        return sorted({i.row() for i in self.table.selectedIndexes()})

    def _play_row(self, row: int) -> None:
        if not self.voice or not (0 <= row < len(self.voice.clips)):
            return
        try:
            import sounddevice as sd
            import soundfile as sf

            audio, sr = sf.read(str(self.voice.path(self.voice.clips[row].file)), dtype="float32")
            sd.stop()
            sd.play(audio, sr)
        except Exception as e:  # noqa: BLE001
            self._log(f"Не удалось воспроизвести: {e}")

    def _make_reference(self) -> None:
        rows = self._selected_rows()
        if not self.voice or not rows:
            return
        clip = self.voice.clips[rows[0]]
        if not 3 <= clip.duration <= 15:
            QMessageBox.warning(self, "Эталон", "Эталонный фрагмент должен длиться 3–15 секунд (лучше 6–10).")
            return
        self.voice.reference, self.voice.reference_text = clip.file, clip.text
        clip.enabled = True
        self.voice.save()
        self._fill_table()
        self._update_summary()
        self.voices_changed.emit()

    def _remove_clips(self) -> None:
        rows = self._selected_rows()
        if not self.voice or not rows or self.busy:
            return
        for r in reversed(rows):
            clip = self.voice.clips.pop(r)
            self.voice.path(clip.file).unlink(missing_ok=True)
            if clip.file == self.voice.reference:
                self.voice.reference, self.voice.reference_text = None, ""
        if not self.voice.ref_path:
            self.voice.pick_reference()
        self.voice.save()
        self._fill_table()
        self._update_summary()
        self.voices_changed.emit()

    def _update_summary(self) -> None:
        v = self.voice
        if not v:
            self.summary.setText("Создайте голос кнопкой «Новый голос…».")
            self._update_train_estimate()
            return
        good = v.total_seconds()
        parts = [f"Фрагментов: {len(v.clips)}, отмечено {len(v.enabled_clips())} — {_fmt_minutes(good)} речи."]
        parts.append(f"Эталон: {Path(v.reference).stem} «{v.reference_text[:80]}»" if v.ref_path else "Эталон не выбран.")
        if v.is_trained:
            parts.append(f"Дообучен: {v.trained_updates} шагов на {_fmt_minutes(v.trained_seconds)} речи.")
        if good < MIN_TRAIN_MINUTES * 60:
            parts.append(f"Для дообучения добавьте ещё записей (нужно от {MIN_TRAIN_MINUTES} мин).")
        self.summary.setText(" ".join(parts))
        self._update_train_estimate()

    # ------------------------------------------------------------------ dataset preparation

    def create_from_files(self, files: list[str]) -> None:
        """The one-step way: a new voice named after the first recording, prepared right away."""
        if self.busy or not files:
            return
        voice = self.library.create(Path(files[0]).stem[:40])
        self._reload_list(voice.id)
        self.voices_changed.emit()
        self._prepare(files)

    def _add_recordings(self) -> None:
        if not self.voice or self.busy:
            return
        last_dir = self.settings.value("train/last_dir", str(Path.home()))
        files, _ = QFileDialog.getOpenFileNames(self, "Записи голоса", last_dir, AUDIO_FILTER)
        if not files:
            return
        self.settings.setValue("train/last_dir", str(Path(files[0]).parent))
        self._prepare(files)

    def _missing_for_recordings(self, files: list[str]) -> tuple[list[str], list[str]]:
        """(packages the app can install itself, problems it can't fix) for turning these recordings into a voice."""
        import importlib.util

        installable, blocking = [], []
        if importlib.util.find_spec("torch") is None:
            blocking.append("PyTorch (полная версия программы: uv sync --extra train или build-installer.bat)")
        if importlib.util.find_spec("transformers") is None:
            installable.append("transformers")
        exotic = (".m4a", ".aac", ".mp4", ".webm", ".wma", ".amr", ".3gp")
        if any(f.lower().endswith(exotic) for f in files) and importlib.util.find_spec("av") is None and not shutil.which("ffmpeg"):
            installable.append("av")
        return installable, blocking

    def _install_then(self, modules: list[str], then: Callable[[], None]) -> None:
        from ..installer import PACKAGES, install

        requirements = [r for m in modules for r in PACKAGES[m]]
        self._busy = "prep"
        self._update_controls()
        sig = _Signals()
        sig.log.connect(self._log)

        def finished(_result, error) -> None:
            self._busy = None
            self._update_controls()
            if error is not None:
                QMessageBox.critical(self, "Не удалось установить", str(error))
                return
            then()

        sig.done.connect(finished)
        self._install_signals = sig

        def target() -> None:
            try:
                install(requirements, sig.log.emit)
            except BaseException as e:  # noqa: BLE001
                sig.done.emit(None, e)
            else:
                sig.done.emit(None, None)

        threading.Thread(target=target, daemon=True, name="ai-reader-install").start()

    def _prepare(self, files: list[str]) -> None:
        if not self.voice or self.busy:
            return
        if self._main_busy():
            QMessageBox.information(self, "Занято", "Дождитесь окончания чтения или экспорта в главном окне.")
            return
        installable, blocking = self._missing_for_recordings(files)
        if blocking:
            QMessageBox.warning(
                self, "Не хватает компонентов",
                "Чтобы сделать голос из записей, нужна полная версия программы — в этой сборке нет:\n• "
                + "\n• ".join(blocking)
                + "\n\nГолосом из записей говорят движки F5-TTS и XTTS-v2, им тоже нужна полная версия.",
            )
            return
        if installable:
            answer = QMessageBox.question(
                self, "Нужно установить пакеты",
                "Для распознавания записей не хватает пакетов: " + ", ".join(installable)
                + ".\nУстановить их сейчас? Нужен интернет; ход установки — в журнале.",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._install_then(installable, lambda: self._prepare(files))
            return
        self._release_gpu()
        voice, device, stress = self.voice, self._device_preference(), self.stress_check.isChecked()
        cancel = threading.Event()
        self._cancel = cancel
        self._busy = "prep"
        self._update_controls()
        self.progress.setValue(0)
        self._log(f"Подготовка {len(files)} записей для голоса «{voice.name}»…")

        from ..voices.dataset import prepare_dataset

        sig = _Signals()
        sig.log.connect(self._log)
        sig.progress.connect(self._on_progress)

        def finished(result, error) -> None:
            self._busy = None
            self._cancel = None
            self._update_controls()
            self.progress.setFormat("")
            if error is not None:
                from ..pipeline import Cancelled

                if isinstance(error, Cancelled):
                    self._log("Подготовка отменена.")
                else:
                    QMessageBox.critical(self, "Ошибка подготовки", str(error) or error.__class__.__name__)
            self.voice = self.library.get(voice.id)
            self._reload_list(voice.id)
            self._suggest_epochs()
            self.voices_changed.emit()
            if error is None and self.voice and self.voice.ref_path:
                self.voice_ready.emit(self.voice.lib_id)

        sig.done.connect(finished)
        self._prep_signals = sig

        def target() -> None:
            try:
                n = prepare_dataset(voice, files, device, sig.log.emit, sig.progress.emit, cancel.is_set, stress)
            except BaseException as e:  # noqa: BLE001
                if e.__class__.__name__ != "Cancelled":
                    sig.log.emit(traceback.format_exc())
                sig.done.emit(None, e)
            else:
                sig.done.emit(n, None)

        threading.Thread(target=target, daemon=True, name="ai-reader-dataset").start()

    def _cancel_prep(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self._log("Отмена…")

    def _on_progress(self, done: int, total: int, what: str) -> None:
        frac = done / max(total, 1)
        self.progress.setValue(int(frac * 1000))
        self.progress.setFormat(f"{what}: {done}/{total}")

    # ------------------------------------------------------------------ training

    def _train_settings(self) -> TrainSettings:
        return TrainSettings(
            base_model=self.base_combo.currentData(),
            epochs=self.epochs_spin.value(),
            learning_rate=self.lr_spin.value(),
            batch_frames=self.batch_spin.value(),
            bnb=self.bnb_check.isChecked(),
        )

    def _batches_per_epoch(self) -> int:
        seconds = self.voice.total_seconds() if self.voice else 0
        return max(1, math.ceil(seconds / (self.batch_spin.value() * FRAME_SECONDS)))

    def _suggest_epochs(self) -> None:
        self.epochs_spin.setValue(max(20, min(300, round(TARGET_UPDATES / self._batches_per_epoch()))))

    def _update_train_estimate(self, *_) -> None:
        if not self.voice:
            self.estimate.setText("")
            return
        updates = self._batches_per_epoch() * self.epochs_spin.value()
        disk = "" if (self.voice.train_dir / "ckpts").exists() else " Понадобится ~7 ГБ на диске."
        self.estimate.setText(
            f"≈ {updates} шагов. На RTX 3060–4070 это примерно {max(1, updates / 3 / 60):.0f}–{max(1, updates / 1.5 / 60):.0f} мин, "
            f"на GTX 10xx в несколько раз дольше.{disk}"
        )

    def _start_training(self) -> None:
        v = self.voice
        if not v or self.busy:
            return
        if v.total_seconds() < MIN_TRAIN_MINUTES * 60:
            answer = QMessageBox.question(
                self, "Мало записей",
                f"Сейчас {_fmt_minutes(v.total_seconds())} речи, а для заметного результата нужно от {MIN_TRAIN_MINUTES} мин "
                f"(лучше {GOOD_TRAIN_MINUTES}+). Всё равно начать?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self._main_busy():
            QMessageBox.information(self, "Занято", "Остановите чтение или экспорт в главном окне: обучению нужна вся видеопамять.")
            return
        if shutil.disk_usage(v.dir).free < 8 * 1024**3 and not (v.train_dir / "ckpts").exists():
            QMessageBox.warning(self, "Мало места", f"На диске с папкой голосов меньше 8 ГБ свободно:\n{v.dir}")
            return
        self._save_voice()
        self._release_gpu()
        s = self.settings
        s.setValue("train/epochs", self.epochs_spin.value())
        s.setValue("train/lr", self.lr_spin.value())
        s.setValue("train/batch", self.batch_spin.value())
        s.setValue("train/bnb", self.bnb_check.isChecked())
        self._run_trainer(export_only=False)

    def _run_trainer(self, export_only: bool) -> None:
        v = self.voice
        self._busy = "train"
        self._train_voice_id = v.id
        self._train_started = time.monotonic()
        self.progress.setValue(0)
        self.progress.setFormat("Запуск…")
        self.train_label.setText("")
        self._update_controls()
        self._log(("Экспорт голоса" if export_only else "Обучение") + f" «{v.name}» — процесс запущен.")
        try:
            self._train = TrainProcess(
                v.dir, self._train_settings(),
                on_line=self.sig.log.emit,
                on_status=lambda st: self.sig.train_status.emit(TrainStatus(**vars(st))),
                on_exit=self.sig.train_exit.emit,
                export_only=export_only,
            )
        except OSError as e:
            self._busy = None
            self._update_controls()
            QMessageBox.critical(self, "Не удалось запустить обучение", str(e))

    def _on_train_status(self, st: TrainStatus) -> None:
        self.progress.setValue(int(st.fraction * 1000))
        self.progress.setFormat(f"Эпоха {st.epoch}/{st.epochs} · {st.fraction * 100:.1f}%")
        loss = f" · loss {st.loss:.3f}" if st.loss is not None else ""
        elapsed = time.monotonic() - self._train_started
        remaining = elapsed * (1 - st.fraction) / st.fraction if st.fraction > 0.002 else 0
        eta = f" · осталось ≈ {remaining / 60:.0f} мин" if remaining else ""
        self.train_label.setText(f"Шаг {st.update}{loss}{eta}")

    def stop_training_and_wait(self, timeout: float = 30.0) -> None:
        """Used when the app quits: stop the trainer (its last checkpoint stays on disk) and wait for it to exit."""
        train = self._train
        self._stop_training()
        if train is not None:
            train.wait(timeout)

    def _stop_training(self) -> None:
        if self._train and self._train.running:
            self._stopping = True
            self._train.stop()
            self._log("Остановка… голос будет сохранён из последней контрольной точки.")

    def _on_train_exit(self, code: int) -> None:
        stopped = self._stopping
        self._stopping = False
        self._train = None
        self._busy = None
        voice = self.library.get(self._train_voice_id)
        if stopped and voice and (voice.train_dir / "ckpts" / "model_last.pt").exists():
            self.voice = voice
            self._run_trainer(export_only=True)  # save what was learned so far
            return
        self._update_controls()
        if code == 0:
            self.progress.setValue(1000)
            self.progress.setFormat("Готово")
            self._log("Голос сохранён. Выберите его в движке F5-TTS как «★ …(дообучен)».")
        elif stopped:
            self.progress.setFormat("Остановлено")
        else:
            self.progress.setFormat("Ошибка")
            self._log(f"Процесс обучения завершился с кодом {code} — подробности выше в журнале.")
            QMessageBox.warning(self, "Обучение", "Обучение завершилось с ошибкой. Подробности — в журнале окна.")
        self._reload_list(self._train_voice_id)
        self.voices_changed.emit()

    def _clean_training(self) -> None:
        v = self.voice
        if not v or self.busy or not v.train_dir.exists():
            return
        msg = "Удалить контрольные точки обучения?"
        msg += " Готовый голос останется, но продолжить обучение будет нельзя." if v.is_trained else " Обучение начнётся заново."
        if QMessageBox.question(self, "Файлы обучения", msg) == QMessageBox.StandardButton.Yes:
            shutil.rmtree(v.train_dir, ignore_errors=True)
            self._log("Файлы обучения удалены.")
            self._update_controls()
            self._update_train_estimate()

    # ------------------------------------------------------------------ misc

    def _update_controls(self) -> None:
        has = self.voice is not None
        busy = self.busy
        self.new_btn.setEnabled(not busy)
        self.voice_list.setEnabled(not busy)
        self.rename_btn.setEnabled(has and not busy)
        self.delete_btn.setEnabled(has and not busy)
        self.add_btn.setEnabled(has and not busy)
        self.stress_check.setEnabled(not busy)
        self.cancel_btn.setEnabled(self._busy == "prep")
        self.table.setEnabled(has and not busy)
        for w in (self.play_btn, self.ref_btn, self.remove_btn):
            w.setEnabled(has and not busy)
        for w in (self.base_combo, self.epochs_spin, self.auto_epochs, self.lr_spin, self.batch_spin, self.bnb_check):
            w.setEnabled(has and not busy)
        self.train_btn.setEnabled(has and not busy and bool(self.voice and self.voice.enabled_clips()))
        self.stop_btn.setEnabled(self._busy == "train" and self._train is not None)
        self.clean_btn.setEnabled(has and not busy and bool(self.voice and self.voice.train_dir.exists()))
        last = self.voice.train_dir / "ckpts" / "model_last.pt" if self.voice else None
        stale = bool(last and last.exists()) and (
            not self.voice.model_path or last.stat().st_mtime > self.voice.model_path.stat().st_mtime
        )
        self.export_btn.setEnabled(has and not busy and stale)

    def _log(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message.rstrip()}"
        self.log_view.appendPlainText(line)
        self._main_log(message)

    def dragEnterEvent(self, event) -> None:
        if _dropped_audio(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        files = _dropped_audio(event.mimeData())
        if self.voice:
            self._prepare(files)
        else:
            self.create_from_files(files)

    def closeEvent(self, event) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_voice()
        self.settings.setValue("train/stress", self.stress_check.isChecked())
        super().closeEvent(event)  # training keeps running in its own process; the window can be reopened
