from __future__ import annotations

import json
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QFont, QKeySequence, QPalette, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFontDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QSystemTrayIcon,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..audio import BITRATES, FORMATS, chapter_chime
from ..books import FILE_FILTER, Book, load_book
from ..installer import PACKAGES, install
from ..pipeline import Cancelled, ExportJob, ReadAloud, SynthSettings, make_chunks, needs_stresser, prepare_voices, run_export
from ..stress import PronunciationDictionary, strip_stress, stresser
from ..tts import EngineManager, TTSEngine
from ..tts.base import Style
from ..tts.silero import RANDOM_PREFIX, SileroEngine
from ..voices.audio_io import AUDIO_EXTENSIONS, AUDIO_FILTER
from ..voices import is_library_voice
from ..winpath import to_local_path, windows_drives
from .dictionary_dialog import DictionaryDialog
from .theme import DEFAULT_THEME, THEMES, apply_theme
from .voices_dialog import VoicesDialog

LANGUAGES = [("ru", "Русский"), ("uk", "Українська"), ("en", "English")]
DEVICES = [("auto", "Авто (GPU, если есть)"), ("cuda", "GPU (CUDA)"), ("cpu", "CPU")]
# reading style -> (speed, pitch %, expressiveness %, pause between phrases, pause between paragraphs)
# pauses now follow every sentence (not every ~500 characters), so these are what you actually hear
STYLE_PRESETS = {
    "Обычное": (0.95, 0, 50, 0.35, 0.8),
    "Спокойное, перед сном": (0.85, -5, 35, 0.5, 1.1),
    "Выразительное, художественное": (0.9, 0, 75, 0.55, 1.1),
    "Быстрое, для нон-фикшн": (1.1, 0, 40, 0.2, 0.5),
}
DEFAULT_PRESET = "Выразительное, художественное"
CUSTOM_PRESET = "Своё"
F5_QUALITY = [(16, "Быстро (16 шагов)"), (32, "Обычно (32 шага)"), (64, "Максимум (64 шага, вдвое дольше)")]
CHARS_PER_MINUTE = 900  # rough speaking rate for Russian prose


class Bridge(QObject):
    """Carries events from worker threads to the GUI thread (signals are queued across threads)."""

    log = Signal(str)
    progress = Signal(int, int, str)
    chunk = Signal(int, int)
    done = Signal(object, object)  # result, exception


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings()
        self.engines = EngineManager()
        self.dictionary = PronunciationDictionary()
        self.voices_dialog: VoicesDialog | None = None
        self.book: Book | None = None
        self.current_index = -1
        self._busy: str | None = None  # None | "load" | "read" | "export" | "stress"
        self._cancel: threading.Event | None = None
        self._bridge: Bridge | None = None
        self._reader: ReadAloud | None = None
        self._read_mode = "chapter"
        self._export_started = 0.0
        self._quitting = False  # True once the user chose «Выход» (as opposed to closing the window into the tray)
        self.tray: QSystemTrayIcon | None = None
        self._applying_preset = False
        self._theme_actions: dict[str, QAction] = {}

        self.setWindowTitle("AI Reader")
        self.resize(1400, 860)
        self.setAcceptDrops(True)
        self._build_ui()
        self._build_menu()
        self._build_tray()
        self._restore_settings()
        self._connect_autosave()
        self._update_controls()

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: book info + chapters
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(6, 6, 6, 6)
        self.book_label = QLabel("Откройте книгу: TXT, FB2, EPUB, PDF, HTML\n(или перетащите файл в окно)")
        self.book_label.setWordWrap(True)
        self.open_button = QPushButton("Открыть книгу…")
        self.open_button.clicked.connect(self.open_book_dialog)
        self.chapter_list = QListWidget()
        self.chapter_list.currentRowChanged.connect(self._on_chapter_selected)
        self.chapter_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.chapter_list.customContextMenuRequested.connect(self._chapter_menu)
        self.chapter_list.setToolTip("Правый щелчок: отметить главы начиная с этой (например, пропустить предисловия)")
        sel_row = QHBoxLayout()
        all_btn, none_btn = QPushButton("Отметить все"), QPushButton("Снять все")
        all_btn.clicked.connect(lambda: self._check_all(True))
        none_btn.clicked.connect(lambda: self._check_all(False))
        sel_row.addWidget(all_btn)
        sel_row.addWidget(none_btn)
        lv.addWidget(self.open_button)
        lv.addWidget(self.book_label)
        lv.addWidget(QLabel("Главы (отмеченные — для экспорта):"))
        lv.addWidget(self.chapter_list, 1)
        lv.addLayout(sel_row)

        # center: editable chapter text
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(6, 6, 6, 6)
        self.chapter_title = QLabel("")
        title_font = self.chapter_title.font()
        title_font.setPointSize(title_font.pointSize() + 3)
        title_font.setBold(True)
        self.chapter_title.setFont(title_font)
        self.chapter_title.setWordWrap(True)
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("Текст главы появится здесь. Его можно править перед озвучкой.")
        editor_font = QFont("Georgia")
        editor_font.setPointSize(12)
        self.editor.setFont(editor_font)
        self.editor.textChanged.connect(self._update_stats)
        self.stats_label = QLabel("")
        tools = QHBoxLayout()
        self.stress_button = QPushButton("Расставить ударения")
        self.stress_button.setToolTip(
            "RUAccent ставит «+» перед ударными гласными в главе (или в выделенном фрагменте) и возвращает «ё».\n"
            "Проверьте и исправьте спорные места до озвучки. Ваши «+» не меняются. Отмена — Ctrl+Z."
        )
        self.stress_button.clicked.connect(self.stress_editor_text)
        self.unstress_button = QPushButton("Убрать ударения")
        self.unstress_button.clicked.connect(self.unstress_editor_text)
        self.to_dict_button = QPushButton("Слово в словарь…")
        self.to_dict_button.setToolTip("Запомнить, как читать выделенное слово, во всех книгах")
        self.to_dict_button.clicked.connect(lambda: self.open_dictionary(self._selected_word()))
        dict_button = QPushButton("Словарь произношений…")
        dict_button.clicked.connect(lambda: self.open_dictionary())
        for b in (self.stress_button, self.unstress_button, self.to_dict_button, dict_button):
            tools.addWidget(b)
        tools.addStretch(1)
        hint = QLabel(
            "Подсказка: чтение начинается с абзаца под курсором или с выделенного фрагмента. "
            "Ударение — «+» перед гласной («з+амок»); его понимают Silero и F5-TTS, остальные движки знак пропускают."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        cv.addWidget(self.chapter_title)
        cv.addWidget(self.editor, 1)
        cv.addLayout(tools)
        cv.addWidget(self.stats_label)
        cv.addWidget(hint)

        # right: voice, reading and export settings
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 6, 6, 6)
        rv.addWidget(self._build_voice_group())
        rv.addWidget(self._build_read_group())
        rv.addWidget(self._build_export_group())
        rv.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(right)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(400)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([280, 720, 400])

        # bottom: log
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setPlaceholderText("Журнал")
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.addWidget(splitter)
        self.vsplit.addWidget(self.log_view)
        self.vsplit.setStretchFactor(0, 1)
        self.vsplit.setStretchFactor(1, 0)
        self.vsplit.setSizes([700, 120])
        self.setCentralWidget(self.vsplit)

        self._highlight_format = QTextCharFormat()
        self._update_highlight_color()

        self.statusBar().showMessage("Модель не загружена")

    def _build_voice_group(self) -> QGroupBox:
        box = QGroupBox("Голос")
        form = QFormLayout(box)
        self.voice_form = form

        self.engine_combo = QComboBox()
        for engine in self.engines.all():
            label = engine.name if not engine.missing_requirements() else f"{engine.name} — не установлен"
            self.engine_combo.addItem(label, engine.id)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        form.addRow("Движок:", self.engine_combo)

        self.engine_info = QLabel()
        self.engine_info.setWordWrap(True)
        self.engine_info.setStyleSheet("color: gray;")
        self.engine_info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(self.engine_info)

        self.language_combo = QComboBox()
        for code, label in LANGUAGES:
            self.language_combo.addItem(label, code)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        form.addRow("Язык текста:", self.language_combo)

        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self._update_reference_rows)
        self.voice_combo.currentIndexChanged.connect(self._update_random_buttons)
        form.addRow("Голос:", self.voice_combo)

        random_button = QPushButton("🎲 Новый случайный голос")
        random_button.setToolTip("Silero придумывает новый голос. Послушайте его; не понравился — удалите и создайте другой.")
        random_button.clicked.connect(self.new_random_voice)
        self.delete_random_button = QPushButton("Удалить")
        self.delete_random_button.clicked.connect(self.delete_random_voice)
        self.random_row = self._button_row(random_button, self.delete_random_button)
        form.addRow(self.random_row)

        from_recording = QPushButton("＋ Голос из записи…")
        from_recording.setToolTip(
            "Выберите голосовое сообщение или другую запись речи — программа сама сделает из неё голос.\n"
            "Можно и просто перетащить запись в окно."
        )
        from_recording.clicked.connect(lambda: self.new_voice_from_recording())
        my_voices = QPushButton("Мои голоса и дообучение…")
        my_voices.setToolTip("Проверить записи и тексты, дообучить F5-TTS на своём голосе")
        my_voices.clicked.connect(self.open_voices)
        self.my_voices_row = self._button_row(from_recording, my_voices)
        form.addRow(self.my_voices_row)
        for combo in (self.engine_combo, self.voice_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(18)

        ref_row = QWidget()
        rl = QHBoxLayout(ref_row)
        rl.setContentsMargins(0, 0, 0, 0)
        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText("WAV 6–30 секунд, чистая речь")
        ref_btn = QToolButton()
        ref_btn.setText("…")
        ref_btn.clicked.connect(self._choose_reference)
        rl.addWidget(self.ref_edit, 1)
        rl.addWidget(ref_btn)
        self.ref_row = ref_row
        form.addRow("Образец голоса:", ref_row)

        self.ref_text_edit = QLineEdit()
        self.ref_text_edit.setPlaceholderText("Точный текст образца (пусто — распознать автоматически)")
        self.ref_text_edit.setToolTip(
            "Что именно звучит в образце голоса, слово в слово. Ударения можно отметить «+» перед гласной.\n"
            "Если оставить пустым, текст распознает Whisper (качество клонирования может быть хуже)."
        )
        form.addRow("Текст образца:", self.ref_text_edit)

        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip(
            "Готовые сочетания скорости, высоты голоса, выразительности и пауз.\n"
            "Подобрали удачные значения — сохраните их кнопкой 💾, они останутся и после перезапуска."
        )
        self.preset_combo.activated.connect(self._apply_preset)
        self.user_presets: dict[str, list] = {}
        save_preset = QToolButton()
        save_preset.setText("💾")
        save_preset.setToolTip("Сохранить текущие скорость, высоту, выразительность и паузы как свой стиль")
        save_preset.clicked.connect(self._save_user_preset)
        self.delete_preset_button = QToolButton()
        self.delete_preset_button.setText("✕")
        self.delete_preset_button.setToolTip("Удалить свой стиль")
        self.delete_preset_button.clicked.connect(self._delete_user_preset)
        self.preset_combo.currentTextChanged.connect(
            lambda name: self.delete_preset_button.setEnabled(name in self.user_presets)
        )
        preset_row = QWidget()
        pl = QHBoxLayout(preset_row)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.preset_combo, 1)
        pl.addWidget(save_preset)
        pl.addWidget(self.delete_preset_button)
        self.preset_row = preset_row
        self._fill_presets()
        form.addRow("Стиль чтения:", preset_row)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setValue(0.9)
        self.speed_spin.setSuffix("×")
        form.addRow("Скорость:", self.speed_spin)

        self.pitch_slider, self.pitch_row = self._slider_row(-30, 30, lambda v: f"{v:+d}%" if v else "обычная")
        self.pitch_row.setToolTip(
            "Silero. Сильное понижение вместе с медленной скоростью (например −5% и 0,85×) может растягивать вдохи\n"
            "и делать их «роботизированными». Если так — верните высоту ближе к нулю."
        )
        form.addRow("Высота голоса:", self.pitch_row)
        self.expression_slider, self.expression_row = self._slider_row(0, 100, lambda v: f"{v}%")
        self.expression_slider.setValue(50)
        self.expression_row.setToolTip(
            "Меньше — ровнее и стабильнее, больше — живее интонации, но выше риск оговорок.\n"
            "Piper: шум генератора, XTTS-v2: температура."
        )
        form.addRow("Выразительность:", self.expression_row)
        self.quality_combo = QComboBox()
        for steps, label in F5_QUALITY:
            self.quality_combo.addItem(label, steps)
        self.quality_combo.setCurrentIndex(1)
        form.addRow("Качество:", self.quality_combo)

        self.device_combo = QComboBox()
        for code, label in DEVICES:
            self.device_combo.addItem(label, code)
        form.addRow("Устройство:", self.device_combo)

        self.numbers_check = QCheckBox("Читать числа словами")
        self.numbers_check.setChecked(True)
        form.addRow(self.numbers_check)

        self.auto_stress_check = QCheckBox("Ударения нейросетью RUAccent (точнее, различает омографы)")
        self.auto_stress_check.setToolTip(
            "Перед синтезом RUAccent расставляет ударения сам. Без галочки Silero ставит их своим словарём.\n"
            "F5-TTS всегда использует RUAccent. Нужен пакет ruaccent (uv sync --extra stress)."
        )
        self.auto_stress_check.setChecked(stresser.available())
        form.addRow(self.auto_stress_check)
        self.install_stress_button = QPushButton("Установить RUAccent (ударения нейросетью)")
        self.install_stress_button.setToolTip("Скачает и установит пакет ruaccent прямо отсюда, без командной строки")
        self.install_stress_button.clicked.connect(lambda: self.install_feature("ruaccent", "ударения нейросетью RUAccent"))
        form.addRow(self.install_stress_button)

        self.dialogue_check = QCheckBox("Диалоги — отдельными голосами (муж./жен.)")
        self.dialogue_check.setToolTip(
            "Реплики после тире («— Привет, — сказала она») читаются другим голосом: женские — женским,\n"
            "мужские — мужским. Пол определяется по словам автора («сказала», «ответил», «она»), а для реплик\n"
            "без них — по ходу разговора. Слова автора и остальной текст читает основной голос (рассказчик)."
        )
        self.dialogue_check.toggled.connect(self._update_dialogue_rows)
        form.addRow(self.dialogue_check)
        self.male_combo = QComboBox()
        self.female_combo = QComboBox()
        for combo in (self.male_combo, self.female_combo):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(18)
        form.addRow("Мужские реплики:", self.male_combo)
        form.addRow("Женские реплики:", self.female_combo)

        self.sentence_pause = QDoubleSpinBox()
        self.sentence_pause.setRange(0.0, 3.0)
        self.sentence_pause.setSingleStep(0.05)
        self.sentence_pause.setValue(0.15)
        self.sentence_pause.setSuffix(" с")
        form.addRow("Пауза между фразами:", self.sentence_pause)

        self.paragraph_pause = QDoubleSpinBox()
        self.paragraph_pause.setRange(0.0, 5.0)
        self.paragraph_pause.setSingleStep(0.1)
        self.paragraph_pause.setValue(0.5)
        self.paragraph_pause.setSuffix(" с")
        form.addRow("Пауза между абзацами:", self.paragraph_pause)

        self.load_button = QPushButton("Загрузить модель")
        self.load_button.clicked.connect(self.load_model)
        form.addRow(self.load_button)
        for signal in (self.speed_spin.valueChanged, self.pitch_slider.valueChanged, self.expression_slider.valueChanged,
                       self.sentence_pause.valueChanged, self.paragraph_pause.valueChanged):
            signal.connect(self._on_style_edited)
        return box

    @staticmethod
    def _button_row(*buttons: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        for button in buttons:
            layout.addWidget(button)
        return row

    @staticmethod
    def _slider_row(low: int, high: int, describe: Callable[[int], str]) -> tuple[QSlider, QWidget]:
        """A horizontal slider with its value shown next to it."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(low, high)
        slider.setSingleStep(5)
        slider.setPageStep(10)
        label = QLabel(describe(slider.value()))
        label.setMinimumWidth(70)
        slider.valueChanged.connect(lambda value: label.setText(describe(value)))
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(label)
        return slider, row

    def _fill_presets(self) -> None:
        try:
            self.user_presets = json.loads(self.settings.value("style/user_presets", "{}") or "{}")
        except ValueError:
            self.user_presets = {}
        current = self.preset_combo.currentText()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems([*STYLE_PRESETS, *self.user_presets, CUSTOM_PRESET])
        self.preset_combo.setCurrentText(current or DEFAULT_PRESET)
        self.preset_combo.blockSignals(False)
        self.delete_preset_button.setEnabled(self.preset_combo.currentText() in self.user_presets)

    def _style_values(self) -> list:
        return [round(self.speed_spin.value(), 2), self.pitch_slider.value(), self.expression_slider.value(),
                round(self.sentence_pause.value(), 2), round(self.paragraph_pause.value(), 2)]

    def _save_user_preset(self) -> None:
        current = self.preset_combo.currentText()
        suggestion = current if current in self.user_presets else "Мой стиль"
        name, ok = QInputDialog.getText(self, "Свой стиль чтения", "Название:", text=suggestion)
        name = name.strip()
        if not ok or not name:
            return
        if name in STYLE_PRESETS or name == CUSTOM_PRESET:
            QMessageBox.information(self, "Свой стиль", "Это название занято готовым стилем — выберите другое.")
            return
        self.user_presets[name] = self._style_values()
        self.settings.setValue("style/user_presets", json.dumps(self.user_presets, ensure_ascii=False))
        self._fill_presets()
        self.preset_combo.setCurrentText(name)
        self._save_settings()
        self._log(f"Стиль «{name}» сохранён.")

    def _delete_user_preset(self) -> None:
        name = self.preset_combo.currentText()
        if name not in self.user_presets:
            return
        del self.user_presets[name]
        self.settings.setValue("style/user_presets", json.dumps(self.user_presets, ensure_ascii=False))
        self._fill_presets()
        self.preset_combo.setCurrentText(CUSTOM_PRESET)

    def _apply_preset(self, index: int) -> None:
        name = self.preset_combo.itemText(index)
        values = STYLE_PRESETS.get(name) or self.user_presets.get(name)
        if values is None:
            return
        speed, pitch, expressiveness, sentence_pause, paragraph_pause = values
        self._applying_preset = True
        self.speed_spin.setValue(speed)
        self.pitch_slider.setValue(pitch)
        self.expression_slider.setValue(expressiveness)
        self.sentence_pause.setValue(sentence_pause)
        self.paragraph_pause.setValue(paragraph_pause)
        self._applying_preset = False

    def _on_style_edited(self, *_) -> None:
        if not self._applying_preset:
            self.preset_combo.setCurrentText(CUSTOM_PRESET)

    def _build_read_group(self) -> QGroupBox:
        box = QGroupBox("Чтение вслух")
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.read_button = QPushButton("▶ Читать (F5)")
        self.read_button.clicked.connect(self.toggle_reading)
        self.stop_button = QPushButton("■ Стоп (F6)")
        self.stop_button.clicked.connect(self.stop_reading)
        row.addWidget(self.read_button)
        row.addWidget(self.stop_button)
        v.addLayout(row)
        self.continue_check = QCheckBox("Продолжать со следующей главы")
        self.continue_check.setChecked(True)
        v.addWidget(self.continue_check)
        return box

    def _build_export_group(self) -> QGroupBox:
        box = QGroupBox("Экспорт аудиокниги")
        form = QFormLayout(box)

        self.format_combo = QComboBox()
        for code, label in FORMATS.items():
            self.format_combo.addItem(label, code)
        self.format_combo.currentIndexChanged.connect(self._update_controls)
        form.addRow("Формат:", self.format_combo)

        self.bitrate_combo = QComboBox()
        for b in BITRATES:
            self.bitrate_combo.addItem(f"{b} кбит/с", b)
        self.bitrate_combo.setCurrentIndex(BITRATES.index(128))
        form.addRow("Битрейт MP3:", self.bitrate_combo)

        self.per_chapter_radio = QRadioButton("Файл на каждую главу")
        self.single_radio = QRadioButton("Один файл на книгу")
        self.single_radio.setChecked(True)
        group = QButtonGroup(box)
        group.addButton(self.per_chapter_radio)
        group.addButton(self.single_radio)
        self.single_radio.setToolTip(
            "Удобнее для телефона: книга не перемешается с музыкой, плеер помнит место.\n"
            "В файл записываются метки глав (MP3, OGG, Opus, FLAC) — по ним можно переходить в плеере."
        )
        self.per_chapter_radio.setToolTip("Удобно на компьютере: 001 - Глава первая.mp3, 002 - …")
        form.addRow(self.per_chapter_radio)
        form.addRow(self.single_radio)

        self.titles_check = QCheckBox("Зачитывать названия глав")
        self.titles_check.setChecked(True)
        form.addRow(self.titles_check)

        self.chime_check = QCheckBox("Мелодия перед каждой главой")
        self.chime_check.setToolTip("Короткий тихий звон — сразу слышно, что началась новая глава. Работает и при чтении вслух.")
        form.addRow(self.chime_check)

        self.skip_check = QCheckBox("Пропускать уже готовые главы")
        self.skip_check.setToolTip("Если экспорт был прерван, повторный запуск продолжит с первой неготовой главы")
        self.skip_check.setChecked(True)
        form.addRow(self.skip_check)

        out_row = QWidget()
        ol = QHBoxLayout(out_row)
        ol.setContentsMargins(0, 0, 0, 0)
        self.out_edit = QLineEdit(str(Path.home() / "Music" / "AI Reader"))
        out_btn = QToolButton()
        out_btn.setText("…")
        out_btn.clicked.connect(self._choose_out_dir)
        ol.addWidget(self.out_edit, 1)
        ol.addWidget(out_btn)
        form.addRow("Папка:", out_row)

        btn_row = QHBoxLayout()
        self.export_button = QPushButton("Экспортировать отмеченные главы")
        self.export_button.clicked.connect(self.start_export)
        self.cancel_button = QPushButton("Отмена")
        self.cancel_button.clicked.connect(self.cancel_task)
        btn_row.addWidget(self.export_button, 1)
        btn_row.addWidget(self.cancel_button)
        form.addRow(btn_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        form.addRow(self.progress)
        self.eta_label = QLabel("")
        self.eta_label.setWordWrap(True)
        form.addRow(self.eta_label)
        return box

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&Файл")
        open_action = QAction("Открыть книгу…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_book_dialog)
        file_menu.addAction(open_action)
        open_path_action = QAction("Открыть по пути…", self)
        open_path_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_path_action.setToolTip("Вставьте путь к книге, в том числе путь Windows вида D:\\Книги\\книга.fb2")
        open_path_action.triggered.connect(self.open_book_by_path)
        file_menu.addAction(open_path_action)
        file_menu.addSeparator()
        quit_action = QAction("Выход", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.quit)
        file_menu.addAction(quit_action)

        play_menu = self.menuBar().addMenu("&Чтение")
        read_action = QAction("Читать / пауза", self)
        read_action.setShortcut(QKeySequence(Qt.Key.Key_F5))
        read_action.triggered.connect(self.toggle_reading)
        stop_action = QAction("Стоп", self)
        stop_action.setShortcut(QKeySequence(Qt.Key.Key_F6))
        stop_action.triggered.connect(self.stop_reading)
        play_menu.addAction(read_action)
        play_menu.addAction(stop_action)

        view_menu = self.menuBar().addMenu("&Вид")
        log_action = QAction("Журнал", self)
        log_action.setCheckable(True)
        log_action.setChecked(True)
        log_action.toggled.connect(self.log_view.setVisible)
        view_menu.addAction(log_action)
        self.log_action = log_action
        theme_menu = view_menu.addMenu("Тема оформления")
        theme_group = QActionGroup(self)
        for theme_id, label in THEMES.items():
            action = QAction(label, self, checkable=True)
            action.triggered.connect(lambda _checked, name=theme_id: self._set_theme(name))
            theme_group.addAction(action)
            theme_menu.addAction(action)
            self._theme_actions[theme_id] = action
        font_action = QAction("Шрифт текста книги…", self)
        font_action.triggered.connect(self._choose_editor_font)
        larger = QAction("Текст крупнее", self)
        larger.setShortcut(QKeySequence("Ctrl+="))
        larger.triggered.connect(lambda: self._zoom_editor(1))
        smaller = QAction("Текст мельче", self)
        smaller.setShortcut(QKeySequence("Ctrl+-"))
        smaller.triggered.connect(lambda: self._zoom_editor(-1))
        view_menu.addActions([font_action, larger, smaller])
        self.tray_action = QAction("При закрытии сворачивать в трей", self, checkable=True)
        self.tray_action.setToolTip("Окно прячется в значок возле часов, экспорт и чтение продолжаются")
        view_menu.addAction(self.tray_action)

        voices_menu = self.menuBar().addMenu("&Голоса")
        voices_action = QAction("Мои голоса и дообучение…", self)
        voices_action.triggered.connect(self.open_voices)
        dict_action = QAction("Словарь произношений…", self)
        dict_action.triggered.connect(lambda: self.open_dictionary())
        recording_action = QAction("Новый голос из записи…", self)
        recording_action.triggered.connect(lambda: self.new_voice_from_recording())
        voices_menu.addAction(recording_action)
        voices_menu.addAction(voices_action)
        voices_menu.addAction(dict_action)

        help_menu = self.menuBar().addMenu("&Справка")
        about = QAction("О программе", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    def _build_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():  # e.g. WSL has no tray
            self.tray_action.setEnabled(False)
            return
        self.tray = QSystemTrayIcon(QApplication.windowIcon(), self)
        self.tray.setToolTip("AI Reader")
        menu = QMenu(self)
        menu.addAction("Открыть окно", self.show_from_tray)
        menu.addAction("Читать / пауза", self.toggle_reading)
        menu.addAction("Стоп", self.stop_reading)
        menu.addSeparator()
        menu.addAction("Выход", self.quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self.show_from_tray()

    def show_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def quit(self) -> None:
        self._quitting = True
        self.close()

    def _notify(self, title: str, text: str) -> None:
        """A system notification, shown only when the window is not in front of the user."""
        if self.tray is not None and not self.isActiveWindow():
            self.tray.showMessage(title, text, QApplication.windowIcon(), 8000)

    # ------------------------------------------------------------------ settings

    def _restore_settings(self) -> None:
        s = self.settings
        geometry = s.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = s.value("window/splitter")
        if state is not None:
            self.vsplit.restoreState(state)
        self.log_action.setChecked(s.value("window/log", True, type=bool))
        self._select_data(self.engine_combo, s.value("engine", "silero"))
        self._select_data(self.language_combo, s.value("language", "ru"))
        self._select_data(self.device_combo, s.value("device", "auto"))
        self._select_data(self.format_combo, s.value("format", "mp3"))
        self._select_data(self.bitrate_combo, int(s.value("bitrate", 128)))
        speed, pitch, expressiveness, sentence_pause, paragraph_pause = STYLE_PRESETS[DEFAULT_PRESET]
        self.speed_spin.setValue(float(s.value("speed", speed)))
        self.sentence_pause.setValue(float(s.value("sentence_pause", sentence_pause)))
        self.paragraph_pause.setValue(float(s.value("paragraph_pause", paragraph_pause)))
        self.numbers_check.setChecked(s.value("numbers", True, type=bool))
        self.auto_stress_check.setChecked(s.value("auto_stress", stresser.available(), type=bool))
        self.titles_check.setChecked(s.value("read_titles", True, type=bool))
        self.skip_check.setChecked(s.value("skip_existing", True, type=bool))
        self.tray_action.setChecked(self.tray is not None and s.value("tray_on_close", True, type=bool))
        self.pitch_slider.setValue(int(s.value("style/pitch", pitch)))
        self.expression_slider.setValue(int(s.value("style/expressiveness", expressiveness)))
        self._select_data(self.quality_combo, int(s.value("style/quality", 32)))
        self.preset_combo.setCurrentText(s.value("style/preset", DEFAULT_PRESET))
        self.chime_check.setChecked(s.value("chime", False, type=bool))
        self.dialogue_check.setChecked(s.value("dialogue/enabled", False, type=bool))
        self._set_theme(s.value("theme", DEFAULT_THEME))
        font = QFont()
        if s.value("editor_font") and font.fromString(s.value("editor_font")):
            self.editor.setFont(font)
        self.continue_check.setChecked(s.value("continue", True, type=bool))
        (self.single_radio if s.value("single_file", True, type=bool) else self.per_chapter_radio).setChecked(True)
        self.out_edit.setText(s.value("out_dir", self.out_edit.text()))
        self.ref_edit.setText(s.value("reference_wav", ""))
        self.ref_text_edit.setText(s.value("reference_text", ""))
        self._on_engine_changed()
        self._restored = True

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("window/geometry", self.saveGeometry())
        s.setValue("window/splitter", self.vsplit.saveState())
        s.setValue("window/log", self.log_action.isChecked())
        s.setValue("engine", self.engine_combo.currentData())
        s.setValue("language", self.language_combo.currentData())
        s.setValue("device", self.device_combo.currentData())
        s.setValue("format", self.format_combo.currentData())
        s.setValue("bitrate", self.bitrate_combo.currentData())
        s.setValue("speed", self.speed_spin.value())
        s.setValue("sentence_pause", self.sentence_pause.value())
        s.setValue("paragraph_pause", self.paragraph_pause.value())
        s.setValue("numbers", self.numbers_check.isChecked())
        s.setValue("auto_stress", self.auto_stress_check.isChecked())
        s.setValue("read_titles", self.titles_check.isChecked())
        s.setValue("skip_existing", self.skip_check.isChecked())
        if self.tray is not None:
            s.setValue("tray_on_close", self.tray_action.isChecked())
        s.setValue("style/pitch", self.pitch_slider.value())
        s.setValue("style/expressiveness", self.expression_slider.value())
        s.setValue("style/quality", self.quality_combo.currentData())
        s.setValue("style/preset", self.preset_combo.currentText())
        s.setValue("editor_font", self.editor.font().toString())
        s.setValue("continue", self.continue_check.isChecked())
        s.setValue("single_file", self.single_radio.isChecked())
        s.setValue("out_dir", self.out_edit.text())
        s.setValue("reference_wav", self.ref_edit.text())
        s.setValue("reference_text", self.ref_text_edit.text())
        s.setValue("chime", self.chime_check.isChecked())
        s.setValue("dialogue/enabled", self.dialogue_check.isChecked())
        self._remember_voice()
        s.sync()  # straight to disk: settings must survive a PC shutdown with the app still in the tray

    def _connect_autosave(self) -> None:
        """Every change is saved a moment later. Before, settings were only written when the window closed,
        so after a reboot or a crash (the app sitting in the tray) everything was back to defaults."""
        self._autosave = QTimer(self)
        self._autosave.setSingleShot(True)
        self._autosave.setInterval(700)
        self._autosave.timeout.connect(self._save_settings)
        schedule = lambda *_: self._autosave.start() if getattr(self, "_restored", False) else None  # noqa: E731
        for combo in self.findChildren(QComboBox):
            combo.currentIndexChanged.connect(schedule)
        for spin in self.findChildren(QDoubleSpinBox):
            spin.valueChanged.connect(schedule)
        for slider in self.findChildren(QSlider):
            slider.valueChanged.connect(schedule)
        for button in (*self.findChildren(QCheckBox), *self.findChildren(QRadioButton)):
            button.toggled.connect(schedule)
        for edit in self.findChildren(QLineEdit):
            if edit.parent() is not None and not isinstance(edit.parent(), QComboBox):
                edit.editingFinished.connect(schedule)
        self.voice_combo.editTextChanged.connect(schedule)
        QApplication.instance().aboutToQuit.connect(self._save_settings)

    @staticmethod
    def _select_data(combo: QComboBox, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ book

    def open_book_dialog(self) -> None:
        start_dir = self.settings.value("last_dir", str(Path.home()))
        dialog = QFileDialog(self, "Открыть книгу", start_dir, FILE_FILTER)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        drives = windows_drives()  # in WSL: /mnt/c, /mnt/d… in the sidebar
        if drives:
            dialog.setSidebarUrls(dialog.sidebarUrls() + [QUrl.fromLocalFile(d) for d in drives])
        if dialog.exec() and dialog.selectedFiles():
            self.open_book(dialog.selectedFiles()[0])

    def open_book_by_path(self) -> None:
        clipboard = QApplication.clipboard().text().strip()
        suggestion = clipboard if clipboard and len(clipboard) < 1000 and "\n" not in clipboard else ""
        text, ok = QInputDialog.getText(
            self, "Открыть по пути",
            "Путь к книге (можно путь Windows, например D:\\Книги\\книга.fb2 —\n"
            "в Проводнике: Shift + правый щелчок по файлу → «Копировать как путь»):",
            text=suggestion,
        )
        if ok and text.strip():
            self.open_book(text)

    def open_book(self, path: str | Path) -> None:
        if self._busy:
            QMessageBox.information(self, "AI Reader", "Сначала остановите чтение или экспорт.")
            return
        path = Path(to_local_path(str(path)))
        if not path.is_file():
            QMessageBox.warning(self, "Файл не найден", f"Нет такого файла:\n{path}")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            book = load_book(path)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Не удалось открыть книгу", f"{path.name}\n\n{e}")
            return
        QApplication.restoreOverrideCursor()

        self.settings.setValue("last_dir", str(path.parent))
        self.book = book
        self.current_index = -1
        if book.language:
            self._select_data(self.language_combo, book.language.lower()[:2])

        info = f"<b>{book.title}</b>"
        if book.author:
            info += f"<br>{book.author}"
        minutes = book.total_chars / CHARS_PER_MINUTE
        info += f"<br><span style='color:gray'>{len(book.chapters)} глав, {book.total_chars:,} знаков, ≈ {_fmt_duration(minutes * 60)} звучания</span>".replace(",", " ")
        self.book_label.setText(info)
        self.setWindowTitle(f"{book.title} — AI Reader")

        self.chapter_list.blockSignals(True)
        self.chapter_list.clear()
        for i, ch in enumerate(book.chapters, 1):
            item = QListWidgetItem(f"{i}. {ch.title}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setToolTip(f"{ch.title}\n{len(ch.text):,} знаков".replace(",", " "))
            self.chapter_list.addItem(item)
        self.chapter_list.blockSignals(False)
        self.chapter_list.setCurrentRow(0)
        self._log(f"Открыта книга: {path.name} — {len(book.chapters)} глав")
        self._update_controls()

    def _chapter_menu(self, pos) -> None:
        row = self.chapter_list.indexAt(pos).row()
        if row < 0:
            return
        count = self.chapter_list.count()
        menu = QMenu(self)
        menu.addAction("Отметить с этой главы до конца", lambda: self._check_range(row, count))
        menu.addAction("Отметить только эту главу", lambda: self._check_range(row, row + 1))
        menu.addAction("Отметить до этой главы включительно", lambda: self._check_range(0, row + 1))
        menu.addSeparator()
        menu.addAction("Отметить все", lambda: self._check_all(True))
        menu.addAction("Снять все", lambda: self._check_all(False))
        menu.exec(self.chapter_list.viewport().mapToGlobal(pos))

    def _check_range(self, first: int, stop: int) -> None:
        for i in range(self.chapter_list.count()):
            state = Qt.CheckState.Checked if first <= i < stop else Qt.CheckState.Unchecked
            self.chapter_list.item(i).setCheckState(state)

    def _check_all(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.chapter_list.count()):
            self.chapter_list.item(i).setCheckState(state)

    def _checked_chapters(self) -> list[int]:
        return [
            i for i in range(self.chapter_list.count())
            if self.chapter_list.item(i).checkState() == Qt.CheckState.Checked
        ]

    def _commit_editor(self) -> None:
        if self.book and 0 <= self.current_index < len(self.book.chapters) and self.editor.document().isModified():
            self.book.chapters[self.current_index].text = self.editor.toPlainText()
            self.editor.document().setModified(False)

    def _on_chapter_selected(self, row: int) -> None:
        if not self.book or row < 0:
            return
        if self._busy == "read":
            # user picked another chapter while reading: stop and stay there
            self._read_mode = "stopped"
            self.stop_reading()
        self._commit_editor()
        self.current_index = row
        ch = self.book.chapters[row]
        self.chapter_title.setText(ch.title)
        self.editor.setPlainText(ch.text)
        self.editor.document().setModified(False)
        self.editor.setExtraSelections([])
        self.editor.moveCursor(QTextCursor.MoveOperation.Start)
        self._update_stats()

    def _update_stats(self) -> None:
        n = len(self.editor.toPlainText()) if self.book else 0
        self.stats_label.setText(
            f"{n:,} знаков · ≈ {_fmt_duration(n / CHARS_PER_MINUTE * 60)} звучания".replace(",", " ") if n else ""
        )

    # ------------------------------------------------------------------ engine / voice

    def _engine(self) -> TTSEngine:
        return self.engines.get(self.engine_combo.currentData())

    def _language(self) -> str:
        return self.language_combo.currentData()

    def _on_engine_changed(self, *_) -> None:
        engine = self._engine()
        info = engine.description
        missing = engine.missing_requirements()
        if missing:
            info += f"<br><b style='color:#c0392b'>Не установлено: {', '.join(missing)}.</b> Команда: <code>{engine.install_hint()}</code>"
        if self._language() not in engine.languages:
            info += f"<br><b style='color:#c0392b'>Этот движок не поддерживает выбранный язык.</b>"
        self.engine_info.setText(info)
        stress_row = engine.stress_marks and not engine.builtin_stress
        self.voice_form.setRowVisible(self.auto_stress_check, stress_row)
        self.voice_form.setRowVisible(self.install_stress_button, not stresser.available())
        self.voice_form.setRowVisible(self.my_voices_row, engine.supports_cloning)
        self.voice_form.setRowVisible(self.random_row, isinstance(engine, SileroEngine))
        self.voice_form.setRowVisible(self.pitch_row, "pitch" in engine.style_controls)
        self.voice_form.setRowVisible(self.expression_row, "expressiveness" in engine.style_controls)
        self.voice_form.setRowVisible(self.quality_combo, "quality" in engine.style_controls)
        self.voice_combo.setEditable(engine.id == "piper")
        self._refresh_voices()
        self._update_reference_rows()

    def _update_random_buttons(self, *_) -> None:
        voice = self.voice_combo.currentData()
        self.delete_random_button.setEnabled(isinstance(voice, str) and voice.startswith(RANDOM_PREFIX))

    def _refresh_voices_keeping_choice(self) -> None:
        current = self.voice_combo.currentData()
        self._refresh_voices()
        index = self.voice_combo.findData(current)
        if index < 0 and isinstance(current, str) and current.endswith(":"):
            # a "more voices after the first load" placeholder was replaced by the real voices
            index = next(
                (i for i in range(self.voice_combo.count()) if str(self.voice_combo.itemData(i)).startswith(current)), -1
            )
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)

    def _update_reference_rows(self, *_) -> None:
        """The WAV/transcript fields are only for one-off cloning; library voices carry their own reference."""
        engine = self._engine()
        voice = self.voice_combo.currentData()
        own = engine.supports_cloning and not is_library_voice(voice) and (engine.id != "xtts" or voice == "__reference__")
        self.voice_form.setRowVisible(self.ref_row, own)
        self.voice_form.setRowVisible(self.ref_text_edit, own and engine.needs_reference_text)

    def _on_language_changed(self, *_) -> None:
        self._on_engine_changed()

    def _refresh_voices(self) -> None:
        engine = self._engine()
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for voice_id, label in engine.voices(self._language()):
            self.voice_combo.addItem(label, voice_id)
        saved = self.settings.value(f"voice/{engine.id}/{self._language()}")
        if saved:
            idx = self.voice_combo.findData(saved)
            if idx >= 0:
                self.voice_combo.setCurrentIndex(idx)
            elif self.voice_combo.isEditable():
                self.voice_combo.setEditText(saved)
        self.voice_combo.blockSignals(False)
        self._refresh_dialogue_voices()

    def _refresh_dialogue_voices(self) -> None:
        engine = self._engine()
        voices = [(v, label) for v, label in engine.voices(self._language()) if not str(v).endswith(":")]
        for combo, gender, marker in ((self.male_combo, "male", "муж"), (self.female_combo, "female", "жен")):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Как у рассказчика", "")
            for voice_id, label in voices:
                combo.addItem(label, voice_id)
            wanted = self.settings.value(f"dialogue/{engine.id}/{gender}") or current
            index = combo.findData(wanted) if wanted else -1
            if index < 0:  # first time: the first voice labelled «(муж.)» / «(жен.)»
                index = next((i for i in range(combo.count()) if f"({marker}" in combo.itemText(i)), 0)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._update_dialogue_rows()

    def _dialogue_available(self) -> bool:
        return self._language() == "ru" and self.male_combo.count() > 1 and self._engine().id != "mms"

    def _update_dialogue_rows(self, *_) -> None:
        available = self._dialogue_available()
        self.voice_form.setRowVisible(self.dialogue_check, available)
        on = available and self.dialogue_check.isChecked()
        self.voice_form.setRowVisible(self.male_combo, on)
        self.voice_form.setRowVisible(self.female_combo, on)

    def _current_voice(self) -> str:
        text = self.voice_combo.currentText().strip()
        idx = self.voice_combo.findText(text)
        if idx >= 0:
            return self.voice_combo.itemData(idx)
        return text

    def _remember_voice(self) -> None:
        voice = self._current_voice()
        if voice:
            self.settings.setValue(f"voice/{self._engine().id}/{self._language()}", voice)
        if self.male_combo.count():
            self.settings.setValue(f"dialogue/{self._engine().id}/male", self.male_combo.currentData())
            self.settings.setValue(f"dialogue/{self._engine().id}/female", self.female_combo.currentData())

    def _synth_settings(self) -> SynthSettings:
        return SynthSettings(
            voice=self._current_voice(),
            speed=self.speed_spin.value(),
            language=self._language(),
            speaker_wav=self.ref_edit.text().strip() or None,
            speaker_text=self.ref_text_edit.text().strip() or None,
            expand_numbers=self.numbers_check.isChecked(),
            sentence_pause=self.sentence_pause.value(),
            paragraph_pause=self.paragraph_pause.value(),
            auto_stress=self.auto_stress_check.isChecked() and stresser.available(),
            style=Style(
                pitch=self.pitch_slider.value(),
                expressiveness=self.expression_slider.value() / 100,
                quality=self.quality_combo.currentData(),
            ),
            dictionary=self.dictionary,
            dialogue_voices=self.dialogue_check.isChecked() and self._dialogue_available(),
            male_voice=self.male_combo.currentData() or "",
            female_voice=self.female_combo.currentData() or "",
        )

    def _prepare_engine(self, engine: TTSEngine, device: str, settings: SynthSettings, bridge: Bridge) -> None:
        self.engines.ensure_loaded(engine, device, settings.voice, bridge.log.emit)
        if settings.dialogue_voices:
            prepare_voices(engine, settings, bridge.log.emit)
        if needs_stresser(engine, settings) and stresser.available():
            stresser.load(bridge.log.emit)

    def _training_blocks(self) -> bool:
        if self.voices_dialog is not None and self.voices_dialog.busy:
            QMessageBox.information(
                self, "Идёт подготовка голоса",
                "Сейчас видеокарта занята подготовкой записей или обучением голоса (окно «Мои голоса»). "
                "Дождитесь окончания или остановите его.",
            )
            return True
        return False

    def _ready_to_synthesize(self, engine: TTSEngine) -> bool:
        if self._training_blocks():
            return False
        missing = engine.missing_requirements()
        if missing:
            QMessageBox.warning(
                self, "Движок не установлен",
                f"Для «{engine.name}» нужны пакеты: {', '.join(missing)}.\n\nУстановите их командой:\n{engine.install_hint()}",
            )
            return False
        if not self._current_voice():
            QMessageBox.warning(self, "Нет голоса", "Для выбранного языка у этого движка нет голосов.")
            return False
        if engine.license_notice and not self.settings.value(f"license/{engine.id}", False, type=bool):
            answer = QMessageBox.question(self, "Лицензия модели", engine.license_notice)
            if answer != QMessageBox.StandardButton.Yes:
                return False
            self.settings.setValue(f"license/{engine.id}", True)
        self._remember_voice()
        return True

    def _choose_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Образец голоса", self.ref_edit.text(), "Аудио (*.wav *.flac *.mp3)")
        if path:
            self.ref_edit.setText(path)

    def _choose_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка для аудиокниг", self.out_edit.text())
        if path:
            self.out_edit.setText(path)

    # ------------------------------------------------------------------ background tasks

    def _run_task(self, kind: str, fn: Callable[[Bridge, threading.Event], object], on_done: Callable[[object, object], None]) -> None:
        bridge = Bridge()
        cancel = threading.Event()
        bridge.log.connect(self._log)
        bridge.progress.connect(self._on_progress)
        bridge.chunk.connect(self._highlight)

        def finished(result, error) -> None:
            self._busy = None
            self._cancel = None
            self._bridge = None
            self._update_controls()
            self._update_status()
            self._refresh_voices_keeping_choice()  # some engines only know all their voices once loaded
            on_done(result, error)

        bridge.done.connect(finished)
        self._bridge, self._cancel, self._busy = bridge, cancel, kind
        self._update_controls()

        def target() -> None:
            try:
                result = fn(bridge, cancel)
            except BaseException as e:  # noqa: BLE001 — reported to the GUI
                if not isinstance(e, Cancelled):
                    bridge.log.emit(traceback.format_exc())
                bridge.done.emit(None, e)
            else:
                bridge.done.emit(result, None)

        threading.Thread(target=target, daemon=True, name=f"ai-reader-{kind}").start()

    def cancel_task(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self._log("Отмена…")
        if self._reader is not None:
            self._reader.stop()

    def _update_controls(self, *_) -> None:
        busy = self._busy is not None
        has_book = self.book is not None
        for w in (self.engine_combo, self.language_combo, self.voice_combo, self.device_combo, self.ref_row,
                  self.ref_text_edit, self.open_button, self.load_button, self.speed_spin, self.numbers_check,
                  self.sentence_pause, self.paragraph_pause, self.preset_combo, self.pitch_row,
                  self.expression_row, self.quality_combo, self.random_row, self.my_voices_row, self.preset_row,
                  self.dialogue_check, self.male_combo, self.female_combo, self.install_stress_button, self.chime_check):
            w.setEnabled(not busy)
        self.export_button.setEnabled(has_book and not busy)
        self.cancel_button.setEnabled(self._busy in ("export", "load", "stress", "install"))
        self.read_button.setEnabled(has_book and self._busy in (None, "read"))
        self.stop_button.setEnabled(self._busy == "read")
        self.editor.setReadOnly(busy)
        for w in (self.stress_button, self.unstress_button, self.to_dict_button):
            w.setEnabled(has_book and not busy)
        self.auto_stress_check.setEnabled(not busy)
        self.bitrate_combo.setEnabled(self.format_combo.currentData() == "mp3")
        if self._busy == "read" and self._reader is not None and self._reader.paused:
            self.read_button.setText("▶ Продолжить (F5)")
        elif self._busy == "read":
            self.read_button.setText("⏸ Пауза (F5)")
        else:
            self.read_button.setText("▶ Читать (F5)")

    def _update_status(self) -> None:
        loaded = [e for e in self.engines.all() if e.loaded]
        if loaded:
            e = loaded[0]
            self.statusBar().showMessage(f"Загружено: {e.name} · {e.device} · {e.sample_rate} Гц")
        else:
            self.statusBar().showMessage("Модель не загружена")

    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {message.rstrip()}")

    def _show_error(self, title: str, error: BaseException) -> None:
        QMessageBox.critical(self, title, str(error) or error.__class__.__name__)

    # ------------------------------------------------------------------ model

    def load_model(self) -> None:
        engine = self._engine()
        if self._busy or not self._ready_to_synthesize(engine):
            return
        settings = self._synth_settings()
        device = self.device_combo.currentData()

        def task(bridge: Bridge, cancel: threading.Event):
            self._prepare_engine(engine, device, settings, bridge)

        def done(_, error) -> None:
            if error:
                self._show_error("Не удалось загрузить модель", error)
            else:
                self._refresh_voices()  # XTTS exposes its speaker list only after loading

        self._run_task("load", task, done)

    # ------------------------------------------------------------------ read aloud

    def toggle_reading(self) -> None:
        if self._busy == "read" and self._reader is not None:
            if self._reader.paused:
                self._reader.resume()
            else:
                self._reader.pause()
            self._update_controls()
            return
        if self._busy or not self.book:
            return
        engine = self._engine()
        if not self._ready_to_synthesize(engine):
            return
        self._commit_editor()

        text = self.editor.toPlainText()
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            start, end = cursor.selectionStart(), cursor.selectionEnd()
            self._read_mode = "selection"
        else:
            start, end = cursor.block().position(), len(text)
            self._read_mode = "chapter"
        settings = self._synth_settings()
        chime = self.chime_check.isChecked() and self._read_mode == "chapter" and start == 0
        device = self.device_combo.currentData()

        def task(bridge: Bridge, cancel: threading.Event):
            self._prepare_engine(engine, device, settings, bridge)
            chunks = make_chunks(engine, text, settings, start, end)
            intro = chapter_chime(engine.sample_rate) if chime else None
            reader = ReadAloud(engine, settings, chunks, bridge.chunk.emit, bridge.log.emit, intro)
            self._reader = reader
            if cancel.is_set():  # stop was pressed while the model was loading
                return False
            reader.run()
            return reader.completed

        def done(completed, error) -> None:
            self._reader = None
            self.editor.setExtraSelections([])
            self._update_controls()
            if error:
                self._show_error("Ошибка чтения", error)
                return
            if completed and self._read_mode == "chapter" and self.continue_check.isChecked():
                nxt = self.current_index + 1
                if self.book and nxt < len(self.book.chapters):
                    self.chapter_list.setCurrentRow(nxt)
                    self.toggle_reading()

        self._run_task("read", task, done)

    def stop_reading(self) -> None:
        if self._busy != "read":
            return
        if self._read_mode == "chapter":
            self._read_mode = "stopped"
        self.cancel_task()

    def _highlight(self, start: int, end: int) -> None:
        doc = self.editor.document()
        if end > doc.characterCount():
            return
        selection = QTextEdit.ExtraSelection()
        cursor = QTextCursor(doc)
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        selection.cursor = cursor
        selection.format = self._highlight_format
        self.editor.setExtraSelections([selection])
        caret = QTextCursor(doc)
        caret.setPosition(start)
        self.editor.setTextCursor(caret)  # so "Читать" resumes from here after a stop
        self.editor.centerCursor()
        self._update_controls()

    # ------------------------------------------------------------------ optional packages

    def install_feature(self, module: str, what: str, then: Callable[[], None] | None = None) -> None:
        """Ask, then install a missing optional package in the background (no console needed)."""
        if self._busy:
            return
        requirements = PACKAGES[module]
        answer = QMessageBox.question(
            self, "Нужно установить пакет",
            f"Для функции «{what}» нужен пакет {', '.join(requirements)}. Он не установлен.\n\n"
            "Установить сейчас? Нужен интернет; ход установки виден в журнале внизу окна.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.progress.setFormat("Установка…")

        def task(bridge: Bridge, cancel: threading.Event):
            install(requirements, bridge.log.emit, cancel.is_set)

        def done(_result, error) -> None:
            self.progress.setFormat("")
            if error:
                self._show_error("Не удалось установить", error)
                return
            if module == "ruaccent":
                self.auto_stress_check.setChecked(True)
            self._on_engine_changed()
            QMessageBox.information(self, "Готово", f"Установлено. Функция «{what}» готова к работе.")
            if then is not None:
                then()

        self._run_task("install", task, done)

    # ------------------------------------------------------------------ export

    def start_export(self) -> None:
        if self._busy or not self.book:
            return
        indices = self._checked_chapters()
        if not indices:
            QMessageBox.information(self, "Экспорт", "Отметьте хотя бы одну главу.")
            return
        engine = self._engine()
        if not self._ready_to_synthesize(engine):
            return
        out_dir = Path(self.out_edit.text().strip() or Path.home() / "Music" / "AI Reader")
        self._commit_editor()

        job = ExportJob(
            book=self.book,
            chapter_indices=indices,
            out_dir=out_dir,
            fmt=self.format_combo.currentData(),
            bitrate=self.bitrate_combo.currentData(),
            single_file=self.single_radio.isChecked(),
            read_titles=self.titles_check.isChecked(),
            skip_existing=self.skip_check.isChecked(),
            chime=self.chime_check.isChecked(),
        )
        settings = self._synth_settings()
        device = self.device_combo.currentData()
        self.progress.setValue(0)
        self.progress.setFormat("Загрузка модели…")
        self.eta_label.setText("")

        def task(bridge: Bridge, cancel: threading.Event):
            self._prepare_engine(engine, device, settings, bridge)
            self._export_started = time.monotonic()
            bridge.log.emit(f"Экспорт {len(indices)} глав → {out_dir}")
            return run_export(job, engine, settings, bridge.progress.emit, bridge.log.emit, cancel.is_set)

        def done(files, error) -> None:
            if isinstance(error, Cancelled):
                self.progress.setFormat("Отменено")
                self.eta_label.setText("")
                self._log("Экспорт отменён.")
                return
            if error:
                self.progress.setFormat("Ошибка")
                self._show_error("Ошибка экспорта", error)
                return
            elapsed = time.monotonic() - self._export_started
            self.progress.setValue(1000)
            self.progress.setFormat("Готово")
            self.eta_label.setText(f"Готово за {_fmt_duration(elapsed)}: {len(files)} файл(ов)")
            self._log(f"Экспорт завершён: {len(files)} файл(ов) за {_fmt_duration(elapsed)}")
            self._notify("Экспорт завершён", f"{job.book.title}: {len(files)} файл(ов) за {_fmt_duration(elapsed)}")
            folder = files[0].parent if files else out_dir
            box = QMessageBox(self)
            box.setWindowTitle("Экспорт завершён")
            box.setText(f"Сохранено файлов: {len(files)}\n{folder}")
            open_btn = box.addButton("Открыть папку", QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Close)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

        self._run_task("export", task, done)

    def _on_progress(self, done: int, total: int, chapter: str) -> None:
        frac = done / max(total, 1)
        self.progress.setValue(int(frac * 1000))
        self.progress.setFormat(f"{frac * 100:.1f}%")
        if self.tray is not None:
            self.tray.setToolTip(f"AI Reader — {frac * 100:.0f}% · {chapter}")
        elapsed = time.monotonic() - self._export_started
        eta = elapsed * (1 - frac) / frac if frac > 0 else 0
        self.eta_label.setText(f"{chapter}\nПрошло {_fmt_duration(elapsed)} · осталось ≈ {_fmt_duration(eta)}")

    # ------------------------------------------------------------------ stress, dictionary, voices

    def _editor_range(self) -> tuple[int, int]:
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            return cursor.selectionStart(), cursor.selectionEnd()
        return 0, len(self.editor.toPlainText())

    def _replace_editor_range(self, start: int, end: int, text: str) -> None:
        cursor = QTextCursor(self.editor.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText(text)  # one undo step
        self.editor.document().setModified(True)

    def stress_editor_text(self) -> None:
        if self._busy or not self.book:
            return
        if not stresser.available():
            self.install_feature("ruaccent", "ударения нейросетью RUAccent")
            return
        start, end = self._editor_range()
        source = self.editor.toPlainText()[start:end]
        if not source.strip():
            return
        self._log(f"Расстановка ударений ({len(source):,} знаков)…".replace(",", " "))

        def task(bridge: Bridge, cancel: threading.Event):
            stresser.load(bridge.log.emit)
            out, pos = [], 0
            lines = source.split("\n")
            for i in range(0, len(lines), 40):
                if cancel.is_set():
                    raise Cancelled
                part = "\n".join(lines[i : i + 40])
                out.append(stresser.process(part))
                pos += len(part) + 1
                bridge.progress.emit(pos, len(source), "Ударения")
            return "\n".join(out)

        def done(result, error) -> None:
            self.progress.setFormat("")
            self.eta_label.setText("")
            if isinstance(error, Cancelled):
                return
            if error:
                self._show_error("Не удалось расставить ударения", error)
                return
            if self.editor.toPlainText()[start:end] != source:
                self._log("Текст изменился во время расстановки ударений — результат не применён.")
                return
            self._replace_editor_range(start, end, result)
            self._log("Ударения расставлены. Проверьте спорные слова; отменить — Ctrl+Z.")

        self._export_started = time.monotonic()
        self._run_task("stress", task, done)

    def unstress_editor_text(self) -> None:
        if self._busy or not self.book:
            return
        start, end = self._editor_range()
        source = self.editor.toPlainText()[start:end]
        cleaned = strip_stress(source)
        if cleaned != source:
            self._replace_editor_range(start, end, cleaned)

    def _selected_word(self) -> str:
        cursor = self.editor.textCursor()
        if not cursor.hasSelection():
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        return cursor.selectedText().strip().strip(".,;:!?«»\"'()—–-")

    def open_dictionary(self, word: str = "") -> None:
        dialog = DictionaryDialog(self, self.dictionary, word)
        if dialog.exec():
            self._log(f"Словарь произношений сохранён: {len(self.dictionary.entries)} слов.")

    def new_voice_from_recording(self, files: list[str] | None = None) -> None:
        """One step from a voice message to a usable voice: pick (or drop) recordings, the rest is automatic."""
        if self._busy or self._training_blocks():
            return
        if not files:
            start = self.settings.value("train/last_dir", str(Path.home()))
            files, _ = QFileDialog.getOpenFileNames(self, "Запись голоса: голосовое сообщение, MP3, WAV…", start, AUDIO_FILTER)
            if not files:
                return
            self.settings.setValue("train/last_dir", str(Path(files[0]).parent))
        self.open_voices()
        self.voices_dialog.create_from_files(files)

    def _on_voice_ready(self, voice_id: str) -> None:
        if self._busy:
            return
        if not self._engine().supports_cloning:
            installed = [e for e in ("f5", "xtts") if not self.engines.get(e).missing_requirements()]
            if installed:
                self._select_data(self.engine_combo, installed[0])
        self._refresh_voices()
        self._select_data(self.voice_combo, voice_id)
        self._update_reference_rows()
        if self.voice_combo.currentData() != voice_id:
            self._log("Голос готов, но говорить им умеют только F5-TTS и XTTS-v2 — установите их: uv sync --extra all")
            return
        self._log(
            "Голос готов и выбран. Нажмите «Читать», чтобы послушать. "
            "Для большего сходства добавьте записей и дообучите его в окне «Мои голоса»."
        )

    def new_random_voice(self) -> None:
        engine = self._engine()
        if self._busy or not isinstance(engine, SileroEngine) or not self._ready_to_synthesize(engine):
            return
        taken = sum(1 for voice_id, _label in engine.voices("ru") if voice_id.startswith(RANDOM_PREFIX))
        name, ok = QInputDialog.getText(self, "Новый случайный голос", "Название голоса:", text=f"Голос {taken + 1}")
        if not ok or not name.strip():
            return
        device = self.device_combo.currentData()

        def task(bridge: Bridge, cancel: threading.Event):
            self.engines.ensure_loaded(engine, device, engine.voices("ru")[0][0], bridge.log.emit)
            return engine.create_random_voice(name.strip(), bridge.log.emit)

        def done(voice_id, error) -> None:
            if error:
                self._show_error("Не удалось создать голос", error)
                return
            self._refresh_voices()
            self._select_data(self.voice_combo, voice_id)
            self._log("Создан новый голос. Нажмите «Читать», чтобы послушать; не понравился — «Удалить» и создайте другой.")

        self._run_task("load", task, done)

    def delete_random_voice(self) -> None:
        voice = self.voice_combo.currentData()
        if self._busy or not isinstance(voice, str) or not voice.startswith(RANDOM_PREFIX):
            return
        answer = QMessageBox.question(self, "Удалить голос", f"Удалить голос «{self.voice_combo.currentText()}»?")
        if answer == QMessageBox.StandardButton.Yes:
            SileroEngine.delete_random_voice(voice)
            self._refresh_voices()

    # ------------------------------------------------------------------ appearance

    def _set_theme(self, name: str) -> None:
        if name not in THEMES:
            name = DEFAULT_THEME
        apply_theme(name)
        self.settings.setValue("theme", name)
        self._theme_actions[name].setChecked(True)
        self._update_highlight_color()

    def _update_highlight_color(self) -> None:
        highlight = QApplication.palette().color(QPalette.ColorRole.Highlight)
        highlight.setAlpha(90)
        self._highlight_format.setBackground(highlight)

    def _choose_editor_font(self) -> None:
        ok, font = QFontDialog.getFont(self.editor.font(), self, "Шрифт текста книги")
        if ok:
            self.editor.setFont(font)

    def _zoom_editor(self, delta: int) -> None:
        font = self.editor.font()
        font.setPointSize(max(8, min(40, font.pointSize() + delta)))
        self.editor.setFont(font)

    def open_voices(self) -> None:
        if self.voices_dialog is None:
            self.voices_dialog = VoicesDialog(
                self,
                device_preference=lambda: self.device_combo.currentData(),
                release_gpu=self._release_gpu,
                main_busy=lambda: self._busy is not None,
                main_log=self._log,
            )
            self.voices_dialog.voices_changed.connect(self._on_voices_changed)
            self.voices_dialog.voice_ready.connect(self._on_voice_ready)
        self.voices_dialog.show()
        self.voices_dialog.raise_()
        self.voices_dialog.activateWindow()

    def _release_gpu(self) -> None:
        for engine in self.engines.all():
            if engine.loaded:
                self._log(f"Выгрузка «{engine.name}», чтобы освободить видеопамять.")
                engine.unload()
        self._update_status()

    def _on_voices_changed(self) -> None:
        if self._busy:
            return
        current = self.voice_combo.currentData()
        self._refresh_voices()
        idx = self.voice_combo.findData(current)
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        self._update_reference_rows()
        for engine in self.engines.all():
            engine.forget_voices()

    # ------------------------------------------------------------------ misc

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "О программе",
            f"<b>AI Reader {__version__}</b><br>Озвучка книг нейросетевыми моделями.<br><br>"
            "Движки: Silero TTS, Piper (rhasspy/piper-voices), XTTS-v2 (coqui), F5-TTS, Meta MMS-TTS.<br>"
            "Форматы книг: TXT, FB2, EPUB, PDF, HTML.<br>Экспорт: MP3, OGG, Opus, FLAC, WAV.",
        )

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        mime = event.mimeData()
        urls = mime.urls()
        recordings = [u.toLocalFile() for u in urls if u.isLocalFile() and u.toLocalFile().lower().endswith(AUDIO_EXTENSIONS)]
        if recordings:
            self.new_voice_from_recording(recordings)
        elif urls and urls[0].isLocalFile():
            self.open_book(urls[0].toLocalFile())
        elif mime.hasText() and mime.text().strip():
            self.open_book(mime.text().strip().splitlines()[0])

    def closeEvent(self, event) -> None:
        # Closing to the tray is still a user close action; keep the latest controls.
        self._save_settings()
        if self.tray is not None and self.tray_action.isChecked() and not self._quitting:
            event.ignore()
            self.hide()
            if not self.settings.value("tray_hint_shown", False, type=bool):
                self.tray.showMessage(
                    "AI Reader работает в трее",
                    "Щёлкните значок, чтобы открыть окно. Выйти: правый щелчок → «Выход».",
                    QApplication.windowIcon(), 8000,
                )
                self.settings.setValue("tray_hint_shown", True)
            return
        self._quitting = False  # the user may answer "no" below and keep working
        if self._busy == "export":
            answer = QMessageBox.question(self, "Выход", "Идёт экспорт. Прервать и выйти?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        if self.voices_dialog is not None and self.voices_dialog.training:
            answer = QMessageBox.question(
                self, "Выход", "Идёт обучение голоса. Остановить его и выйти? Прогресс сохранится, обучение можно продолжить позже."
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.voices_dialog.stop_training_and_wait()
        self.cancel_task()
        self._commit_editor()
        self._save_settings()
        if self.tray is not None:
            self.tray.hide()
        super().closeEvent(event)
        QApplication.quit()  # with the tray icon alive Qt would not quit by itself
