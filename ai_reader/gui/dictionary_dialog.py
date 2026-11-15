"""Pronunciation dictionary editor: fixed stresses and respellings applied to every book."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..stress import Entry, PronunciationDictionary


class DictionaryDialog(QDialog):
    def __init__(self, parent: QWidget, dictionary: PronunciationDictionary, add_word: str = "") -> None:
        super().__init__(parent)
        self.dictionary = dictionary
        self.setWindowTitle("Словарь произношений")
        self.resize(640, 560)

        info = QLabel(
            "Здесь задаётся, как читать слова во всех книгах и всеми движками. "
            "Ударение — «+» перед гласной: <b>Герми+она</b>, <b>з+амок</b>. "
            "Можно переписать слово так, как оно звучит: <b>ПВО → пэ вэ +о</b>, <b>Worcester → в+устер</b>. "
            "Регистр букв не важен, ищутся целые слова."
        )
        info.setWordWrap(True)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Поиск…")
        self.filter_edit.textChanged.connect(self._filter)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["В тексте", "Читать как"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        for e in dictionary.entries:
            self._add_row(e.word, e.say)

        add_btn = QPushButton("Добавить")
        add_btn.clicked.connect(lambda: self._add_row("", "", edit=True))
        del_btn = QPushButton("Удалить выбранные")
        del_btn.clicked.connect(self._delete_rows)
        row = QHBoxLayout()
        row.addWidget(add_btn)
        row.addWidget(del_btn)
        row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        v = QVBoxLayout(self)
        v.addWidget(info)
        v.addWidget(self.filter_edit)
        v.addWidget(self.table, 1)
        v.addLayout(row)
        v.addWidget(buttons)

        if add_word:
            existing = self._find(add_word)
            if existing >= 0:
                self.table.selectRow(existing)
                self.table.editItem(self.table.item(existing, 1))
            else:
                self._add_row(add_word, add_word, edit=True)

    def _find(self, word: str) -> int:
        word = word.replace("+", "").lower()
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.text().replace("+", "").lower() == word:
                return r
        return -1

    def _add_row(self, word: str, say: str, edit: bool = False) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(word))
        self.table.setItem(r, 1, QTableWidgetItem(say))
        if edit:
            self.table.scrollToItem(self.table.item(r, 0))
            self.table.setCurrentCell(r, 1 if word else 0)
            self.table.editItem(self.table.item(r, 1 if word else 0))

    def _delete_rows(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def _filter(self, text: str) -> None:
        text = text.lower().strip()
        for r in range(self.table.rowCount()):
            cells = [self.table.item(r, c).text().lower() if self.table.item(r, c) else "" for c in (0, 1)]
            self.table.setRowHidden(r, bool(text) and not any(text in c for c in cells))

    def _save(self) -> None:
        entries = []
        for r in range(self.table.rowCount()):
            w, s = self.table.item(r, 0), self.table.item(r, 1)
            if w and s and w.text().strip() and s.text().strip():
                entries.append(Entry(w.text(), s.text()))
        self.dictionary.set_entries(entries)
        self.accept()
