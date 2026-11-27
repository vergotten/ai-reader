"""Colour themes for the whole application, some with a faint tiled background pattern.

"Fancy" themes (marked ✦) also scatter small line icons (books, headphones, notes…) over the background and
add a stylesheet: rounded translucent cards, accent buttons, slim scrollbars."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import QApplication

PATTERN_ALPHA = 16  # 0–255: how visible the pattern is over the window colour; keep it barely there
ICON_ALPHA = 44  # the icon pattern of fancy themes: a little more visible, still in the background


@dataclass(frozen=True)
class Theme:
    label: str
    window: str = ""  # empty: the system's own look
    base: str = ""  # text fields and lists
    alternate: str = ""
    text: str = ""
    button: str = ""
    highlight: str = ""
    link: str = ""
    disabled: str = ""
    pattern: str = ""  # a key of _PATTERNS, "icons", or empty for a plain background
    ink: str = ""  # pattern colour; defaults to the text colour
    icons: tuple[str, ...] = ()  # pattern "icons": keys of _ICONS scattered over the background
    fancy: bool = False  # rounded cards, accent buttons and slim scrollbars via a stylesheet


_LIGHT = {"window": "#f3f3f3", "base": "#ffffff", "alternate": "#f7f7f7", "text": "#1e1e1e", "button": "#e9e9e9",
              "highlight": "#2f6fd6", "link": "#1a5fb4", "disabled": "#9a9a9a"}
_DARK = {"window": "#2b2b2b", "base": "#1e1e1e", "alternate": "#262626", "text": "#e2e2e2", "button": "#3a3a3a",
             "highlight": "#3d7be0", "link": "#6ea8ff", "disabled": "#7b7b7b"}
_SEPIA = {"window": "#eadfc8", "base": "#f6eedb", "alternate": "#efe5cf", "text": "#3b2f22", "button": "#e0d3b8",
              "highlight": "#a9743a", "link": "#7a4e1d", "disabled": "#a3957c"}
_NIGHT = {"window": "#1d2433", "base": "#161b27", "alternate": "#1a2030", "text": "#dfe5f2", "button": "#2a3348",
              "highlight": "#5b8def", "link": "#8fb4ff", "disabled": "#6c7890"}
_FOREST = {"window": "#e6ece2", "base": "#f7faf4", "alternate": "#eef3ea", "text": "#23301f", "button": "#d7e0d1",
               "highlight": "#4f8a4b", "link": "#2f6b2c", "disabled": "#93a08d"}
_WINE = {"window": "#2a2024", "base": "#1f171a", "alternate": "#261c20", "text": "#eadfe2", "button": "#3b2c32",
             "highlight": "#b0566e", "link": "#e39ab0", "disabled": "#806c73"}

_MIDNIGHT = {"window": "#161826", "base": "#10121c", "alternate": "#151827", "text": "#e6e8f5", "button": "#262a45",
                 "highlight": "#7b68ee", "link": "#b1a6ff", "disabled": "#6b7090"}
_AURORA = {"window": "#0f1d1f", "base": "#0a1517", "alternate": "#0f1f20", "text": "#def3ee", "button": "#1b3336",
               "highlight": "#1fa883", "link": "#6fe3c4", "disabled": "#5f7d78"}
_EMBER = {"window": "#1f1714", "base": "#16100d", "alternate": "#1c1411", "text": "#f2e6dc", "button": "#36271f",
              "highlight": "#c96a2e", "link": "#f2a76f", "disabled": "#85705f"}
_OCEAN = {"window": "#0f1a2b", "base": "#0a1321", "alternate": "#0e1828", "text": "#e1ecfa", "button": "#1d2d47",
              "highlight": "#2a80d0", "link": "#7cc2ff", "disabled": "#62728c"}
_ROSE = {"window": "#21161f", "base": "#180f17", "alternate": "#1e141c", "text": "#f5e4ef", "button": "#38243a",
             "highlight": "#c94882", "link": "#ff9cc8", "disabled": "#846c80"}
_MORNING = {"window": "#f4f1ea", "base": "#fffdf8", "alternate": "#f8f5ee", "text": "#2a2633", "button": "#e8e2d5",
                "highlight": "#6a4fe0", "link": "#5a3fd1", "disabled": "#a19c90"}
# warm cream paper with brown ink: light, but softer on the eyes than white
_HONEY = {"window": "#efe5cf", "base": "#f8f0de", "alternate": "#f3e9d5", "text": "#3a2f22", "button": "#e3d5b8",
              "highlight": "#a5622a", "link": "#8a5418", "disabled": "#a8997e"}
_MINT = {"window": "#eaf4ef", "base": "#fbfffd", "alternate": "#f2f9f5", "text": "#1d2b26", "button": "#d6e8df",
             "highlight": "#118a69", "link": "#0d7a5d", "disabled": "#8fa59b"}

THEME_SPECS = {
    "honey": Theme("✦ Медовая бумага · книги и чай", **_HONEY, pattern="icons", ink="#a5622a",
                   icons=("book", "cup", "quill", "note", "star"), fancy=True),
    "midnight": Theme("✦ Полночь · книги и ноты", **_MIDNIGHT, pattern="icons", ink="#b1a6ff",
                      icons=("book", "note", "headphones", "quill", "star"), fancy=True),
    "aurora": Theme("✦ Северное сияние · звёзды", **_AURORA, pattern="icons", ink="#6fe3c4",
                    icons=("star", "moon", "note", "wave", "star"), fancy=True),
    "ember": Theme("✦ У камина · книги и чай", **_EMBER, pattern="icons", ink="#f2a76f",
                   icons=("book", "cup", "quill", "star"), fancy=True),
    "ocean": Theme("✦ Глубина · волны звука", **_OCEAN, pattern="icons", ink="#7cc2ff",
                   icons=("wave", "headphones", "note", "star"), fancy=True),
    "rose": Theme("✦ Розовый вечер · луна", **_ROSE, pattern="icons", ink="#ff9cc8",
                  icons=("moon", "star", "book", "note"), fancy=True),
    "morning": Theme("✦ Утро · светлая с книгами", **_MORNING, pattern="icons", ink="#6a4fe0",
                     icons=("book", "quill", "note", "cup"), fancy=True),
    "mint": Theme("✦ Мята · светлая с наушниками", **_MINT, pattern="icons", ink="#118a69",
                  icons=("headphones", "note", "wave", "book"), fancy=True),
    "system": Theme("Системная"),
    "light": Theme("Светлая", **_LIGHT),
    "dark": Theme("Тёмная", **_DARK),
    "sepia": Theme("Сепия", **_SEPIA),
    "light-lattice": Theme("Светлая · ромбы", **_LIGHT, pattern="lattice"),
    "paper": Theme("Старая бумага · волны", **_SEPIA, pattern="waves"),
    "forest": Theme("Лесная · листья", **_FOREST, pattern="leaves"),
    "dark-lattice": Theme("Тёмная · ромбы", **_DARK, pattern="lattice"),
    "night": Theme("Ночное небо · звёзды", **_NIGHT, pattern="stars", ink="#ffffff"),
    "wine": Theme("Бордо · орнамент", **_WINE, pattern="ornament"),
}
DEFAULT_THEME = "honey"
THEMES = {theme_id: theme.label for theme_id, theme in THEME_SPECS.items()}

_system_style: str | None = None  # the platform style the app started with


# ------------------------------------------------------------------ patterns: each paints one seamless tile


def _lattice(p: QPainter, size: int) -> None:
    half = size / 2
    for a, b in (((0, half), (half, 0)), ((half, 0), (size, half)), ((size, half), (half, size)), ((half, size), (0, half))):
        p.drawLine(QPointF(*a), QPointF(*b))
    p.drawEllipse(QPointF(half, half), 1.2, 1.2)


def _waves(p: QPainter, size: int) -> None:
    for row in range(3):
        y0 = size * (row + 0.5) / 3
        points = [QPointF(x, y0 + 3.0 * math.sin(2 * math.pi * x / size)) for x in range(size + 1)]
        p.drawPolyline(points)


def _leaves(p: QPainter, size: int) -> None:
    for cx, cy, angle in ((size * 0.25, size * 0.25, -35), (size * 0.75, size * 0.75, 35)):
        p.save()
        p.translate(cx, cy)
        p.rotate(angle)
        p.drawEllipse(QPointF(0, 0), size * 0.13, size * 0.055)
        p.drawLine(QPointF(-size * 0.13, 0), QPointF(size * 0.2, 0))
        p.restore()


def _stars(p: QPainter, size: int) -> None:
    rng = random.Random(7)  # the same sky every time
    p.setBrush(p.pen().color())
    for _ in range(14):
        x, y = rng.uniform(4, size - 4), rng.uniform(4, size - 4)  # away from the edges: nothing is cut by the seam
        p.drawEllipse(QPointF(x, y), *([rng.choice((0.6, 0.6, 0.9, 1.3))] * 2))


def _ornament(p: QPainter, size: int) -> None:
    half, r = size / 2, size * 0.22
    for cx, cy in ((half, 0), (half, size), (0, half), (size, half)):  # halves meet across tile edges
        p.drawEllipse(QPointF(cx, cy), r, r)
    p.drawEllipse(QPointF(half, half), r * 0.45, r * 0.45)


# name -> (tile size in pixels, painter)
_PATTERNS: dict[str, tuple[int, Callable[[QPainter, int], None]]] = {
    "lattice": (36, _lattice),
    "waves": (48, _waves),
    "leaves": (56, _leaves),
    "stars": (160, _stars),
    "ornament": (44, _ornament),
}


# ------------------------------------------------------------------ line icons, each drawn in the box -1..1


def _icon_book(p: QPainter) -> None:
    for side in (-1, 1):
        page = QPainterPath(QPointF(0, -0.55))
        page.cubicTo(QPointF(side * 0.35, -0.75), QPointF(side * 0.7, -0.7), QPointF(side * 0.95, -0.55))
        page.lineTo(QPointF(side * 0.95, 0.6))
        page.cubicTo(QPointF(side * 0.7, 0.45), QPointF(side * 0.35, 0.5), QPointF(0, 0.7))
        page.closeSubpath()
        p.drawPath(page)
        for y in (-0.3, -0.05, 0.2):
            p.drawLine(QPointF(side * 0.2, y - 0.03), QPointF(side * 0.75, y - 0.1))


def _icon_headphones(p: QPainter) -> None:
    p.drawArc(QRectF(-0.8, -0.85, 1.6, 1.6), 0, 180 * 16)
    for x in (-0.95, 0.6):
        p.drawRoundedRect(QRectF(x, -0.05, 0.35, 0.75), 0.12, 0.12)


def _icon_note(p: QPainter) -> None:
    for x in (-0.55, 0.45):
        p.drawEllipse(QPointF(x, 0.6), 0.27, 0.2)
        p.drawLine(QPointF(x + 0.26, 0.6), QPointF(x + 0.26, -0.6))
    p.drawLine(QPointF(-0.29, -0.6), QPointF(0.71, -0.85))
    p.drawLine(QPointF(-0.29, -0.4), QPointF(0.71, -0.65))


def _icon_quill(p: QPainter) -> None:
    feather = QPainterPath(QPointF(-0.75, 0.85))
    feather.cubicTo(QPointF(-0.4, 0.1), QPointF(0.2, -0.6), QPointF(0.85, -0.9))
    feather.cubicTo(QPointF(0.7, -0.3), QPointF(0.2, 0.3), QPointF(-0.45, 0.45))
    p.drawPath(feather)
    p.drawLine(QPointF(-0.9, 1.0), QPointF(0.5, -0.55))


def _icon_star(p: QPainter) -> None:
    points = []
    for k in range(10):
        r = 0.9 if k % 2 == 0 else 0.4
        a = math.pi * (0.5 + k / 5)
        points.append(QPointF(math.cos(a) * r, -math.sin(a) * r))
    p.drawPolygon(QPolygonF(points))


def _icon_moon(p: QPainter) -> None:
    moon = QPainterPath()
    moon.addEllipse(QPointF(0, 0), 0.8, 0.8)
    bite = QPainterPath()
    bite.addEllipse(QPointF(0.38, -0.25), 0.68, 0.68)
    p.drawPath(moon.subtracted(bite))


def _icon_wave(p: QPainter) -> None:
    for i, h in enumerate((0.3, 0.6, 0.95, 0.55, 0.8, 0.35, 0.15)):
        x = -0.9 + i * 0.3
        p.drawLine(QPointF(x, -h), QPointF(x, h))


def _icon_cup(p: QPainter) -> None:
    cup = QPainterPath(QPointF(-0.7, -0.15))
    cup.lineTo(QPointF(0.5, -0.15))
    cup.cubicTo(QPointF(0.5, 0.55), QPointF(0.25, 0.75), QPointF(-0.1, 0.75))
    cup.cubicTo(QPointF(-0.45, 0.75), QPointF(-0.7, 0.55), QPointF(-0.7, -0.15))
    p.drawPath(cup)
    p.drawArc(QRectF(0.3, 0.0, 0.45, 0.4), -90 * 16, 180 * 16)
    for x in (-0.35, 0.05):
        steam = QPainterPath(QPointF(x, -0.35))
        steam.cubicTo(QPointF(x - 0.15, -0.55), QPointF(x + 0.15, -0.7), QPointF(x, -0.95))
        p.drawPath(steam)


_ICONS: dict[str, Callable[[QPainter], None]] = {
    "book": _icon_book, "headphones": _icon_headphones, "note": _icon_note, "quill": _icon_quill,
    "star": _icon_star, "moon": _icon_moon, "wave": _icon_wave, "cup": _icon_cup,
}
ICON_TILE = 264  # px; every icon stays inside its own cell of a 4x4 grid, so nothing is cut at the tile seam


def _icon_tile(theme: Theme) -> QPixmap:
    tile = QPixmap(ICON_TILE, ICON_TILE)
    tile.fill(QColor(theme.window))
    painter = QPainter(tile)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    ink = QColor(theme.ink or theme.text)
    rng = random.Random(theme.label)  # the same layout every time for a theme
    cells = 4
    cell = ICON_TILE / cells
    for row in range(cells):
        for col in range(cells):
            cx, cy = (col + 0.5) * cell, (row + 0.5) * cell
            if rng.random() < 0.3:  # a few sparkles instead of icons, so the pattern breathes
                dot = QColor(ink)
                dot.setAlpha(ICON_ALPHA + 25)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(dot)
                for _ in range(rng.randint(1, 3)):
                    painter.drawEllipse(QPointF(cx + rng.uniform(-22, 22), cy + rng.uniform(-22, 22)),
                                        *([rng.choice((0.8, 1.1, 1.5))] * 2))
                continue
            scale = rng.uniform(10, 15)
            color = QColor(ink)
            color.setAlpha(int(ICON_ALPHA * rng.uniform(0.7, 1.3)))
            painter.save()
            painter.translate(cx + rng.uniform(-11, 11), cy + rng.uniform(-11, 11))
            painter.rotate(rng.uniform(-28, 28))
            painter.scale(scale, scale)
            painter.setPen(QPen(color, 1.3 / scale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            _ICONS[rng.choice(theme.icons)](painter)
            painter.restore()
    painter.end()
    return tile


def _tile(theme: Theme) -> QPixmap:
    if theme.pattern == "icons":
        return _icon_tile(theme)
    size, paint = _PATTERNS[theme.pattern]
    tile = QPixmap(size, size)
    tile.fill(QColor(theme.window))
    ink = QColor(theme.ink or theme.text)
    ink.setAlpha(PATTERN_ALPHA * (3 if theme.pattern == "stars" else 1))  # tiny dots need more contrast than lines
    painter = QPainter(tile)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(ink, 1.0))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    paint(painter, size)
    painter.end()
    return tile


# ------------------------------------------------------------------ stylesheet of the fancy themes


def _rgba(color: QColor, alpha: float) -> str:
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha:.2f})"


def _stylesheet(theme: Theme) -> str:
    """Rounded translucent cards over the patterned background, accent buttons, slim scrollbars."""
    window, base, text = QColor(theme.window), QColor(theme.base), QColor(theme.text)
    accent, button, link = QColor(theme.highlight), QColor(theme.button), QColor(theme.link)
    hover = button.lighter(125) if window.lightness() < 128 else button.darker(108)
    border = _rgba(text, 0.12)
    white = QColor("#ffffff")
    return f"""
    QGroupBox {{
        background: {_rgba(base, 0.72)}; border: 1px solid {border}; border-radius: 12px;
        margin-top: 14px; padding: 12px 8px 8px 8px; font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 12px; padding: 1px 8px; color: {link.name()};
        background: {window.name()}; border: 1px solid {border}; border-radius: 8px;
    }}
    QPushButton {{
        background: {button.name()}; border: 1px solid {border}; border-radius: 8px; padding: 5px 12px;
        font-weight: normal;
    }}
    QPushButton:hover {{ background: {hover.name()}; border-color: {_rgba(accent, 0.7)}; }}
    QPushButton:pressed {{ background: {accent.darker(130).name()}; color: #ffffff; }}
    QPushButton:disabled {{ color: {theme.disabled}; background: {_rgba(button, 0.5)}; }}
    QPushButton[primary="true"] {{
        color: #ffffff; font-weight: 600; border: 1px solid {accent.lighter(125).name()};
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {accent.lighter(118).name()}, stop:1 {accent.darker(125).name()});
    }}
    QPushButton[primary="true"]:hover {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {accent.lighter(138).name()}, stop:1 {accent.name()});
    }}
    QPushButton[primary="true"]:disabled {{ background: {_rgba(accent, 0.35)}; color: {_rgba(white, 0.6)}; }}
    QLineEdit, QPlainTextEdit, QTextEdit, QListWidget, QListView, QTreeWidget, QTreeView, QTableWidget, QTableView {{
        background: {_rgba(base, 0.94)}; border: 1px solid {border}; border-radius: 10px;
        selection-background-color: {accent.name()}; selection-color: #ffffff;
    }}
    QLineEdit {{ padding: 3px 8px; }}
    QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border: 1px solid {_rgba(accent, 0.8)}; }}
    QListWidget::item, QTreeWidget::item {{ padding: 3px 4px; border-radius: 6px; }}
    QListWidget::item:hover, QTreeWidget::item:hover {{ background: {_rgba(accent, 0.15)}; }}
    QListWidget::item:selected, QTreeWidget::item:selected {{ background: {_rgba(accent, 0.8)}; color: #ffffff; }}
    QHeaderView::section {{ background: {button.name()}; border: none; padding: 4px 6px; }}
    QProgressBar {{
        background: {_rgba(base, 0.9)}; border: 1px solid {border}; border-radius: 8px;
        text-align: center; min-height: 16px;
    }}
    QProgressBar::chunk {{
        border-radius: 7px;
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent.darker(110).name()}, stop:1 {link.name()});
    }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {_rgba(text, 0.25)}; border-radius: 3px; min-height: 28px; min-width: 28px; }}
    QScrollBar::handle:hover {{ background: {_rgba(accent, 0.85)}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: none; background: none; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {_rgba(accent, 0.4)}; }}
    QMenuBar {{ background: {_rgba(base, 0.85)}; border-bottom: 1px solid {border}; }}
    QMenuBar::item {{ padding: 4px 10px; border-radius: 6px; background: transparent; }}
    QMenuBar::item:selected {{ background: {_rgba(accent, 0.3)}; }}
    QMenu {{ background: {base.name()}; border: 1px solid {border}; padding: 6px; }}
    QMenu::item {{ padding: 5px 24px 5px 24px; border-radius: 6px; }}
    QMenu::item:selected {{ background: {accent.name()}; color: #ffffff; }}
    QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}
    QStatusBar {{ background: {_rgba(base, 0.85)}; border-top: 1px solid {border}; }}
    QToolTip {{
        background: {base.name()}; color: {text.name()}; border: 1px solid {_rgba(accent, 0.6)};
        border-radius: 6px; padding: 4px;
    }}
    QTabWidget::pane {{ border: 1px solid {border}; border-radius: 10px; background: {_rgba(base, 0.72)}; }}
    QTabBar::tab {{
        background: {_rgba(button, 0.85)}; padding: 5px 14px; margin-right: 3px;
        border-top-left-radius: 8px; border-top-right-radius: 8px;
    }}
    QTabBar::tab:selected {{ background: {accent.name()}; color: #ffffff; }}
    """


# ------------------------------------------------------------------ applying


def apply_theme(name: str) -> None:
    global _system_style
    app = QApplication.instance()
    if _system_style is None:
        _system_style = app.style().objectName()
    theme = THEME_SPECS.get(name, THEME_SPECS["system"])
    app.setStyleSheet("")  # the palette must be in place before a stylesheet polishes the widgets
    if not theme.window:
        app.setStyle(_system_style)
        app.setPalette(app.style().standardPalette())
        return
    app.setStyle("Fusion")  # the only built-in style that follows a custom palette everywhere
    text, disabled, base = QColor(theme.text), QColor(theme.disabled), QColor(theme.base)
    palette = QPalette()
    roles = QPalette.ColorRole
    for role, color in (
        (roles.Window, QColor(theme.window)), (roles.Base, base), (roles.AlternateBase, QColor(theme.alternate)),
        (roles.Text, text), (roles.WindowText, text), (roles.ButtonText, text), (roles.ToolTipText, text),
        (roles.Button, QColor(theme.button)), (roles.ToolTipBase, base), (roles.Highlight, QColor(theme.highlight)),
        (roles.HighlightedText, QColor("#ffffff")), (roles.Link, QColor(theme.link)), (roles.PlaceholderText, disabled),
    ):
        palette.setColor(role, color)
    for role in (roles.Text, roles.WindowText, roles.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    if theme.pattern:
        palette.setBrush(roles.Window, QBrush(_tile(theme)))  # tiled behind panels; text fields keep a plain base
    app.setPalette(palette)
    if theme.fancy:
        app.setStyleSheet(_stylesheet(theme))  # re-polishes every widget with the new palette
