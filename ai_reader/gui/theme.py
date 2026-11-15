"""Colour themes for the whole application, some with a faint tiled background pattern."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

PATTERN_ALPHA = 16  # 0–255: how visible the pattern is over the window colour; keep it barely there


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
    pattern: str = ""  # a key of _PATTERNS, or empty for a plain background
    ink: str = ""  # pattern colour; defaults to the text colour


_LIGHT = dict(window="#f3f3f3", base="#ffffff", alternate="#f7f7f7", text="#1e1e1e", button="#e9e9e9",
              highlight="#2f6fd6", link="#1a5fb4", disabled="#9a9a9a")
_DARK = dict(window="#2b2b2b", base="#1e1e1e", alternate="#262626", text="#e2e2e2", button="#3a3a3a",
             highlight="#3d7be0", link="#6ea8ff", disabled="#7b7b7b")
_SEPIA = dict(window="#eadfc8", base="#f6eedb", alternate="#efe5cf", text="#3b2f22", button="#e0d3b8",
              highlight="#a9743a", link="#7a4e1d", disabled="#a3957c")
_NIGHT = dict(window="#1d2433", base="#161b27", alternate="#1a2030", text="#dfe5f2", button="#2a3348",
              highlight="#5b8def", link="#8fb4ff", disabled="#6c7890")
_FOREST = dict(window="#e6ece2", base="#f7faf4", alternate="#eef3ea", text="#23301f", button="#d7e0d1",
               highlight="#4f8a4b", link="#2f6b2c", disabled="#93a08d")
_WINE = dict(window="#2a2024", base="#1f171a", alternate="#261c20", text="#eadfe2", button="#3b2c32",
             highlight="#b0566e", link="#e39ab0", disabled="#806c73")

THEME_SPECS = {
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
DEFAULT_THEME = "dark"
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


def _tile(theme: Theme) -> QPixmap:
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


# ------------------------------------------------------------------ applying


def apply_theme(name: str) -> None:
    global _system_style
    app = QApplication.instance()
    if _system_style is None:
        _system_style = app.style().objectName()
    theme = THEME_SPECS.get(name, THEME_SPECS["system"])
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
