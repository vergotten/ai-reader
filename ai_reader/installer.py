"""Install a missing optional package from inside the app, so nobody has to open a console for
«Установите пакет ruaccent». Works for the uv project venv, the installed app and a plain venv."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from typing import Callable

Log = Callable[[str], None]

# what to install for a missing feature: module name -> pip requirement(s)
PACKAGES = {
    "ruaccent": ["ruaccent"],  # neural stress marks (~0.5 GB model is downloaded on first use)
    "av": ["av"],  # decoding M4A/AAC/AMR voice messages
    "transformers": ["transformers>=4.52,<5"],  # Whisper, to make a voice from recordings
}


def _commands(requirements: list[str]) -> list[list[str]]:
    cmds = []
    uv = shutil.which("uv")
    if uv:  # the uv-made venv has no pip at all
        cmds.append([uv, "pip", "install", "--python", sys.executable, *requirements])
    cmds.append([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *requirements])
    return cmds


def install(requirements: list[str], log: Log, cancelled: Callable[[], bool] = lambda: False) -> None:
    """pip-install requirements into the running interpreter's environment; raises RuntimeError on failure."""
    python = sys.executable.replace("pythonw.exe", "python.exe")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    errors = []
    for cmd in _commands(requirements):
        cmd = [python if c == sys.executable else c for c in cmd]
        if cmd[1:3] == ["-m", "pip"]:
            # the bundled Python may come without pip: bootstrap it first
            subprocess.run([python, "-m", "ensurepip", "--upgrade"], capture_output=True, creationflags=flags, check=False)
        log("Установка: " + " ".join(requirements) + " …")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", creationflags=flags)
        tail: list[str] = []
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-15:]
                log(line)
            if cancelled():
                proc.kill()
                raise RuntimeError("Установка отменена")
        if proc.wait() == 0:
            importlib.invalidate_caches()
            log("Установлено: " + " ".join(requirements))
            return
        errors.append("\n".join(tail[-6:]))
    raise RuntimeError("Не удалось установить " + " ".join(requirements) + ":\n" + "\n\n".join(errors))
