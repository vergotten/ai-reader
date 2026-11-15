"""Windows paths in WSL: "D:\\Книги\\x.fb2" -> "/mnt/d/Книги/x.fb2", so books can be opened by their Windows path."""

from __future__ import annotations

import os
import re
from pathlib import Path

_DRIVE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$", re.DOTALL)
_WSL_SHARE = re.compile(r"^[\\/]{2}wsl(?:\$|\.localhost)[\\/]+[^\\/]+[\\/]*(.*)$", re.IGNORECASE | re.DOTALL)


def is_wsl() -> bool:
    if os.name == "nt":
        return False
    try:
        return "microsoft" in Path("/proc/sys/kernel/osrelease").read_text().lower()
    except OSError:
        return False


def clean_path_text(text: str) -> str:
    """Trim what people paste: spaces and the quotes added by Explorer's «Копировать как путь»."""
    return text.strip().strip("\"'").strip()


def to_local_path(text: str, wsl: bool | None = None) -> str:
    """Convert a Windows path to the WSL mount path when running in WSL; other paths are returned unchanged."""
    path = clean_path_text(text)
    if wsl is None:
        wsl = is_wsl()
    if not wsl:
        return path
    if m := _WSL_SHARE.match(path):
        return "/" + m.group(1).replace("\\", "/")
    if m := _DRIVE.match(path):
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{m.group(1).lower()}/{rest}".rstrip("/") if rest else f"/mnt/{m.group(1).lower()}"
    return path


def windows_drives() -> list[str]:
    """Mounted Windows drives in WSL (/mnt/c, /mnt/d...), for the file dialog's sidebar."""
    if not is_wsl():
        return []
    return sorted(str(p) for p in Path("/mnt").glob("[a-z]") if p.is_dir())
