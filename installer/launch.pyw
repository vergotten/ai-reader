"""Entry point of the installed app: started by the shortcuts as  python\\pythonw.exe launch.pyw [book]."""

import os
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))


def _report(error: BaseException) -> None:
    """pythonw has no console: keep the traceback in a file and tell the user where it is."""
    log_dir = Path(os.environ.get("LOCALAPPDATA", APP_DIR)) / "ai-reader"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / "crash.log"
    log.write_text("".join(traceback.format_exception(error)), encoding="utf-8")
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, f"AI Reader could not start:\n{error}\n\nDetails: {log}", "AI Reader", 0x10)


try:
    from ai_reader.__main__ import main

    main()
except Exception as error:  # noqa: BLE001 — SystemExit from a normal quit passes through
    _report(error)
    raise SystemExit(1) from error
