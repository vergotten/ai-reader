"""Runs f5_train in a child process and turns its output into progress callbacks."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_EPOCH = re.compile(r"Epoch (\d+)/(\d+)")
_STEP = re.compile(r"\|\s*(\d+)/(\d+) \[([^<\]]*)<([^,\]]*)")
_LOSS = re.compile(r"loss=([\d.]+)")
_UPDATE = re.compile(r"update=(\d+)")


@dataclass
class TrainSettings:
    base_model: str = "misha24-v2"
    epochs: int = 60
    learning_rate: float = 1e-5
    batch_frames: int = 3200
    grad_accum: int = 1
    bnb: bool = False


@dataclass
class TrainStatus:
    fraction: float = 0.0  # 0..1 over all epochs
    epoch: int = 0
    epochs: int = 0
    update: int = 0
    loss: float | None = None
    eta: str = ""


def python_executable() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":  # the GUI runs windowless; the trainer needs real stdout
        console = exe.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(exe)


class TrainProcess:
    def __init__(
        self,
        voice_dir: Path,
        settings: TrainSettings,
        on_line: Callable[[str], None],
        on_status: Callable[[TrainStatus], None],
        on_exit: Callable[[int], None],
        export_only: bool = False,
    ) -> None:
        self.on_line, self.on_status, self.on_exit = on_line, on_status, on_exit
        args = [
            python_executable(), "-m", "ai_reader.voices.f5_train",
            "--voice-dir", str(voice_dir),
            "--base", settings.base_model,
            "--epochs", str(settings.epochs),
            "--learning-rate", str(settings.learning_rate),
            "--batch-frames", str(settings.batch_frames),
            "--grad-accum", str(settings.grad_accum),
        ]
        if settings.bnb:
            args.append("--bnb")
        if export_only:
            args.append("--export-only")
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self._status = TrainStatus(epochs=settings.epochs)
        self.proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=flags,
        )
        self._thread = threading.Thread(target=self._read, daemon=True, name="ai-reader-train")
        self._thread.start()

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def stop(self) -> None:
        if self.running:
            self.proc.terminate()

    def wait(self, timeout: float) -> None:
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def _read(self) -> None:
        buf = b""
        last_progress_line = ""
        stream = self.proc.stdout
        while True:
            data = stream.read1(4096)
            if not data:
                break
            buf += data
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                if self._parse(line):
                    last_progress_line = line  # tqdm redraws: don't flood the log
                    continue
                self.on_line(line)
        if buf.strip():
            self.on_line(buf.decode("utf-8", "replace").strip())
        if last_progress_line:
            self.on_line(last_progress_line)
        self.on_exit(self.proc.wait())

    def _parse(self, line: str) -> bool:
        epoch = _EPOCH.search(line)
        if not epoch:
            return False
        s = self._status
        s.epoch, s.epochs = int(epoch.group(1)), int(epoch.group(2))
        step = _STEP.search(line)
        within = 0.0
        if step and int(step.group(2)):
            within = int(step.group(1)) / int(step.group(2))
            s.eta = step.group(4).strip()
        s.fraction = min(1.0, ((s.epoch - 1) + within) / max(s.epochs, 1))
        if m := _LOSS.search(line):
            s.loss = float(m.group(1))
        if m := _UPDATE.search(line):
            s.update = int(m.group(1))
        self.on_status(s)
        return True
