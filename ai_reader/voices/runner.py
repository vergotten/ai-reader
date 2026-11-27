"""Heavy voice jobs (preparing recordings, fine-tuning) run in child processes.

A child process gives all of its memory back when it ends, can be stopped at once, and can't freeze the window.
Its output is read line by line on a background thread and turned into callbacks.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

LineHandler = Callable[[str], None]
ExitHandler = Callable[[int], None]

_EPOCH = re.compile(r"Epoch (\d+)/(\d+)")
_STEP = re.compile(r"\|\s*(\d+)/(\d+) \[([^<\]]*)<([^,\]]*)")
_LOSS = re.compile(r"loss=([\d.]+)")
_UPDATE = re.compile(r"update=(\d+)")
_PROGRESS = re.compile(r"^@@ progress (\d+) (\d+) (.*)$")
_ERROR = re.compile(r"^@@ error (.*)$")
_MESSAGE = re.compile(r"^@@ (\w+) ?(.*)$")


def python_executable() -> str:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":  # the GUI runs windowless; the child needs a real stdout
        console = exe.with_name("python.exe")
        if console.exists():
            return str(console)
    return str(exe)


class ChildProcess:
    """`python -m <module> <args>` with output passed to on_line (minus the lines handle() consumes)."""

    def __init__(self, module: str, args: list[str], on_line: LineHandler, on_exit: ExitHandler) -> None:
        self.on_line, self.on_exit = on_line, on_exit
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        package_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(
            [python_executable(), "-m", module, *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            creationflags=flags,
        )
        self._thread = threading.Thread(target=self._read, daemon=True, name=f"ai-reader-{module}")
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

    def handle(self, line: str) -> bool:
        """Subclasses: True when the line was consumed (a progress line) and should not reach the log."""
        return False

    def finished(self) -> None:
        """Subclasses: called after the output ends, before on_exit."""

    def _read(self) -> None:
        buf = b""
        stream = self.proc.stdout
        while data := stream.read1(4096):
            buf += data
            *lines, buf = re.split(rb"[\r\n]", buf)
            for raw in lines:
                self._line(raw)
        self._line(buf)
        self.finished()
        self.on_exit(self.proc.wait())

    def _line(self, raw: bytes) -> None:
        line = raw.decode("utf-8", "replace").strip()
        if line and not self.handle(line):
            self.on_line(line)


# ------------------------------------------------------------------ preparing recordings


class PrepareProcess(ChildProcess):
    """ai_reader.voices.dataset: cut, transcribe and stress recordings for one voice."""

    def __init__(
        self,
        voice_dir: Path,
        files: list[str],
        device: str,
        add_stress: bool,
        on_line: LineHandler,
        on_progress: Callable[[int, int, str], None],
        on_exit: ExitHandler,
    ) -> None:
        self.on_progress = on_progress
        self.error = ""  # the message of an "@@ error" line
        self.cancelled = False  # set by the window before stop(), so the exit isn't reported as a failure
        # the file list goes through a file: a folder of voice messages can exceed the command line limit
        listing = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        with listing:
            listing.write("\n".join(files))
        self._listing = Path(listing.name)
        args = ["--voice-dir", str(voice_dir), "--files-from", str(self._listing), "--device", device]
        if not add_stress:
            args.append("--no-stress")
        super().__init__("ai_reader.voices.dataset", args, on_line, on_exit)

    def handle(self, line: str) -> bool:
        if m := _PROGRESS.match(line):
            self.on_progress(int(m.group(1)), int(m.group(2)), m.group(3))
            return True
        if m := _ERROR.match(line):
            self.error = m.group(1).replace("\\n", "\n")
            return True
        return False

    def finished(self) -> None:
        self._listing.unlink(missing_ok=True)


# ------------------------------------------------------------------ fine-tuning


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


class TrainProcess(ChildProcess):
    """ai_reader.voices.f5_train; tqdm progress lines become TrainStatus updates."""

    def __init__(
        self,
        voice_dir: Path,
        settings: TrainSettings,
        on_line: LineHandler,
        on_status: Callable[[TrainStatus], None],
        on_exit: ExitHandler,
        export_only: bool = False,
    ) -> None:
        self.on_status = on_status
        self.error = ""  # the message of an "@@ error" line
        self.out_of_memory = False  # the GPU ran out of memory: retry with a smaller batch
        self.cancelled = False
        self._status = TrainStatus(epochs=settings.epochs)
        self._last_progress = ""
        args = [
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
        super().__init__("ai_reader.voices.f5_train", args, on_line, on_exit)

    def handle(self, line: str) -> bool:
        if m := _MESSAGE.match(line):  # "@@ <kind> <text>" lines from f5_train.say()
            kind, text = m.group(1), m.group(2)
            if kind == "error":
                self.error = text
            elif kind == "oom":
                self.out_of_memory = True
            if text and kind != "exported":
                self.on_line(text)
            return True
        epoch = _EPOCH.search(line)
        if not epoch:
            return False
        s = self._status
        s.epoch, s.epochs = int(epoch.group(1)), int(epoch.group(2))
        within = 0.0
        if (step := _STEP.search(line)) and int(step.group(2)):
            within = int(step.group(1)) / int(step.group(2))
            s.eta = step.group(4).strip()
        s.fraction = min(1.0, ((s.epoch - 1) + within) / max(s.epochs, 1))
        if m := _LOSS.search(line):
            s.loss = float(m.group(1))
        if m := _UPDATE.search(line):
            s.update = int(m.group(1))
        self.on_status(TrainStatus(**vars(s)))
        self._last_progress = line  # tqdm redraws constantly: log only the last state
        return True

    def finished(self) -> None:
        if self._last_progress:
            self.on_line(self._last_progress)
