import logging
import sys


def _ensure_output_streams() -> None:
    """Started without a console (pythonw, the installed app) Python has no stdout/stderr at all,
    and libraries that write progress there (torch.hub, tqdm) crash. Send their output to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    from ai_reader.paths import data_dir

    log = open(data_dir() / "ai-reader.log", "w", encoding="utf-8", errors="replace", buffering=1)  # noqa: SIM115
    sys.stdout = sys.stdout or log
    sys.stderr = sys.stderr or log


def main() -> None:
    _ensure_output_streams()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from ai_reader.paths import use_models_dir

    use_models_dir()  # before anything imports huggingface_hub / torch

    from ai_reader.voices.library import install_bundled_voices

    try:
        install_bundled_voices()
    except OSError:
        logging.getLogger(__name__).exception("Could not copy the voices shipped with the app")

    from pathlib import Path

    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from ai_reader.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AI Reader")
    app.setOrganizationName("ai-reader")
    app.setWindowIcon(QIcon(str(Path(__file__).with_name("icon.ico"))))
    if sys.platform == "win32":
        import ctypes

        # without its own id Windows shows the app in the taskbar under Python's icon
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ai-reader.app")

    app.setQuitOnLastWindowClosed(False)  # the window may hide into the tray; MainWindow quits explicitly
    window = MainWindow()
    window.show()
    if len(sys.argv) > 1:
        window.open_book(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
