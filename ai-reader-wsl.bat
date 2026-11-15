@echo off
rem AI Reader in WSL: double-click to start, or drag a book file onto this .bat (also works with "Open with").
rem Windows paths are converted by the app (D:\Books\x.fb2 -> /mnt/d/Books/x.fb2).
start "AI Reader" /min wsl.exe --cd "%~dp0." bash run.sh %*
