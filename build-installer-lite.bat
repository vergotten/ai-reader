@echo off
rem Small installer for slow connections: Silero only, CPU PyTorch, one .exe of about 0.5 GB in dist\.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File installer\build.ps1 -Lite %*
pause
