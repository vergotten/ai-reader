@echo off
rem The smallest full installer: F5-TTS in fp16 and the strongest compression. Takes a few hours: start it for the night.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File installer\build.ps1 -Max %*
pause
