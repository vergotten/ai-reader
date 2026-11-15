@echo off
rem Builds the Windows installer into dist\. Arguments are passed on, e.g.  build-installer.bat -Models clone
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File installer\build.ps1 %*
pause
