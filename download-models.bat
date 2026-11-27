@echo off
rem Downloads every model (~7 GB) into the models\ folder, so the app never downloads anything while in use.
rem Already downloaded models are skipped. Needs uv: https://docs.astral.sh/uv/
cd /d "%~dp0"
uv run python -m ai_reader.fetch_models all
pause
