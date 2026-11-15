#!/usr/bin/env bash
# Start AI Reader in WSL / Linux. First run setup-wsl.sh once.
# Usage:  bash run.sh ["/mnt/d/Книги/книга.fb2"]
set -euo pipefail
cd "$(dirname "$0")"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/ai-reader}"
export PATH="$HOME/.local/bin:$PATH"
if [ -z "${PULSE_SERVER:-}" ] && [ -e /mnt/wslg/PulseServer ]; then
    export PULSE_SERVER=unix:/mnt/wslg/PulseServer  # WSLg sound
fi

if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    echo "Нет графического дисплея. Нужен WSLg (Windows 11 или Windows 10 21H2+ с 'wsl --update')." >&2
    exit 1
fi
if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
    echo "Окружение не установлено — сначала выполните: bash setup-wsl.sh" >&2
    exit 1
fi

exec uv run --no-sync ai-reader "$@"
