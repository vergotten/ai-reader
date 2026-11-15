#!/usr/bin/env bash
# One-time setup of AI Reader in WSL (Ubuntu): system libraries for the Qt window and sound, uv, Python packages.
# Usage:  bash setup-wsl.sh            (all engines)
#         bash setup-wsl.sh silero     (only the extras you name)
set -euo pipefail
cd "$(dirname "$0")"

# The Linux environment lives in the WSL home: imports from /mnt/f are very slow,
# and the project's .venv belongs to Windows.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/ai-reader}"

echo "== Системные библиотеки (нужен пароль sudo)"
# "apt-get update" also fails when only a post-update hook breaks (e.g. python3 switched to a deadsnakes
# version, so /usr/lib/cnf-update-db can't import apt_pkg); the package lists are still fresh, so go on.
sudo apt-get update || echo "!! apt-get update вернул ошибку (см. выше) — продолжаю установку."
sudo apt-get install -y \
    libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libfontconfig1 libdbus-1-3 \
    libxcb-cursor0 libxcb-icccm4 libxcb-keysyms1 libxcb-shape0 libxcb-randr0 \
    libxcb-render-util0 libxcb-xinerama0 libxcb-image0 \
    libportaudio2 libasound2-plugins ffmpeg fonts-dejavu-core

# PortAudio talks to ALSA; route ALSA's default device to WSLg's PulseAudio server
if [ -e /mnt/wslg/PulseServer ] && ! grep -qs "type pulse" "$HOME/.asoundrc"; then
    echo "== Звук: ALSA -> PulseAudio WSLg (~/.asoundrc)"
    printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' >> "$HOME/.asoundrc"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "== Установка uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

extras=("$@")
[ ${#extras[@]} -eq 0 ] && extras=(all)
args=()
for e in "${extras[@]}"; do args+=(--extra "$e"); done

echo "== Python-пакеты в $UV_PROJECT_ENVIRONMENT (${extras[*]})"
uv sync "${args[@]}"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "== Видеокарта:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
else
    echo "== nvidia-smi не найден: модели будут работать на CPU. Для GPU установите драйвер NVIDIA в Windows (WSL подхватит его сам)."
fi

echo
echo "Готово. Запуск:  bash run.sh"
