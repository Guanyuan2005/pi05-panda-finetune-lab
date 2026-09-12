#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
# Avoid LeRobot real-hardware extras (for example evdev) in this MuJoCo-only project.
"${VENV_DIR}/bin/python" -m pip install --no-deps lerobot==0.4.4
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements-local.txt"

echo "Local environment ready: ${VENV_DIR}"
