#!/usr/bin/env bash
# Idempotent Cloud Agent install: Python 3.12 venv + requirements.txt into a persistent venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PINNS_VENV:-${HOME}/.pinns/venv}"

ensure_python_venv_apt () {
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-pip python3-venv python3-full \
      || true
  fi
}

if ! python3 -m venv -h >/dev/null 2>&1; then
  ensure_python_venv_apt
fi

mkdir -p "$(dirname "$VENV")"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}" || {
    ensure_python_venv_apt
    python3 -m venv "${VENV}"
  }
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel
# torch pins setuptools<82; upgrade pip first without breaking that constraint
python -m pip install 'setuptools>=70,<82'
python -m pip install -r "${ROOT}/requirements.txt"

MARKER="# pinns-venv-activate (managed by .cursor/install.sh)"
ACTIVATE_LINE="[ -f '${VENV}/bin/activate' ] && . '${VENV}/bin/activate'"
if [[ -f "${HOME}/.bashrc" ]] && ! grep -qF "${MARKER}" "${HOME}/.bashrc" 2>/dev/null; then
  printf '\n%s\n%s\n' "${MARKER}" "${ACTIVATE_LINE}" >> "${HOME}/.bashrc"
fi

echo "PINNS cloud env ready: ${VENV} ($(python -c 'import sys; print(sys.version.split()[0])'))"
