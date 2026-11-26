#!/usr/bin/env bash
# Start script to create virtual env on first run or when requirements.txt changes
#   ./start.sh            start the Streamlit UI; run the pipeline stages from its sidebar
#   extra args are passed to streamlit, e.g. ./start.sh --server.port 8600
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV=.venv
STAMP="$VENV/.requirements.sha256"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: $PYTHON not found. Install Python 3.12 or newer." >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
  echo "Error: Python 3.12+ required. Fund $("$PYTHON" --version 2>&1)." >&2
  exit 1
fi

req_hash=$(shasum -a 256 requirements.txt | cut -d' ' -f1)

if [ ! -x "$VENV/bin/python" ]; then
  echo "Creating virtual env in $VENV ..."
  "$PYTHON" -m venv "$VENV"
fi

if [ "$(cat "$STAMP" 2>/dev/null)" != "$req_hash" ]; then
  echo "Installing deps..."
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements.txt
  echo "$req_hash" > "$STAMP"
fi

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "No .env found. copying .env.example. Edit to configure the LLM endpoint & FalkorDB."
  cp .env.example .env
fi

exec "$VENV/bin/streamlit" run phase_ui.py "$@"
