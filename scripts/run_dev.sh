#!/usr/bin/env bash
set -euo pipefail

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "[!] Activate your venv first: source .venv/bin/activate" >&2
  exit 1
fi

export PYTHONPATH=.
PORT=$(grep -E '^APP_PORT=' config/.env | cut -d'=' -f2 | tr -d '\r' || echo 8000)
exec uvicorn src.api.app.main:app --reload --port "${PORT:-8000}"
