#!/usr/bin/env bash
# Сервис распознавания на хосте после scripts/setup.sh.
#
#   scripts/run.sh                 # http://127.0.0.1:8080
#   HOST=0.0.0.0 scripts/run.sh    # доступ с телефона в той же сети
#
# Первый старт скачивает Qwen3-VL-4B (около 8.3 ГБ) и модели EasyOCR.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"
if [ ! -s data/index/clean_mv.npy ] || [ ! -s data/catalog/catalog.json ]; then
  echo "нет каталога или индекса — сначала scripts/setup.sh" >&2
  exit 1
fi
exec "$PY" -m uvicorn service.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}"
