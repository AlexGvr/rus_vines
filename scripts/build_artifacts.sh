#!/usr/bin/env bash
# Каталог и индексы из датасета кейса — то, что сервис читает при старте.
# В git их нет: они производные от дампа и пересобираются одинаково
# (привязка фото к позициям лежит в data/datapack/photo_map.csv).
#
#   scripts/build_artifacts.sh            # собрать недостающее
#   FORCE=1 scripts/build_artifacts.sh    # пересобрать всё
#
# Интерпретатор — $PYTHON (по умолчанию .venv/bin/python, если он есть,
# иначе python). Этот же скрипт запускает сервис indexer в docker compose.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python; fi
fi

need() { [ "${FORCE:-0}" = "1" ] || [ ! -s "$1" ]; }

if [ ! -f dataset/strapi_output0709.csv ]; then
  echo "нет dataset/strapi_output0709.csv — сначала подготовьте датасет (scripts/setup.sh или README, «Данные»)" >&2
  exit 1
fi
if ! find dataset/uploads_root -type d -name uploads -print -quit 2>/dev/null | grep -q .; then
  echo "нет dataset/uploads_root/**/uploads — распакуйте дамп Strapi (README, «Данные»)" >&2
  exit 1
fi

if need data/catalog/catalog.json; then
  echo "== каталог (секунды)"
  "$PY" pipeline/build_catalog.py
fi
if need data/index/clean_mv.npy; then
  echo "== векторный индекс SigLIP 2: кадр целиком и кроп по бутылке (GPU — минуты, CPU — до получаса)"
  "$PY" pipeline/build_index.py --variant clean --views raw,detect --name clean_mv
fi
if need data/index/sift/desc.npy; then
  echo "== признаки SIFT эталонов (CPU, несколько минут)"
  "$PY" pipeline/build_rerank_index.py
fi
echo "каталог и индексы готовы"
