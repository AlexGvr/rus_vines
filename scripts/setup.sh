#!/usr/bin/env bash
# Установка с нуля на хосте: датасет → окружение → каталог и индексы.
#
#   scripts/setup.sh            # видеокарта NVIDIA (torch cu128, драйвер 570+)
#   TORCH=cpu scripts/setup.sh  # без видеокарты
#   TORCH=cu126 scripts/setup.sh  # старый драйвер NVIDIA
#
# Датасет кейса кладётся в корень репозитория как выдан (Датасет.zip)
# или уже распакованным в dataset/. После установки — scripts/run.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- 1. Датасет
if [ ! -f dataset/strapi_output0709.csv ] && [ -f Датасет.zip ]; then
  echo "== распаковка Датасет.zip"
  tmp=$(mktemp -d)
  unzip -q Датасет.zip -d "$tmp"
  mkdir -p dataset
  mv "$tmp"/Датасет/* dataset/
  rm -rf "$tmp"
fi
if [ ! -f dataset/strapi_output0709.csv ]; then
  echo "нет датасета: положите Датасет.zip в корень репозитория или распакуйте его в dataset/" >&2
  exit 1
fi
if [ ! -d dataset/eval ] && [ -f dataset/eval.zip ]; then
  unzip -q dataset/eval.zip -d dataset/eval
fi
if ! find dataset/uploads_root -type d -name uploads -print -quit 2>/dev/null | grep -q .; then
  echo "== распаковка дампа Strapi (RAR5, 2.2 ГБ) в dataset/uploads_root"
  mkdir -p dataset/uploads_root
  rar=dataset/prod-svoe-vino-strapi.part1.rar
  if command -v unrar >/dev/null; then
    unrar x -idq "$rar" dataset/uploads_root/
  elif command -v docker >/dev/null; then
    # unrar на хосте нет — берём его из контейнера Ubuntu (multiverse).
    docker run --rm -v "$PWD/dataset:/d" ubuntu:24.04 sh -c \
      "apt-get update -qq && apt-get install -y -qq --no-install-recommends unrar >/dev/null && unrar x -idq /d/prod-svoe-vino-strapi.part1.rar /d/uploads_root/"
  elif command -v 7zz >/dev/null; then
    7zz x -y -bso0 "$rar" -odataset/uploads_root
  else
    echo "нужен распаковщик RAR5: unrar (sudo apt install unrar), 7-Zip 7zz или Docker" >&2
    exit 1
  fi
  # unar и 7-Zip без кодека RAR распаковывают RAR5 с пустыми файлами —
  # такой дамп лучше остановить здесь, чем получить индекс с дырами.
  if find dataset/uploads_root -type f -size 0 -print -quit | grep -q .; then
    echo "после распаковки есть пустые файлы — распаковщик не справился с RAR5, возьмите unrar" >&2
    exit 1
  fi
fi

# --- 2. Окружение
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --index-url "https://download.pytorch.org/whl/${TORCH:-cu128}" \
  torch==2.11.0 torchvision==0.26.0
.venv/bin/pip install -q -r requirements.txt

# --- 3. Каталог и индексы (первый запуск скачает SigLIP 2, около 4.5 ГБ, и YOLO11m)
PYTHON=.venv/bin/python scripts/build_artifacts.sh

echo
echo "готово. Запуск сервиса: scripts/run.sh (порт 8080), проверка: curl http://127.0.0.1:8080/health"
