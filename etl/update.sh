#!/usr/bin/env bash
# Серверный контур: обновление датапака.
# Крон раз в сутки: докачивает новые вина (resume-safe) и пересобирает датапак.
# Новая версия = новая дата + hash в meta; приложение сравнивает hash и
# перекачивает data/wines.json + недостающие фото.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 etl/scrape.py
python3 etl/build_datapack.py
echo "OK: датапак пересобран, артефакты в data/datapack/ и app/"
