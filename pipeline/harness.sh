#!/usr/bin/env bash
# S0.7 — одна команда для полного замера сервиса.
#
#   1) официальный скрипт кейсодержателя на публичных фото
#   2) метрики ТЗ на синтетическом наборе
#   3) полевой набор, финальная часть
#
# Прогон либо проходит целиком, либо падает. Раньше полевая часть глушила
# ошибку через `|| true`, и отсутствующий кадр или лежащий сервис давали
# успешный отчёт с неполными числами — то есть отчёт, которому нельзя верить.
#
# Полный журнал пишется в data/index/harness.log, на экран идёт сокращённый
# вывод: искать причину падения по обрезанному тексту невозможно.
#
# Сервис должен быть уже запущен (см. README, раздел «Запуск»).
set -euo pipefail
cd "$(dirname "$0")/.."

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8080/v1/eval/predict}"
SUBSET="${SUBSET:-sharp}"
SPLIT="${SPLIT:-test}"
PY=.venv/bin/python
LOG=data/index/harness.log
mkdir -p data/index
: > "$LOG"

# Весь вывод дублируется в журнал: на экране короткая версия, в файле полная.
run() { "$@" >>"$LOG" 2>&1; }

if ! curl -sf "${ENDPOINT%/v1/eval/predict}/health" >/dev/null; then
  echo "сервис не отвечает на ${ENDPOINT%/v1/eval/predict}/health" >&2
  exit 1
fi

echo "=== 0. Что именно меряем ==="
$PY pipeline/stamp.py | tee -a "$LOG"

echo
echo "=== 1. Скрипт кейсодержателя (публичный датасет) ==="
rm -f dataset/eval/predictions.jsonl
(cd dataset/eval && ./participant_test.sh \
  --images-dir ./queries --manifest ./queries.tsv \
  --endpoint "$ENDPOINT" --output ./predictions.jsonl) >>"$LOG" 2>&1
$PY - <<'PY' | tee -a "$LOG"
import csv
import json
import sys

rows = [json.loads(line) for line in open("dataset/eval/predictions.jsonl")]
expected = sum(1 for _ in csv.DictReader(
    open("dataset/eval/queries.tsv", encoding="utf-8"), delimiter="\t"))
if len(rows) != expected:
    sys.exit(f"обработано {len(rows)} снимков из {expected} по манифесту")
nulls = sum(1 for r in rows if r["predicted_slug"] is None)
lat = sorted(r["latency_ms"] for r in rows)
print(f"запросов {len(rows)}, без ответа {nulls}, "
      f"латентность med {lat[len(lat)//2]} мс, max {lat[-1]} мс")
for row in rows:
    print(f"  {row['image_path']:18} -> {row['predicted_slug']}")
PY

echo
echo "=== 2. Метрики ТЗ на синтетическом наборе (${SUBSET}) ==="
$PY pipeline/report.py --subset "$SUBSET" --held-out 2>&1 \
  | tee -a "$LOG" | grep -v -e '^Loading weights' -e 'token_id'

echo
echo "=== 3. Полевой набор, часть ${SPLIT} ==="
if [ ! -f data/field/manifest.csv ]; then
  echo "полевой набор не собран, см. pipeline/collect_field.py" >&2
  exit 1
fi
$PY pipeline/eval_field.py --split "$SPLIT" 2>&1 \
  | tee -a "$LOG" | grep -v -e '^Loading weights' -e 'token_id' -e '^  [a-z0-9_]*\.jpg'

echo
echo "полный журнал: $LOG"
