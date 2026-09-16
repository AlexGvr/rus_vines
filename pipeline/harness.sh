#!/usr/bin/env bash
# S0.7 — одна команда для полного замера сервиса.
#
#   1) официальный скрипт кейсодержателя на публичных фото
#   2) наш валидационный набор: top-1 / top-5 / p95 латентности
#
# Сервис должен быть уже запущен (см. README, раздел «Запуск»).
set -euo pipefail
cd "$(dirname "$0")/.."

ENDPOINT="${ENDPOINT:-http://127.0.0.1:8080/v1/eval/predict}"
SUBSET="${SUBSET:-sharp}"
PY=.venv/bin/python

if ! curl -sf "${ENDPOINT%/v1/eval/predict}/health" >/dev/null; then
  echo "сервис не отвечает на ${ENDPOINT%/v1/eval/predict}/health" >&2
  exit 1
fi

echo "=== 1. Скрипт кейсодержателя (публичный датасет) ==="
rm -f dataset/eval/predictions.jsonl
(cd dataset/eval && ./participant_test.sh \
  --images-dir ./queries --manifest ./queries.tsv \
  --endpoint "$ENDPOINT" --output ./predictions.jsonl >/dev/null)
$PY - <<'PY'
import json
rows = [json.loads(l) for l in open("dataset/eval/predictions.jsonl")]
nulls = sum(1 for r in rows if r["predicted_slug"] is None)
lat = sorted(r["latency_ms"] for r in rows)
print(f"запросов {len(rows)}, без ответа {nulls}, "
      f"латентность med {lat[len(lat)//2]} мс, max {lat[-1]} мс")
for r in rows:
    print(f"  {r['image_path']:18} -> {r['predicted_slug']}")
PY

echo
echo "=== 2. Метрики ТЗ на валидационном наборе (${SUBSET}) ==="
# Считаем по артефакту калибровки: он получен прогоном того же конвейера,
# что работает в сервисе. Отдельный прогон eval_index здесь только сбивал бы
# с толку — он меряет старые одновидовые индексы clean/dirty, которых в
# работе давно нет.
$PY pipeline/report.py --subset "$SUBSET" --held-out 2>&1 \
  | grep -v -e '^Loading weights' -e 'token_id'

echo
echo "=== 3. Полевой набор ==="
if [ -f data/field/manifest.csv ]; then
  $PY pipeline/eval_field.py 2>&1 \
    | grep -v -e '^Loading weights' -e 'token_id' -e '^  ' || true
else
  echo "полевой набор не собран, см. pipeline/collect_field.py"
fi
