#!/bin/bash
# Regression part, same variant as ocr_grouped_max2000_tune_20260923:
# detector input capped at 2000 px, CPU, virtual memory capped at ~19 GB.
cd /home/alex/projects/rus_vines
T=data/validation/ocr_grouped_max2000_test_20260923
ulimit -v 20000000
echo "=== launch $(date '+%F %T') pid $$" >> $T/run.log
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
/tmp/rus-vines-paddle-ocr/bin/python \
pipeline/experiment_paddle_groups.py \
--resume \
--det-limit-type max --det-limit-side 2000 \
--views $T/views.json \
--output $T/observations.jsonl \
>> $T/run.log 2>&1
echo "=== python exit $? $(date '+%F %T')" >> $T/run.log
