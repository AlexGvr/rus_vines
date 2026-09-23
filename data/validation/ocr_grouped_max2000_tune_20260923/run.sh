#!/bin/bash
# Variant: detector input limited to max side 2000 px (text_det_limit_type=max,
# text_det_limit_side_len=2000). Everything else as in ocr_grouped_tune_20260923.
# Virtual memory capped at ~19 GB so an allocation failure surfaces as a
# MemoryError traceback in run.log instead of a kernel OOM kill.
cd /home/alex/projects/rus_vines
V=data/validation/ocr_grouped_max2000_tune_20260923
ulimit -v 20000000
echo "=== launch $(date '+%F %T') pid $$" >> $V/run.log
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
/tmp/rus-vines-paddle-ocr/bin/python \
pipeline/experiment_paddle_groups.py \
--resume \
--det-limit-type max --det-limit-side 2000 \
--views $V/views.json \
--output $V/observations.jsonl \
>> $V/run.log 2>&1
echo "=== python exit $? $(date '+%F %T')" >> $V/run.log
