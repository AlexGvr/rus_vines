#!/bin/bash
# GPU timing run: same models, same detector limit (max side 2000 px), same tune
# views as ocr_grouped_max2000_tune_20260923; device=gpu in the separate
# /tmp/rus-vines-paddle-gpu environment (paddlepaddle-gpu 3.3.1, cu129).
cd /home/alex/projects/rus_vines
G=data/validation/ocr_grouped_gpu_20260923
echo "=== launch $(date '+%F %T') pid $$" >> $G/run.log
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> $G/run.log
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
/tmp/rus-vines-paddle-gpu/bin/python \
pipeline/experiment_paddle_groups.py \
--resume --device gpu \
--det-limit-type max --det-limit-side 2000 \
--views $G/views.json \
--output $G/observations.jsonl \
>> $G/run.log 2>&1
echo "=== python exit $? $(date '+%F %T')" >> $G/run.log
