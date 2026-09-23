#!/bin/bash
# Sequential probes with a 24 GB virtual-memory cap; RSS polled from outside every 2 s.
cd /home/alex/projects/rus_vines
P=data/validation/ocr_grouped_memory_20260923
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
run() { # tag, extra args...
  tag=$1; shift
  echo "=== $tag $(date +%T)"
  ( ulimit -v 25000000; nice -n 5 /tmp/rus-vines-paddle-ocr/bin/python $P/probe.py --tag $tag "$@" ) &
  pid=$!
  peak=0
  while kill -0 $pid 2>/dev/null; do
    rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
    sleep 2
  done
  wait $pid; echo "exit=$? peak_rss_mb=$((peak/1024)) $(date +%T)"
}
run native_frame11 data/validation/ocr_grouped_tune_20260923/crops/010.png
run max2000_frame11 data/validation/ocr_grouped_tune_20260923/crops/010.png --limit-type max --limit-side 2000
run max1600_frame11 data/validation/ocr_grouped_tune_20260923/crops/010.png --limit-type max --limit-side 1600
run native_frame10 data/validation/ocr_grouped_tune_20260923/crops/009.png
echo "=== probe done $(date +%T)"
