"""GPU vs CPU run on the same tune views: timing and output identity."""
import json, sys
from pathlib import Path
import numpy as np
G = Path('data/validation/ocr_grouped_gpu_20260923'); C = Path('data/validation/ocr_grouped_max2000_tune_20260923')
g = [json.loads(l) for l in (G / 'observations.jsonl').read_text().splitlines() if l.strip()]
c = [json.loads(l) for l in (C / 'observations.jsonl').read_text().splitlines() if l.strip()]
views = json.loads((G / 'views.json').read_text()); big = np.array([max(v['size']) > 2000 for v in views])[:len(g)]
c = c[:len(g)]
assert [x['image'] for x in g] == [x['image'] for x in c]
key = lambda w: (w['text'], round(w['conf'], 2), tuple(w['box']))
same_boxes = sum(x['boxes'] == y['boxes'] for x, y in zip(g, c))
same_words = sum(sorted(map(key, x['words'])) == sorted(map(key, y['words'])) for x, y in zip(g, c))
same_admitted = sum(sorted(w['text'] for w in x['words'] if w['conf'] >= .85) == sorted(w['text'] for w in y['words'] if w['conf'] >= .85) for x, y in zip(g, c))
sg = np.array([x['seconds'] for x in g]); sc = np.array([x['seconds'] for x in c])
stat = lambda a: dict(median=round(float(np.median(a)), 2), p95=round(float(np.percentile(a, 95)), 2), max=round(float(a.max()), 2), total_min=round(float(a.sum() / 60), 1))
out = dict(n=len(g), gpu=stat(sg), cpu=stat(sc),
           gpu_over_2000px=stat(sg[big]) if big.any() else None, gpu_upto_2000px=stat(sg[~big]) if (~big).any() else None,
           gpu_first_frame_s=round(float(sg[0]), 2), gpu_excluding_first=stat(sg[1:]) if len(sg) > 1 else None,
           identical_boxes=same_boxes, identical_words=same_words, identical_admitted_text=same_admitted)
(G / 'comparison_cpu_gpu.json').write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))
