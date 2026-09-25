#!/usr/bin/env python3
"""Инлаеры и согласие цвета в совпавших точках для 20 кандидатов каждого кадра.

Кроп строится так же, как в сервисе; кандидаты берутся из сохранённых
прогонов eval_field.py. Результат читает experiment_rerank_rules.py.

    python pipeline/experiment_color.py data/validation/tune.json \
        data/validation/test.json data/validation/color_stats.json
"""
import sys, json, csv, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core
from embed import embed_images
from eval_field import locate
from imageprep import normalize_batch
from rerank import PrecomputedReranker

vectors = np.load('data/index/clean_mv.npy')
rr = PrecomputedReranker('data/index/sift', with_color=True)
out = []; t0 = time.time()
for path in sys.argv[1:-1]:
    split = 'tune' if 'tune' in Path(path).name else 'test'
    rows = [r for r in json.load(open(path))['results'] if r.get('candidates')]
    for n, row in enumerate(rows, 1):
        image = Image.open(locate(row['image'])).convert('RGB')
        views = [normalize_batch([image], steps=s)[0] for s in [(), ('detect',)]]
        sim = vectors @ embed_images(views).T
        shot = views[core.pick_view(views, sim)]
        slugs = [c['slug'] for c in row['candidates']]
        st = rr.stats(shot, slugs)
        out.append({'image': row['image'], 'split': split,
                    'geom': {s: [int(st[s].inliers), float(st[s].coverage), float(st[s].color)] for s in slugs},
                    'crop_matches': {c['slug']: c['inliers'] for c in row['candidates']} == {s: int(st[s].inliers) for s in slugs}})
        if n % 50 == 0: print(split, n, len(rows), f'{time.time()-t0:.0f}s', flush=True)
json.dump(out, open(sys.argv[-1], 'w'))
print('done', len(out), 'crop match', sum(r['crop_matches'] for r in out))
