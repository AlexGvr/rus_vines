"""Synthetic sets (sharp, neardup, hard) on the current pipeline, present queries only.

Mirrors the present-pass of pipeline/calibrate.py (same shortlist, view choice,
geometry, OCR, settle, check_leader). Geometry statistics are computed once per
query and the decisiveness rule is applied for two margins (0.25 — before
2026-09-22, 0.50 — current) so both orders share the same OCR and candidates.
Writes per-query results; touches nothing under data/index.
"""
import copy, csv, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / 'pipeline'))
import searchcore as core
from embed import embed_paths
from imageprep import normalize_batch
from ocr import read_words
import ocr as ocr_module
from rerank import PrecomputedReranker
from text_match import TextChannel

INDEX = ROOT / 'data/index'; QUERIES = ROOT / 'data/queries'; CATALOG = ROOT / 'data/catalog/catalog.json'
MARGINS = (0.25, 0.50); TOPK = 20; WINDOW = 0.80; MIN_CONF = 0.60
views = [(), ('detect',)]
vectors = np.load(INDEX / 'clean_mv.npy')
index_slugs = [r['slug'] or None for r in csv.DictReader((INDEX / 'clean_mv.csv').open(encoding='utf-8'))]
reranker = PrecomputedReranker(INDEX / 'sift')
wines = json.loads(CATALOG.read_text(encoding='utf-8'))['wines']
channel = TextChannel.from_catalog(CATALOG); by_slug = {w['slug']: w for w in wines}
out = {}
for subset in sys.argv[1:] or ('sharp', 'neardup', 'hard'):
    manifest = list(csv.DictReader((QUERIES / f'manifest_{subset}.csv').open(encoding='utf-8')))
    paths = [str(QUERIES / subset / r['image']) for r in manifest]; gold = {p: r['slug'] for p, r in zip(paths, manifest)}
    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps) for steps in views]
    kept = per_view[0][1]; per_view_scores = [qv @ vectors.T for qv, _ in per_view]
    scores = np.maximum.reduce(per_view_scores)
    rows = []; t0 = time.perf_counter()
    for i, path in enumerate(kept):
        truth = gold[path]
        with Image.open(path) as raw: source = raw.convert('RGB')
        shots = [normalize_batch([source], steps=steps)[0] for steps in views]
        base = core.shortlist(scores[i].copy(), index_slugs, TOPK)
        similarity = np.stack([block[i] for block in per_view_scores], axis=1)
        crop = shots[core.pick_view(shots, similarity)]
        stats = reranker.stats(crop, [c.slug for c in base])
        label = core.LabelText(crop, read_words)
        row = {'image': Path(path).name, 'gold': truth, 'findable': truth in index_slugs,
               'rank_cosine': next((k for k, c in enumerate(base) if c.slug == truth), -1)}
        for margin in MARGINS:
            cands = copy.deepcopy(base)
            for c in cands:
                m = stats.get(c.slug)
                if m is not None: c.inliers, c.coverage, c.color = int(m.inliers), float(m.coverage), float(m.color)
            geo = sorted(cands, key=lambda c: (-c.inliers, -c.cv))
            decisive = len(geo) < 2 or geo[0].inliers >= geo[1].inliers * (1.0 + margin)
            cands.sort(key=(lambda c: (-c.inliers, -c.cv)) if decisive else (lambda c: -c.cv))
            rank_geo = next((k for k, c in enumerate(cands) if c.slug == truth), -1)
            core.settle(cands, crop, channel, label, window=WINDOW, min_conf=MIN_CONF, by_slug=by_slug)
            core.check_leader(cands, crop, channel, label, MIN_CONF, WINDOW, by_slug)
            order = [c.slug for c in cands]
            row[f'm{margin}'] = {'decisive': decisive, 'rank_geometry': rank_geo, 'top1': order[0] if order else '',
                                 'rank_final': order.index(truth) if truth in order else -1, 'inliers_top': cands[0].inliers if cands else 0}
        rows.append(row)
        if (i + 1) % 50 == 0: print(f'  {subset} {i + 1}/{len(kept)} {time.perf_counter() - t0:.0f}s', flush=True)
    n = len(rows)
    summary = {'n': n, 'findable': sum(r['findable'] for r in rows), 'recall@20': sum(r['rank_cosine'] >= 0 for r in rows) / n,
               'cosine_top1': sum(r['rank_cosine'] == 0 for r in rows) / n}
    for margin in MARGINS:
        k = f'm{margin}'
        summary[k] = {'geometry_top1': sum(r[k]['rank_geometry'] == 0 for r in rows) / n,
                      'top1': sum(r[k]['rank_final'] == 0 for r in rows) / n, 'top1_n': sum(r[k]['rank_final'] == 0 for r in rows),
                      'top3': sum(0 <= r[k]['rank_final'] < 3 for r in rows) / n, 'top5': sum(0 <= r[k]['rank_final'] < 5 for r in rows) / n,
                      'decisive_share': sum(r[k]['decisive'] for r in rows) / n}
    summary['seconds_per_query'] = (time.perf_counter() - t0) / n; summary['ocr_failures'] = len(ocr_module.FAILURES)
    out[subset] = {'summary': summary, 'rows': rows}
    print(subset, json.dumps(summary, ensure_ascii=False), flush=True)
    Path(__file__).with_name('results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print('measure done', flush=True)
