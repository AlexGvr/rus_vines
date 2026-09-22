"""Paired OCR view experiment; freeze retrieval and geometry from baseline."""
import argparse
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

import searchcore as core
from embed import embed_images
from imageprep import normalize_batch, find_label_band
from ocr import read_words
from text_match import TextChannel

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/validation/ranker_features_tune.json')
    ap.add_argument('--output', default='data/validation/ocr_views_tune.json')
    ap.add_argument('--variants', nargs='+', choices=['selected', 'bottle', 'label_band'],
                    default=['selected', 'bottle', 'label_band'])
    args = ap.parse_args()
    source = json.loads(Path(args.input).read_text())
    manifest = ROOT / 'data/field/manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != source['provenance']['sha256']['data/field/manifest.csv']:
        raise ValueError('manifest changed since baseline')
    wines = json.loads((ROOT/'data/catalog/catalog.json').read_text())['wines']
    by_slug = {w['slug']: w for w in wines}
    channel = TextChannel(wines, verified_attributes=None, correct_sweetness=False)
    vectors = np.load(ROOT/'data/index/clean_mv.npy')
    results = []
    for n, r in enumerate(source['results'], 1):
        path = next(p for base in ('data/field', 'dataset/eval/queries')
                    if (p := ROOT/base/r['image']).exists())
        image = Image.open(path).convert('RGB')
        bottle = normalize_batch([image], steps=('detect',))[0]
        views = [image, bottle]
        similarity = vectors @ embed_images(views).T
        selected_index = core.pick_view(views, similarity)
        selected = views[selected_index]
        crops = {'selected': selected, 'bottle': bottle,
                 'label_band': find_label_band(bottle)}
        crops = {k:v for k,v in crops.items() if k == 'selected' or k in args.variants}
        candidates = [core.Candidate(c['slug'], c['cv'], c['inliers'], c['coverage'])
                      for c in r['candidates']]
        stats = {c.slug: SimpleNamespace(inliers=c.inliers, coverage=c.coverage, color=1.)
                 for c in candidates}
        core.rerank(selected, candidates, SimpleNamespace(stats=lambda *_: stats))
        record = dict(image=r['image'], gold=r['gold'], findable=r['findable'],
                      selected_view=selected_index, variants={})
        for name, crop in crops.items():
            if name == 'bottle' and selected_index == 1:
                record['variants'][name] = copy.deepcopy(record['variants']['selected'])
                continue
            start = time.perf_counter()
            words = read_words(crop)
            elapsed = time.perf_counter() - start
            order = copy.deepcopy(candidates)
            core.settle(order, crop, channel, lambda *_: words, .8, .6, by_slug)
            record['variants'][name] = dict(
                order=[c.slug for c in order], ocr_seconds=elapsed,
                words=[dict(text=w.text, confidence=w.conf, box=w.box) for w in words],
                size=crop.size)
        if record['variants']['selected']['order'][0] != r['top1']:
            raise ValueError(f"baseline not reproduced: {r['image']}")
        results.append(record)
        print(f"{n}/{len(source['results'])} {r['image']}", flush=True)
    summary = {}
    for name in crops:
        summary[name] = {}
        for group, rows in [('all_positives', [r for r in results if r['gold']]),
                            ('findable', [r for r in results if r['findable']])]:
            summary[name][group] = dict(n=len(rows), **{
                f'top{k}': sum(r['gold'] in r['variants'][name]['order'][:k] for r in rows)
                for k in (1,3,5)})
        summary[name]['ocr_median_seconds'] = float(np.median([
            r['variants'][name]['ocr_seconds'] for r in results]))
    out = dict(summary=summary, results=results,
               input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
               note='Retrieval and geometry frozen; OCR latency only, not end-to-end SLA.')
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
