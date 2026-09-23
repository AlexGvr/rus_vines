"""Offline OCR experiment: overlapping strips and small deskew angles.

No production hooks. Retrieval/geometry and original OCR are frozen from
final3; extra words map back to the selected view before text ranking.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np
from PIL import Image

import searchcore as core
from ocr import Word, load
from text_match import TextChannel

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {'ir_ochen-vkusnoe-legkoe-vino-dlya-semeinogo_0.jpg',
           '13978352_0.jpg', '15598335_0.jpg', '16151511_0.jpg'}


def transformed_strip(image, start, end, angle):
    width, height = image.size
    y0, y1 = int(start * height), int(end * height)
    patch = np.asarray(image.convert('RGB'))[y0:y1]
    scale = min(4., 960 / width, 1600 / (y1-y0))
    pw, ph = max(1, round(width*scale)), max(1, round((y1-y0)*scale))
    patch = cv2.resize(patch, (pw, ph), interpolation=cv2.INTER_CUBIC)
    matrix = cv2.getRotationMatrix2D((pw/2, ph/2), angle, 1.)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(np.ceil(ph*sine+pw*cosine)), int(np.ceil(ph*cosine+pw*sine))
    matrix[:, 2] += [(nw-pw)/2, (nh-ph)/2]
    rotated = cv2.warpAffine(patch, matrix, (nw, nh), borderValue=(255,255,255))
    return rotated, cv2.invertAffineTransform(matrix), (pw/width, ph/(y1-y0)), y0


def original_box(polygon, inverse, scale, y0, size):
    points = np.asarray(polygon, dtype=float)
    points = points @ inverse[:, :2].T + inverse[:, 2]
    points /= np.asarray(scale)
    points[:, 1] += y0
    points = np.clip(points, (0, 0), size)
    lo, hi = np.floor(points.min(axis=0)), np.ceil(points.max(axis=0))
    return tuple(int(x) for x in (*lo, *hi))


def additional_words(image, reader):
    words, scans = [], []
    for start, end in ((0., .6), (.4, 1.)):
        for angle in (-12, 0, 12):
            arr, inv, scale, y0 = transformed_strip(image, start, end, angle)
            tick = time.perf_counter()
            # Small text detector threshold lowered; recognition admission stays strict.
            blocks = reader.readtext(arr, detail=1, paragraph=False, min_size=8,
                                     text_threshold=.5, low_text=.3)
            scan = {'strip': [start, end], 'angle': angle,
                    'seconds': time.perf_counter()-tick, 'words': []}
            for polygon, text, conf in blocks:
                if float(conf) < .30 or not text.strip():
                    continue
                word = Word(text.strip(), float(conf), original_box(polygon,inv,scale,y0,image.size))
                scan['words'].append(asdict(word))
                if word.conf >= .85:
                    words.append(word)
            scans.append(scan)
    return words, scans


def merge(original, extra):
    # Keep original observations; deduplicate exact text at the same location.
    merged = list(original)
    for word in extra:
        key = word.text.casefold().strip()
        duplicate = next((i for i, old in enumerate(merged)
                          if old.text.casefold().strip() == key
                          and abs(old.center_x-word.center_x) < max(12,word.box[2]-word.box[0])
                          and abs((old.box[1]+old.box[3]-word.box[1]-word.box[3])/2)
                          < max(12,word.box[3]-word.box[1])), None)
        if duplicate is None:
            merged.append(word)
        elif merged[duplicate].conf < word.conf:
            merged[duplicate] = word
    return merged


def rank(row, image, words, channel, by_slug):
    cs = [core.Candidate(x['slug'], x['cv'], x['inliers'], x['coverage'])
          for x in row['candidates']]
    cs.sort(key=lambda c: -c.cv)
    geo = sorted(cs,key=lambda c: (-c.inliers,-c.cv))
    decisive = len(geo)<2 or geo[0].inliers >= geo[1].inliers*1.5
    cs.sort(key=(lambda c: (-c.inliers,-c.cv)) if decisive else (lambda c:-c.cv))
    core.settle(cs, image, channel, lambda *_: words, .8, .6, by_slug)
    return [c.slug for c in cs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['pilot','tune'], default='pilot')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    baseline_path = ROOT/'data/validation/final3_tune.json'
    baseline = json.loads(baseline_path.read_text())
    for name in ('data/field/manifest.csv','data/index/clean_mv.npy','data/catalog/catalog.json'):
        with (ROOT/name).open('rb') as fh:
            if hashlib.file_digest(fh,'sha256').hexdigest() != baseline['provenance']['sha256'][name]:
                raise RuntimeError(f'Baseline input changed: {name}')
    rows = baseline['results']
    if args.scope == 'pilot':
        rows = [r for r in rows if r['image'] in TARGETS]
    wines = json.loads((ROOT/'data/catalog/catalog.json').read_text())['wines']
    by_slug = {w['slug']:w for w in wines}
    channel = TextChannel(wines)
    cache = ROOT/'data/validation/transcript_final3_assets'
    view_meta = json.loads((cache/'views.json').read_text())
    reader = load()
    results=[]
    out=Path(args.output)
    for i,row in enumerate(rows):
        if row['image'] in view_meta:
            image=Image.open(cache/row['image']).convert('RGB')
        else:
            from eval_field import locate
            from imageprep import normalize_batch
            from embed import embed_images
            raw=Image.open(locate(row['image'])).convert('RGB')
            views=[raw,normalize_batch([raw],steps=('detect',))[0]]
            vectors=np.load(ROOT/'data/index/clean_mv.npy',mmap_mode='r')
            image=views[core.pick_view(views,vectors@embed_images(views).T)]
        words=[Word(w['text'],w['conf'],tuple(w['box'])) for w in row['words']]
        before=rank(row,image,words,channel,by_slug)
        if before[0]!=row['top1']:
            raise RuntimeError(f'Baseline replay mismatch: {row["image"]}')
        tick=time.perf_counter()
        extra,scans=additional_words(image,reader)
        elapsed=time.perf_counter()-tick
        merged=merge(words,extra)
        after=rank(row,image,merged,channel,by_slug)
        result=dict(image=row['image'],gold=row['gold'],findable=row['findable'],
                    before=before,after=after,extra_seconds=elapsed,scans=scans,
                    admitted_words=[asdict(w) for w in extra])
        results.append(result)
        print(f'{i+1}/{len(rows)} {row["image"]}: {before[0]==row["gold"]} -> {after[0]==row["gold"]} {elapsed:.2f}s',flush=True)
        # Crash leaves an explicitly partial artifact, never a completed report.
        out.write_text(json.dumps({'complete':False,'results':results},ensure_ascii=False,indent=1))
    positive=[r for r in results if r['gold']]
    summary={stage:{f'top{k}':sum(r['gold'] in r[stage][:k] for r in positive)
                    for k in (1,3,5)} for stage in ('before','after')}
    summary.update(n=len(results),positives=len(positive),
                   fixed=[r['image'] for r in positive if r['before'][0]!=r['gold'] and r['after'][0]==r['gold']],
                   broken=[r['image'] for r in positive if r['before'][0]==r['gold'] and r['after'][0]!=r['gold']],
                   extra_seconds_median=float(np.median([r['extra_seconds'] for r in results])),
                   extra_seconds_p95=float(np.percentile([r['extra_seconds'] for r in results],95)))
    out.write_text(json.dumps({'complete':True,'summary':summary,'results':results,
                              'scope':args.scope,'baseline_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              'note':'Frozen visual candidates/geometry. Added OCR latency only; no recalibration or HTTP SLA measurement.'},ensure_ascii=False,indent=1))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
