"""Post-analysis of the paired text replay: per-SKU distribution, timing, checks.

Reads comparison.json (from pipeline/eval_paddle_groups.py), observations.jsonl,
views.json, run.log and the frozen baseline. Prints a JSON summary and writes
analysis.json next to comparison.json.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

V = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/validation/ocr_grouped_max2000_tune_20260923')
cmp_ = json.loads((V / 'comparison.json').read_text())
obs = [json.loads(l) for l in (V / 'observations.jsonl').read_text().splitlines() if l.strip()]
views = json.loads((V / 'views.json').read_text())
base = json.loads(Path('data/validation/final3_tune.json').read_text())['results']
wines = {w['slug']: w for w in json.loads(Path('data/catalog/catalog.json').read_text())['wines']}
title = lambda s: wines.get(s, {}).get('title', s or '—')
log = (V / 'run.log').read_text()

checks = {
    'observations': len(obs), 'unique_images': len({o['image'] for o in obs}),
    'views_match': [o['image'] for o in obs] == [v['image'] for v in views],
    'sha_match': all(o['source_sha256'] == __import__('hashlib').sha256(Path(v['source']).read_bytes()).hexdigest()
                     for o, v in zip(obs, views)),
    'complete_json': (V / 'observations.complete.json').exists(),
    'log_tracebacks': log.count('Traceback'), 'log_memory_errors': log.count('MemoryError'),
    'log_done_lines': len(re.findall(r'^\d+/187 .* [0-9.]+s$', log, re.M)),
    'launches': log.count('=== launch'),
}
by_image = {r['image']: r for r in cmp_['results']}
base_by = {r['image']: r for r in base}
pos = [r for r in cmp_['results'] if r['gold']]
def block(rows):
    n = len(rows)
    return {stage: {f'top{k}': f"{sum(r['gold'] in r[stage][:k] for r in rows)}/{n} = {sum(r['gold'] in r[stage][:k] for r in rows)/n:.1%}"
                    for k in (1, 3, 5)} for stage in ('before', 'after')}
metrics = {'all_positives': block(pos), 'findable': block([r for r in pos if r['findable']])}

def describe(img):
    r = by_image[img]; return {'image': img, 'gold': title(r['gold']), 'before': title(r['before'][0]),
                               'after': title(r['after'][0]), 'admitted': r['admitted'],
                               'admitted_words': [w['text'] for w in next(o for o in obs if o['image'] == img)['words'] if w['conf'] >= .85]}
fixed = [describe(i) for i in cmp_['summary']['fixed']]
broken = [describe(i) for i in cmp_['summary']['broken']]
wrong_to_wrong = [describe(r['image']) for r in pos if r['before'][0] != r['after'][0]
                  and r['before'][0] != r['gold'] and r['after'][0] != r['gold']]
neg_changes = [describe(i) for i in cmp_['summary']['negative_leader_changes']]
per_sku = defaultdict(lambda: Counter())
for r in pos:
    per_sku[r['gold']]['frames'] += 1
    per_sku[r['gold']]['correct_before'] += r['before'][0] == r['gold']
    per_sku[r['gold']]['correct_after'] += r['after'][0] == r['gold']
    per_sku[r['gold']]['leader_changed'] += r['before'][0] != r['after'][0]
sku_changes = {title(s): dict(c) for s, c in per_sku.items() if c['leader_changed']}
series = lambda img: re.sub(r'_\d+\.jpg$', '', img)
series_changes = Counter(series(i) for i in cmp_['summary']['fixed'] + cmp_['summary']['broken'])

secs = np.array([o['seconds'] for o in obs]); big = np.array([max(v['size']) > 2000 for v in views])
timing = {'median_s': round(float(np.median(secs)), 2), 'p95_s': round(float(np.percentile(secs, 95)), 2),
          'max_s': round(float(secs.max()), 2), 'total_min': round(float(secs.sum() / 60), 1),
          'median_s_frames_over_2000px': round(float(np.median(secs[big])), 2) if big.any() else None,
          'median_s_frames_upto_2000px': round(float(np.median(secs[~big])), 2) if (~big).any() else None,
          'n_over_2000px': int(big.sum())}
admitted = [r['admitted'] for r in cmp_['results']]
words = {'frames_with_admitted': sum(a > 0 for a in admitted), 'admitted_total': sum(admitted),
         'grouped_total': sum(len(o['words']) for o in obs),
         'frames_with_any_group': sum(bool(o['words']) for o in obs),
         'boxes_median': float(np.median([len(o['boxes']) for o in obs]))}
out = {'checks': checks, 'metrics': metrics, 'fixed': fixed, 'broken': broken, 'wrong_to_wrong': wrong_to_wrong,
       'negative_leader_changes': neg_changes, 'sku_with_leader_changes': sku_changes,
       'series_with_fixed_or_broken': dict(series_changes), 'timing': timing, 'words': words}
(V / 'analysis.json').write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print(json.dumps(out, ensure_ascii=False, indent=1))
