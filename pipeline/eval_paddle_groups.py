"""Paired ranking replay with frozen candidates and extra grouped OCR words."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiment_ocr_strips import rank, merge
from ocr import Word
from text_match import TextChannel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',default='data/validation/final3_tune.json')
    ap.add_argument('--observations',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    observations=Path(args.observations)
    completion=json.loads(observations.with_suffix('.complete.json').read_text())
    rows=json.loads(Path(args.baseline).read_text())['results']
    added=[json.loads(line) for line in observations.read_text().splitlines()]
    extra={r['image']:r for r in added}
    if len(added)!=completion['n'] or len(extra)!=len(added) or set(extra)!={r['image'] for r in rows}:
        raise ValueError('Incomplete, duplicated, or different sample')
    wines=json.loads(Path('data/catalog/catalog.json').read_text())['wines']
    by_slug={r['slug']:r for r in wines}
    channel=TextChannel(wines)
    results=[]
    for row in rows:
        original=[Word(w['text'],w['conf'],tuple(w['box'])) for w in row['words']]
        # Replay text-only decisions. Reader closure supplies coordinates in the
        # original selected view; no image operation occurs inside this path.
        placeholder=Image.new('RGB',(1,1))
        before=rank(row,placeholder,original,channel,by_slug)
        if before[0]!=row['top1']:
            raise ValueError(f'Baseline replay mismatch: {row["image"]}')
        admitted=[Word(w['text'],w['conf'],tuple(w['box']))
                  for w in extra[row['image']]['words'] if w['conf']>=.85 and w['text'].strip()]
        after=rank(row,placeholder,merge(original,admitted),channel,by_slug)
        results.append(dict(image=row['image'],gold=row['gold'],findable=row['findable'],
                            before=before,after=after,admitted=len(admitted)))
    def group(selected):
        return dict(n=len(selected),**{stage:{f'top{k}':sum(r['gold'] in r[stage][:k] for r in selected)
                                            for k in (1,3,5)} for stage in ('before','after')})
    positives=[r for r in results if r['gold']]
    summary=dict(all_positives=group(positives),findable=group([r for r in positives if r['findable']]),
                 fixed=[r['image'] for r in positives if r['before'][0]!=r['gold'] and r['after'][0]==r['gold']],
                 broken=[r['image'] for r in positives if r['before'][0]==r['gold'] and r['after'][0]!=r['gold']],
                 negative_leader_changes=[r['image'] for r in results if not r['gold'] and r['before'][0]!=r['after'][0]],
                 added_ocr_seconds_median=float(np.median([r['seconds'] for r in added])),
                 added_ocr_seconds_p95=float(np.percentile([r['seconds'] for r in added],95)))
    files=[args.baseline,args.observations,'pipeline/text_match.py','pipeline/searchcore.py',__file__]
    report=dict(summary=summary,results=results,admission_confidence=.85,
                sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
                note='Frozen visual retrieval/geometry, paired text replay. Extra OCR CPU time only. Confidence/F1/SLA not re-evaluated.')
    Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
