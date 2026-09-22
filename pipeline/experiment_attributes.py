"""Replay frozen visual candidates and OCR with verified catalog attributes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import searchcore as core
from ocr import Word
from text_match import TextChannel, VERIFIED_ATTRIBUTES

ROOT = Path(__file__).resolve().parent.parent


def replay(source, cached):
    wines = json.loads((ROOT/'data/catalog/catalog.json').read_text())['wines']
    by_slug = {w['slug']:w for w in wines}
    old_channel = TextChannel(wines, verified_attributes=None, correct_sweetness=False)
    new_channel = TextChannel(wines)
    ocr = {r['image']:r for r in cached['results']}
    if set(ocr) != {r['image'] for r in source['results']}:
        raise ValueError('OCR and visual datasets differ')
    results = []
    weights = json.loads((ROOT/'data/index/confidence.json').read_text())['outcome_weights']
    for r in source['results']:
        v = ocr[r['image']]['variants']['selected']
        words = [Word(w['text'],w['confidence'],tuple(w['box'])) for w in v['words']]
        candidates = [core.Candidate(c['slug'],c['cv'],c['inliers'],c['coverage'])
                      for c in r['candidates']]
        stats = {c.slug: SimpleNamespace(inliers=c.inliers,coverage=c.coverage,color=1.)
                 for c in candidates}
        core.rerank(None,candidates,SimpleNamespace(stats=lambda *_:stats))
        orders = {}
        for key,ch in [('baseline',old_channel),('verified',new_channel)]:
            order = copy.deepcopy(candidates)
            reader = lambda *_:words
            core.settle(order,None,ch,reader,.8,.6,by_slug)
            core.check_leader(order,None,ch,reader,.6,.8,by_slug)
            orders[key] = dict(order=[c.slug for c in order],
                               probability=core.outcomes(core.features(order),weights)[0])
        if orders['baseline']['order'][0] != r['top1']:
            raise ValueError('baseline replay differs: '+r['image'])
        results.append(dict(image=r['image'],gold=r['gold'],findable=r['findable'],**orders))
    pos = [r for r in results if r['gold']]
    summary = dict(n=len(pos), baseline_top1=sum(r['baseline']['order'][0]==r['gold'] for r in pos),
                   verified_top1=sum(r['verified']['order'][0]==r['gold'] for r in pos),
                   fixed=[r['image'] for r in pos if r['baseline']['order'][0]!=r['gold']==r['verified']['order'][0]],
                   broken=[r['image'] for r in pos if r['baseline']['order'][0]==r['gold']!=r['verified']['order'][0]])
    return dict(summary=summary,results=results,
                attributes_sha256=hashlib.sha256(VERIFIED_ATTRIBUTES.read_bytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default='data/validation/ranker_features_tune.json')
    ap.add_argument('--ocr',default='data/validation/ocr_views_tune.json')
    ap.add_argument('--output',default='data/validation/verified_attributes_tune.json')
    a=ap.parse_args()
    out=replay(json.loads(Path(a.features).read_text()),json.loads(Path(a.ocr).read_text()))
    Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out['summary'],ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
