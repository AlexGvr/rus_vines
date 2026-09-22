"""Conservative same-producer tie-break experiment over cached OCR.

Fixed rule: top five, exact normalized tokens, OCR >= .85; no fuzzy match.
Conflicting unique words or structured attribute conflicts mean abstention.
"""
import argparse
import json
from pathlib import Path

from ocr import Word
from text_match import (TextChannel, card_features, central_words, clean, best_match,
                        GENERIC_WORDS, CATEGORY_WORDS, STOPWORDS)

ROOT = Path(__file__).resolve().parent.parent


def refine(order, words, channel):
    if not order:
        return order, []
    brand = channel.by_slug[order[0]].get('manufacturer')
    if not brand:
        return order, []
    group = [s for s in order[:5] if channel.by_slug[s].get('manufacturer') == brand]
    if len(group) < 2:
        return order, []
    stop = GENERIC_WORDS | CATEGORY_WORDS | STOPWORDS
    marks = {}
    for s in group:
        f = card_features(s, channel)
        marks[s] = (f.name | f.grapes) - stop
    evidence = []
    owners_seen = set()
    for w in central_words(words, .85):
        for token in clean(w.text).split():
            if len(token) < 4 or not token.isalpha() or token in stop:
                continue
            owners = {s for s in group if best_match(token, marks[s], 1.0)}
            if len(owners) == 1:
                owner = next(iter(owners))
                evidence.append(dict(token=token, owner=owner, confidence=w.conf))
                owners_seen.add(owner)
    if len(owners_seen) != 1:
        return order, evidence
    owner = next(iter(owners_seen))
    if channel.hard_conflicts(owner, words):
        return order, evidence
    return [owner] + [s for s in order if s != owner], evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/validation/ocr_views_tune.json')
    ap.add_argument('--output', default='data/validation/exact_text_tune.json')
    args = ap.parse_args()
    source = json.loads(Path(args.input).read_text())
    wines = json.loads((ROOT/'data/catalog/catalog.json').read_text())['wines']
    channel = TextChannel(wines)
    results = []
    for r in source['results']:
        variants = {}
        for key, v in r['variants'].items():
            words = [Word(w['text'], w['confidence'], tuple(w['box'])) for w in v['words']]
            order, evidence = refine(v['order'], words, channel)
            variants[key] = dict(order=order, evidence=evidence)
        results.append(dict(image=r['image'], gold=r['gold'], findable=r['findable'],
                            baseline=r['variants']['selected']['order'], variants=variants))
    summary = {}
    for name in ('selected', 'bottle', 'label_band'):
        pos = [r for r in results if r['gold']]
        summary[name] = dict(n=len(pos), **{f'top{k}':sum(r['gold'] in r['variants'][name]['order'][:k]
                                                      for r in pos) for k in (1,3,5)},
                            fixed=[r['image'] for r in pos if r['baseline'][0]!=r['gold']==r['variants'][name]['order'][0]],
                            broken=[r['image'] for r in pos if r['baseline'][0]==r['gold']!=r['variants'][name]['order'][0]])
    Path(args.output).write_text(json.dumps(dict(summary=summary, results=results),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
