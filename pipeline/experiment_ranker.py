"""Fixed pairwise logistic ranker; grouped out-of-fold evaluation on tune only.

No candidate identifier or ground-truth-derived attribute enters the features.
This experiment never updates the service or its confidence calibration.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from verify_field import assign_splits

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ('cv', 'log_inliers', 'coverage', 'relative_inliers',
            'inverse_cv_rank', 'hard_conflicts', 'text_conflicts', 'text_support')


def features(candidates):
    order = sorted(range(len(candidates)), key=lambda i: -candidates[i]['cv'])
    ranks = {i: rank + 1 for rank, i in enumerate(order)}
    best = max((c['inliers'] for c in candidates), default=0)
    return np.array([[c['cv'], np.log1p(c['inliers']), c['coverage'],
                      c['inliers'] / max(best, 1), 1 / ranks[i],
                      c['hard_conflicts'], c['text_conflicts'], c['text_support']]
                     for i, c in enumerate(candidates)], dtype=float)


def fold_assignments(rows, n_folds=5):
    _, find = assign_splits(rows)
    groups = {r['image']: find('серия:' + r['series']) for r in rows}
    ordered = sorted(set(groups.values()), key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    parts = {g: i % n_folds for i, g in enumerate(ordered)}
    return {name: parts[g] for name, g in groups.items()}


def fit_pairwise(records, regularization=.01):
    differences = []
    for r in records:
        if not r['findable']:
            continue
        candidates = r['candidates']
        ids = [c['slug'] for c in candidates]
        if r['gold'] not in ids:
            continue
        x = features(candidates)
        gold = ids.index(r['gold'])
        differences.extend(x[gold] - np.delete(x, gold, axis=0))
    if not differences:
        raise ValueError('no positive/negative candidate pairs in training fold')
    d = np.array(differences)
    scale = np.maximum(np.std(d, axis=0), 1e-6)
    d = d / scale

    def objective(w):
        margins = d @ w
        loss = np.logaddexp(0, -margins).mean() + regularization * (w @ w) / 2
        grad = -(d.T @ expit(-margins)) / len(d) + regularization * w
        return loss, grad

    fit = minimize(objective, np.zeros(len(FEATURES)), jac=True, method='L-BFGS-B')
    if not fit.success:
        raise ValueError(f'optimizer failed: {fit.message}')
    return fit.x / scale


def summarize_predictions(predictions):
    summary = {}
    for name, data in [('all_positives', [r for r in predictions if r['gold']]),
                       ('findable', [r for r in predictions if r['findable']])]:
        summary[name] = {'n': len(data)}
        for k in (1, 3, 5):
            summary[name][f'baseline_top{k}'] = sum(r['gold'] in r['baseline_order'][:k] for r in data)
            summary[name][f'ranker_top{k}'] = sum(r['gold'] in r['order'][:k] for r in data)
        summary[name]['fixed'] = [r['image'] for r in data if r['baseline'] != r['gold'] == r['prediction']]
        summary[name]['broken'] = [r['image'] for r in data if r['baseline'] == r['gold'] != r['prediction']]
    return summary


def predict(records, weights, fold=None, preserve_hard_veto=False):
    predictions = []
    for r in records:
        candidates = r['candidates']
        scores = features(candidates) @ weights
        indices = list(np.argsort(-scores, kind='stable'))
        if preserve_hard_veto:
            # Same depth and move as searchcore.demote_contradicted; cached OCR
            # conflicts are candidate-specific and independent of list order.
            for _ in range(4):
                if not candidates[indices[0]]['hard_conflicts']:
                    break
                nxt = next((i for i in range(1, len(indices))
                            if not candidates[indices[i]]['hard_conflicts']), None)
                if nxt is None:
                    break
                indices.insert(nxt, indices.pop(0))
        order = [candidates[i]['slug'] for i in indices]
        predictions.append(dict(image=r['image'], gold=r['gold'], fold=fold,
                                findable=r['findable'], baseline=r['top1'],
                                prediction=order[0], order=order[:5],
                                baseline_order=[c['slug'] for c in candidates[:5]]))
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/validation/ranker_features_tune.json')
    ap.add_argument('--manifest', default='data/field/manifest.csv')
    ap.add_argument('--output', default='data/validation/ranker_oof.json')
    ap.add_argument('--validation-input', help='optional fixed regression features; never used to fit')
    ap.add_argument('--model-output', default='data/validation/ranker_candidate.json')
    args = ap.parse_args()
    artifact = json.loads(Path(args.input).read_text())
    if artifact['provenance']['args']['split'] != 'tune':
        raise ValueError('only tune may be used for this experiment')
    manifest = Path(args.manifest)
    recorded = artifact['provenance']['sha256'].get('data/field/manifest.csv')
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != recorded:
        raise ValueError('manifest differs from feature collection')
    rows = [r for r in csv.DictReader(manifest.open()) if r['split'] == 'tune']
    records = artifact['results']
    if {r['image'] for r in records} != {r['image'] for r in rows}:
        raise ValueError('feature collection is incomplete')
    labels = {r['image']: r['slug'] for r in csv.DictReader(manifest.open())}
    if any(r['gold'] != labels[r['image']] for r in records):
        raise ValueError('training labels differ from the frozen manifest')
    folds = fold_assignments(rows)
    predictions, guarded_predictions, weights = [], [], []
    for fold in range(5):
        train = [r for r in records if folds[r['image']] != fold]
        test = [r for r in records if folds[r['image']] == fold]
        w = fit_pairwise(train)
        weights.append(w.tolist())
        predictions.extend(predict(test, w, fold))
        guarded_predictions.extend(predict(test, w, fold, preserve_hard_veto=True))
    summary = summarize_predictions(predictions)
    out = dict(summary=summary, features=FEATURES, weights_by_fold=weights,
               with_existing_hard_veto=dict(summary=summarize_predictions(guarded_predictions),
                                            predictions=guarded_predictions),
               predictions=predictions, folds=folds, regularization=.01,
               role='grouped development OOF; not independent acceptance test',
               input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest())
    final_weights = fit_pairwise(records)
    Path(args.model_output).write_text(json.dumps(dict(
        features=FEATURES, weights=final_weights.tolist(),
        training_sha256=out['input_sha256'], status='experimental_not_deployed'), indent=2)+'\n')
    if args.validation_input:
        validation = json.loads(Path(args.validation_input).read_text())
        expected = {r['image'] for r in csv.DictReader(manifest.open()) if r['split'] == 'test'}
        if {r['image'] for r in validation['results']} != expected:
            raise ValueError('regression input is incomplete or not the frozen test partition')
        if any(r['gold'] != labels[r['image']] for r in validation['results']):
            raise ValueError('regression labels differ from the frozen manifest')
        for key in ('topk', 'views', 'index', 'weights'):
            if validation['provenance']['args'][key] != artifact['provenance']['args'][key]:
                raise ValueError(f'train and regression argument {key} differs')
        # Only split/output paths may differ; model inputs and feature code must match.
        for key in ('sha256', 'environment'):
            if validation['provenance'][key] != artifact['provenance'][key]:
                raise ValueError(f'train and regression {key} differ')
        regression = predict(validation['results'], final_weights)
        guarded = predict(validation['results'], final_weights, preserve_hard_veto=True)
        out['regression'] = dict(summary=summarize_predictions(regression), predictions=regression,
                                with_existing_hard_veto=dict(summary=summarize_predictions(guarded),
                                                             predictions=guarded))
        print('Regression:', json.dumps(out['regression']['summary'], ensure_ascii=False))
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
