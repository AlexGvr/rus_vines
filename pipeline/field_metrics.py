"""Metrics with explicit denominators; no exclusion of missing references."""


def summarize(results, show_p):
    positives = [r for r in results if r['gold']]
    negatives = [r for r in results if not r['gold']]
    usable = [r for r in positives if r['findable']]

    def fraction(n, d):
        return n / d if d else None

    def group(rows):
        return {
            'n': len(rows),
            'top1': fraction(sum(r['correct'] for r in rows), len(rows)),
            'top3': fraction(sum(r['in_top3'] for r in rows), len(rows)),
            'top5': fraction(sum(r['in_top5'] for r in rows), len(rows)),
            'retrieval_recall': {str(k): fraction(sum(
                0 <= r['rank_retrieval100'] < k for r in rows), len(rows))
                for k in (20, 50, 100)},
            'stage_top1_counts': {k: sum(r[k] == 0 for r in rows)
                                 for k in ('rank_shortlist', 'rank_geometry', 'rank_text')},
        }

    shown = [r for r in results if r['probability'] >= show_p]
    quality = {}
    for k, key in ((1, 'correct'), (3, 'in_top3'), (5, 'in_top5')):
        tp = sum(bool(r['gold']) and r[key] for r in shown)
        fp, fn = len(shown) - tp, len(positives) - tp
        quality[str(k)] = dict(tp=tp, fp=fp, fn=fn,
                              precision=fraction(tp, tp + fp),
                              recall=fraction(tp, tp + fn),
                              f1=fraction(2 * tp, 2 * tp + fp + fn))
    return {'all_positives': group(positives), 'findable_diagnostic': group(usable),
            'with_rejection_all_queries': quality,
            'show_p': show_p, 'absent_n': len(negatives),
            'false_accept_rate': fraction(sum(r['probability'] >= show_p for r in negatives),
                                          len(negatives))}
