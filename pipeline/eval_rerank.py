#!/usr/bin/env python3
"""S1.5 + S2.5 — замер геометрической проверки и калибровка порога отсечки.

Схема поиска: SigLIP отбирает десятку кандидатов, локальные признаки решают,
какой из них та же самая этикетка. Число инлаеров — величина абсолютная,
в отличие от косинуса, поэтому по нему же калибруется ответ «такого вина
в каталоге нет»: мало совпавших точек значит, что похожего просто не нашлось.

Оговорка к цифрам: запросы синтетические и получены из эталонных фото, так
что локальные признаки здесь в более выгодном положении, чем на реальной
пересъёмке. Направление вывода это не меняет, абсолютные значения — да.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import Reranker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
STEPS = ("detect",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_detect")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    paths, gold = [], {}
    for row in manifest:
        path = QUERIES / args.subset / row["image"]
        if path.exists():
            paths.append(str(path))
            gold[str(path)] = row["slug"]

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    qvec, kept = embed_paths(paths, batch_size=32, progress_every=0, steps=STEPS)
    truth = [gold[p] for p in kept]
    scores = qvec @ vectors.T
    order = np.argsort(-scores, axis=1)[:, :200]

    candidates = []
    for i in range(len(kept)):
        seen, picked = set(), []
        for j in order[i]:
            slug = index_slugs[j]
            if slug is None or slug in seen:
                continue
            seen.add(slug)
            picked.append((slug, float(scores[i, j])))
            if len(picked) == args.topk:
                break
        candidates.append(picked)

    reranker = Reranker()
    cv_top1 = cv_topk = rr_top1 = rr_top5 = 0
    inliers_right, inliers_wrong, latencies = [], [], []
    records = []          # (инлаеры лидера, отрыв, доминирование, верен ли лидер)

    for i, path in enumerate(kept):
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=STEPS)[0]
        t0 = time.perf_counter()
        pairs = [(slug, wines[slug]["photo"]) for slug, _ in candidates[i]
                 if wines.get(slug, {}).get("photo")]
        inliers = reranker.scores(crop, pairs)
        latencies.append((time.perf_counter() - t0) * 1000)

        if candidates[i] and candidates[i][0][0] == truth[i]:
            cv_top1 += 1
        if truth[i] in [slug for slug, _ in candidates[i]]:
            cv_topk += 1
        if not inliers:
            continue
        ranked = sorted(inliers, key=lambda s: -inliers[s])
        best = inliers[ranked[0]]
        second = inliers[ranked[1]] if len(ranked) > 1 else 0
        records.append((best, best - second,
                        best / (best + second) if best + second else 0.0,
                        ranked[0] == truth[i]))
        if ranked[0] == truth[i]:
            rr_top1 += 1
            inliers_right.append(inliers[ranked[0]])
        else:
            inliers_wrong.append(inliers[ranked[0]])
        if truth[i] in ranked[:5]:
            rr_top5 += 1

    n = max(len(kept), 1)
    print(f"\nнабор {args.subset}: запросов {n}, топ-{args.topk} кандидатов на реранк")
    print(f"  CV top-1            {cv_top1 / n * 100:5.1f}%")
    print(f"  CV top-{args.topk:<2}           {cv_topk / n * 100:5.1f}%   (потолок реранка)")
    print(f"  после реранка top-1 {rr_top1 / n * 100:5.1f}%")
    print(f"  после реранка top-5 {rr_top5 / n * 100:5.1f}%")
    print(f"  реранк: {np.mean(latencies):.0f} мс в среднем, "
          f"{np.percentile(latencies, 95):.0f} мс на 95-м процентиле")

    if inliers_right and inliers_wrong:
        print(f"\nинлаеры лучшего кандидата: верный ответ "
              f"медиана {np.median(inliers_right):.0f}, 10-й процентиль "
              f"{np.percentile(inliers_right, 10):.0f}; "
              f"неверный медиана {np.median(inliers_wrong):.0f}, "
              f"90-й процентиль {np.percentile(inliers_wrong, 90):.0f}")
        print(f"{'порог':>7} {'верных отклонено':>18} {'неверных отсечено':>19}")
        for threshold in (5, 10, 15, 20, 30, 40):
            lost = sum(1 for v in inliers_right if v < threshold) / len(inliers_right)
            cut = sum(1 for v in inliers_wrong if v < threshold) / len(inliers_wrong)
            print(f"{threshold:>7} {lost * 100:>17.1f}% {cut * 100:>18.1f}%")

    if records:
        print(f"\nправило «вино найдено»: инлаеров не меньше N и доминирование не ниже D")
        print(f"{'N':>4} {'D':>6} {'верных принято':>16} {'неверных принято':>18}")
        right = [r for r in records if r[3]]
        wrong = [r for r in records if not r[3]]
        for n_min in (10, 15, 25):
            for dom in (0.0, 0.6, 0.65, 0.7):
                keep_r = sum(1 for b, _, d, _ in right if b >= n_min and d >= dom)
                keep_w = sum(1 for b, _, d, _ in wrong if b >= n_min and d >= dom)
                print(f"{n_min:>4} {dom:>6.2f} {keep_r / len(right) * 100:>15.1f}% "
                      f"{keep_w / max(len(wrong), 1) * 100:>17.1f}%")

    out = INDEX_DIR / f"rerank_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "n": n, "topk": args.topk,
        "cv_top1": cv_top1 / n, "cv_topk": cv_topk / n,
        "rerank_top1": rr_top1 / n, "rerank_top5": rr_top5 / n,
        "rerank_ms_mean": float(np.mean(latencies)),
        "rerank_ms_p95": float(np.percentile(latencies, 95)),
        "inliers_right_median": float(np.median(inliers_right)) if inliers_right else 0,
        "inliers_wrong_median": float(np.median(inliers_wrong)) if inliers_wrong else 0,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
