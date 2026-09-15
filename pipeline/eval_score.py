#!/usr/bin/env python3
"""S5.6 — чем ранжировать кандидатов помимо абсолютного числа инлаеров.

Сейчас порядок задаёт одно число: сколько точек совпало. Оно смещено в
сторону подробных этикеток — у гравюры точек втрое больше, чем у минимализма,
и сравнивать по нему кандидатов с разной плотностью деталей нечестно.
Проверяются альтернативы, каждая дешёвая и считающаяся из того же прохода:

  inliers    — как сейчас
  precision  — доля инлаеров среди прошедших тест Лоу: устойчива к плотности
  coverage   — доля ячеек сетки эталона, где есть совпадения: сто точек на
               логотипе винодельни и сто по всей этикетке — разные события
  spread     — площадь охвата инлаеров
  color      — близость цветовых гистограмм кропа и эталона: SIFT работает
               по яркости и слеп к тому, что вино красное, а не белое

Комбинации берутся произведением: у величин разный масштаб, а произведение
не требует подбирать веса под него. Всё меряется на одном шортлисте, поэтому
сравнивается ровно ранжирование, а не отбор.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker, descriptors, match_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
HIST_BINS = (8, 8)      # тон и насыщенность; яркость выкинута как самая шумная


def color_signature(image: Image.Image) -> np.ndarray:
    """Гистограмма тона и насыщенности по центральной части кадра.

    Края кропа — это фон и соседние бутылки, они смазывают сигнатуру,
    поэтому берётся центральная половина.
    """
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    arr = arr[h // 4:h * 3 // 4, w // 4:w * 3 // 4]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def build_color_index(wines: dict) -> dict[str, np.ndarray]:
    cache = INDEX_DIR / "color.npz"
    if cache.exists():
        data = np.load(cache)
        return {slug: data[slug] for slug in data.files}
    out = {}
    for n, (slug, wine) in enumerate(wines.items(), 1):
        if not wine.get("photo"):
            continue
        try:
            with Image.open(ROOT / wine["photo"]) as im:
                out[slug] = color_signature(im)
        except Exception:
            continue
        if n % 500 == 0:
            print(f"  цвет {n}/{len(wines)}", flush=True)
    np.savez_compressed(cache, **out)
    return out


SCORES = {
    "inliers": lambda s, color: float(s.inliers),
    "precision": lambda s, color: s.precision,
    "coverage": lambda s, color: s.coverage,
    "spread": lambda s, color: s.spread,
    "color": lambda s, color: color,
    "inliers×precision": lambda s, color: s.inliers * s.precision,
    "inliers×coverage": lambda s, color: s.inliers * (0.3 + s.coverage),
    "inliers×spread": lambda s, color: s.inliers * (0.3 + s.spread),
    "inliers×color": lambda s, color: s.inliers * (0.5 + color),
    "inliers×coverage×color": lambda s, color: s.inliers * (0.3 + s.coverage) * (0.5 + color),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--views", default="raw,detect")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    views = [tuple(s for s in spec.split("+") if s and s != "raw")
             for spec in args.views.split(",")]

    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    if args.limit:
        manifest = manifest[:args.limit]
    paths, gold = [], {}
    for row in manifest:
        path = QUERIES / args.subset / row["image"]
        if path.exists():
            paths.append(str(path))
            gold[str(path)] = row["slug"]

    colors = build_color_index(wines)
    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in views]
    kept = per_view[0][1]
    truth = [gold[p] for p in kept]
    scores_matrix = np.maximum.reduce([qv @ vectors.T for qv, _ in per_view])
    order = np.argsort(-scores_matrix, axis=1)[:, :200]

    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    hits = {name: 0 for name in SCORES}
    top5 = {name: 0 for name in SCORES}
    t0 = time.time()

    for i, path in enumerate(kept):
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]
        query_kp, query_desc = descriptors(crop)
        query_color = color_signature(crop)

        shortlist, seen = [], set()
        for j in order[i]:
            slug = index_slugs[j]
            if slug is None or slug in seen:
                continue
            seen.add(slug)
            shortlist.append(slug)
            if len(shortlist) == args.topk:
                break

        rows = []
        for slug in shortlist:
            ref_kp, ref_desc = reranker.reference(slug)
            stats = match_stats(query_kp, query_desc, ref_kp, ref_desc)
            reference = colors.get(slug)
            similarity = (float(cv2.compareHist(query_color, reference,
                                                cv2.HISTCMP_CORREL))
                          if reference is not None else 0.0)
            rows.append((slug, stats, max(similarity, 0.0)))

        for name, formula in SCORES.items():
            ranked = sorted(rows, key=lambda r: -formula(r[1], r[2]))
            names = [slug for slug, _, _ in ranked]
            hits[name] += names[:1] == [truth[i]]
            top5[name] += truth[i] in names[:5]
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(kept)}", flush=True)

    n = max(len(kept), 1)
    print(f"\nнабор {args.subset}, запросов {n}, шортлист {args.topk}, "
          f"{time.time() - t0:.0f} с")
    print(f"{'оценка':24} {'top-1':>8} {'top-5':>8}")
    for name in SCORES:
        print(f"{name:24} {hits[name] / n * 100:>7.1f}% {top5[name] / n * 100:>7.1f}%")

    out = INDEX_DIR / f"score_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "n": n, "topk": args.topk, "index": args.index,
        "top1": {name: hits[name] / n for name in SCORES},
        "top5": {name: top5[name] / n for name in SCORES},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
