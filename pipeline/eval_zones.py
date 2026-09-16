#!/usr/bin/env python3
"""S5.9 — взвешивание совпавших точек по зонам этикетки.

Гипотеза из разбора полевых промахов. У соседей по линейке одной винодельни
верх этикетки общий: бренд, год основания, герб. Различаются они низом —
сортом, категорией, годом урожая. Инлаеры считаются по всей этикетке, общая
часть даёт больше точек, и лидером становится не то вино. На «Русском
Игристом» это стоило семи ответов из восьми.

Проверяются три способа учесть зону:

  низ тяжелее      вес растёт линейно сверху вниз: 1 + weight · y
  только низ       точки выше середины не считаются вовсе
  редкость зоны    вес зависит от того, насколько эта часть эталона
                   отличает его от соседей по шортлисту. Считается на лету:
                   точка, совпавшая у многих кандидатов сразу, ничего не
                   различает и весит мало

Третий способ — единственный, не опирающийся на догадку о раскладке
этикетки. Первые два дёшевы и нужны как проверка самой гипотезы: если
низ действительно различает, простое линейное взвешивание это покажет.

Считается на одном и том же шортлисте, поэтому сравнивается ровно
взвешивание, а не отбор.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_images, embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker, descriptors, match_stats  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
GRID = 6          # сетка для «редкости зоны»: мельче — шумнее, крупнее — тупее


def weighted(points, weight: float) -> float:
    """Точки тем тяжелее, чем ниже они на этикетке."""
    return sum(1.0 + weight * y for _, y in points)


def bottom_only(points, cut: float = 0.5) -> float:
    return float(sum(1 for _, y in points if y >= cut))


def rarity_weighted(per_candidate: dict[str, tuple]) -> dict[str, float]:
    """Вес точки обратен тому, скольким кандидатам она досталась.

    Клетка сетки, совпавшая сразу у половины шортлиста, — это общая часть
    серии: логотип, герб, год основания. Она не различает ничего, и точки
    в ней должны весить меньше, чем точки в клетке, совпавшей у одного.
    """
    owners: dict[tuple[int, int], set[str]] = defaultdict(set)
    cells: dict[str, list[tuple[int, int]]] = {}
    for slug, points in per_candidate.items():
        chosen = [(min(int(x * GRID), GRID - 1), min(int(y * GRID), GRID - 1))
                  for x, y in points]
        cells[slug] = chosen
        owners_of = owners
        for cell in set(chosen):
            owners_of[cell].add(slug)
    return {slug: sum(1.0 / len(owners[cell]) for cell in chosen)
            for slug, chosen in cells.items()}


SCHEMES = {
    "инлаеры": None,
    "низ ×2": lambda stats, rare: weighted(stats.points, 1.0),
    "низ ×4": lambda stats, rare: weighted(stats.points, 3.0),
    "только низ": lambda stats, rare: bottom_only(stats.points),
    "редкость зоны": lambda stats, rare: rare,
    "инлаеры × редкость": lambda stats, rare: stats.inliers * (rare + 1.0) ** 0.5,
}


def load_queries(subset: str, limit: int):
    """Пары (путь, правильный slug) и вид набора."""
    if subset == "field":
        rows = list(csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8")))
        wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
        out = []
        for row in rows:
            if not row["slug"] or not wines.get(row["slug"], {}).get("photo"):
                continue
            path = FIELD / row["image"]
            if not path.exists():
                path = PUBLIC / row["image"]
            if path.exists():
                out.append((path, row["slug"]))
        return out[:limit] if limit else out
    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{subset}.csv").open(encoding="utf-8")))
    out = [(QUERIES / subset / row["image"], row["slug"]) for row in manifest]
    out = [(p, s) for p, s in out if p.exists()]
    return out[:limit] if limit else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="field")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sift", default=str(INDEX_DIR / "sift"),
                    help="папка предрассчитанных признаков; масштаб должен "
                         "совпадать с SIFT_MAX_SIDE, иначе сравниваются разные вещи")
    args = ap.parse_args()

    queries = load_queries(args.subset, args.limit)
    if not queries:
        sys.exit(f"нет запросов для набора {args.subset}")

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(args.sift)

    # Запросы синтетических наборов лежат пачкой — их дешевле считать разом.
    paths = [str(p) for p, _ in queries]
    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in ((), ("detect",))]
    kept = per_view[0][1]
    index_of = {p: i for i, p in enumerate(kept)}
    scores = np.maximum.reduce([qv @ vectors.T for qv, _ in per_view])

    hits = {name: 0 for name in SCHEMES}
    total = 0
    for path, gold in queries:
        row = index_of.get(str(path))
        if row is None:
            continue
        total += 1
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]
        query_kp, query_desc = descriptors(crop)
        candidates = core.shortlist(scores[row], index_slugs, args.topk)

        stats = {}
        for candidate in candidates:
            stats[candidate.slug] = match_stats(query_kp, query_desc,
                                                *reranker.reference(candidate.slug))
        rare = rarity_weighted({slug: st.points for slug, st in stats.items()})

        for name, formula in SCHEMES.items():
            if formula is None:
                ranked = sorted(stats, key=lambda s: -stats[s].inliers)
            else:
                ranked = sorted(stats, key=lambda s: -formula(stats[s], rare.get(s, 0.0)))
            hits[name] += ranked[:1] == [gold]
        if total % 50 == 0:
            print(f"  {total}/{len(queries)}", flush=True)

    print(f"\nнабор {args.subset}, запросов {total}, шортлист {args.topk}")
    print(f"{'схема':22} {'top-1':>8}")
    for name in SCHEMES:
        print(f"{name:22} {hits[name] / max(total, 1) * 100:>7.1f}%")

    out = INDEX_DIR / f"zones_{args.subset}_{Path(args.sift).name}.json"
    out.write_text(json.dumps(
        {"subset": args.subset, "n": total, "grid": GRID,
         "sift": args.sift,
         "top1": {name: hits[name] / max(total, 1) for name in SCHEMES}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
