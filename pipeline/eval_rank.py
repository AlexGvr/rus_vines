#!/usr/bin/env python3
"""Чем упорядочивать кандидатов: косинусом, геометрией или их сочетанием.

Разбор полевых промахов показал неприятное: правильное вино попадает
в шортлист, часто первым по косинусу, а перестановка по числу совпавших
точек уводит его вниз. На «Кокуре» (11813420_0.jpg) у пяти кандидатов
46-52 инлаера — разница внутри шума, а порядок она задаёт целиком.

Причина в том, что у линейки одной винодельни общая часть этикетки больше
различающей: гравюра дворца, зелёная лента, «1894 ВИНО РОССИИ» совпадают
у всех, а название сорта — три слова мелким шрифтом. Взвешивание точек
по зонам этикетки эту разницу не ловит (pipeline/eval_zones.py: инлаеры
верного и неверного кандидата лежат в одной области). Значит, вопрос не
в том, где точки, а в том, когда вообще верить их числу.

Скрипт считает один и тот же шортлист и сравнивает правила упорядочивания:

  косинус              вообще без геометрии
  инлаеры              как сейчас
  инлаеры × косинус    произведение, геометрия как вес
  косинус в спорных    инлаеры решают, но внутри группы, где их число
                       отличается меньше чем на порог, порядок берётся
                       по косинусу
  геометрия при отрыве если лидер по инлаерам не оторвался от второго,
                       весь порядок берётся по косинусу

Последние два — разные ответы на один вопрос: первое чинит порядок внутри
спорной группы, второе отключает геометрию целиком, когда она не уверена.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def order_cosine(cands):
    return sorted(cands, key=lambda c: -c.cv)


def order_inliers(cands):
    return sorted(cands, key=lambda c: (-c.inliers, -c.cv))


def order_product(cands):
    return sorted(cands, key=lambda c: -(c.inliers * max(c.cv, 0.0)))


def order_cosine_in_close(cands, window: float):
    """Геометрия задаёт группы, косинус — порядок внутри спорной группы."""
    ranked = order_inliers(cands)
    top = ranked[0].inliers
    close = [c for c in ranked if c.inliers >= top * window]
    rest = [c for c in ranked if c.inliers < top * window]
    return order_cosine(close) + rest


def order_geometry_if_clear(cands, margin: float):
    """Если лидер по инлаерам не оторвался от второго, геометрия молчит."""
    ranked = order_inliers(cands)
    if len(ranked) < 2 or ranked[0].inliers >= ranked[1].inliers * (1.0 + margin):
        return ranked
    return order_cosine(cands)


def order_inside_five(cands, depth: int = 5):
    """Косинус решает, кто попадёт в пятёрку; геометрия — порядок в ней.

    Разделение ролей по тому, что каждая ступень умеет. Косинус узнаёт
    линейку и держит правильное вино в первой пятёрке; геометрия внутри
    пятёрки отвечает на вопрос «та же ли это этикетка». Промахи мимо
    пятёрки при этом невозможны по построению: состав пятёрки геометрия
    не меняет.
    """
    ranked = order_cosine(cands)
    return order_inliers(ranked[:depth]) + ranked[depth:]


def schemes(window: float, margin: float) -> dict:
    return {
        "косинус": order_cosine,
        "инлаеры": order_inliers,
        "инлаеры × косинус": order_product,
        f"косинус в спорных ({window:.2f})": lambda c: order_cosine_in_close(c, window),
        **{f"геометрия при отрыве {m:.0%}":
           (lambda m: lambda c: order_geometry_if_clear(c, m))(m)
           for m in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.50, 0.75)},
        "геометрия внутри пятёрки": order_inside_five,
        "геометрия внутри тройки": lambda c: order_inside_five(c, 3),
        "геометрия внутри десятки": lambda c: order_inside_five(c, 10),
    }


def load_queries(subset: str, split: str):
    """Пары (путь, правильный slug)."""
    if subset != "field":
        manifest = list(csv.DictReader(
            (QUERIES / f"manifest_{subset}.csv").open(encoding="utf-8")))
        pairs = [(QUERIES / subset / row["image"], row["slug"]) for row in manifest]
        return [(p, s) for p, s in pairs if p.exists()]

    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
    extra = {row["slug"] for row in csv.DictReader(
        line for line in (ROOT / "data" / "catalog" / "extra_refs.csv")
        .open(encoding="utf-8") if not line.startswith("#"))}
    out = []
    for row in csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8")):
        if split != "all" and row["split"] != split:
            continue
        slug = row["slug"]
        # Позиция без эталона не находится ни при каком порядке — такие
        # кадры меряют полноту каталога, а не правило упорядочивания.
        if not slug or not (wines.get(slug, {}).get("photo") or slug in extra):
            continue
        path = FIELD / row["image"]
        if not path.exists():
            path = PUBLIC / row["image"]
        if path.exists():
            out.append((path, slug))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="field")
    ap.add_argument("--split", choices=["tune", "test", "all"], default="tune",
                    help="часть полевого набора; test для подбора правил не трогаем")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--window", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.subset == "field" and args.split == "test":
        sys.exit("тестовая часть существует ровно для того, чтобы правила "
                 "подбирались не на ней")

    queries = load_queries(args.subset, args.split)
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        sys.exit(f"нет запросов: {args.subset}/{args.split}")

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(INDEX_DIR / "sift")

    paths = [str(p) for p, _ in queries]
    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in ((), ("detect",))]
    kept = per_view[0][1]
    row_of = {p: i for i, p in enumerate(kept)}
    scores = np.maximum.reduce([qv @ vectors.T for qv, _ in per_view])

    rules = schemes(args.window, args.margin)
    hits = {name: [0, 0] for name in rules}     # top-1, top-5
    total = 0
    for path, gold in queries:
        row = row_of.get(str(path))
        if row is None:
            continue
        total += 1
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]
        candidates = core.shortlist(scores[row], index_slugs, args.topk)
        core.rerank(crop, candidates, reranker)
        for name, rule in rules.items():
            order = [c.slug for c in rule(list(candidates))]
            hits[name][0] += order[:1] == [gold]
            hits[name][1] += gold in order[:5]

    print(f"\nнабор {args.subset}, часть {args.split}: {total} кадров, "
          f"шортлист {args.topk}")
    print(f"{'правило порядка':34} {'top-1':>8} {'top-5':>8}")
    for name, (one, five) in hits.items():
        print(f"{name:34} {one / max(total, 1) * 100:>7.1f}% "
              f"{five / max(total, 1) * 100:>7.1f}%")

    out = INDEX_DIR / f"rank_{args.subset}_{args.split}.json"
    out.write_text(json.dumps(
        {"subset": args.subset, "split": args.split, "n": total,
         "window": args.window, "margin": args.margin,
         "top1": {k: v[0] / max(total, 1) for k, v in hits.items()},
         "top5": {k: v[1] / max(total, 1) for k, v in hits.items()}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
