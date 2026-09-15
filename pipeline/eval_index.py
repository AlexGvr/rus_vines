#!/usr/bin/env python3
"""S1.6 — замер индекса на валидационном наборе и сравнение вариантов.

Считает top-1 / top-5 по slug, отрыв первого результата от второго и —
для варианта dirty — долю запросов, где на первое место вышел файл,
не привязанный ни к одной позиции каталога (то есть мусор).

Отрыв важен отдельно от точности: ТЗ требует показывать одну карточку,
а не список вариантов, и решение об этом принимается именно по отрыву.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"


def load_index(variant: str) -> tuple[np.ndarray, list[str | None]]:
    vectors = np.load(INDEX_DIR / f"{variant}.npy")
    with (INDEX_DIR / f"{variant}.csv").open(encoding="utf-8") as fh:
        slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    return vectors, slugs


def query_vectors(subset: str, cache: dict) -> tuple[np.ndarray, list[str]]:
    """Эмбеддинги запросов считаются один раз и переиспользуются всеми вариантами."""
    if subset in cache:
        return cache[subset]
    manifest = list(csv.DictReader((QUERIES / f"manifest_{subset}.csv").open(encoding="utf-8")))
    paths = [str(QUERIES / subset / row["image"]) for row in manifest]
    t0 = time.time()
    vectors, kept = embed_paths(paths, batch_size=32, progress_every=0)
    truth = [manifest[paths.index(p)]["slug"] for p in kept]
    print(f"запросов: {len(kept)}, эмбеддинги за {time.time() - t0:.0f} с")
    cache[subset] = (vectors, truth)
    return cache[subset]


def evaluate(variant: str, qvec: np.ndarray, truth: list[str]) -> dict:
    index, slugs = load_index(variant)
    t0 = time.time()
    scores = qvec @ index.T                      # косинус: векторы L2-нормированы
    order = np.argsort(-scores, axis=1)[:, :50]
    search_ms = (time.time() - t0) / max(len(qvec), 1) * 1000

    top1 = top5 = garbage_top1 = 0
    gaps = []
    for i, gold in enumerate(truth):
        # Сервис не знает, какие векторы мусорные, поэтому мусор занимает
        # слот выдачи наравне с позициями каталога: выпавший на первое место
        # мусор — это промах, а не повод взять следующего кандидата.
        ranked, seen = [], set()
        for j in order[i]:
            slug = slugs[j]
            if slug is not None:
                if slug in seen:
                    continue              # одна позиция каталога — один кандидат
                seen.add(slug)
            ranked.append((slug, float(scores[i, j])))
            if len(ranked) == 5:
                break
        if ranked[0][0] is None:
            garbage_top1 += 1
        if ranked[0][0] == gold:
            top1 += 1
        if any(slug == gold for slug, _ in ranked):
            top5 += 1
        if len(ranked) >= 2:
            gaps.append(ranked[0][1] - ranked[1][1])

    n = max(len(truth), 1)
    return {
        "variant": variant,
        "index_size": len(slugs),
        "garbage_in_index": sum(1 for s in slugs if s is None),
        "top1": top1 / n,
        "top5": top5 / n,
        "garbage_top1": garbage_top1 / n,
        "gap_median": float(np.median(gaps)) if gaps else 0.0,
        "search_ms": search_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["clean", "dirty"])
    ap.add_argument("--subset", default="hard")
    args = ap.parse_args()

    cache: dict = {}
    qvec, truth = query_vectors(args.subset, cache)

    results = []
    for variant in args.variants:
        if not (INDEX_DIR / f"{variant}.npy").exists():
            print(f"пропуск {variant}: индекс не собран")
            continue
        results.append(evaluate(variant, qvec, truth))

    print(f"\nнабор: {args.subset}, запросов: {len(truth)}")
    print(f"{'вариант':8} {'в индексе':>10} {'мусора':>8} {'top-1':>8} {'top-5':>8} "
          f"{'мусор@1':>9} {'отрыв':>8} {'поиск':>9}")
    for r in results:
        print(f"{r['variant']:8} {r['index_size']:>10} {r['garbage_in_index']:>8} "
              f"{r['top1'] * 100:>7.1f}% {r['top5'] * 100:>7.1f}% "
              f"{r['garbage_top1'] * 100:>8.1f}% {r['gap_median']:>8.4f} "
              f"{r['search_ms']:>7.1f}мс")

    if len(results) == 2:
        d1 = (results[1]["top1"] - results[0]["top1"]) * 100
        d5 = (results[1]["top5"] - results[0]["top5"]) * 100
        print(f"\nразница {results[1]['variant']} минус {results[0]['variant']}: "
              f"top-1 {d1:+.1f} п.п., top-5 {d5:+.1f} п.п.")

    out = ROOT / "data" / "index" / f"eval_{args.subset}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
