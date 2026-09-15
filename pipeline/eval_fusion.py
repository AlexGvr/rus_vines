#!/usr/bin/env python3
"""S2.3 + S2.5 — сравнение каналов и подбор веса слияния.

Отдельно меряется подмножество почти-дублей: позиции, у которых в каталоге
есть «сосед» той же винодельни с тем же названием, отличающийся категорией,
крепостью или годом. ТЗ называет их главным источником ошибок, и общая
точность их прячет — их в каталоге меньше десятой части.

OCR считается один раз и кэшируется: это самая дорогая часть замера.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from text_match import TextChannel, extract_attributes, fuse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
STEPS = ("detect",)
CV_TOPK = 50


def near_duplicate_slugs(wines: list[dict]) -> set[str]:
    """Позиции, у которых есть тёзка той же винодельни."""
    groups = defaultdict(list)
    for wine in wines:
        key = (wine["manufacturer"].strip().lower(), wine["title"].strip().lower())
        groups[key].append(wine["slug"])
    return {slug for slugs in groups.values() if len(slugs) > 1 for slug in slugs}


def load_index(name: str) -> tuple[np.ndarray, list[str | None]]:
    vectors = np.load(INDEX_DIR / f"{name}.npy")
    with (INDEX_DIR / f"{name}.csv").open(encoding="utf-8") as fh:
        slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    return vectors, slugs


def ocr_texts(paths: list[str], cache_path: Path) -> dict[str, str]:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = [p for p in paths if p not in cache]
    if missing:
        from ocr import read_text
        print(f"OCR: считаю {len(missing)} новых кадров…", flush=True)
        t0 = time.time()
        for n, path in enumerate(missing, 1):
            try:
                with Image.open(path) as raw:
                    crop = normalize_batch([raw.convert("RGB")], steps=STEPS)[0]
                cache[path] = read_text(crop)
            except Exception:
                cache[path] = ""
            if n % 100 == 0:
                print(f"  {n}/{len(missing)}", flush=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"OCR готов за {time.time() - t0:.0f} с", flush=True)
    return cache


def accuracy(predictions: list[list[str]], truth: list[str],
             subset: set[str] | None = None) -> tuple[float, float, int]:
    idx = [i for i, gold in enumerate(truth) if subset is None or gold in subset]
    if not idx:
        return 0.0, 0.0, 0
    top1 = sum(1 for i in idx if predictions[i] and predictions[i][0] == truth[i])
    top5 = sum(1 for i in idx if truth[i] in predictions[i])
    return top1 / len(idx), top5 / len(idx), len(idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hard")
    ap.add_argument("--index", default="clean_detect")
    ap.add_argument("--weights", nargs="+", type=float,
                    default=[0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--attr-weights", nargs="+", type=float, default=[0.0])
    args = ap.parse_args()

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    near_dup = near_duplicate_slugs(wines)
    channel = TextChannel(wines)

    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    paths, gold_by_path = [], {}
    for row in manifest:
        path = QUERIES / args.subset / row["image"]
        if path.exists():                     # часть кадров могли удалить вручную
            paths.append(str(path))
            gold_by_path[str(path)] = row["slug"]

    vectors, index_slugs = load_index(args.index)
    qvec, kept = embed_paths(paths, batch_size=32, progress_every=0, steps=STEPS)
    truth = [gold_by_path[p] for p in kept]

    scores = qvec @ vectors.T
    order = np.argsort(-scores, axis=1)[:, :200]
    cv_lists = []
    for i in range(len(kept)):
        seen, candidates = set(), []
        for j in order[i]:
            slug = index_slugs[j]
            if slug is None or slug in seen:
                continue
            seen.add(slug)
            candidates.append((slug, float(scores[i, j])))
            if len(candidates) == CV_TOPK:
                break
        cv_lists.append(candidates)

    texts = ocr_texts(kept, INDEX_DIR / f"ocr_{args.subset}.json")
    text_scores, attributes = [], []
    for path in kept:
        text = texts.get(path, "")
        text_scores.append(channel.scores(text))
        attributes.append(extract_attributes(text))
    empty = sum(1 for p in kept if not texts.get(p, "").strip())

    print(f"\nнабор {args.subset}: запросов {len(kept)}, "
          f"из них почти-дублей {sum(1 for t in truth if t in near_dup)}, "
          f"без распознанного текста {empty}")
    print(f"{'текст':>7} {'атриб':>7} {'top-1':>8} {'top-5':>8} "
          f"{'top-1 дубли':>13} {'top-5 дубли':>13}")

    results = []
    for weight in args.weights:
        for attr_w in args.attr_weights:
            preds = [[slug for slug, _ in
                      fuse(cv, ts, at, channel, weight, top=5, weight_attr=attr_w)]
                     for cv, ts, at in zip(cv_lists, text_scores, attributes)]
            t1, t5, _ = accuracy(preds, truth)
            d1, d5, n_dup = accuracy(preds, truth, near_dup)
            results.append({"weight_text": weight, "weight_attr": attr_w,
                            "top1": t1, "top5": t5,
                            "top1_neardup": d1, "top5_neardup": d5, "n_neardup": n_dup})
            print(f"{weight:>7.2f} {attr_w:>7.2f} {t1 * 100:>7.1f}% {t5 * 100:>7.1f}% "
                  f"{d1 * 100:>12.1f}% {d5 * 100:>12.1f}%")

    out = INDEX_DIR / f"fusion_{args.subset}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
