#!/usr/bin/env python3
"""S5.5 — OCR как разрешитель противоречий, а не как второй канал поиска.

Прошлый заход (`eval_fusion.py`) смешивал текстовый и визуальный скоры по
всему шортлисту и дал +1.0 п.п. в лучшей точке при нуле на почти-дублях:
ошибки каналов коррелируют, и общий вес текста только шумит.

Здесь у текста роль уже: он вмешивается там, где геометрия не разделила
кандидатов. «Мускатель белый» против «Мускатель розовый» отличаются одним
словом на этикетке, и инлаеров у них поровну — вот ровно этот случай.
Правило одностороннее: уверенно прочитанное противоречие опускает кандидата,
неуверенное чтение не делает ничего. Награды за совпадение нет намеренно —
она поднимала бы кандидата за случайно совпавшее слово.

Замеряется на почти-дублях, где эффект должен быть виден, и на общем наборе,
где важно убедиться, что правило ничего не ломает.
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
from ocr import read_words  # noqa: E402
from rerank import PrecomputedReranker, descriptors, inlier_count  # noqa: E402
from text_match import TextChannel, resolve_close  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="neardup")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--views", default="raw,detect")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--window", type=float, nargs="+", default=[0.70, 0.80, 0.90],
                    help="доля инлаеров лидера, ниже которой кандидат уже не близкий")
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    views = [tuple(s for s in spec.split("+") if s and s != "raw")
             for spec in args.views.split(",")]

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

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in views]
    kept = per_view[0][1]
    truth = [gold[p] for p in kept]
    scores = np.maximum.reduce([qv @ vectors.T for qv, _ in per_view])
    order = np.argsort(-scores, axis=1)[:, :200]

    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    channel = TextChannel.from_catalog(CATALOG)

    base_hits = 0
    after = {w: 0 for w in args.window}
    touched = {w: 0 for w in args.window}
    fixed = {w: 0 for w in args.window}
    broken = {w: 0 for w in args.window}
    ocr_ms, groups = [], []
    examples: list[str] = []

    for i, path in enumerate(kept):
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]
        query_kp, query_desc = descriptors(crop)

        shortlist, seen = [], set()
        for j in order[i]:
            slug = index_slugs[j]
            if slug is None or slug in seen:
                continue
            seen.add(slug)
            shortlist.append(slug)
            if len(shortlist) == args.topk:
                break
        scored = []
        for slug in shortlist:
            ref_kp, ref_desc = reranker.reference(slug)
            scored.append((slug, float(inlier_count(query_kp, query_desc,
                                                    ref_kp, ref_desc))))
        scored.sort(key=lambda c: -c[1])
        if scored and scored[0][0] == truth[i]:
            base_hits += 1

        # OCR запускается только если группа близких кандидатов вообще есть:
        # это и есть «по близким оценкам», а не на каждом запросе.
        widest = max(args.window)
        best = scored[0][1] if scored else 0.0
        close = [c for c in scored if best > 0 and c[1] >= best * widest]
        groups.append(len(close))
        words = None
        if len(close) > 1:
            t0 = time.perf_counter()
            words = read_words(crop)
            ocr_ms.append((time.perf_counter() - t0) * 1000)

        for window in args.window:
            if words is None:
                ranked, found = scored, {}
            else:
                ranked, found = resolve_close(scored, words, channel, window=window,
                                              min_conf=args.min_conf)
            hit = bool(ranked and ranked[0][0] == truth[i])
            after[window] += hit
            if ranked[:1] != scored[:1]:
                touched[window] += 1
                was = bool(scored and scored[0][0] == truth[i])
                fixed[window] += hit and not was
                broken[window] += was and not hit
                if window == 0.80 and len(examples) < 8:
                    mark = "исправил" if hit and not was else (
                        "сломал" if was and not hit else "переставил")
                    examples.append(
                        f"  {mark}: {scored[0][0]} → {ranked[0][0]} "
                        f"(инлаеры {scored[0][1]:.0f}/{ranked[0][1]:.0f}, "
                        f"причина: {'; '.join(found.get(scored[0][0], [])) or '—'})")

    n = max(len(kept), 1)
    print(f"\nнабор {args.subset}, запросов {n}, индекс {args.index}")
    print(f"группа близких кандидатов возникает у {sum(1 for g in groups if g > 1)} "
          f"запросов ({sum(1 for g in groups if g > 1) / n * 100:.1f}%), "
          f"OCR по ним: {np.mean(ocr_ms) if ocr_ms else 0:.0f} мс в среднем")
    print(f"\nбез разрешения противоречий top-1 {base_hits / n * 100:.1f}%")
    print(f"{'окно':>6} {'top-1':>8} {'переставлено':>14} {'исправлено':>12} {'сломано':>9}")
    for window in args.window:
        print(f"{window:>6.2f} {after[window] / n * 100:>7.1f}% {touched[window]:>14} "
              f"{fixed[window]:>12} {broken[window]:>9}")
    if examples:
        print("\nчто именно менялось (окно 0.80):")
        print("\n".join(examples))

    out = INDEX_DIR / f"tiebreak_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "n": n, "index": args.index,
        "min_conf": args.min_conf,
        "base_top1": base_hits / n,
        "ocr_ms_mean": float(np.mean(ocr_ms)) if ocr_ms else 0.0,
        "with_close_group": sum(1 for g in groups if g > 1),
        "by_window": {str(w): {"top1": after[w] / n, "touched": touched[w],
                               "fixed": fixed[w], "broken": broken[w]}
                      for w in args.window},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
