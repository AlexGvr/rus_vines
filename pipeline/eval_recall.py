#!/usr/bin/env python3
"""S5.1 — где именно теряется правильный ответ: на отборе или на ранжировании.

Итоговая точность — произведение двух вещей: попал ли нужный slug в шортлист
и вытащила ли его наверх геометрия. Общая метрика их смешивает, поэтому здесь
для каждого запроса сохраняется позиция правильного slug ДО уточнения (ранг
по косинусу) и ПОСЛЕ (ранг по инлаерам внутри шортлиста). Промахи делятся на
два непересекающихся класса:

  отбор       — правильного slug нет в шортлисте, геометрии его не дать;
  ранжирование — slug в шортлисте был, но наверх вышел другой.

Первый класс лечится расширением шортлиста и дополнительными видами запроса,
второй — только качеством сопоставления этикеток. Лечить их одним средством
бессмысленно, поэтому сначала измеряем пропорцию.

Шортлисты разных размеров считаются за один проход: инлаеры набираются для
самого длинного варианта, а короткие получаются его префиксом. Иначе замер
времени утонул бы в повторном счёте дескрипторов запроса.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker, descriptors, inlier_count  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
STEPS = ("detect",)
DEEP = 500      # до какого ранга ищем правильный slug, чтобы промах был измерим


def shortlist(scores_row: np.ndarray, order_row: np.ndarray,
              index_slugs: list[str | None], limit: int) -> list[tuple[str, float]]:
    """Кандидаты по убыванию косинуса, по одному на slug.

    Одно фото каталога может принадлежать двум позициям, и один slug может
    иметь несколько векторов, поэтому свёртка по slug обязательна: без неё
    шортлист на 20 мест занимают несколько представлений одной позиции.
    """
    seen: set[str] = set()
    picked: list[tuple[str, float]] = []
    for j in order_row:
        slug = index_slugs[j]
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        picked.append((slug, float(scores_row[j])))
        if len(picked) == limit:
            break
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_detect")
    ap.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    ap.add_argument("--limit", type=int, default=0, help="взять только N запросов")
    args = ap.parse_args()
    sizes = sorted(args.topk)
    deepest = max(sizes)

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

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    qvec, kept = embed_paths(paths, batch_size=32, progress_every=0, steps=STEPS)
    truth = [gold[p] for p in kept]
    scores = qvec @ vectors.T
    order = np.argsort(-scores, axis=1)[:, :DEEP * 2]

    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    stats = {k: Counter() for k in sizes}
    desc_ms: list[float] = []
    match_ms = {k: [] for k in sizes}
    cv_rank_hist: list[int] = []      # ранг правильного slug до уточнения, -1 если глубже DEEP
    rr_rank_hist = {k: [] for k in sizes}

    for i, path in enumerate(kept):
        deep = shortlist(scores[i], order[i], index_slugs, DEEP)
        deep_slugs = [s for s, _ in deep]
        cv_rank = deep_slugs.index(truth[i]) if truth[i] in deep_slugs else -1
        cv_rank_hist.append(cv_rank)

        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=STEPS)[0]
        t0 = time.perf_counter()
        query_kp, query_desc = descriptors(crop)
        desc_ms.append((time.perf_counter() - t0) * 1000)

        # Инлаеры считаем для самого длинного шортлиста; короткие — его префикс.
        per_candidate: list[tuple[str, int, float]] = []
        for slug, _ in deep[:deepest]:
            t1 = time.perf_counter()
            ref_kp, ref_desc = reranker.reference(slug)
            value = inlier_count(query_kp, query_desc, ref_kp, ref_desc)
            per_candidate.append((slug, value, (time.perf_counter() - t1) * 1000))

        for k in sizes:
            prefix = per_candidate[:k]
            match_ms[k].append(sum(ms for _, _, ms in prefix))
            in_list = 0 <= cv_rank < k
            stats[k]["в шортлисте"] += in_list
            if not prefix:
                continue
            ranked = [slug for slug, _, _ in sorted(prefix, key=lambda r: -r[1])]
            rank = ranked.index(truth[i]) if truth[i] in ranked else -1
            rr_rank_hist[k].append(rank)
            if rank == 0:
                stats[k]["top-1"] += 1
            if 0 <= rank < 5:
                stats[k]["top-5"] += 1
            if rank != 0:
                stats[k]["промах отбора" if not in_list else "промах ранжирования"] += 1

    n = max(len(kept), 1)
    print(f"\nнабор {args.subset}, запросов {n}, индекс {args.index}")
    print(f"дескрипторы запроса: {np.mean(desc_ms):.0f} мс в среднем\n")
    print(f"{'шортлист':>9} {'top-1':>7} {'top-5':>7} {'в списке':>9} "
          f"{'промах отбора':>15} {'промах ранж.':>14} {'реранк, мс':>11} {'p95':>7}")
    for k in sizes:
        s = stats[k]
        total_ms = np.array(match_ms[k]) + np.array(desc_ms)
        print(f"{k:>9} {s['top-1'] / n * 100:>6.1f}% {s['top-5'] / n * 100:>6.1f}% "
              f"{s['в шортлисте'] / n * 100:>8.1f}% {s['промах отбора']:>15} "
              f"{s['промах ранжирования']:>14} {total_ms.mean():>11.0f} "
              f"{np.percentile(total_ms, 95):>7.0f}")

    beyond = sum(1 for r in cv_rank_hist if r < 0)
    print(f"\nранг правильного slug по косинусу (до уточнения):")
    for lo, hi in ((0, 1), (1, 5), (5, 20), (20, 50), (50, 100), (100, DEEP)):
        count = sum(1 for r in cv_rank_hist if lo <= r < hi)
        print(f"  {lo:>4}..{hi - 1:<4} {count:>5}  ({count / n * 100:4.1f}%)")
    print(f"  глубже {DEEP}    {beyond:>5}  ({beyond / n * 100:4.1f}%)  — расширение "
          f"шортлиста их не вернёт")

    out = INDEX_DIR / f"recall_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "index": args.index, "n": n,
        "desc_ms_mean": float(np.mean(desc_ms)),
        "by_topk": {str(k): {
            "top1": stats[k]["top-1"] / n,
            "top5": stats[k]["top-5"] / n,
            "in_shortlist": stats[k]["в шортлисте"] / n,
            "miss_retrieval": stats[k]["промах отбора"],
            "miss_ranking": stats[k]["промах ранжирования"],
            "ms_mean": float((np.array(match_ms[k]) + np.array(desc_ms)).mean()),
            "ms_p95": float(np.percentile(np.array(match_ms[k]) + np.array(desc_ms), 95)),
        } for k in sizes},
        "cv_rank": cv_rank_hist,
        "rerank_rank": {str(k): rr_rank_hist[k] for k in sizes},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
