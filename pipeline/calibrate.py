#!/usr/bin/env python3
"""S5.8 — калибровка уверенности и отказа на полном конвейере.

Прежние пороги подбирались по одному признаку — числу инлаеров — и по
набору, где нужное вино всегда есть в каталоге. Из-за этого не измерялась
главная ошибка отказа: сколько раз сервис уверенно покажет карточку вина,
которого в каталоге нет вовсе.

Отрицательные примеры строятся честно, без отдельной съёмки: для каждого
запроса правильная позиция вычёркивается из индекса, и тот же снимок
превращается в запрос к отсутствующему вину. Соседи по серии при этом
остаются на месте — то есть случай получается трудным, а не показательным.

Меряются одновременно четыре величины, потому что улучшать одну за счёт
другой легко и бессмысленно:
  точность карточек   — доля верных среди автоматически открытых
  охват               — доля запросов, получивших карточку
  ложный отказ        — вино в каталоге есть, а сервис его не показал
  ложное принятие     — вина в каталоге нет, а карточка показана

Конвейер прогоняется целиком, вместе с перестановкой близких кандидатов по
тексту: калибровать надо ту выдачу, которую увидит пользователь. Признаки
считает pipeline/searchcore.py — тот же модуль, которым считает сервис.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


@dataclass
class Case:
    """Один замер: что решил конвейер и что было на самом деле."""
    present: bool          # есть ли правильная позиция в индексе
    correct: bool          # лидер совпал с правильной позицией
    in_top5: bool          # правильная позиция попала в показанную пятёрку
    inliers: int
    feats: dict


def evaluate(cases: list[Case], shown: np.ndarray) -> dict:
    """Доля верных, охват и обе ошибки отказа для одной маски показа."""
    present = np.array([c.present for c in cases])
    correct = np.array([c.correct for c in cases])
    shown_present = shown & present
    return {
        "precision": float(correct[shown_present].mean()) if shown_present.any() else 0.0,
        "coverage": float(shown_present.sum() / max(present.sum(), 1)),
        "false_refusal": float((present & correct & ~shown).sum() / max(present.sum(), 1)),
        "false_accept": float((~present & shown).sum() / max((~present).sum(), 1)),
        "shown": int(shown.sum()),
    }


def collect(subset: str, index: str, views: list[tuple[str, ...]], topk: int,
            window: float, min_conf: float, limit: int) -> list[Case]:
    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{subset}.csv").open(encoding="utf-8")))
    if limit:
        manifest = manifest[:limit]
    paths, gold = [], {}
    for row in manifest:
        path = QUERIES / subset / row["image"]
        if path.exists():
            paths.append(str(path))
            gold[str(path)] = row["slug"]

    vectors = np.load(INDEX_DIR / f"{index}.npy")
    with (INDEX_DIR / f"{index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in views]
    kept = per_view[0][1]
    scores = np.maximum.reduce([qv @ vectors.T for qv, _ in per_view])
    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    channel = TextChannel.from_catalog(CATALOG)

    cases: list[Case] = []
    for i, path in enumerate(kept):
        truth = gold[path]
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]

        for present in (True, False):
            row = scores[i].copy()
            if not present:
                # Вычёркиваем правильную позицию: тот же снимок становится
                # запросом к вину, которого в каталоге нет.
                row[[j for j, slug in enumerate(index_slugs) if slug == truth]] = -1.0
            candidates = core.shortlist(row, index_slugs, topk)
            core.rerank(crop, candidates, reranker)
            core.resolve(candidates, crop, channel, read_words,
                         window=window, min_conf=min_conf)
            if not candidates:
                continue
            slugs = [c.slug for c in candidates]
            cases.append(Case(
                present=present,
                correct=present and slugs[0] == truth,
                in_top5=present and truth in slugs[:5],
                inliers=candidates[0].inliers,
                feats=core.features(candidates)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(kept)}", flush=True)
    return cases


def fit_logistic(rows: np.ndarray, labels: np.ndarray, steps: int = 6000,
                 rate: float = 0.5) -> np.ndarray:
    """Логистическая регрессия на признаках ядра, градиентным спуском.

    Модель намеренно крошечная. Признаков пять, данных — сотни примеров,
    а набор синтетический: всё, что сложнее, запомнит особенности генератора
    запросов вместо свойств задачи. Веса при этом читаемы, и видно, какой
    признак на что влияет.
    """
    mean, std = rows.mean(axis=0), rows.std(axis=0) + 1e-6
    scaled = np.hstack([(rows - mean) / std, np.ones((len(rows), 1))])
    weights = np.zeros(scaled.shape[1])
    for _ in range(steps):
        prediction = 1.0 / (1.0 + np.exp(-scaled @ weights))
        weights -= rate * scaled.T @ (prediction - labels) / len(labels)
    return np.concatenate([weights[:-1] / std,
                           [weights[-1] - float(weights[:-1] @ (mean / std))]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--views", default="raw,detect")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--window", type=float, default=0.80)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    views = [tuple(s for s in spec.split("+") if s and s != "raw")
             for spec in args.views.split(",")]

    cases = collect(args.subset, args.index, views, args.topk,
                    args.window, args.min_conf, args.limit)
    present = [c for c in cases if c.present]
    absent = [c for c in cases if not c.present]
    print(f"\nнабор {args.subset}: {len(present)} запросов с вином в каталоге, "
          f"{len(absent)} с вычеркнутым")

    rows = np.array([[c.feats[name] for name in core.FEATURE_ORDER] for c in cases])
    labels = np.array([float(c.correct) for c in cases])
    half = len(cases) // 2
    train = np.zeros(len(cases), dtype=bool)
    train[:half] = True
    # Обучение на половине запросов, проверка на другой: иначе порог
    # подгонится под те же данные, на которых его меряют.
    raw_weights = fit_logistic(rows[train], labels[train])
    weights = dict(zip(core.FEATURE_ORDER, raw_weights[:-1].tolist()))
    weights["bias"] = float(raw_weights[-1])
    probability = np.array([core.probability(c.feats, weights) for c in cases])
    print("веса: " + ", ".join(f"{k} {v:+.3f}" for k, v in weights.items()))

    held_cases = [c for c, keep in zip(cases, ~train) if keep]
    held = ~train
    print(f"\nпроверка на отложенной половине ({len(held_cases)} замеров)")
    print(f"{'порог':>7} {'точность':>10} {'охват':>8} {'ложный отказ':>14} "
          f"{'ложное принятие':>17}")
    grid = []
    for threshold in (0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        result = evaluate(held_cases, (probability >= threshold)[held])
        result["threshold"] = threshold
        grid.append(result)
        print(f"{threshold:>7.2f} {result['precision'] * 100:>9.1f}% "
              f"{result['coverage'] * 100:>7.1f}% "
              f"{result['false_refusal'] * 100:>13.1f}% "
              f"{result['false_accept'] * 100:>16.1f}%")

    out = INDEX_DIR / f"calibration_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "index": args.index, "topk": args.topk,
        "window": args.window, "min_conf": args.min_conf,
        "n_present": len(present), "n_absent": len(absent),
        "features": list(core.FEATURE_ORDER),
        "weights": weights,
        "grid": grid,
        "cases": [{"present": c.present, "correct": c.correct,
                   "in_top5": c.in_top5, "inliers": c.inliers, **c.feats}
                  for c in cases],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
