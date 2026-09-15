#!/usr/bin/env python3
"""S5.8 — калибровка уверенности и отказа.

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

Признаки решения: число инлаеров, доминирование лидера над вторым,
косинус лучшего кандидата и раскладка совпавших точек по эталону.
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
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker, descriptors, match_stats  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


@dataclass
class Case:
    """Один замер: что решил конвейер и что было на самом деле.

    Признаки намеренно относительные — сравнивают лидера с остальным
    шортлистом, а не с абсолютной шкалой. Абсолютные величины не переносятся
    с синтетики на съёмку: у синтетического запроса, полученного из того же
    файла, что и эталон, косинус около 0.9 и полторы сотни инлаеров, а у фото
    с полки — 0.75 и полсотни. Правило, обученное на абсолютных значениях,
    на реальном кадре отказывает всегда.
    """
    present: bool          # есть ли правильная позиция в индексе
    correct: bool          # лидер совпал с правильной позицией
    inliers: int
    second: int
    cv: float
    cv_second: float
    cv_tail: float         # средний косинус хвоста шортлиста: уровень шума
    coverage: float

    @property
    def dominance(self) -> float:
        total = self.inliers + self.second
        return self.inliers / total if total else 0.0

    @property
    def cv_margin(self) -> float:
        """Отрыв лидера от второго кандидата по косинусу."""
        return self.cv - self.cv_second

    @property
    def cv_lift(self) -> float:
        """Насколько лидер выше шума шортлиста."""
        return self.cv - self.cv_tail


def evaluate(cases: list[Case], min_inliers: int, dominance: float,
             min_cv: float = 0.0) -> dict:
    """Доля верных, охват и обе ошибки отказа для одного правила."""
    shown = [c for c in cases
             if c.inliers >= min_inliers and c.dominance >= dominance and c.cv >= min_cv]
    present = [c for c in cases if c.present]
    absent = [c for c in cases if not c.present]
    shown_present = [c for c in shown if c.present]
    return {
        "precision": (sum(c.correct for c in shown_present) / len(shown_present)
                      if shown_present else 0.0),
        "coverage": len(shown_present) / len(present) if present else 0.0,
        "false_refusal": (sum(1 for c in present if c.correct and c not in shown)
                          / len(present) if present else 0.0),
        "false_accept": (sum(1 for c in absent if c in shown) / len(absent)
                         if absent else 0.0),
        "shown": len(shown),
    }


def collect(subset: str, index: str, views: list[tuple[str, ...]],
            topk: int, limit: int) -> list[Case]:
    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
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

    cases: list[Case] = []
    for i, path in enumerate(kept):
        truth = gold[path]
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]
        query_kp, query_desc = descriptors(crop)

        for present in (True, False):
            row = scores[i].copy()
            if not present:
                # Вычёркиваем правильную позицию: тот же снимок становится
                # запросом к вину, которого в каталоге нет.
                for j, slug in enumerate(index_slugs):
                    if slug == truth:
                        row[j] = -1.0
            order = np.argsort(-row)[:200]
            shortlist, seen = [], set()
            for j in order:
                slug = index_slugs[j]
                if slug is None or slug in seen:
                    continue
                seen.add(slug)
                shortlist.append((slug, float(row[j])))
                if len(shortlist) == topk:
                    break

            ranked = []
            for slug, cv in shortlist:
                stats = match_stats(query_kp, query_desc, *reranker.reference(slug))
                ranked.append((slug, stats, cv))
            ranked.sort(key=lambda r: -r[1].inliers)
            if not ranked:
                continue
            best, stats, cv = ranked[0]
            second = ranked[1][1].inliers if len(ranked) > 1 else 0
            by_cv = sorted((c for _, _, c in ranked), reverse=True)
            cases.append(Case(
                present=present, correct=present and best == truth,
                inliers=stats.inliers, second=second, cv=cv,
                cv_second=by_cv[1] if len(by_cv) > 1 else cv,
                cv_tail=float(np.mean(by_cv[5:])) if len(by_cv) > 5 else
                        float(np.mean(by_cv)),
                coverage=stats.coverage))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(kept)}", flush=True)
    return cases


FEATURES = ("доминирование", "log инлаеров", "отрыв косинуса",
            "превышение над шумом", "раскладка")


def feature_row(case: Case) -> list[float]:
    return [case.dominance, float(np.log1p(case.inliers)),
            case.cv_margin, case.cv_lift, case.coverage]


def fit_logistic(rows: np.ndarray, labels: np.ndarray, steps: int = 4000,
                 rate: float = 0.5) -> np.ndarray:
    """Логистическая регрессия на четырёх признаках, градиентным спуском.

    Модель намеренно крошечная. Признаков четыре, данных — сотни примеров,
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


def apply_logistic(rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(rows @ weights[:-1] + weights[-1])))


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

    cases = collect(args.subset, args.index, views, args.topk, args.limit)
    present = [c for c in cases if c.present]
    absent = [c for c in cases if not c.present]
    print(f"\nнабор {args.subset}: {len(present)} запросов с вином в каталоге, "
          f"{len(absent)} с вычеркнутым")
    print(f"инлаеры лидера: верный ответ медиана "
          f"{np.median([c.inliers for c in present if c.correct]):.0f}; "
          f"вино вычеркнуто медиана {np.median([c.inliers for c in absent]):.0f}, "
          f"90-й процентиль {np.percentile([c.inliers for c in absent], 90):.0f}")

    grid = []
    for min_inliers in (10, 15, 20, 25, 30, 40, 50):
        for dominance in (0.0, 0.55, 0.60, 0.65, 0.70):
            result = evaluate(cases, min_inliers, dominance)
            grid.append(((min_inliers, dominance, 0.0), result))

    print(f"\n{'инлаеры':>8} {'домин.':>7} {'точность':>10} {'охват':>8} "
          f"{'ложный отказ':>14} {'ложное принятие':>17}")
    for (min_inliers, dominance, _), r in grid:
        print(f"{min_inliers:>8} {dominance:>7.2f} {r['precision'] * 100:>9.1f}% "
              f"{r['coverage'] * 100:>7.1f}% {r['false_refusal'] * 100:>13.1f}% "
              f"{r['false_accept'] * 100:>16.1f}%")

    # Совместное решение по четырём признакам. Обучается на половине запросов,
    # проверяется на другой: иначе порог подгонится под те же данные, на
    # которых его меряют, и число окажется красивым, но неправдой.
    rows = np.array([feature_row(c) for c in cases])
    labels = np.array([float(c.correct) for c in cases])
    half = len(cases) // 2
    train = np.zeros(len(cases), dtype=bool)
    train[:half] = True
    weights = fit_logistic(rows[train], labels[train])
    probability = apply_logistic(rows, weights)
    print("\nвеса модели: " + ", ".join(
        f"{name} {w:+.2f}" for name, w in zip(FEATURES, weights[:-1])))

    held = ~train
    print(f"\nсовместное правило, проверка на отложенной половине "
          f"({int(held.sum())} замеров)")
    print(f"{'порог':>7} {'точность':>10} {'охват':>8} {'ложный отказ':>14} "
          f"{'ложное принятие':>17}")
    model_grid = []
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        shown = held & (probability >= threshold)
        present_mask = held & np.array([c.present for c in cases])
        correct_mask = np.array([c.correct for c in cases])
        absent_mask = held & ~np.array([c.present for c in cases])
        shown_present = shown & present_mask
        result = {
            "threshold": threshold,
            "precision": float(correct_mask[shown_present].mean())
                         if shown_present.any() else 0.0,
            "coverage": float(shown_present.sum() / max(present_mask.sum(), 1)),
            "false_refusal": float((present_mask & correct_mask & ~shown).sum()
                                   / max(present_mask.sum(), 1)),
            "false_accept": float((absent_mask & shown).sum()
                                  / max(absent_mask.sum(), 1)),
        }
        model_grid.append(result)
        print(f"{threshold:>7.2f} {result['precision'] * 100:>9.1f}% "
              f"{result['coverage'] * 100:>7.1f}% "
              f"{result['false_refusal'] * 100:>13.1f}% "
              f"{result['false_accept'] * 100:>16.1f}%")

    out = INDEX_DIR / f"calibration_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "index": args.index, "topk": args.topk,
        "n_present": len(present), "n_absent": len(absent),
        "grid": [{"min_inliers": a, "dominance": b, "min_cv": c, **r}
                 for (a, b, c), r in grid],
        "features": list(FEATURES),
        "weights": weights.tolist(),
        "model_grid": model_grid,
        "cases": [{"present": c.present, "correct": c.correct, "inliers": c.inliers,
                   "second": c.second, "cv": c.cv, "cv_second": c.cv_second,
                   "cv_tail": c.cv_tail, "coverage": c.coverage}
                  for c in cases],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
