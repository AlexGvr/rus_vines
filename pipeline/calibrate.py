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
    plain: bool = True     # можно ли снять оговорку по правилам выдачи
    field: bool = False    # снято людьми, а не собрано из каталожного фото

    @property
    def outcome(self) -> int:
        """Исход: 0 — лидер верен, 1 — верный во второй половине пятёрки,
        2 — верного в пятёрке нет. Для вина не из каталога верного нет
        по определению, и это тот же исход 2."""
        if self.correct:
            return 0
        return 1 if self.in_top5 else 2


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


def field_queries(split: str) -> tuple[list[str], dict[str, str]]:
    """Реальные снимки настроечной части: путь и правильный slug.

    Кадр без вина из каталога — это готовый отрицательный пример, причём
    честный: там снято похожее вино, которого в каталоге нет. Синтетика
    такие примеры подделывает вычёркиванием позиции из индекса, и соседи
    по серии остаются на месте — задача выходит легче настоящей.
    """
    manifest = ROOT / "data" / "field" / "manifest.csv"
    if not manifest.exists():
        return [], {}
    paths, gold = [], {}
    for row in csv.DictReader(manifest.open(encoding="utf-8")):
        if row["split"] != split or row["kind"] != "field":
            continue
        path = ROOT / "data" / "field" / row["image"]
        if not path.exists():
            path = ROOT / "dataset" / "eval" / "queries" / row["image"]
        if path.exists():
            paths.append(str(path))
            gold[str(path)] = row["slug"]
    return paths, gold


def collect(subset: str, index: str, views: list[tuple[str, ...]], topk: int,
            window: float, min_conf: float, limit: int,
            field_split: str = "") -> list[Case]:
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
    real = set()
    if field_split:
        field_paths, field_gold = field_queries(field_split)
        paths.extend(field_paths)
        gold.update(field_gold)
        real = set(field_paths)

    vectors = np.load(INDEX_DIR / f"{index}.npy")
    with (INDEX_DIR / f"{index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in views]
    kept = per_view[0][1]
    per_view_scores = [qv @ vectors.T for qv, _ in per_view]
    scores = np.maximum.reduce(per_view_scores)
    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    channel = TextChannel.from_catalog(CATALOG)
    by_slug = {w["slug"]: w for w in json.loads(
        CATALOG.read_text(encoding="utf-8"))["wines"]}

    cases: list[Case] = []
    for i, path in enumerate(kept):
        truth = gold[path]
        with Image.open(path) as raw:
            source = raw.convert("RGB")
        shots = [normalize_batch([source], steps=steps)[0] for steps in views]

        # У полевого кадра без вина в каталоге правильного ответа нет вовсе,
        # вычёркивать нечего — такой кадр даёт ровно один случай.
        variants = (True, False) if truth else (False,)
        for present in variants:
            row = scores[i].copy()
            if not present:
                # Вычёркиваем правильную позицию: тот же снимок становится
                # запросом к вину, которого в каталоге нет.
                row[[j for j, slug in enumerate(index_slugs) if slug == truth]] = -1.0
            candidates = core.shortlist(row, index_slugs, topk)
            # Тот же выбор кропа для геометрии, что в сервисе.
            similarity = np.stack([block[i] for block in per_view_scores], axis=1)
            crop = shots[core.pick_view(shots, similarity)]
            core.rerank(crop, candidates, reranker)
            core.resolve(candidates, crop, channel, read_words,
                         window=window, min_conf=min_conf)
            if not candidates:
                continue
            # Улики против лидера — часть конвейера, а не отдельный замер:
            # калибровать надо ту выдачу, которую увидит пользователь.
            core.check_leader(candidates, crop, channel, read_words, min_conf, window)
            slugs = [c.slug for c in candidates]
            cases.append(Case(
                present=present,
                correct=bool(present and slugs[0] == truth),
                in_top5=bool(present and truth in slugs[:5]),
                inliers=candidates[0].inliers,
                feats=core.features(candidates),
                plain=core.may_drop_caveat(candidates, by_slug, window),
                field=path in real))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(kept)}", flush=True)
    return cases


def fit_softmax(rows: np.ndarray, classes: np.ndarray, steps: int = 6000,
                rate: float = 0.5) -> np.ndarray:
    """Модель трёх взаимоисключающих исходов, веса как матрица.

    Исходы: правильное вино первое, правильное на местах со второго по
    пятое, правильного в пятёрке нет. Так top-5 по построению не бывает
    ниже top-1 — раньше это были две независимые модели, и ничто не мешало
    им разойтись.
    """
    mean, std = rows.mean(axis=0), rows.std(axis=0) + 1e-6
    scaled = np.hstack([(rows - mean) / std, np.ones((len(rows), 1))])
    weights = np.zeros((scaled.shape[1], 3))
    onehot = np.zeros((len(classes), 3))
    onehot[np.arange(len(classes)), classes] = 1.0
    for _ in range(steps):
        logits = scaled @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        weights -= rate * scaled.T @ (probability - onehot) / len(classes)
    # Разворачиваем нормировку обратно в исходные единицы признаков.
    raw = weights[:-1] / std[:, None]
    bias = weights[-1] - (mean / std) @ weights[:-1]
    return np.vstack([raw, bias])


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


def sweep(cases: list[Case], top1: np.ndarray, plain: np.ndarray,
          thresholds) -> list[dict]:
    """Что даёт каждый порог показа: точность, охват и обе ошибки."""
    grid = []
    for threshold in thresholds:
        result = evaluate(cases, top1 >= threshold)
        result["threshold"] = round(float(threshold), 3)
        confident = (top1 >= threshold) & plain
        present = np.array([c.present for c in cases])
        correct = np.array([c.correct for c in cases])
        result["confident"] = int(confident.sum())
        result["confident_precision"] = (float(correct[confident].mean())
                                         if confident.any() else 0.0)
        result["confident_absent"] = int((confident & ~present).sum())
        grid.append(result)
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--views", default="raw,detect")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--window", type=float, default=0.80)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--field-split", default="tune",
                    help="какую часть полевого набора подмешать в обучение; "
                         "test не указывать — он для итоговой проверки")
    ap.add_argument("--write", action="store_true",
                    help="записать веса и пороги в data/index/confidence.json")
    args = ap.parse_args()
    if args.field_split == "test":
        sys.exit("на тестовой части калибровать нельзя: она для итогового прогона")
    views = [tuple(s for s in spec.split("+") if s and s != "raw")
             for spec in args.views.split(",")]

    cases = collect(args.subset, args.index, views, args.topk,
                    args.window, args.min_conf, args.limit, args.field_split)
    present = [c for c in cases if c.present]
    real = [c for c in cases if c.field]
    print(f"\nнабор {args.subset}: {len(present)} запросов с вином в каталоге, "
          f"{len(cases) - len(present)} без; из них реальных снимков {len(real)}")

    rows = np.array([[c.feats[name] for name in core.FEATURE_ORDER] for c in cases])
    classes = np.array([c.outcome for c in cases])
    plain = np.array([c.plain for c in cases])

    # Реальных снимков на два порядка меньше синтетических, и без веса
    # модель их просто не заметит. Вес подобран так, чтобы полевая часть
    # весила примерно четверть обучения: синтетика задаёт форму зависимости,
    # реальные кадры — уровень.
    synthetic = np.array([not c.field for c in cases])
    share = max(len(real), 1)
    repeat = max(1, int(round(0.33 * synthetic.sum() / share)))
    print(f"реальные снимки входят в обучение с весом ×{repeat}")

    # Обучение на половине запросов, проверка на другой: иначе порог
    # подгонится под те же данные, на которых его меряют. Полевые кадры
    # делятся отдельно, чтобы они попали в обе половины.
    train = np.zeros(len(cases), dtype=bool)
    for mask in (synthetic, ~synthetic):
        where = np.flatnonzero(mask)
        train[where[:len(where) // 2]] = True

    index = np.flatnonzero(train)
    weight = np.concatenate([index] + [np.flatnonzero(train & ~synthetic)]
                            * (repeat - 1))
    raw = fit_softmax(rows[weight], classes[weight])
    outcome_weights = raw.tolist()
    probabilities = np.array([core.outcomes(c.feats, outcome_weights) for c in cases])
    top1, top5 = probabilities[:, 0], probabilities[:, 1]
    print("веса модели трёх исходов:")
    for name, row in zip(list(core.FEATURE_ORDER) + ["bias"], raw):
        print(f"  {name:14} лидер верен {row[0]:+8.3f}  "
              f"верный 2-5 {row[1]:+8.3f}  вне пятёрки {row[2]:+8.3f}")
    print(f"top-5 нигде не ниже top-1: {bool((top5 >= top1 - 1e-9).all())}")

    held = ~train
    held_cases = [c for c, keep in zip(cases, held) if keep]
    grid_all = sweep(held_cases, top1[held], plain[held],
                     (0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60))
    print(f"\nотложенная половина, все замеры ({len(held_cases)})")
    print(f"{'порог':>7} {'точность':>10} {'охват':>8} {'ложный отказ':>14} "
          f"{'ложное принятие':>17} {'без оговорок':>14}")
    for result in grid_all:
        print(f"{result['threshold']:>7.2f} {result['precision'] * 100:>9.1f}% "
              f"{result['coverage'] * 100:>7.1f}% "
              f"{result['false_refusal'] * 100:>13.1f}% "
              f"{result['false_accept'] * 100:>16.1f}% "
              f"{result['confident']:>6} шт "
              f"{result['confident_precision'] * 100:>4.0f}%")

    # Пороги выбираются по реальным снимкам. Синтетика систематически проще:
    # на ней любой порог выглядит лучше, чем окажется на съёмке.
    field_cases = [c for c in cases if c.field]
    grid_field = []
    if field_cases:
        mask = np.array([c.field for c in cases])
        grid_field = sweep(field_cases, top1[mask], plain[mask],
                           (0.02, 0.04, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60))
        print(f"\nнастроечная часть полевого набора ({len(field_cases)} кадров)")
        print(f"{'порог':>7} {'точность':>10} {'охват':>8} {'ложный отказ':>14} "
              f"{'ложное принятие':>17} {'без оговорок':>14}")
        for result in grid_field:
            print(f"{result['threshold']:>7.2f} {result['precision'] * 100:>9.1f}% "
                  f"{result['coverage'] * 100:>7.1f}% "
                  f"{result['false_refusal'] * 100:>13.1f}% "
                  f"{result['false_accept'] * 100:>16.1f}% "
                  f"{result['confident']:>6} шт "
                  f"{result['confident_precision'] * 100:>4.0f}%")

    out = INDEX_DIR / f"calibration_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "index": args.index, "topk": args.topk,
        "window": args.window, "min_conf": args.min_conf,
        "n_present": len(present), "n_field": len(real), "field_weight": repeat,
        "features": list(core.FEATURE_ORDER),
        "outcome_weights": outcome_weights,
        "grid": grid_all, "grid_field": grid_field,
        "cases": [{"present": c.present, "correct": c.correct, "field": c.field,
                   "in_top5": c.in_top5, "inliers": c.inliers, "plain": c.plain,
                   **c.feats}
                  for c in cases],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзамеры: {out}")

    if args.write:
        artifact = INDEX_DIR / "confidence.json"
        previous = (json.loads(artifact.read_text(encoding="utf-8"))["thresholds"]
                    if artifact.exists() else {"show_p": 0.06, "confident_p": 0.30})
        artifact.write_text(json.dumps({
            "outcome_weights": outcome_weights,
            "features": list(core.FEATURE_ORDER),
            "thresholds": previous,
            "trained_on": f"{args.subset} + полевая часть {args.field_split}",
            "n_cases": len(cases), "n_field": len(real), "field_weight": repeat,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"веса записаны: {artifact} (пороги пока прежние, "
              f"выбрать по таблице выше)")


if __name__ == "__main__":
    main()
