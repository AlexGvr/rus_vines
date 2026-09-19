#!/usr/bin/env python3
"""Обучаемое упорядочивание пятёрки кандидатов вместо правила на глаз.

Порядок кандидатов сейчас задаёт правило из одной строки: геометрия меняет
порядок, если у лидера по совпавшим точкам отрыв от второго не меньше 25%,
иначе остаётся порядок по косинусу. Правило подобрано по одному замеру
и на другом делении полевого набора уже не выигрывает — то есть разница
между правилами внутри шума, а сам выбор «чему верить» никуда не делся.

Здесь тот же выбор делает модель. Признаки нарочно относительные: доля
от лучшего по шортлисту, место в порядке, отрыв от соседа. Абсолютные
величины с синтетики на съёмку не переносятся — у запроса, сделанного
из файла эталона, и косинус, и число совпавших точек заметно выше, чем
у снимка с полки. Относительные переносятся лучше: «инлаеров вдвое
больше, чем у второго» значит одно и то же на обоих наборах.

Обучение — CatBoost в режиме ранжирования: группа это один запрос,
метка — правильный ли кандидат. Ранжирование, а не классификация,
потому что задача сравнительная: важно, кто первый в группе, а не
абсолютная вероятность каждого.

Проверяется на полевой настроечной части. Тестовая не используется:
она существует ровно для того, чтобы правила и пороги подбирались не
на ней.

Результат замера, чтобы его не повторяли заново. Модель, обученная на
синтетике: 94.5% top-1 против 92.0% у правила на отложенной синтетике
и 40.6% против 50.0% на съёмке. Подмешивание половины полевой
настроечной части (32 группы) картину не меняет: 31.2% против 34.4%
top-1 и 71.9% против 65.6% top-5 на второй половине — разница внутри
шума на тридцати двух кадрах. Упирается не в класс модели, а в объём
реальных данных: тридцати двух групп для четырнадцати признаков мало,
а синтетика переносится плохо. Скрипт оставлен рабочим: когда полевых
кадров станет заметно больше, замер надо повторить.
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
from build_index import extra_references  # noqa: E402
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"

FEATURES = (
    "cv", "cv_to_best", "cv_to_second", "cv_rank",
    "inliers", "inliers_to_best", "inliers_to_second", "geometry_rank",
    "coverage", "spread", "same_brand_as_cv_leader", "contradicted",
    "group_size", "geometry_decisive",
)


def rows_for(candidates: list, by_slug: dict) -> np.ndarray:
    """Признаки каждого кандидата относительно остальных в шортлисте."""
    if not candidates:
        return np.zeros((0, len(FEATURES)))
    cvs = np.array([c.cv for c in candidates])
    inl = np.array([float(c.inliers) for c in candidates])
    best_cv = cvs.max() or 1e-9
    order_cv = np.argsort(-cvs)
    order_in = np.argsort(-inl)
    place_cv = {i: r for r, i in enumerate(order_cv)}
    place_in = {i: r for r, i in enumerate(order_in)}
    top_in = inl.max()
    second_in = np.sort(inl)[-2] if len(inl) > 1 else 0.0
    second_cv = np.sort(cvs)[-2] if len(cvs) > 1 else 0.0
    leader_brand = (by_slug.get(candidates[int(order_cv[0])].slug) or {}
                    ).get("manufacturer")
    close = int((inl >= top_in * 0.8).sum())
    decisive = float(top_in >= second_in * 1.25)

    out = []
    for i, candidate in enumerate(candidates):
        brand = (by_slug.get(candidate.slug) or {}).get("manufacturer")
        out.append([
            float(candidate.cv),
            float(candidate.cv - best_cv),
            float(candidate.cv - second_cv),
            float(place_cv[i]),
            float(candidate.inliers),
            float(candidate.inliers / top_in) if top_in else 0.0,
            float(candidate.inliers - second_in),
            float(place_in[i]),
            float(candidate.coverage),
            float(getattr(candidate, "color", 1.0)),
            float(brand is not None and brand == leader_brand),
            float(bool(candidate.contradictions)),
            float(close),
            decisive,
        ])
    return np.array(out, dtype=np.float32)


def collect(subset: str, split: str, index: str, topk: int, limit: int):
    """Группы кандидатов: признаки, метка «это правильный ответ», запрос."""
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    findable = ({s for s, w in by_slug.items() if w.get("photo")}
                | {s for _, s in extra_references()})

    pairs = []
    if subset == "field":
        for row in csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8")):
            if (split != "all" and row["split"] != split) or not row["slug"]:
                continue
            if row["slug"] not in findable:
                continue
            path = FIELD / row["image"]
            if not path.exists():
                path = PUBLIC / row["image"]
            if path.exists():
                pairs.append((path, row["slug"]))
    else:
        manifest = list(csv.DictReader(
            (QUERIES / f"manifest_{subset}.csv").open(encoding="utf-8")))
        for row in manifest:
            path = QUERIES / subset / row["image"]
            if path.exists():
                pairs.append((path, row["slug"]))
    if limit:
        pairs = pairs[:limit]

    vectors = np.load(INDEX_DIR / f"{index}.npy")
    with (INDEX_DIR / f"{index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(INDEX_DIR / "sift")

    paths = [str(p) for p, _ in pairs]
    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in ((), ("detect",))]
    kept = per_view[0][1]
    row_of = {p: i for i, p in enumerate(kept)}
    blocks = [qv @ vectors.T for qv, _ in per_view]

    groups = []
    for path, gold in pairs:
        i = row_of.get(str(path))
        if i is None:
            continue
        similarity = np.stack([block[i] for block in blocks], axis=1)
        with Image.open(path) as raw:
            source = raw.convert("RGB")
        shots = [normalize_batch([source], steps=steps)[0]
                 for steps in ((), ("detect",))]
        shot = shots[core.pick_view(shots, similarity)]
        candidates = core.shortlist(similarity.max(axis=1), index_slugs, topk)
        core.rerank(shot, candidates, reranker)
        groups.append((rows_for(candidates, by_slug),
                       np.array([c.slug == gold for c in candidates], dtype=int),
                       [c.slug for c in candidates], gold, str(path)))
        if len(groups) % 100 == 0:
            print(f"  {len(groups)}/{len(pairs)}", flush=True)
    return groups


def score(groups, predict) -> tuple[float, float]:
    """top-1 и top-5 при заданной функции упорядочивания."""
    one = five = 0
    for features, _, slugs, gold, _ in groups:
        order = [slugs[i] for i in np.argsort(-predict(features))]
        one += order[:1] == [gold]
        five += gold in order[:5]
    return one / max(len(groups), 1), five / max(len(groups), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--train-subset", default="sharp")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=400)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--field-train", action="store_true",
                    help="подмешать половину полевой настроечной части в обучение "
                         "и проверять на второй половине")
    ap.add_argument("--out", default=str(INDEX_DIR / "rank_model.cbm"))
    args = ap.parse_args()

    from catboost import CatBoost, Pool

    print("синтетические группы для обучения")
    train = collect(args.train_subset, "all", args.index, args.topk, args.limit)
    print("полевая настроечная часть для проверки")
    tune = collect("field", "tune", args.index, args.topk, 0)

    def flatten(groups):
        features = np.concatenate([g[0] for g in groups])
        labels = np.concatenate([g[1] for g in groups])
        ids = np.concatenate([[i] * len(g[0]) for i, g in enumerate(groups)])
        return Pool(features, labels, group_id=ids, feature_names=list(FEATURES))

    # Половина синтетики на обучение, половина на проверку: без неё нельзя
    # отличить выученное правило от запомненной выборки.
    half = len(train) // 2
    # Полевая часть делится по кадрам, а не по винам: внутри настроечной
    # части вина уже отделены от тестовых, и здесь проверяется другое —
    # перенос правила, а не перенос на новые вина.
    field_half = len(tune) // 2
    pool = train[:half] + (tune[:field_half] if args.field_train else [])
    check = tune[field_half:] if args.field_train else tune
    model = CatBoost({"loss_function": "YetiRank", "iterations": args.iterations,
                      "depth": args.depth, "verbose": 200, "random_seed": 17})
    model.fit(flatten(pool))
    print(f"обучено на {len(pool)} группах "
          f"(синтетика {half}, съёмка {len(pool) - half})")

    def predict(features):
        return model.predict(features) if len(features) else np.zeros(0)

    def rule(features):
        """Текущее правило в тех же признаках — чтобы сравнивать честно."""
        decisive = features[:, FEATURES.index("geometry_decisive")]
        if len(features) and decisive[0] > 0.5:
            return -features[:, FEATURES.index("geometry_rank")]
        return -features[:, FEATURES.index("cv_rank")]

    print(f"\n{'набор':34} {'правило':>16} {'модель':>16}")
    for name, groups in (("синтетика, отложенная половина", train[half:]),
                         ("полевая часть для проверки", check)):
        r1, r5 = score(groups, rule)
        m1, m5 = score(groups, predict)
        print(f"{name:34} {r1 * 100:>7.1f}/{r5 * 100:<8.1f} "
              f"{m1 * 100:>7.1f}/{m5 * 100:<8.1f}  (top-1/top-5, кадров {len(groups)})")

    model.save_model(args.out)
    print(f"\nмодель: {args.out}")
    weights = model.get_feature_importance(type="PredictionValuesChange")
    print("вклад признаков:")
    for name, value in sorted(zip(FEATURES, weights), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:26} {value:6.1f}")


if __name__ == "__main__":
    main()
