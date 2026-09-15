#!/usr/bin/env python3
"""S5.4 — несколько векторов на позицию вместо одного.

Одиночный вектор заставляет выбрать одно представление эталона и надеяться,
что запрос попадёт в него же. Разбор промахов показал, чем это кончается:
если детектор на запросе промахнулся мимо целевой бутылки, кроп описывает
соседа, и правильная позиция улетает ниже сотого места.

Здесь позиция описывается набором векторов (кадр целиком, кроп по бутылке,
полоса этикетки), а выдача сворачивается по slug максимумом косинуса. Запрос
тоже может идти несколькими видами. Смысл в том, что видам не нужно совпасть
все сразу — достаточно одной пары, чтобы позиция попала в шортлист.

Цена индексная, не запросная: векторов больше, но поиск остаётся одним
матричным умножением. Запрос дорожает ровно на число своих видов.
"""
from __future__ import annotations

import argparse
import csv
import itertools
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

# Вид — это пара «индекс, шаги обработки запроса». Шаги обязаны совпадать
# с теми, которыми собран индекс, иначе сравниваются разные сущности.
VIEWS = {
    "raw": ("clean_raw", ()),
    "detect": ("clean_detect", ("detect",)),
    "label": ("clean_detect_label", ("detect", "label")),
}


def load_index(name: str) -> tuple[np.ndarray, list[str | None]]:
    vectors = np.load(INDEX_DIR / f"{name}.npy")
    with (INDEX_DIR / f"{name}.csv").open(encoding="utf-8") as fh:
        slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    return vectors, slugs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20, 50])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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

    # Матрица «запрос × позиция» для каждой пары видов: индексного и запросного.
    index_cache, query_cache, kept_ref = {}, {}, None
    # Пул кандидатов — весь каталог, а не позиции выборки: иначе recall считается
    # по сотне вариантов вместо двух тысяч и получается завышенным.
    for iname in VIEWS:
        index_cache[VIEWS[iname][0]] = load_index(VIEWS[iname][0])
    slugs = sorted({s for _, rows in index_cache.values() for s in rows if s})
    column = {s: i for i, s in enumerate(slugs)}

    pair_scores: dict[tuple[str, str], np.ndarray] = {}
    for iname, qname in itertools.product(VIEWS, VIEWS):
        index_name, _ = VIEWS[iname]
        _, steps = VIEWS[qname]
        if index_name not in index_cache:
            index_cache[index_name] = load_index(index_name)
        vectors, row_slugs = index_cache[index_name]
        if steps not in query_cache:
            query_cache[steps] = embed_paths(paths, batch_size=32,
                                             progress_every=0, steps=steps)
            print(f"запросы, вид {qname}: посчитаны", flush=True)
        qvec, kept = query_cache[steps]
        if kept_ref is None:
            kept_ref = kept
            missing = {gold[p] for p in kept} - set(column)
            if missing:
                sys.exit(f"позиций нет в разметке: {len(missing)}")
        raw = qvec @ vectors.T
        out = np.full((len(kept), len(slugs)), -1.0, dtype=np.float32)
        for j, slug in enumerate(row_slugs):
            if slug is None or slug not in column:
                continue
            col = column[slug]
            np.maximum(out[:, col], raw[:, j], out=out[:, col])
        pair_scores[(iname, qname)] = out

    gold_col = np.array([column[gold[p]] for p in kept_ref])
    rows_idx = np.arange(len(gold_col))

    def rank_of(matrix: np.ndarray) -> np.ndarray:
        order = np.argsort(-matrix, axis=1)
        ranks = np.empty_like(order)
        ranks[rows_idx[:, None], order] = np.arange(matrix.shape[1])[None, :]
        return ranks[rows_idx, gold_col]

    # Набор видов = максимум по всем допустимым парам. Индексные виды дают
    # дополнительные строки, запросные — дополнительные прогоны модели.
    combos = []
    for iviews in (("detect",), ("detect", "raw"), ("detect", "raw", "label")):
        for qviews in (("detect",), ("detect", "raw")):
            combos.append((iviews, qviews, "все пары"))
    # Отдельно — только совпадающие пары видов: запрос-кроп против индекса-кропа,
    # запрос-кадр против индекса-кадра. Перекрёстные пары сравнивают кроп
    # с целым кадром, и если они дают шум, их стоит выключить.
    combos.append((("detect", "raw"), ("detect", "raw"), "по парам"))

    print(f"\nнабор {args.subset}, запросов {len(kept_ref)}, позиций {len(slugs)}")
    print(f"{'индекс':24} {'запрос':12} {'режим':9} " + "  ".join(f"recall@{k}" for k in args.ks))
    report = {}
    base_rank = None
    for iviews, qviews, mode in combos:
        pairs = ([(v, v) for v in iviews if v in qviews] if mode == "по парам"
                 else [(i, q) for i in iviews for q in qviews])
        matrix = np.maximum.reduce([pair_scores[p] for p in pairs])
        rank = rank_of(matrix)
        if base_rank is None:
            base_rank = rank
        values = [float((rank < k).mean()) for k in args.ks]
        report[f"{'+'.join(iviews)} / {'+'.join(qviews)} / {mode}"] = values
        cells = "  ".join(f"{v * 100:8.1f}%" for v in values)
        gained = int(((base_rank >= 20) & (rank < 20)).sum())
        lost = int(((base_rank < 20) & (rank >= 20)).sum())
        print(f"{'+'.join(iviews):24} {'+'.join(qviews):12} {mode:9} {cells}"
              f"   вернул {gained}, потерял {lost}")

    out = INDEX_DIR / f"multiview_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "n": len(kept_ref), "ks": args.ks, "recall": report,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
