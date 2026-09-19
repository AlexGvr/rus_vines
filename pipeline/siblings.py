#!/usr/bin/env python3
"""Соседи по линейке: насколько вообще их различает эмбеддинг.

Вопрос, на который отвечает скрипт, один: стоит ли дообучать модель.
Все дешёвые приёмы на соседях по линейке уже проверены и не сработали —
взвешивание совпавших точек по зонам этикетки, цвет вокруг совпавших
точек, чтение различающей надписи, дополнительные эталоны. Прежде чем
браться за дообучение, надо убедиться, что дело в самом представлении,
а не в том, как им пользуются.

Соседями считаются позиции одной винодельни, названия которых после
вычёркивания слов о цвете и сладости совпадают. Это и есть та пара,
которую покупатель различает по одной строке мелким шрифтом: «Русское
Игристое полусладкое красное» против «Русское Игристое полусухое».

Печатает три числа, и смысл в их сравнении:

  внутри линейки   средний косинус между эталонами соседей
  та же винодельня средний косинус с позициями той же винодельни вне линейки
  чужая винодельня средний косинус со случайными позициями

Если первое близко к единице, а разброс внутри линейки меньше разницы
между ракурсами одного вина, то никакое правило поверх готовых векторов
соседей не разведёт.

Заодно выкладывает пары в CSV — готовый набор сложных отрицательных
примеров для дообучения.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

INDEX_DIR = ROOT / "data" / "index"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"

# Слова, которыми соседи по линейке и различаются. Из ключа группировки
# они вычёркиваются: иначе каждая позиция окажется в своей группе.
VARIANT_WORDS = {
    "белое", "красное", "розовое", "оранжевое", "белый", "красный", "розовый",
    "сухое", "полусухое", "полусладкое", "сладкое", "брют", "брут",
    "экстра", "натюр", "кошерное", "выдержанное", "молодое",
    "blanc", "rose", "rosso", "bianco", "brut", "extra", "demi", "sec",
    "dry", "sweet", "nature",
}


def series_key(wine: dict) -> str:
    """Название без слов о цвете и сладости плюс винодельня."""
    title = re.sub(r"[^\w\s-]", " ", (wine.get("title") or "").lower())
    words = [w for w in title.split() if w not in VARIANT_WORDS and len(w) > 2]
    return f"{(wine.get('manufacturer') or '').lower()}|{' '.join(words)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--out", default=str(ROOT / "data" / "catalog" / "siblings.csv"))
    ap.add_argument("--sample", type=int, default=4000,
                    help="сколько случайных пар взять для фона")
    args = ap.parse_args()

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    rows = list(csv.DictReader((INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8")))
    # Берём один вид — целый кадр: сравниваем представления позиций,
    # а не действие кадрирования.
    whole = {row["slug"]: i for i, row in enumerate(rows)
             if row.get("view", "raw") == "raw" and row["slug"]}

    groups: dict[str, list[str]] = defaultdict(list)
    for wine in wines:
        if wine["slug"] in whole:
            groups[series_key(wine)].append(wine["slug"])
    series = {key: slugs for key, slugs in groups.items() if len(slugs) > 1}

    by_brand: dict[str, list[str]] = defaultdict(list)
    for slug in whole:
        by_brand[(by_slug[slug].get("manufacturer") or "").lower()].append(slug)

    def cosine(a: str, b: str) -> float:
        return float(vectors[whole[a]] @ vectors[whole[b]])

    inside, brand, stranger = [], [], []
    pairs = []
    for key, slugs in series.items():
        for i, left in enumerate(slugs):
            for right in slugs[i + 1:]:
                value = cosine(left, right)
                inside.append(value)
                pairs.append((left, right, round(value, 4), key.split("|")[0]))

    rng = random.Random(17)
    keys = list(series)
    for _ in range(args.sample):
        key = rng.choice(keys)
        left = rng.choice(series[key])
        family = by_brand[(by_slug[left].get("manufacturer") or "").lower()]
        outside = [s for s in family if s not in series[key]]
        if outside:
            brand.append(cosine(left, rng.choice(outside)))
        other = rng.choice(list(whole))
        if (by_slug[other].get("manufacturer") or "").lower() != \
           (by_slug[left].get("manufacturer") or "").lower():
            stranger.append(cosine(left, other))

    def describe(name: str, values: list[float]) -> None:
        if not values:
            print(f"{name:26} нет пар")
            return
        arr = np.array(values)
        print(f"{name:26} пар {len(arr):>6}  средний {arr.mean():.3f}  "
              f"медиана {np.median(arr):.3f}  "
              f"90-й процентиль {np.quantile(arr, 0.9):.3f}")

    print(f"линеек с двумя и более позициями: {len(series)}, "
          f"позиций в них {sum(len(v) for v in series.values())}")
    describe("внутри линейки", inside)
    describe("та же винодельня", brand)
    describe("чужая винодельня", stranger)

    pairs.sort(key=lambda p: -p[2])
    print("\nсамые неразличимые пары:")
    for left, right, value, brand_name in pairs[:12]:
        print(f"  {value:.3f}  {by_slug[left]['title'][:34]:34} | "
              f"{by_slug[right]['title'][:34]:34} | {brand_name[:18]}")

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slug_a", "slug_b", "cosine", "manufacturer"])
        writer.writerows(pairs)
    print(f"\nсложные отрицательные пары: {out} ({len(pairs)} строк)")


if __name__ == "__main__":
    main()
