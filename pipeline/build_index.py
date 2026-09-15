#!/usr/bin/env python3
"""S1.3 — векторный индекс каталога. Два варианта для сравнения.

clean — только эталонные фото, привязанные к позициям каталога (по одному
        на позицию). Индекс «вычищен»: каждый вектор отвечает за конкретный slug.
dirty — все оригиналы из дампа Strapi. Кроме эталонов там лежат фото
        виноградников, гроздей, блюд, людей, логотипов и иллюстраций к статьям.
        Такие векторы не отвечают ни за какую позицию и могут только увести
        выдачу — это и есть «мусор».

Сравнение двух индексов отвечает на вопрос: нужна ли отдельная чистка дампа
перед индексацией, или достаточно залить uploads целиком.
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
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
MAPPING = ROOT / "data" / "catalog" / "mapping.csv"
OUT_DIR = ROOT / "data" / "index"
PREVIEW_PREFIXES = ("thumbnail_", "small_", "medium_", "large_")


def uploads_dir() -> Path:
    found = next((ROOT / "dataset" / "uploads_root").rglob("uploads"), None)
    if found is None:
        sys.exit("не найдена папка uploads")
    return found


def collect(variant: str) -> tuple[list[str], list[str | None]]:
    """Список файлов индекса и slug каждого файла (None — мусор)."""
    catalog = json.loads(CATALOG.read_text())["wines"]
    file2slug = {}
    for wine in catalog:
        if wine.get("photo"):
            file2slug[Path(wine["photo"]).name] = wine["slug"]

    if variant == "clean":
        updir = uploads_dir()
        paths = [str(updir / name) for name in file2slug]
        return paths, [file2slug[Path(p).name] for p in paths]

    updir = uploads_dir()
    names = [n for n in sorted(updir.iterdir())
             if n.is_file() and not n.name.startswith(PREVIEW_PREFIXES)]
    paths = [str(p) for p in names]
    return paths, [file2slug.get(p.name) for p in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "dirty"], required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", default="", help="шаги нормализации через запятую")
    ap.add_argument("--name", default="", help="имя индекса (по умолчанию = вариант)")
    args = ap.parse_args()
    steps = tuple(s for s in args.steps.split(",") if s)
    name = args.name or args.variant

    paths, slugs = collect(args.variant)
    garbage = sum(1 for s in slugs if s is None)
    print(f"индекс {name} (шаги: {','.join(steps) or 'нет'}): файлов {len(paths)}, "
          f"из них без привязки к каталогу (мусор): {garbage}")

    t0 = time.time()
    vectors, kept = embed_paths(paths, batch_size=args.batch_size, steps=steps)
    kept_slugs = [slugs[paths.index(p)] for p in kept] if len(kept) != len(paths) else slugs
    print(f"эмбеддингов: {len(kept)} за {time.time() - t0:.0f} с, размерность {vectors.shape[1]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}.npy", vectors)
    with (OUT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "slug"])
        for path, slug in zip(kept, kept_slugs):
            writer.writerow([str(Path(path).relative_to(ROOT)), slug or ""])
    print(f"сохранено: {OUT_DIR / (name + '.npy')}")


if __name__ == "__main__":
    main()
