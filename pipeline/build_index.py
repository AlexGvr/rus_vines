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
    """Строки индекса: путь к файлу и slug позиции (None — мусор).

    Единица индекса — позиция каталога, а не файл. Связь не взаимно
    однозначная: четыре фото платформа повесила на две карточки каждое
    (два винтажа или два сорта одной винодельни). Группировка по файлу
    оставила бы от такой пары один slug, и вторая позиция выпала бы из
    поиска целиком. Поэтому путь в списке может повторяться.
    """
    catalog = json.loads(CATALOG.read_text())["wines"]
    updir = uploads_dir()

    if variant == "clean":
        rows = [(str(updir / Path(w["photo"]).name), w["slug"])
                for w in catalog if w.get("photo")]
        return [p for p, _ in rows], [s for _, s in rows]

    # В dirty строка — файл дампа, поэтому обратная свёртка неизбежна:
    # общему фото достаётся первый по каталогу slug.
    file2slug: dict[str, str] = {}
    for wine in catalog:
        if wine.get("photo"):
            file2slug.setdefault(Path(wine["photo"]).name, wine["slug"])
    names = [n for n in sorted(updir.iterdir())
             if n.is_file() and not n.name.startswith(PREVIEW_PREFIXES)]
    paths = [str(p) for p in names]
    return paths, [file2slug.get(p.name) for p in names]


def embed_view(paths: list[str], slugs: list[str | None], steps: tuple[str, ...],
               batch_size: int) -> tuple[np.ndarray, list[tuple[str, str | None]]]:
    """Один вид индекса: вектор на каждую строку и её (путь, slug)."""
    # Считаем каждый файл один раз, даже если на него ссылаются две позиции:
    # эмбеддинг зависит только от картинки.
    unique = list(dict.fromkeys(paths))
    t0 = time.time()
    vectors, kept = embed_paths(unique, batch_size=batch_size, steps=steps)
    # Битые файлы embed_paths пропускает молча, поэтому строки собираем по
    # тем путям, для которых вектор действительно посчитан.
    row_of_path = {path: i for i, path in enumerate(kept)}
    rows = [(path, slug) for path, slug in zip(paths, slugs) if path in row_of_path]
    print(f"  вид «{'+'.join(steps) or 'кадр целиком'}»: эмбеддингов {len(kept)} "
          f"за {time.time() - t0:.0f} с, строк {len(rows)}")
    return vectors[[row_of_path[path] for path, _ in rows]], rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["clean", "dirty"], required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", default="", help="шаги нормализации через запятую")
    ap.add_argument("--views", default="",
                    help="несколько видов через запятую, шаги внутри вида через '+': "
                         "'raw,detect' даст кадр целиком и кроп по бутылке. Позиция "
                         "получает вектор на каждый вид, выдача сворачивается по slug")
    ap.add_argument("--name", default="", help="имя индекса (по умолчанию = вариант)")
    args = ap.parse_args()
    if args.views:
        views = [tuple(s for s in spec.split("+") if s and s != "raw")
                 for spec in args.views.split(",")]
    else:
        views = [tuple(s for s in args.steps.split(",") if s)]
    name = args.name or args.variant

    paths, slugs = collect(args.variant)
    garbage = sum(1 for s in slugs if s is None)
    print(f"индекс {name}: видов {len(views)}, строк на вид {len(paths)}, "
          f"из них без привязки к каталогу (мусор): {garbage}")

    blocks, rows = [], []
    for steps in views:
        block, block_rows = embed_view(paths, slugs, steps, args.batch_size)
        blocks.append(block)
        rows.extend((path, slug, "+".join(steps) or "raw") for path, slug in block_rows)
    vectors = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    print(f"строк индекса: {len(rows)}, размерность {vectors.shape[1]}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{name}.npy", vectors)
    with (OUT_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "slug", "view"])
        for path, slug, view in rows:
            writer.writerow([str(Path(path).relative_to(ROOT)), slug or "", view])
    print(f"сохранено: {OUT_DIR / (name + '.npy')}")


if __name__ == "__main__":
    main()
