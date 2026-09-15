#!/usr/bin/env python3
"""S5.3 — отбор кандидатов упирается в детектор, а не в ёмкость модели.

Разбор глубоких промахов (`recall_sharp.json`) показал: у запросов, где
правильный slug оказался ниже сотого места, детектор либо не нашёл бутылку
вовсе, либо выбрал соседнюю — в кроп уходила полоска шириной 134 пикселя
с чужой этикеткой. Расширять шортлист в такой ситуации бесполезно: вектор
описывает не то вино.

Скрипт прогоняет запросы через разные настройки кадрирования и сравнивает
полноту шортлиста по одному и тому же индексу. Индекс здесь намеренно
зафиксирован: эталон — студийный снимок одной бутылки, настройки детектора
на нём почти ничего не меняют, а сравнивать варианты нужно на одной шкале.
Победивший вариант потом проверяется честно, с пересборкой индекса.

Варианты кадрирования:
  full     — кадр целиком, без детекции (нижняя граница)
  single   — текущее поведение: одна бутылка, ранг score·√площадь·центральность
  tuned    — то же, но детектор видит кадр крупнее и порог ниже
  edge     — tuned плюс штраф боксам, обрезанным краем кадра: соседи по полке
             почти всегда срезаны, целевая бутылка — почти никогда
  union    — кандидаты объединяются по нескольким найденным бутылкам: если
             целевая among них, она получает свой шанс независимо от того,
             какая выиграла ранжирование
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
import imageprep  # noqa: E402
from embed import embed_images  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"


def boxes_for(images: list[Image.Image], conf: float, imgsz: int,
              edge_penalty: bool, limit: int) -> list[list[imageprep.Box]]:
    """До `limit` боксов на кадр, по убыванию правдоподобия «это целевая бутылка»."""
    model = imageprep.load_detector()
    out: list[list[imageprep.Box]] = []
    for i in range(0, len(images), 16):
        chunk = [np.array(im.convert("RGB"))[:, :, ::-1] for im in images[i:i + 16]]
        results = model.predict(chunk, classes=[imageprep.BOTTLE_CLASS], conf=conf,
                                imgsz=imgsz, verbose=False, device=imageprep._device())
        for im, res in zip(images[i:i + 16], results):
            width, height = im.size
            scored = []
            for box, score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = (int(v) for v in box)
                cand = imageprep.Box(x1, y1, x2, y2)
                rel_area = cand.area / max(width * height, 1)
                cx = (x1 + x2) / 2 / max(width, 1)
                centrality = 1.0 - min(abs(cx - 0.5) * 2, 1.0)
                rank = score * (rel_area ** 0.5) * (0.35 + 0.65 * centrality)
                if edge_penalty:
                    # Только горизонтальная срезка: целевая бутылка часто занимает
                    # кадр по высоте целиком, и штраф за верх и низ бил бы по ней.
                    margin = max(1, int(width * 0.01))
                    cut = (x1 <= margin) + (x2 >= width - margin)
                    rank *= (1.0, 0.7, 0.45)[cut]
                    # Проверка формы: у бутылки в кадре высота больше ширины,
                    # но не в разы. Полоска 134×1015 — срез соседней бутылки.
                    ratio = (y2 - y1) / max(x2 - x1, 1)
                    if ratio > 5.0:
                        rank *= 0.3
                    elif ratio < 1.2:
                        rank *= 0.5
                scored.append((rank, cand))
            scored.sort(key=lambda r: -r[0])
            out.append([c for _, c in scored[:limit]])
    return out


VARIANTS = {
    "full":   dict(conf=0.20, imgsz=640,  edge=False, limit=0),
    "single": dict(conf=0.20, imgsz=640,  edge=False, limit=1),
    "tuned":  dict(conf=0.10, imgsz=1024, edge=False, limit=1),
    "edge":   dict(conf=0.10, imgsz=1024, edge=True,  limit=1),
    "union":  dict(conf=0.10, imgsz=1024, edge=True,  limit=3),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_detect")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20, 50])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    if args.limit:
        manifest = manifest[:args.limit]
    rows = [(QUERIES / args.subset / r["image"], r["slug"]) for r in manifest]
    rows = [(p, s) for p, s in rows if p.exists()]

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    slugs = sorted({s for s in index_slugs if s})
    column = {s: i for i, s in enumerate(slugs)}
    gold_col = np.array([column[s] for _, s in rows if s in column])
    if len(gold_col) != len(rows):
        sys.exit("в индексе нет части правильных ответов — пересоберите индекс")

    results = {}
    for name in args.variants:
        cfg = VARIANTS[name]
        best = np.full((len(rows), len(slugs)), -1.0, dtype=np.float32)
        t0 = time.time()
        for start in range(0, len(rows), 32):
            chunk = [Image.open(p).convert("RGB") for p, _ in rows[start:start + 32]]
            if cfg["limit"] == 0:
                views = [[im] for im in chunk]
            else:
                found = boxes_for(chunk, cfg["conf"], cfg["imgsz"],
                                  cfg["edge"], cfg["limit"])
                views = [[imageprep.crop_box(im, b) for b in bs] or [im]
                         for im, bs in zip(chunk, found)]
            flat = [v for group in views for v in group]
            qvec = embed_images(flat, batch_size=32)
            # Позиция получает лучший косинус среди видов кадра: объединение
            # идёт по slug, иначе один вид занял бы весь шортлист.
            cursor = 0
            for row, group in enumerate(views, start=start):
                take = qvec[cursor:cursor + len(group)]
                cursor += len(group)
                raw = (take @ vectors.T).max(axis=0)
                for j, slug in enumerate(index_slugs):
                    if slug is not None:
                        col = column[slug]
                        if raw[j] > best[row, col]:
                            best[row, col] = raw[j]
        order = np.argsort(-best, axis=1)
        rank = np.array([int(np.where(order[i] == gold_col[i])[0][0])
                         for i in range(len(rows))])
        results[name] = (rank, time.time() - t0)
        cells = "  ".join(f"{float((rank < k).mean()) * 100:8.1f}%" for k in args.ks)
        print(f"{name:8} {cells}   {(time.time() - t0) / len(rows) * 1000:6.0f} мс/запрос",
              flush=True)

    print(f"\nнабор {args.subset}, запросов {len(rows)}, индекс {args.index}")
    print(f"{'вариант':8} " + "  ".join(f"recall@{k}" for k in args.ks))
    for name, (rank, _) in results.items():
        cells = "  ".join(f"{float((rank < k).mean()) * 100:8.1f}%" for k in args.ks)
        print(f"{name:8} {cells}")

    if "single" in results:
        base = results["single"][0]
        for name, (rank, _) in results.items():
            if name == "single":
                continue
            gained = int(((base >= 20) & (rank < 20)).sum())
            lost = int(((base < 20) & (rank >= 20)).sum())
            print(f"{name} против single на k=20: вернул {gained}, потерял {lost}")

    out = INDEX_DIR / f"detect_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "index": args.index, "n": len(rows),
        "variants": {name: {
            "config": VARIANTS[name],
            "recall": {str(k): float((rank < k).mean()) for k in args.ks},
            "ms_per_query": seconds / len(rows) * 1000,
        } for name, (rank, seconds) in results.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
