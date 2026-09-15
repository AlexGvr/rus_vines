#!/usr/bin/env python3
"""S0.6 — валидационный набор «полевых» запросов с известными ответами.

Правильные ответы кейса закрыты, публичных фото три штуки, поэтому
собственный набор строится из эталонных фото каталога: каждое проводится
через цепочку искажений, повторяющих съёмку у полки — перспектива, блик,
мелкий масштаб, посторонние бутылки в кадре, шум, слабый свет, JPEG.

Набор нужен прежде всего для сравнения вариантов пайплайна между собой:
запросы одинаковые, значит разница в метрике — это разница в алгоритме.
Абсолютные числа на нём оптимистичнее реальных: искажения синтетические.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFile, ImageFilter

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
OUT_DIR = ROOT / "data" / "queries"
CANVAS = (900, 1200)


def perspective(im: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    """Наклон бутылки: слабое проективное искажение через quad-трансформ.

    PIL ждёт исходный четырёхугольник в порядке NW, SW, SE, NE — при другом
    порядке кадр выворачивает наизнанку.
    """
    w, h = im.size
    d = max(1, int(min(w, h) * strength))
    nw = (rng.randint(0, d), rng.randint(0, d))
    sw = (rng.randint(0, d), h - rng.randint(0, d))
    se = (w - rng.randint(0, d), h - rng.randint(0, d))
    ne = (w - rng.randint(0, d), rng.randint(0, d))
    quad = [c for corner in (nw, sw, se, ne) for c in corner]
    return im.transform((w, h), Image.QUAD, quad, Image.BICUBIC)


def add_glare(im: Image.Image, rng: random.Random) -> Image.Image:
    """Блик от витринного света: светлое пятно поверх стекла."""
    glare = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(glare)
    w, h = im.size
    cx, cy = rng.randint(0, w), rng.randint(0, h)
    rx, ry = rng.randint(w // 8, w // 3), rng.randint(h // 8, h // 3)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=rng.randint(90, 190))
    glare = glare.filter(ImageFilter.GaussianBlur(min(w, h) // 12))
    return Image.composite(Image.new("RGB", im.size, "white"), im, glare)


def shelf_background(neighbours: list[Path], rng: random.Random) -> Image.Image:
    """Фон полки: соседние бутылки по краям кадра, как в реальной съёмке."""
    bg = Image.new("RGB", CANVAS, (rng.randint(24, 60),) * 3)
    for side in (0, 1):
        if not neighbours or rng.random() < 0.25:
            continue
        try:
            with Image.open(rng.choice(neighbours)) as nb:
                nb = nb.convert("RGB")
                # Соседи ниже целевой бутылки и уходят за край: в кадре видна
                # только их часть, иначе они перетягивают внимание модели.
                scale = CANVAS[1] * rng.uniform(0.55, 0.8) / max(nb.height, 1)
                nb = nb.resize((max(1, int(nb.width * scale)), max(1, int(nb.height * scale))))
                x = -nb.width * 0.55 if side == 0 else CANVAS[0] - nb.width * 0.45
                bg.paste(nb, (int(x), rng.randint(CANVAS[1] // 5, CANVAS[1] // 2)))
        except Exception:
            continue
    return bg


def make_query(ref: Path, neighbours: list[Path], rng: random.Random,
               hard: bool) -> Image.Image | None:
    try:
        with Image.open(ref) as raw:
            bottle = raw.convert("RGB")
    except Exception:
        return None

    # Целевая бутылка остаётся главным объектом кадра — пользователь наводит
    # камеру именно на неё; соседи лишь попадают по краям.
    scale_range = (0.62, 0.86) if hard else (0.78, 0.95)
    target_h = int(CANVAS[1] * rng.uniform(*scale_range))
    ratio = target_h / max(bottle.height, 1)
    bottle = bottle.resize((max(1, int(bottle.width * ratio)), target_h), Image.LANCZOS)
    bottle = perspective(bottle, rng, 0.06 if hard else 0.03)

    canvas = shelf_background(neighbours, rng)
    canvas.paste(bottle, ((CANVAS[0] - bottle.width) // 2 + rng.randint(-45, 45),
                          (CANVAS[1] - bottle.height) // 2 + rng.randint(-40, 40)))

    canvas = canvas.rotate(rng.uniform(-7, 7) if hard else rng.uniform(-3, 3),
                           resample=Image.BICUBIC, fillcolor=(30, 30, 30))
    if rng.random() < (0.7 if hard else 0.35):
        canvas = add_glare(canvas, rng)
    canvas = ImageEnhance.Brightness(canvas).enhance(
        rng.uniform(0.55, 1.15) if hard else rng.uniform(0.8, 1.1))
    canvas = ImageEnhance.Color(canvas).enhance(rng.uniform(0.75, 1.2))
    if rng.random() < (0.55 if hard else 0.25):
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(0.6, 1.8)))
    return canvas


def near_duplicate_slugs(catalog: list[dict]) -> set[str]:
    """Позиции, у которых есть тёзка той же винодельни."""
    groups: dict[tuple[str, str], list[str]] = {}
    for wine in catalog:
        key = (wine["manufacturer"].strip().lower(), wine["title"].strip().lower())
        groups.setdefault(key, []).append(wine["slug"])
    return {slug for slugs in groups.values() if len(slugs) > 1 for slug in slugs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=400, help="сколько позиций взять")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--hard", action="store_true", help="жёсткий режим искажений")
    ap.add_argument("--name", default="", help="имя набора (папка и манифест)")
    ap.add_argument("--repeats", type=int, default=1, help="кадров на позицию")
    ap.add_argument("--min-side", type=int, default=0,
                    help="брать только эталоны с длинной стороной не меньше N: "
                         "на мелких фото текст этикетки не переживает пересъёмку "
                         "и замер текстового канала получается заниженным")
    ap.add_argument("--only-neardup", action="store_true",
                    help="только позиции, у которых в каталоге есть тёзка")
    ap.add_argument("--canvas", default="900x1200", help="размер кадра запроса, ШxВ")
    args = ap.parse_args()

    global CANVAS
    CANVAS = tuple(int(v) for v in args.canvas.lower().split("x"))

    catalog = json.loads(CATALOG.read_text())["wines"]
    with_photo = [w for w in catalog if w.get("photo") and (ROOT / w["photo"]).exists()]

    if args.min_side:
        kept = []
        for wine in with_photo:
            try:
                with Image.open(ROOT / wine["photo"]) as im:
                    if max(im.size) >= args.min_side:
                        kept.append(wine)
            except Exception:
                continue
        with_photo = kept
    if args.only_neardup:
        near = near_duplicate_slugs(catalog)
        with_photo = [w for w in with_photo if w["slug"] in near]

    if not with_photo:
        raise SystemExit("под заданные условия не нашлось ни одной позиции")

    rng = random.Random(args.seed)
    picked = rng.sample(with_photo, min(args.count, len(with_photo)))
    pool = [ROOT / w["photo"] for w in rng.sample(with_photo, min(200, len(with_photo)))]

    name = args.name or ("hard" if args.hard else "easy")
    out_img = OUT_DIR / name
    out_img.mkdir(parents=True, exist_ok=True)
    for old in out_img.glob("*.jpg"):
        old.unlink()

    manifest = []
    for i, wine in enumerate(picked):
        for rep in range(args.repeats):
            img = make_query(ROOT / wine["photo"], pool, rng, args.hard)
            if img is None:
                continue
            filename = f"q{i:04d}_{rep}.jpg"
            img.save(out_img / filename, quality=rng.randint(62, 90))
            manifest.append({"query_id": f"q-{i:06d}-{rep}", "image": filename,
                             "slug": wine["slug"]})

    path = OUT_DIR / f"manifest_{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["query_id", "image", "slug"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"позиций {len(picked)}, запросов {len(manifest)} → {out_img}")
    print(f"манифест: {path}")


if __name__ == "__main__":
    main()
