#!/usr/bin/env python3
"""Проверка эталонов каталога на устаревание.

Повод. На полевом наборе девять из десяти неудачных кадров пришлись на одно
вино — «Абрау Купаж тёмный», у которого эталон каталога показывает упаковку
другого поколения. Отсюда вопрос: это единичный случай или каталог в целом
отстал от полки. Отвечать на него догадкой нельзя, поэтому берётся выборка
позиций и для каждой сравнивается эталон с тем, что сейчас в продаже.

Источник «как сейчас» — товарная страница производителя или магазина.
Адреса перечислены в data/freshness.csv и добывались руками: устойчивого
способа найти актуальное фото по названию нет, а ошибка тут дороже, чем
ручная работа на десяток позиций.

Численная оценка совпадения — инлаеры SIFT между эталоном и текущим фото,
но доверять ей одной нельзя: у «Нового Света» два студийных снимка одной
и той же бутылки дали всего десять инлаеров из-за разного света и кадра.
Поэтому скрипт печатает числа и собирает лист для просмотра глазами,
а вывод делает человек.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from embed import embed_images  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import descriptors, match_stats  # noqa: E402

CATALOG = ROOT / "data" / "catalog" / "catalog.json"
SOURCES = ROOT / "data" / "freshness.csv"
CACHE = ROOT / "data" / "freshness"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")


def download(url: str, path: Path) -> bool:
    if path.exists():
        return True
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read()
    except Exception as error:
        print(f"  не скачалось: {error}")
        return False
    if len(data) < 8000:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(SOURCES))
    ap.add_argument("--sheet", default=str(ROOT / "data" / "freshness" / "sheet.png"))
    args = ap.parse_args()

    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
    rows = list(csv.DictReader(Path(args.sources).open(encoding="utf-8")))

    results, sheet_rows = [], []
    for row in rows:
        wine = wines.get(row["slug"])
        if wine is None:
            sys.exit(f"нет такой позиции в каталоге: {row['slug']}")
        current = CACHE / f"{row['slug'][:40]}_{row['source']}.img"
        if not download(row["url"], current):
            continue

        reference = normalize_batch(
            [Image.open(ROOT / wine["photo"]).convert("RGB")], steps=("detect",))[0]
        shelf = normalize_batch(
            [Image.open(current).convert("RGB")], steps=("detect",))[0]
        ref_kp, ref_desc = descriptors(reference)
        shelf_kp, shelf_desc = descriptors(shelf)
        stats = match_stats(shelf_kp, shelf_desc, ref_kp, ref_desc)
        vectors = embed_images([reference, shelf])
        cosine = float(vectors[0] @ vectors[1])

        results.append({"slug": row["slug"], "title": wine["title"],
                        "source": row["source"], "url": row["url"],
                        "inliers": stats.inliers, "cosine": round(cosine, 4)})
        sheet_rows.append((wine, current, stats.inliers, cosine))
        print(f"{wine['title'][:34]:34} {row['source']:12} "
              f"инлаеры {stats.inliers:>4}  косинус {cosine:.3f}", flush=True)

    if sheet_rows:
        font = ImageFont.truetype(FONT, 14)
        cell = 250
        sheet = Image.new("RGB", (cell * 2 + 300, cell * len(sheet_rows) + 16), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (wine, current, inliers, cosine) in enumerate(sheet_rows):
            y = i * cell + 8
            draw.text((6, y + cell // 2 - 24), wine["title"][:28], fill="black", font=font)
            draw.text((6, y + cell // 2 - 4), wine["manufacturer"][:28],
                      fill="#777", font=font)
            draw.text((6, y + cell // 2 + 16),
                      f"инлаеры {inliers}, косинус {cosine:.3f}", fill="#a00", font=font)
            for j, (path, label) in enumerate(((ROOT / wine["photo"], "каталог"),
                                               (current, "в продаже"))):
                image = Image.open(path).convert("RGB")
                image.thumbnail((cell - 16, cell - 16))
                sheet.paste(image, (300 + j * cell + (cell - image.width) // 2,
                                    y + (cell - image.height) // 2))
                draw.text((300 + j * cell + 4, y + 2), label, fill="#a00", font=font)
        Path(args.sheet).parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.sheet)
        print(f"\nлист для просмотра: {args.sheet}")

    out = ROOT / "data" / "index" / "freshness.json"
    out.write_text(json.dumps({"checked": len(results), "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
