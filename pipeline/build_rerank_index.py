#!/usr/bin/env python3
"""Предрасчёт локальных признаков эталонов.

Без него первый запрос к позиции читает её фото с диска и считает SIFT прямо
в обработчике: замер показал до 2.8 с на холодном кэше при SLA в 3 с.
Признаки эталонов не меняются между запросами, поэтому считаются один раз.

Формат хранения — три массива подряд с таблицей смещений: дескрипторы
кладём в uint8 (SIFT и так лежит в 0..255), иначе каталог занял бы больше
гигабайта вместо трёхсот мегабайт.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank import descriptors, keypoint_colors  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
OUT_DIR = Path(os.environ.get("SIFT_DIR", ROOT / "data" / "index" / "sift"))


def main() -> None:
    wines = [w for w in json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
             if w.get("photo")]
    # Дополнительные эталоны идут отдельными записями с тем же slug:
    # реранкер берёт лучшее совпадение среди всех записей позиции.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_index import extra_references
    wines += [{"slug": slug, "photo": path} for path, slug in extra_references()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_desc, all_kpts, all_colors, offsets, slugs = [], [], [], [0], []
    t0 = time.time()
    for n, wine in enumerate(wines, 1):
        try:
            with Image.open(ROOT / wine["photo"]) as im:
                image = im.convert("RGB")
                keypoints, desc = descriptors(image)
        except Exception:
            image, keypoints, desc = None, [], None
        if desc is None or len(desc) == 0:
            desc = np.zeros((0, 128), dtype=np.float32)
            points = np.zeros((0, 2), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)
        else:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            # Цвет вокруг точки: SIFT его не видит, а соседей по линейке
            # он часто и разводит — синяя обёртка против золотой.
            colors = keypoint_colors(image, keypoints)
        all_desc.append(np.clip(desc, 0, 255).astype(np.uint8))
        all_kpts.append(points)
        all_colors.append(colors)
        offsets.append(offsets[-1] + len(desc))
        slugs.append(wine["slug"])
        if n % 400 == 0:
            print(f"  {n}/{len(wines)}", flush=True)

    np.save(OUT_DIR / "desc.npy", np.concatenate(all_desc) if all_desc
            else np.zeros((0, 128), np.uint8))
    np.save(OUT_DIR / "kpts.npy", np.concatenate(all_kpts) if all_kpts
            else np.zeros((0, 2), np.float32))
    np.save(OUT_DIR / "colors.npy", np.concatenate(all_colors) if all_colors
            else np.zeros((0, 3), np.uint8))
    np.save(OUT_DIR / "offsets.npy", np.array(offsets, dtype=np.int64))
    with (OUT_DIR / "slugs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["slug"])
        writer.writerows([[s] for s in slugs])

    size_mb = sum(p.stat().st_size for p in OUT_DIR.iterdir()) / 1e6
    print(f"эталонов {len(slugs)}, точек {offsets[-1]}, "
          f"{size_mb:.0f} МБ за {time.time() - t0:.0f} с → {OUT_DIR}")


if __name__ == "__main__":
    main()
