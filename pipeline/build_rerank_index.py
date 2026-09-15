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
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerank import descriptors  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
OUT_DIR = ROOT / "data" / "index" / "sift"


def main() -> None:
    wines = [w for w in json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
             if w.get("photo")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_desc, all_kpts, offsets, slugs = [], [], [0], []
    t0 = time.time()
    for n, wine in enumerate(wines, 1):
        try:
            with Image.open(ROOT / wine["photo"]) as im:
                keypoints, desc = descriptors(im)
        except Exception:
            keypoints, desc = [], None
        if desc is None or len(desc) == 0:
            desc = np.zeros((0, 128), dtype=np.float32)
            points = np.zeros((0, 2), dtype=np.float32)
        else:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        all_desc.append(np.clip(desc, 0, 255).astype(np.uint8))
        all_kpts.append(points)
        offsets.append(offsets[-1] + len(desc))
        slugs.append(wine["slug"])
        if n % 400 == 0:
            print(f"  {n}/{len(wines)}", flush=True)

    np.save(OUT_DIR / "desc.npy", np.concatenate(all_desc) if all_desc
            else np.zeros((0, 128), np.uint8))
    np.save(OUT_DIR / "kpts.npy", np.concatenate(all_kpts) if all_kpts
            else np.zeros((0, 2), np.float32))
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
