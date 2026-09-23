#!/usr/bin/env python3
"""Фото первого MVP (app/, mobile/) — из файлов датасета кейса.

Раньше `etl/scrape.py` скачивал их с ресайз-прокси платформы. По
рекомендации организатора решение использует только изображения из
датасета кейса, поэтому фото пересобираются из эталонов дампа Strapi
(`data/catalog/catalog.json`, поле photo): вписываются в 400×400, как
делал прокси, и пишутся в webp. У позиций без эталона в дампе фото нет,
и в `wines.json` приложений поле photo становится null.

    python etl/photos_from_dataset.py [--out-root DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
TARGETS = (("app/photos", "app/data/wines.json"),
           ("mobile/assets/photos", "mobile/assets/wines.json"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT), help="куда писать (по умолчанию — репозиторий)")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    dump = {w["slug"]: w["photo"] for w in json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
            if (w.get("photo") or "").startswith("dataset/")}
    for photos_rel, data_rel in TARGETS:
        data_path = ROOT / data_rel
        data = json.loads(data_path.read_text(encoding="utf-8"))
        photos = out_root / photos_rel
        photos.mkdir(parents=True, exist_ok=True)
        for old in photos.glob("*.webp"):
            old.unlink()
        made = missing = 0
        for wine in data["wines"]:
            src = dump.get(wine["slug"])
            if not src:
                wine["photo"] = None
                missing += 1
                continue
            im = Image.open(ROOT / src).convert("RGBA")
            im.thumbnail((400, 400), Image.LANCZOS)
            im.save(photos / f"{wine['slug']}.webp", "WEBP", quality=82, method=6)
            wine["photo"] = f"photos/{wine['slug']}.webp"
            made += 1
        out_data = out_root / data_rel
        out_data.parent.mkdir(parents=True, exist_ok=True)
        out_data.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                            encoding="utf-8")
        print(f"{photos_rel}: из датасета {made}, без эталона в дампе {missing}")


if __name__ == "__main__":
    main()
