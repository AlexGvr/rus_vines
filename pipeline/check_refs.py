#!/usr/bin/env python3
"""Позиции каталога, которых нет в поисковом индексе.

Позиция без эталонной фотографии не индексируется и не находится никогда,
при любом качестве поиска. В отчётах такие кадры легко пропадают: они
выглядят как обычные промахи, и их пытаются чинить алгоритмом. Список
нужен, чтобы отделить дефект данных от дефекта поиска.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from build_index import extra_references  # noqa: E402

CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def main() -> None:
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    uploads = next((ROOT / "dataset" / "uploads_root").rglob("uploads"))
    extra = {slug for _, slug in extra_references()}
    missing = []
    for wine in wines:
        photo = wine.get("photo")
        if wine["slug"] in extra:
            continue
        if not photo or not (uploads / Path(photo).name).exists():
            missing.append(wine)

    print(f"позиций без эталона в индексе: {len(missing)} из {len(wines)}")
    for wine in missing:
        print(f"  {wine['slug'][:52]:52} {wine['title'][:34]:34} "
              f"{(wine.get('manufacturer') or '')[:20]}")
    if extra:
        print(f"\nэталон добавлен вручную (data/catalog/extra_refs.csv): "
              f"{len(extra)} позиций")


if __name__ == "__main__":
    main()
