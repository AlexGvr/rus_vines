#!/usr/bin/env python3
"""Позиции, появившиеся на платформе после выгрузки кейса.

Выгрузка `strapi_output0709.csv` снята 7 сентября, а платформа прирастает
десятками позиций в день. На 23 сентября в sitemap 73 slug, которых в
выгрузке нет, — и среди них вина, снятые организатором для публичного
набора и размеченные у нас как «нет в каталоге» (Литавщук «Совиньон Блан»
полусладкое, «Каберне Совиньон» полусухое). Если закрытая таблица
размечена по текущей платформе, такие кадры сервис обязан находить.

Скрипт читает sitemap, для новых slug — HTML-страницу карточки (не /api/,
закрытый robots.txt), разбирает payload Nuxt и скачивает фото карточки.
Результат — `data/datapack/platform_additions.json` в формате скрейпа
`wines.json` плюс путь и SHA-256 фото в `data/refs/platform/`;
`build_catalog.py` добавляет эти позиции в каталог последними. Позиции,
которые с платформы убраны, из каталога не удаляются: как размечена
закрытая таблица, неизвестно, и лишний кандидат дешевле пропавшего.

    python pipeline/sync_platform.py            # запрос к платформе, ~1 с на карточку
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://vino-svoe.ru"
IMG = "https://api.vino-svoe.ru/v1/img/str-api/1000/1000/resize"
CSV_PATH = ROOT / "dataset" / "strapi_output0709.csv"
OUT = ROOT / "data" / "datapack" / "platform_additions.json"
PHOTOS = ROOT / "data" / "refs" / "platform"
UA = {"User-Agent": "Mozilla/5.0 (rus_vines catalog sync)"}

SWEETNESS = ("сухое", "полусухое", "полусладкое", "сладкое", "брют", "экстра брют",
             "брют натюр", "десертное", "ликерное", "ликёрное")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def payload_card(html: str, slug: str) -> dict | None:
    """Карточка из __NUXT_DATA__: плоский массив, ссылки — индексы в нём."""
    m = re.search(r'<script[^>]*id="__NUXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        return None
    data = json.loads(m.group(1))

    def res(value, depth=0):
        if depth > 8:
            return None
        if isinstance(value, dict):
            return {k: res(data[v], depth + 1) if isinstance(v, int) else v
                    for k, v in value.items()}
        if isinstance(value, list):
            if value and value[0] in ("ShallowReactive", "Reactive", "Ref", "ShallowRef"):
                return res(data[value[1]], depth + 1)
            return [res(data[v], depth + 1) if isinstance(v, int) else v for v in value]
        return value

    for item in data:
        if isinstance(item, dict) and "slug" in item and "title" in item:
            card = res(item)
            if card.get("slug") == slug and card.get("manufacturer"):
                return card
    return None


def flatten(card: dict) -> dict:
    """Карточка платформы в формате data/datapack/wines.json."""
    category = ((card.get("category") or {}).get("name") or "").strip()
    low = category.lower()
    color = category.split()[0] if category else ""
    sweet = next((s for s in sorted(SWEETNESS, key=len, reverse=True) if s in low), "")
    return {
        "slug": card["slug"],
        "title": (card.get("title") or "").strip(),
        "manufacturer": ((card.get("manufacturer") or {}).get("name") or "").strip(),
        "region": ((card.get("region") or {}).get("name") or "").strip(),
        "category": category,
        "color": color,
        "sweetness": sweet.capitalize(),
        "wineColor": card.get("color") or "",
        "alcohol": card.get("alcohol"),
        "temperature": card.get("temperature"),
        "rating": card.get("publicRating"),
        "description": card.get("description") or "",
        "grapes": [g.get("name") for g in card.get("grapes") or [] if g.get("name")],
        "dishes": [d.get("name") for d in card.get("dishes") or [] if d.get("name")],
        "image_url": ((card.get("image") or {}).get("url") or ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=1.0, help="пауза между запросами, с")
    args = ap.parse_args()

    sitemap = fetch(f"{BASE}/wines-sitemap.xml").decode("utf-8")
    live = set(re.findall(r"<loc>https://vino-svoe.ru/wines/([^<]+)</loc>", sitemap))
    dump = {r["Slug"].strip() for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))}
    new, removed = sorted(live - dump), sorted(dump - live)
    print(f"на платформе {len(live)}, в выгрузке {len(dump)}: новых {len(new)}, убрано {len(removed)}")

    PHOTOS.mkdir(parents=True, exist_ok=True)
    wines, failed = [], []
    for n, slug in enumerate(new, 1):
        try:
            card = payload_card(fetch(f"{BASE}/wines/{slug}").decode("utf-8"), slug)
            if card is None:
                raise ValueError("нет карточки в payload")
            wine = flatten(card)
            if wine["image_url"]:
                ext = Path(wine["image_url"]).suffix or ".webp"
                path = PHOTOS / f"{slug}{ext}"
                path.write_bytes(fetch(f"{IMG}{wine['image_url']}"))
                wine["photo"] = str(path.relative_to(ROOT))
                wine["photo_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            wines.append(wine)
            print(f"  {n}/{len(new)} {slug:44} {wine['manufacturer'][:30]:30} {wine['category']}")
        except Exception as exc:  # noqa: BLE001 — сеть: пропускаем и перечисляем
            failed.append({"slug": slug, "error": str(exc)[:200]})
            print(f"  {n}/{len(new)} {slug}: ОШИБКА {exc}", file=sys.stderr)
        time.sleep(args.delay)

    OUT.write_text(json.dumps({
        "meta": {"date": date.today().isoformat(), "source": f"{BASE}/wines-sitemap.xml",
                 "live": len(live), "dump": len(dump), "added": len(wines),
                 "removed_from_platform": removed, "failed": failed},
        "wines": wines,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записано: {OUT} ({len(wines)} позиций, ошибок {len(failed)})")


if __name__ == "__main__":
    main()
