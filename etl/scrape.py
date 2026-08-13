#!/usr/bin/env python3
"""Выгрузка базы вин vino-svoe.ru: карточки JSON + фото.

Источники:
  - https://vino-svoe.ru/wines-sitemap.xml          — список всех slug
  - https://vino-svoe.ru/api/wines/{slug}           — полная карточка
  - https://api.vino-svoe.ru/v1/img/str-api/...     — фото (ресайз-прокси)

Повторный запуск безопасен: уже скачанное пропускается.
"""
import json
import re
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://vino-svoe.ru"
IMG = "https://api.vino-svoe.ru/v1/img/str-api/400/400/resize"
UA = "RusVinesMVP/0.1 (educational prototype; contact: prey2999@gmail.com)"
DELAY = 0.25  # пауза на воркер между запросами
WORKERS = 2

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "raw" / "cards"
PHOTOS = ROOT / "data" / "raw" / "photos"
CARDS.mkdir(parents=True, exist_ok=True)
PHOTOS.mkdir(parents=True, exist_ok=True)


def fetch(url: str, retries: int = 4) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except Exception as e:
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise last


def get_slugs() -> list[str]:
    xml = fetch(f"{BASE}/wines-sitemap.xml").decode()
    slugs = re.findall(r"<loc>https://vino-svoe\.ru/wines/([^<]+)</loc>", xml)
    return sorted(set(slugs))


def grab(slug: str) -> str:
    card_path = CARDS / f"{slug}.json"
    if not card_path.exists():
        try:
            raw = fetch(f"{BASE}/api/wines/{slug}")
            json.loads(raw)  # валидация
            card_path.write_bytes(raw)
        except urllib.error.HTTPError as e:
            return f"card {slug}: HTTP {e.code}"
        time.sleep(DELAY)

    photo_path = PHOTOS / f"{slug}.webp"
    if not photo_path.exists():
        try:
            card = json.loads(card_path.read_bytes())
            img = (card.get("image") or {}).get("url")
            if img:
                data = fetch(f"{IMG}{img}")
                if len(data) > 500:
                    photo_path.write_bytes(data)
                time.sleep(DELAY)
        except urllib.error.HTTPError as e:
            return f"photo {slug}: HTTP {e.code}"
    return ""


def main() -> None:
    slugs = get_slugs()
    print(f"slugs in sitemap: {len(slugs)}", flush=True)
    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for err in pool.map(grab, slugs):
            done += 1
            if err:
                errors.append(err)
            if done % 100 == 0:
                print(f"progress: {done}/{len(slugs)}, errors: {len(errors)}", flush=True)
    print(f"done: {done}, cards: {len(list(CARDS.glob('*.json')))}, "
          f"photos: {len(list(PHOTOS.glob('*.webp')))}, errors: {len(errors)}", flush=True)
    for e in errors[:30]:
        print("ERR:", e, flush=True)
    (ROOT / "data" / "raw" / "errors.log").write_text("\n".join(errors))


if __name__ == "__main__":
    sys.exit(main())
