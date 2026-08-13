#!/usr/bin/env python3
"""Сборка датапака из сырых карточек.

Артефакты:
  data/datapack/wines.json    — для PWA (все карточки + мета)
  data/datapack/wines.sqlite  — SQLite + FTS5 (серверный/нативный вариант)
  app/data/wines.json         — копия для приложения
  app/photos/{slug}.webp      — фото для приложения
"""
import hashlib
import json
import shutil
import sqlite3
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "core"))
from normalize import clean  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CARDS = ROOT / "data" / "raw" / "cards"
PHOTOS = ROOT / "data" / "raw" / "photos"
OUT = ROOT / "data" / "datapack"
APP_DATA = ROOT / "app" / "data"
APP_PHOTOS = ROOT / "app" / "photos"

COLORS = ("белое", "красное", "оранжевое", "розовое")


def parse_category(name: str) -> tuple[str, str]:
    """'Белое брют' -> ('Белое', 'Брют')."""
    low = clean(name)
    color = next((c for c in COLORS if low.startswith(c)), "")
    sweetness = low[len(color):].strip() if color else low
    return color.capitalize(), sweetness.capitalize()


def name_of(value) -> str:
    if isinstance(value, dict):
        return value.get("name") or ""
    return value or ""


def build_wine(card: dict, has_photo: bool) -> dict:
    cat = (card.get("category") or {}).get("name") or ""
    color, sweetness = parse_category(cat)
    return {
        "slug": card["slug"],
        "title": card.get("title") or card["slug"],
        "manufacturer": name_of(card.get("manufacturer")),
        "region": name_of(card.get("region")),
        "category": cat,
        "color": color,
        "sweetness": sweetness,
        "wineColor": card.get("color") or "",
        "alcohol": card.get("alcohol"),
        "temperature": card.get("temperature") or "",
        "rating": card.get("publicRating"),
        "description": card.get("description") or "",
        "grapes": [g["name"] for g in (card.get("grapes") or [])],
        "dishes": [d["name"] for d in (card.get("dishes") or [])],
        "gradient": (card.get("category") or {}).get("backgroundGradient") or "",
        "photo": f"photos/{card['slug']}.webp" if has_photo else None,
    }


def main() -> None:
    wines = []
    for path in sorted(CARDS.glob("*.json")):
        card = json.loads(path.read_bytes())
        slug = path.stem
        card["slug"] = card.get("slug") or slug
        wines.append(build_wine(card, (PHOTOS / f"{slug}.webp").exists()))

    OUT.mkdir(parents=True, exist_ok=True)
    APP_DATA.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "version": date.today().isoformat(),
            "count": len(wines),
            "source": "vino-svoe.ru",
        },
        "wines": wines,
    }
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload["meta"]["hash"] = hashlib.sha256(blob.encode()).hexdigest()[:12]
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    (OUT / "wines.json").write_text(blob)
    (APP_DATA / "wines.json").write_text(blob)

    db_path = OUT / "wines.sqlite"
    db_path.unlink(missing_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE wines (
            slug TEXT PRIMARY KEY, title TEXT, manufacturer TEXT, region TEXT,
            category TEXT, color TEXT, sweetness TEXT, wine_color TEXT,
            alcohol REAL, temperature TEXT, rating REAL, description TEXT,
            grapes TEXT, dishes TEXT, photo TEXT
        );
        CREATE VIRTUAL TABLE wines_fts USING fts5(
            slug UNINDEXED, title, manufacturer, region, grapes,
            content='', tokenize='unicode61'
        );
    """)
    for w in wines:
        db.execute(
            "INSERT INTO wines VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (w["slug"], w["title"], w["manufacturer"], w["region"], w["category"],
             w["color"], w["sweetness"], w["wineColor"], w["alcohol"],
             w["temperature"], w["rating"], w["description"],
             ", ".join(w["grapes"]), ", ".join(w["dishes"]), w["photo"]))
        db.execute(
            "INSERT INTO wines_fts (slug, title, manufacturer, region, grapes) VALUES (?,?,?,?,?)",
            (w["slug"], w["title"], w["manufacturer"], w["region"], ", ".join(w["grapes"])))
    db.commit()
    db.close()

    if PHOTOS.exists():
        APP_PHOTOS.mkdir(parents=True, exist_ok=True)
        copied = 0
        for photo in PHOTOS.glob("*.webp"):
            target = APP_PHOTOS / photo.name
            if not target.exists():
                shutil.copy2(photo, target)
            copied += 1
        print(f"photos copied to app: {copied}")

    size_json = (OUT / "wines.json").stat().st_size
    size_db = db_path.stat().st_size
    print(f"wines: {len(wines)}")
    print(f"wines.json: {size_json / 1e6:.1f} MB, wines.sqlite: {size_db / 1e6:.1f} MB")
    print(f"version: {payload['meta']['version']}, hash: {payload['meta']['hash']}")


if __name__ == "__main__":
    main()
