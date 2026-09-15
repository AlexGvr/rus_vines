#!/usr/bin/env python3
"""S0.2 + S0.3 — каталог кейса и маппинг позиций на файлы эталонных фото.

Вход:
  dataset/strapi_output0709.csv                  — официальный дамп каталога
  dataset/uploads_root/**/strapi/uploads/        — файлы из дампа Strapi
  data/datapack/wines.json                       — наш скрейп (доливка полей,
                                                   которых нет в CSV)

Выход:
  data/catalog/catalog.json    — позиции каталога + путь к эталонному фото
  data/catalog/mapping.csv     — отчёт по маппингу (стратегия, размер, файл)

Проблема, которую решает маппинг: в CSV лежит ИСХОДНОЕ имя фото
("DSC09173.webp", "Спуманте белый брют.webp"), а Strapi хранит файл как
slugify(имя) + "_" + hash10 + ext, плюс превью thumbnail_/small_/medium_/large_.
Прямого ключа связи в выгрузке нет, поэтому идём каскадом стратегий и
падаем на превью, если оригинал не сохранился.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset" / "strapi_output0709.csv"
UPLOADS = next((ROOT / "dataset" / "uploads_root").rglob("uploads"), None)
SCRAPE = ROOT / "data" / "datapack" / "wines.json"
OUT_DIR = ROOT / "data" / "catalog"

# Размеры превью в порядке убывания качества: оригинал предпочтительнее,
# но для CV-индекса пригодно и превью — лучше, чем позиция без фото.
SIZES = ["", "large", "medium", "small", "thumbnail"]

# Транслитерация под @sindresorhus/slugify, которым пользуется Strapi.
CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def translit(text: str) -> str:
    return "".join(CYR.get(ch, CYR.get(ch.lower(), ch).upper() if ch.isupper() else ch)
                   if ch.lower() in CYR else ch
                   for ch in text)


def norm(text: str) -> str:
    """Имя файла → ключ поиска: без расширения, не-словарные символы в '_'."""
    text = unicodedata.normalize("NFKD", text)
    return re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_").lower()


def index_uploads(uploads: Path) -> dict[str, dict[str, str]]:
    """{стем без hash: {размер: имя файла}}. Стем — то, что Strapi сделал из имени."""
    idx: dict[str, dict[str, str]] = defaultdict(dict)
    for name in os.listdir(uploads):
        core, size = name, ""
        for prefix in ("thumbnail", "small", "medium", "large"):
            if name.startswith(prefix + "_"):
                core, size = name[len(prefix) + 1:], prefix
                break
        m = re.match(r"^(.*)_([0-9a-f]{10})(\.\w+)$", core)
        if not m:
            continue
        idx[m.group(1).lower()].setdefault(size, name)
    return idx


def candidate_keys(row: dict) -> list[tuple[str, str]]:
    """Ключи поиска файла, от самого надёжного к запасному."""
    photo = os.path.splitext(row["Название фото"].strip())[0]
    title = row["Название вина"].strip()
    return [
        ("photo", norm(photo)),
        ("photo_translit", norm(translit(photo))),
        ("slug", row["Slug"].strip().replace("-", "_").lower()),
        ("title_translit", norm(translit(title))),
    ]


def load_scrape_extras() -> dict[str, dict]:
    """Поля, которых нет в CSV кейса: рейтинг, блюда, крепость, температура."""
    if not SCRAPE.exists():
        return {}
    wines = json.loads(SCRAPE.read_text())["wines"]
    return {
        w["slug"]: {
            "rating": w.get("rating"),
            "alcohol": w.get("alcohol"),
            "temperature": w.get("temperature"),
            "dishes": w.get("dishes") or [],
        }
        for w in wines
    }


def main() -> int:
    if UPLOADS is None or not UPLOADS.is_dir():
        sys.exit("не найдена папка uploads — распакуйте дамп в dataset/uploads_root")

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    positions: dict[str, dict] = {}
    for row in rows:
        slug = row["Slug"].strip()
        if slug:
            positions[slug] = row  # дубли строк в CSV схлопываем по slug

    idx = index_uploads(UPLOADS)
    extras = load_scrape_extras()

    catalog, report = [], []
    stats = defaultdict(int)
    for slug, row in positions.items():
        photo_file = photo_size = strategy = None
        for name, key in candidate_keys(row):
            found = idx.get(key)
            if not found:
                continue
            for size in SIZES:
                if size in found:
                    photo_file, photo_size, strategy = found[size], size or "original", name
                    break
            break

        stats[f"{strategy or 'НЕТ'}/{photo_size or '-'}"] += 1
        stats["с фото" if photo_file else "без фото"] += 1

        grapes = [g.strip() for g in row["Сорт винограда"].split(",") if g.strip()]
        catalog.append({
            "slug": slug,
            "title": row["Название вина"].strip(),
            "manufacturer": row["Винодельня"].strip(),
            "region": row["Регион"].strip(),
            "category": row["Категория"].strip(),
            "color": row["Цвет"].strip(),
            "grapes": grapes,
            "description": row["Описание"].strip(),
            "photo": f"{UPLOADS.relative_to(ROOT)}/{photo_file}" if photo_file else None,
            "photo_size": photo_size,
            **extras.get(slug, {}),
        })
        report.append({
            "slug": slug, "csv_photo": row["Название фото"].strip(),
            "matched_file": photo_file or "", "size": photo_size or "",
            "strategy": strategy or "none",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "catalog.json").write_text(json.dumps(
        {"meta": {"source": "strapi_output0709.csv", "count": len(catalog),
                  "with_photo": stats["с фото"]},
         "wines": catalog}, ensure_ascii=False), encoding="utf-8")
    with (OUT_DIR / "mapping.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)

    enriched = sum(1 for w in catalog if w.get("rating") is not None)
    print(f"позиций каталога: {len(catalog)}")
    print(f"с эталонным фото: {stats['с фото']} "
          f"({stats['с фото'] / len(catalog) * 100:.1f}%), без фото: {stats['без фото']}")
    print(f"обогащено из скрейпа (рейтинг и прочее): {enriched}")
    print("\nразбивка стратегия/размер:")
    for key, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        if "/" in key:
            print(f"  {key:26} {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
