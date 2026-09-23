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
# Позиции платформы, которых нет в выгрузке кейса (pipeline/sync_platform.py):
# добавляются последними, фото сверяется по SHA-256.
PLATFORM_ADDITIONS = ROOT / "data" / "datapack" / "platform_additions.json"
RAW_CARDS = ROOT / "data" / "raw" / "cards"     # обход каталога, не в репозитории
PHOTO_MAP = ROOT / "data" / "datapack" / "photo_map.csv"   # его дистиллят, в репозитории
# Ручные исправления привязки: проверенные глазами эталоны из официальных
# источников и зафиксированные пробелы. Применяются последними и переживают
# пересборку каталога — иначе каждая пересборка возвращала бы известные ошибки.
PHOTO_OVERRIDES = ROOT / "data" / "datapack" / "photo_overrides.csv"
REFS_DIR = ROOT / "data" / "refs"
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


def collapse(text: str) -> str:
    """Ключ без разделителей: только латиница и цифры после транслитерации.

    Strapi режет имя по границам слов, которых нет в исходном имени:
    «AligRiesl» → «Alig_Riesl», «DSC00839» → «DSC_00839», «10_44PM» →
    «10_44_PM», а «_ (1)» схлопывает в «_1». Ключ с подчёркиваниями такие
    файлы не находит, хотя они лежат в дампе под именем из CSV. Ключ без
    разделителей терпим к любой расстановке границ; ложные склейки не
    возникают, потому что сравнение остаётся точным по всей строке.
    """
    return re.sub(r"[^a-z0-9]", "", norm(translit(text)))


def index_uploads(uploads: Path) -> dict[str, list[dict[str, str]]]:
    """{стем без hash: [{размер: имя файла}, ...]}.

    Важно хранить ВСЕ файлы с одинаковым стемом, а не первый попавшийся:
    у «Shardone» в дампе восемь разных файлов — по одному на винодельню.
    Если оставить один, три карточки разных производителей получат одно фото.
    """
    by_hash: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for name in os.listdir(uploads):
        core, size = name, ""
        for prefix in ("thumbnail", "small", "medium", "large"):
            if name.startswith(prefix + "_"):
                core, size = name[len(prefix) + 1:], prefix
                break
        m = re.match(r"^(.*)_([0-9a-f]{10})(\.\w+)$", core)
        if not m:
            continue
        by_hash[(m.group(1).lower(), m.group(2))][size] = name

    idx: dict[str, list[dict[str, str]]] = defaultdict(list)
    for (stem, _hash), sizes in by_hash.items():
        idx[stem].append(sizes)
    return idx


def scrape_photo_map(raw_cards: Path) -> dict[str, str]:
    """Привязка из обхода каталога: `image.url` карточки — имя файла Strapi.

    Это авторитетный ключ связи, которого нет в выгрузке кейса. Но сам обход
    в репозиторий не коммитится, поэтому результат складывается в PHOTO_MAP
    и дальше используется оттуда — иначе сборка на чужой машине молча
    теряет 160 позиций и даёт другой каталог.
    """
    mapping: dict[str, str] = {}
    if not raw_cards.is_dir():
        return mapping
    for path in sorted(raw_cards.glob("*.json")):
        try:
            card = json.loads(path.read_bytes())
        except Exception:
            continue
        url = (card.get("image") or {}).get("url")
        if url:
            mapping[card.get("slug") or path.stem] = url.rsplit("/", 1)[-1]
    return mapping


def api_photo_map(available: set[str], refresh: bool = False) -> dict[str, str]:
    """Точная привязка позиции к файлу. Источник — PHOTO_MAP из репозитория.

    Файлы, которых нет в дампе, отбрасываются: у соседа может не совпасть
    версия выгрузки, и ссылаться на отсутствующий файл нельзя.
    """
    if refresh:
        scraped = scrape_photo_map(RAW_CARDS)
        if not scraped:
            sys.exit(f"нет данных обхода в {RAW_CARDS} — нечем обновлять привязку")
        PHOTO_MAP.parent.mkdir(parents=True, exist_ok=True)
        with PHOTO_MAP.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["slug", "file"])
            writer.writerows(sorted(scraped.items()))
        print(f"привязка обновлена из обхода: {len(scraped)} позиций → {PHOTO_MAP}")

    if not PHOTO_MAP.exists():
        print(f"ВНИМАНИЕ: нет {PHOTO_MAP.relative_to(ROOT)}, точная привязка "
              f"недоступна — каталог будет заметно беднее фотографиями")
        return {}
    with PHOTO_MAP.open(encoding="utf-8") as fh:
        return {row["slug"]: row["file"] for row in csv.DictReader(fh)
                if row["file"] in available}


# Префикс ключа без разделителей: не пересекается с настоящими стемами дампа.
COLLAPSED = "~"


def candidate_keys(row: dict) -> list[tuple[str, str]]:
    """Ключи поиска файла, от самого надёжного к запасному."""
    photo = os.path.splitext(row["Название фото"].strip())[0]
    title = row["Название вина"].strip()
    return [
        ("photo", norm(photo)),
        ("photo_translit", norm(translit(photo))),
        ("photo_collapsed", COLLAPSED + collapse(photo)),
        ("slug", row["Slug"].strip().replace("-", "_").lower()),
        ("title_translit", norm(translit(title))),
    ]


def with_collapsed(idx: dict[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    """Тот же индекс дампа плюс ключи без разделителей."""
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for stem, groups in idx.items():
        out[stem].extend(groups)
        out[COLLAPSED + collapse(stem)].extend(groups)
    return out


def load_overrides() -> dict[str, dict[str, str]]:
    """slug → строка photo_overrides.csv.

    Пустой file — зафиксированный пробел: эталон платформы показывает другое
    вино, а верного изображения найти не удалось. Такая позиция остаётся без
    фото, и эвристика подбора по имени к ней не применяется — иначе она бы
    снова подставила тот же неверный файл.
    """
    if not PHOTO_OVERRIDES.exists():
        return {}
    with PHOTO_OVERRIDES.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        slug = (row.get("slug") or "").strip()
        if not slug:
            continue
        if slug in out:
            sys.exit(f"{PHOTO_OVERRIDES}: slug {slug} встречается дважды")
        out[slug] = {k: (v or "").strip() for k, v in row.items()}
    return out


def resolve_override(slug: str, row: dict[str, str]) -> str | None:
    """Путь к файлу переопределения относительно корня или None для пробела.

    Файл сверяется по SHA-256: подмена картинки под тем же именем испортила бы
    индекс молча, а причину искали бы в алгоритме.
    """
    if not row["file"]:
        return None
    path = REFS_DIR / row["file"]
    if not path.exists():
        sys.exit(f"{PHOTO_OVERRIDES}: для {slug} нет файла {path}")
    if row.get("sha256"):
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            sys.exit(f"{PHOTO_OVERRIDES}: у {path} другой SHA-256 ({digest[:12]}…), "
                     f"ожидался {row['sha256'][:12]}…")
    return str(path.relative_to(ROOT))


def guess_photos(positions: dict[str, dict], api_map: dict[str, str],
                 idx: dict[str, list[dict[str, str]]],
                 claimed: set[str], skip: set[str] = frozenset()) -> dict[str, tuple[str, str, str]]:
    """Подбор файла по имени для позиций, которых нет в обходе каталога.

    Неоднозначность здесь меряется конкуренцией позиций за имя, а не числом
    файлов под ним. Несколько файлов с одним стемом — обычная перезаливка:
    у «Frizante_beloe_suhoe» их четыре, и на всех одна и та же бутылка,
    отличаются кроп и сжатие. А вот если на одно имя претендуют две разные
    позиции, выбрать не из чего — ни имя, ни хеш не говорят, чьё это фото,
    и ошибка испортит разом и индекс, и разметку тестов. Такие пропускаем.
    """
    # Каждой позиции — первый ключ, под которым в дампе есть свободный файл.
    wanted: dict[str, list[str]] = defaultdict(list)
    keyname: dict[str, tuple[str, str]] = {}
    for slug, row in positions.items():
        if slug in api_map or slug in skip:
            continue
        for name, key in candidate_keys(row):
            free = [g for g in idx.get(key, []) if not set(g.values()) & claimed]
            if free:
                wanted[key].append(slug)
                keyname[slug] = (name, key)
                break

    out: dict[str, tuple[str, str, str]] = {}
    for slug, (name, key) in keyname.items():
        if len(wanted[key]) != 1:
            continue
        free = [g for g in idx.get(key, []) if not set(g.values()) & claimed]
        # Из перезаливок берём самую полную: оригинал предпочтительнее превью,
        # среди оригиналов — самый тяжёлый файл, он же наименее пережатый.
        best = max(free, key=lambda g: (
            "" in g, (UPLOADS / g[next(s for s in SIZES if s in g)]).stat().st_size))
        size = next(s for s in SIZES if s in best)
        out[slug] = (best[size], size or "original", name)
        claimed.add(best[size])
    return out


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


def main(refresh: bool = False) -> int:
    if UPLOADS is None or not UPLOADS.is_dir():
        sys.exit("не найдена папка uploads — распакуйте дамп в dataset/uploads_root")

    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    positions: dict[str, dict] = {}
    for row in rows:
        slug = row["Slug"].strip()
        if slug:
            positions[slug] = row  # дубли строк в CSV схлопываем по slug

    idx = with_collapsed(index_uploads(UPLOADS))
    extras = load_scrape_extras()
    overrides = load_overrides()
    available = {name for sizes in idx.values() for group in sizes
                 for name in group.values()}
    api_map = api_photo_map(available, refresh=refresh)
    # Файлы, занятые точной привязкой, эвристике больше не предлагаем:
    # иначе она снова раздаст одно фото нескольким винодельням.
    claimed = set(api_map.values())

    guessed = guess_photos(positions, api_map, idx, claimed, skip=set(overrides))

    catalog, report = [], []
    stats = defaultdict(int)
    for slug, row in positions.items():
        photo_file = photo_size = strategy = photo_rel = None

        if slug in overrides:
            photo_rel = resolve_override(slug, overrides[slug])
            if photo_rel:
                photo_file, photo_size, strategy = Path(photo_rel).name, "original", "override"
            else:
                strategy = "override-gap"

        elif slug in api_map:
            photo_file, photo_size, strategy = api_map[slug], "original", "api"

        elif slug in guessed:
            photo_file, photo_size, strategy = guessed[slug]

        if photo_file and not photo_rel:
            photo_rel = f"{UPLOADS.relative_to(ROOT)}/{photo_file}"

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
            "photo": photo_rel,
            "photo_size": photo_size,
            **extras.get(slug, {}),
        })
        report.append({
            "slug": slug, "csv_photo": row["Название фото"].strip(),
            "matched_file": photo_file or "", "size": photo_size or "",
            "strategy": strategy or "none",
        })

    # Позиции, появившиеся на платформе после выгрузки. Slug выгрузки не
    # перекрываются: выгрузка — официальный источник, дополнение только
    # добавляет то, чего в ней нет.
    added = 0
    if PLATFORM_ADDITIONS.exists():
        import hashlib
        for w in json.loads(PLATFORM_ADDITIONS.read_text(encoding="utf-8"))["wines"]:
            if w["slug"] in positions:
                continue
            photo = w.get("photo")
            if photo:
                path = ROOT / photo
                if not path.exists():
                    sys.exit(f"{PLATFORM_ADDITIONS.name}: для {w['slug']} нет файла {photo}")
                if hashlib.sha256(path.read_bytes()).hexdigest() != w.get("photo_sha256"):
                    sys.exit(f"{PLATFORM_ADDITIONS.name}: у {photo} другой SHA-256")
            catalog.append({
                "slug": w["slug"], "title": w["title"], "manufacturer": w["manufacturer"],
                "region": w["region"], "category": w["color"], "color": w["wineColor"],
                "grapes": w["grapes"], "description": w["description"],
                "photo": photo, "photo_size": "original" if photo else None,
                "rating": w.get("rating"), "alcohol": w.get("alcohol"),
                "temperature": w.get("temperature"), "dishes": w.get("dishes") or [],
                "source": "platform-sync",
            })
            report.append({"slug": w["slug"], "csv_photo": "", "matched_file": photo or "",
                           "size": "original" if photo else "", "strategy": "platform-sync"})
            stats["с фото" if photo else "без фото"] += 1
            stats[f"platform-sync/{'original' if photo else '-'}"] += 1
            added += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "catalog.json").write_text(json.dumps(
        {"meta": {"source": "strapi_output0709.csv", "platform_additions": added,
                  "count": len(catalog),
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-map", action="store_true",
                        help="пересобрать data/datapack/photo_map.csv из обхода "
                             "каталога (нужен data/raw/cards)")
    sys.exit(main(refresh=parser.parse_args().refresh_map))
