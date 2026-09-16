#!/usr/bin/env python3
"""Сбор полевого набора: реальные фотографии бутылок из отзывов покупателей.

Зачем. Все метрики проекта посчитаны на синтетике — запросы получены из тех
же файлов, что эталоны каталога. Такой набор годится сравнивать варианты
между собой, но абсолютные значения завышает, и пороги выдачи, обученные
на нём, на съёмке отказывают. Публичных фотографий кейса три, этого мало
даже чтобы оценить сдвиг.

Отзывы покупателей дают ровно то, чего не хватает: бутылка на столе, тёплый
свет кухни, блик, рука в кадре, снято телефоном. Ни одна из этих
фотографий не происходит из каталога, поэтому набор независим от индекса.

Правила, которых скрипт придерживается:
  * только страницы, открытые в robots.txt площадки (списки отзывов и сами
    отзывы; поиск площадки закрыт и не используется);
  * пауза между запросами, ограничение на число отзывов и фотографий;
  * фотографии кладутся в data/field/, который не коммитится — набор
    пересобирается скриптом, а не тиражируется.

Привязка фотографии к позиции каталога делается в два шага: сначала
по названию страницы отзывов, потом глазами (pipeline/verify_field.py).
Автоматической привязке одной доверять нельзя — у каталога есть позиции,
различающиеся одним словом.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "field"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
REVIEW_RE = re.compile(r"/review_(\d+)\.html")
IMAGE_RE = re.compile(r"i\d{4}\.otzovik\.com/[0-9/]+/\d+/img/[0-9a-z_]+\.(?:jpe?g|png)")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)

# Вторая площадка. Устроена иначе: страница бренда сразу даёт список отзывов,
# а фотографии покупателей лежат в product-images. Аватары (pictures/) и
# превью (imagecache/) отбрасываются — нужен оригинал.
IR_REVIEW_RE = re.compile(r'href="(/content/[a-z0-9-]{8,})"')
IR_IMAGE_RE = re.compile(
    r"https://[^\"' ]*?/files/(?:imagecache/[^/]+/)?(product-images/\d+/[^\"' ]+?\.(?:jpe?g|png))")


def ir_product(html: str) -> str:
    """Название товара из заголовка отзыва: «товар - «заголовок отзыва»»."""
    title = clean_title(html)
    return re.split(r"\s+[-–—]\s+[«\"]", title)[0].strip()


def collect_irecommend(pages: list[str], per_page: int, per_review: int,
                       pause: float, known: dict) -> int:
    saved = 0
    for number, page in enumerate(pages, 1):
        try:
            html = fetch(page)
        except Exception as error:
            print(f"[{number}/{len(pages)}] не открылось: {page} ({error})")
            continue
        reviews = list(dict.fromkeys(IR_REVIEW_RE.findall(html)))[:per_page]
        print(f"[{number}/{len(pages)}] {page.rsplit('/', 1)[-1]:28} "
              f"отзывов: {len(reviews)}", flush=True)
        time.sleep(pause)

        for path in reviews:
            url = "https://irecommend.ru" + path
            try:
                review = fetch(url)
            except Exception:
                time.sleep(pause)
                continue
            product = ir_product(review)
            images = list(dict.fromkeys(IR_IMAGE_RE.findall(review)))[:per_review]
            key = path.rsplit("/", 1)[-1][:40]
            for order, tail in enumerate(images):
                name = f"ir_{key}_{order}.jpg"
                target = OUT_DIR / name
                if target.exists():
                    continue
                full = f"https://cdn-irec.r-99.com/sites/default/files/{tail}"
                if download(full, target):
                    known[name] = {"listing": page, "listing_title": product,
                                   "review": url, "image": full}
                    saved += 1
                time.sleep(0.8)
            time.sleep(pause)
    return saved


def fetch(url: str, timeout: int = 25, tries: int = 4, pause: float = 8.0) -> str:
    """Запрос с отступом при отказе.

    Площадка отвечает 507, если ходить часто: это просьба сбавить темп,
    а не ошибка. Уважаем — ждём вдвое дольше после каждого отказа.
    """
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = pause
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def download(url: str, path: Path, timeout: int = 25) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except Exception:
        return False
    if len(data) < 8000:          # превью и заглушки нам не нужны
        return False
    path.write_bytes(data)
    return True


def catalog_guess(title: str, matcher, by_slug, floor: float):
    """Позиция каталога, на которую похож товар, либо None.

    Проверяется не только сходство названия, но и совпадение винодельни:
    без этого «Российское шампанское Абрау-Дюрсо» уходит в позицию «Нового
    Света» — общие слова «российское шампанское выдержанное» перевешивают,
    а производителя матчер не взвешивает вовсе.

    Догадка нужна только чтобы не тратить обращения к площадке на вина
    не из каталога. Настоящая привязка делается глазами в verify_field.py.
    """
    from normalize import clean as normalize_clean

    found = matcher.search(title, top=1)
    if not found or found[0]["confidence"] < floor:
        return None
    wine = by_slug[found[0]["slug"]]
    words = [w for w in normalize_clean(wine["manufacturer"]).split() if len(w) >= 4]
    low = normalize_clean(title)
    if words and not any(w in low for w in words):
        return None
    return wine


def clean_title(html: str) -> str:
    match = TITLE_RE.search(html)
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    title = re.sub(r"^Отзывы о\s*", "", title)
    return title.split("|")[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listings", required=True,
                    help="файл со списком адресов страниц отзывов, по одному в строке")
    ap.add_argument("--reviews-per-wine", type=int, default=6)
    ap.add_argument("--photos-per-review", type=int, default=3)
    ap.add_argument("--pause", type=float, default=7.0, help="пауза между запросами, с")
    ap.add_argument("--source", choices=["otzovik", "irecommend"], default="otzovik")
    ap.add_argument("--require-catalog", type=float, default=0.0,
                    help="пропускать товары, не похожие ни на одну позицию "
                         "каталога с такой уверенностью (0 — не проверять)")
    args = ap.parse_args()

    matcher = by_slug = None
    if args.require_catalog:
        sys.path.insert(0, str(ROOT / "core"))
        from matcher import Matcher
        wines = json.loads((ROOT / "data" / "catalog" / "catalog.json")
                           .read_text(encoding="utf-8"))["wines"]
        matcher, by_slug = Matcher(wines), {w["slug"]: w for w in wines}

    listings = [line.strip() for line in Path(args.listings).read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_path = OUT_DIR / "sources.json"
    known = (json.loads(index_path.read_text(encoding="utf-8"))
             if index_path.exists() else {})

    saved = 0
    if args.source == "irecommend":
        saved = collect_irecommend(listings, args.reviews_per_wine,
                                   args.photos_per_review, args.pause, known)
        listings = []
    for number, listing in enumerate(listings, 1):
        try:
            html = fetch(listing)
        except Exception as error:
            print(f"[{number}/{len(listings)}] не открылось: {listing} ({error})")
            continue
        title = clean_title(html)
        if matcher is not None:
            wine = catalog_guess(title, matcher, by_slug, args.require_catalog)
            if wine is None:
                print(f"[{number}/{len(listings)}] {title[:60]:60} нет в каталоге, "
                      f"пропуск", flush=True)
                time.sleep(args.pause)
                continue
        reviews = list(dict.fromkeys(REVIEW_RE.findall(html)))[:args.reviews_per_wine]
        print(f"[{number}/{len(listings)}] {title[:60]:60} отзывов: {len(reviews)}",
              flush=True)
        time.sleep(args.pause)

        for review_id in reviews:
            url = f"https://otzovik.com/review_{review_id}.html"
            try:
                page = fetch(url)
            except Exception:
                time.sleep(args.pause)
                continue
            images = list(dict.fromkeys(IMAGE_RE.findall(page)))[:args.photos_per_review]
            for order, image in enumerate(images):
                name = f"{review_id}_{order}.jpg"
                path = OUT_DIR / name
                if path.exists():
                    continue
                if download("https://" + image, path):
                    known[name] = {"listing": listing, "listing_title": title,
                                   "review": url, "image": "https://" + image}
                    saved += 1
                time.sleep(1.0)
            time.sleep(args.pause)

    index_path.write_text(json.dumps(known, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    print(f"\nсохранено новых фотографий: {saved}, всего в наборе: {len(known)}")
    print(f"папка: {OUT_DIR}")


if __name__ == "__main__":
    main()
