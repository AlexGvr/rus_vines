#!/usr/bin/env python3
"""Сбор полевых кадров с площадок отзывов по всем позициям каталога.

Зачем отдельный скрипт. collect_field.py умеет ходить на обе площадки,
но только по списку адресов, написанному руками: годится добрать кадры
для десятка вин, не годится пройти каталог целиком. Здесь сбор начинается
не со списка ссылок, а с каталога — 2103 позиции, — и сам ищет, у каких
из них на площадке есть отзывы с фотографиями.

Площадки две, устроены по-разному, но задача одна, поэтому различия
собраны в адаптеры, а обход, сопоставление и учёт скачанного общие.

  otzovik      рубрики /food/alcohol/wines/ и соседние, постранично;
               товар показывает превью всех своих отзывов, и полный кадр
               получается подменой суффикса: _t (150 px) на _b (1500 px).
  irecommend   рубрика «Алкоголь» разбита на фасеты по типу напитка
               и по производителю; превью и полный кадр — один файл
               в разных пресетах imagecache: 200i против copyright1.

Обе площадки закрывают поиск в robots.txt, поэтому искать вино по имени
нельзя, и обход идёт по открытым спискам рубрик. У irecommend закрыт
ещё и фильтр каталога (/catalog_filter/) и цепочки фасетов длиннее трёх
частей, у otzovik — галереи (/*/gallery/) и адреса отзывов с запросом.

Три шага, каждый со своим состоянием и каждый можно повторять:

  discover  обойти рубрики, собрать товары: адрес, название, число
            отзывов. У irecommend новые фасеты берутся с уже скачанных
            страниц, так что перечень виноделен не нужен заранее.
  match     сопоставить название товара с позицией каталога. Это только
            догадка — она решает, к кому идти за фотографиями, а не что
            изображено на кадре. Настоящая привязка делается глазами
            в verify_field.py.
  harvest   скачать фотографии покупателей у сопоставленных товаров.

Экономия обращений. Обе площадки показывают превью отзывов прямо
на странице товара, а превью и полный кадр — один и тот же файл. Значит
одно обращение к товару даёт всю его съёмку, и на страницы самих отзывов
ходить не нужно.

Файлы кладутся в data/field/ и называются так, чтобы verify_field.py
опознал фотосерию: кадры одного отзыва — одна серия, и серия целиком
уходит либо в настройку, либо в тест.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELD = ROOT / "data" / "field"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Слова, по которым товар опознаётся как винный или как заведомо чужой.
# Рубрики у площадок общие для всего алкоголя, и без отсева обход уходит
# в пиво и виски, которых в каталоге нет.
WINE_WORDS = re.compile(
    r"вин[оаы]|игрист|шампан|портвейн|вермут|кагор|мадер|херес|мускат|"
    r"шардоне|каберне|саперави|мерло|совиньон|рислинг|пино|санджовезе", re.I)
SKIP_WORDS = re.compile(
    r"пиво|пивн|виски|водка|коньяк|ликер|ликёр|сидр|джин|ром\b|текила|"
    r"настойка|бальзам|самогон|абсент|бренди|шнапс|граппа|чача|сак[еэ]|"
    r"безалкоголь|энергетич", re.I)

LEGAL_RE = re.compile(r"\b(ооо|оао|зао|ао|пао|ип|фгуп|кфх|агрофирма|"
                      r"винодельня|винзавод|завод|торговый дом)\b")

# Слова, которые в названии винодельни ничего не различают: «вино» есть
# и в «Дербент Вино», и в заголовке любого товара, поэтому проверку
# совпадения производителя оно пропускает вхолостую.
MAKER_STOP = {"вино", "вина", "винный", "винодельня", "винзавод", "завод",
              "дом", "компания", "агрофирма", "холдинг", "групп", "групе"}


def fetch(url: str, timeout: int = 25, tries: int = 4, backoff: float = 8.0) -> str:
    """Запрос с отступом при отказе.

    Площадка отвечает 507 или 521, если ходить часто или если ей плохо:
    это просьба сбавить темп, а не ошибка. Ждём вдвое дольше после
    каждого отказа.
    """
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    delay = backoff
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            # 404 и 410 — товар убран с площадки, повтор ничего не изменит,
            # а отступ съедает минуту на каждой такой странице. Ждём только
            # там, где ожидание помогает: перегрузка и отказ сервера.
            if error.code not in (429,) and error.code < 500:
                raise
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""


def download(url: str, path: Path, referer: str = "", timeout: int = 25,
             floor: int = 8000) -> bool:
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except Exception:
        return False
    if len(data) < floor:          # превью и заглушки нам не нужны
        return False
    path.write_bytes(data)
    return True


def text_of(raw: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()


def squeeze(name: str) -> str:
    """Голые буквы подписи: без правовой формы, кавычек, пробелов и дефисов."""
    low = LEGAL_RE.sub(" ", name.lower())
    return re.sub(r"[^a-zа-яё0-9]+", "", low)


def maker_keys() -> set[str]:
    """Имена виноделен каталога в виде, годном для сравнения с подписью."""
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    return {k for k in (squeeze(w.get("manufacturer", "")) for w in wines)
            if len(k) >= 5}


# --- площадки -------------------------------------------------------------

class Otzovik:
    """Отзовик: рубрики товаров, пагинация номером в пути."""

    name = "otzovik"
    site = "https://otzovik.com"
    seeds = ("/food/alcohol/wines/", "/food/alcohol/sparkling_wines/",
             "/food/alcohol/vermouths/")

    ITEM_SPLIT = re.compile(r'<div class="item" data-pid="(\d+)"')
    NAME_RE = re.compile(r'<a href="/reviews/([a-z_0-9-]+)/"[^>]*'
                         r"class='product-name'>(.*?)</a>", re.S)
    COUNT_RE = re.compile(r"class='reviews-counter'>(\d+)")
    # Превью кадров отзыва на странице товара: .../<номер отзыва>/img/..._t.jpeg
    THUMB_RE = re.compile(r"//(i\d*\.otzovik\.com/[0-9/]+/(\d+)/img/"
                          r"[0-9a-z_]+)_t\.(jpe?g|png)", re.I)

    def page_url(self, listing: str, number: int) -> str:
        return self.site + listing + (f"{number}/" if number > 1 else "")

    def parse_listing(self, page: str) -> tuple[list[dict], list[str], int]:
        products = []
        parts = self.ITEM_SPLIT.split(page)
        for pid, chunk in zip(parts[1::2], parts[2::2]):
            link = self.NAME_RE.search(chunk)
            if not link:
                continue
            count = self.COUNT_RE.search(chunk)
            products.append({"slug": link.group(1), "title": text_of(link.group(2)),
                             "nid": pid,
                             "reviews": int(count.group(1)) if count else 0})
        pages = [int(n) for n in re.findall(r'/food/alcohol/[a-z_]+/(\d+)/', page)]
        return products, [], max(pages) if pages else 1

    def product_url(self, item: dict) -> str:
        return f"{self.site}/reviews/{item['slug']}/"

    def parse_product(self, page: str) -> list[tuple[str, str, list[str]]]:
        """Отзывы товара: ключ серии, адрес отзыва, полноразмерные кадры.

        Превью и полный кадр — один файл: суффикс _t даёт 150 px, _b —
        1500 px. Поэтому страницы самих отзывов не нужны.
        """
        found: dict[str, list[str]] = {}
        for stem, review_id, ext in self.THUMB_RE.findall(page):
            found.setdefault(review_id, [])
            full = f"https://{stem}_b.{ext}"
            if full not in found[review_id]:
                found[review_id].append(full)
        return [(rid, f"{self.site}/review_{rid}.html", urls)
                for rid, urls in found.items()]

    def frame_name(self, key: str, order: int) -> str:
        return f"{key}_{order}.jpg"


class Irecommend:
    """irecommend: фасеты внутри рубрики «Алкоголь», пагинация запросом."""

    name = "irecommend"
    site = "https://irecommend.ru"
    seeds = ("/catalog/list/938",)
    cdn = "https://cdn-irec.r-99.com/sites/default/files"

    FACET_RE = re.compile(
        r'href="(/catalog/list/938[0-9-]*)"[^>]*title="([^"]*)"[^>]*data-vid="(\d+)"')
    TIZER_SPLIT = re.compile(r'<div class="ProductTizer[^"]*"[^>]*data-nid="(\d+)"')
    TITLE_LINK_RE = re.compile(r'<div class="title"><a href="/content/([^"]+)">(.*?)</a>')
    COUNTER_RE = re.compile(r'class="read-all-reviews-link"[^>]*>.*?'
                            r'<span class="counter">(\d+)</span>', re.S)
    REVIEW_SPLIT = re.compile(r'<div\s+class="[^"]*reviews-list-item[^"]*"')
    REVIEW_LINK_RE = re.compile(r'class="reviewTextSnippet"[^>]*href="/content/([a-z0-9-]+)"')
    PREVIEW_RE = re.compile(r'imagecache/200i/(user-images/\d+/[^"\' ]+?\.(?:jpe?g|png))')

    def __init__(self) -> None:
        self.makers = maker_keys()

    def page_url(self, listing: str, number: int) -> str:
        return self.site + listing + (f"?page={number}" if number else "")

    def facet_wanted(self, title: str, vid: str) -> bool:
        """Идти ли в этот фасет.

        По типу напитка — всё винное и ничего крепкого. По производителю —
        только те, чьё имя есть в каталоге: у иностранных виноделен отзывы
        тоже есть, но искать по ним нечего. Одна винодельня подписана
        на площадке и в каталоге по-разному — «ООО "Кубань - вино"» против
        «Кубань-Вино», — поэтому сравниваются голые буквы обеих подписей.
        """
        low = title.lower()
        if vid == "38":
            return bool(WINE_WORDS.search(low)) and not SKIP_WORDS.search(low)
        if vid == "37":
            bare = squeeze(title)
            return any(maker in bare or bare in maker for maker in self.makers)
        return False

    def parse_listing(self, page: str) -> tuple[list[dict], list[str], int]:
        products = []
        parts = self.TIZER_SPLIT.split(page)
        for nid, chunk in zip(parts[1::2], parts[2::2]):
            link = self.TITLE_LINK_RE.search(chunk)
            if not link:
                continue
            counter = self.COUNTER_RE.search(chunk)
            products.append({"slug": link.group(1), "title": text_of(link.group(2)),
                             "nid": nid,
                             "reviews": int(counter.group(1)) if counter else 0})
        more = [link for link, title, vid in self.FACET_RE.findall(page)
                if self.facet_wanted(html.unescape(title), vid)]
        pages = [int(n) for n in re.findall(r"[?&]page=(\d+)", page)]
        return products, more, max(pages) if pages else 0

    def product_url(self, item: dict) -> str:
        return f"{self.site}/content/{item['slug']}"

    def parse_product(self, page: str) -> list[tuple[str, str, list[str]]]:
        out = []
        for chunk in self.REVIEW_SPLIT.split(page)[1:]:
            link = self.REVIEW_LINK_RE.search(chunk)
            if not link:
                continue
            # Кусок последнего отзыва тянется до конца страницы и захватывает
            # боковую колонку с чужими отзывами. Галерея всегда лежит выше
            # текста отзыва, поэтому ищем превью только до него.
            tails = list(dict.fromkeys(self.PREVIEW_RE.findall(chunk[:link.start()])))
            if not tails:
                continue
            # Снимки одного отзыва грузит один человек — папка у них общая.
            folder = tails[0].rsplit("/", 1)[0]
            urls = [f"{self.cdn}/imagecache/copyright1/{t}"
                    for t in tails if t.startswith(folder)]
            out.append((link.group(1)[:40],
                        f"{self.site}/content/{link.group(1)}", urls))
        return out

    def frame_name(self, key: str, order: int) -> str:
        return f"ir_{key}_{order}.jpg"


SITES = {"otzovik": Otzovik, "irecommend": Irecommend}


# --- состояние ------------------------------------------------------------

def state_dir(site) -> Path:
    return FIELD / site.name


def load_state(site, name: str) -> dict:
    path = state_dir(site) / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(site, name: str, data: dict) -> None:
    state_dir(site).mkdir(parents=True, exist_ok=True)
    (state_dir(site) / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# --- шаг 1: обход рубрик --------------------------------------------------

def discover(site, pause: float, max_pages: int, budget: int) -> None:
    products = load_state(site, "products.json")
    visited = set(load_state(site, "listings.json").get("done", []))
    queue = [s for s in site.seeds if s not in visited]
    queued = set(queue)
    requests = misses = 0

    while queue and requests < budget:
        listing = queue.pop(0)
        if listing in visited:
            continue
        first = 0 if site.name == "irecommend" else 1
        page_no, last, failed = first, first, False
        while page_no <= last and requests < budget:
            url = site.page_url(listing, page_no)
            try:
                body = fetch(url)
            except Exception as error:
                print(f"не открылось: {url} ({error})", flush=True)
                failed = True
                break
            requests += 1
            misses = 0
            found, more, pager = site.parse_listing(body)
            if page_no == first:
                # Корень рубрики у irecommend общий для всего алкоголя:
                # нужен только ради ссылок на винные фасеты, листать его
                # страницы с пивом и виски незачем.
                root = listing.rstrip("/").endswith("938")
                last = first if root else min(pager, max_pages)
            fresh = 0
            for item in found:
                if item["slug"] not in products:
                    fresh += 1
                entry = products.setdefault(item["slug"], item)
                entry["reviews"] = max(entry.get("reviews", 0), item["reviews"])
            for link in more:
                if link not in queued and link not in visited:
                    queue.append(link)
                    queued.add(link)
            print(f"{listing}{page_no} из {last}: товаров {len(found)} "
                  f"(новых {fresh}), всего {len(products)}, очередь {len(queue)}",
                  flush=True)
            page_no += 1
            time.sleep(pause)
        if failed:
            # Отказ касается всех адресов сразу, и перебор очереди только
            # добавляет площадке нагрузки. Рубрика остаётся недойденной:
            # следующий запуск возьмёт её снова.
            queue.append(listing)
            misses += 1
            if misses >= 3:
                print("площадка отказывает подряд — останавливаюсь, "
                      "повторить запуск позже", flush=True)
                break
            continue
        visited.add(listing)
        save_state(site, "products.json", products)
        save_state(site, "listings.json",
                   {"done": sorted(visited),
                    "queue": [x for x in queue if x not in visited]})

    print(f"\nтоваров найдено: {len(products)}, рубрик пройдено: {len(visited)}, "
          f"обращений: {requests}")


# --- шаг 2: догадка о позиции каталога ------------------------------------

def match(site, floor: float) -> None:
    """Сопоставление названия товара с каталогом.

    Проверяется не только сходство названия, но и совпадение винодельни:
    без этого «Российское шампанское Абрау-Дюрсо» уходит в позицию «Нового
    Света» — общие слова перевешивают, а производителя матчер не взвешивает.
    Порог намеренно щадящий: пропустить товар дороже, чем сходить на него
    зря, а лишнее отсеет человек в verify_field.py.
    """
    sys.path.insert(0, str(ROOT / "core"))
    from matcher import Matcher
    from normalize import clean as normalize_clean

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    matcher, by_slug = Matcher(wines), {w["slug"]: w for w in wines}
    products = load_state(site, "products.json")
    if not products:
        sys.exit(f"сначала discover: {state_dir(site)}/products.json пуст")

    # Дешёвый отсев перед матчером. Товаров тринадцать тысяч, поиск
    # по каталогу стоит пятую долю секунды каждый — это час работы,
    # причём почти весь он тратится на импорт, который всё равно не
    # пройдёт проверку винодельни ниже. Проверка эта требует, чтобы
    # в названии товара нашлось слово производителя, поэтому товар
    # без единого слова какой-нибудь винодельни каталога можно
    # отбросить не считая: результат тот же, времени в разы меньше.
    tokens = {w for wine in wines
              for w in normalize_clean(wine["manufacturer"]).split()
              if len(w) >= 4 and w not in MAKER_STOP}

    hit = skipped = 0
    for item in products.values():
        item.pop("match", None)
        if SKIP_WORDS.search(item["title"]):
            continue          # коньяк и чача той же винодельни нам не нужны
        low = normalize_clean(item["title"])
        if not any(token in low for token in tokens):
            skipped += 1
            continue
        found = matcher.search(item["title"], top=1)
        if not found or found[0]["confidence"] < floor:
            continue
        wine = by_slug[found[0]["slug"]]
        words = [w for w in normalize_clean(wine["manufacturer"]).split()
                 if len(w) >= 4 and w not in MAKER_STOP]
        low = normalize_clean(item["title"])
        if words and not any(w in low for w in words):
            continue
        item["match"] = {"slug": wine["slug"], "confidence": found[0]["confidence"],
                         "catalog_title": wine["title"],
                         "manufacturer": wine["manufacturer"]}
        hit += 1

    save_state(site, "products.json", products)
    covered = {i["match"]["slug"] for i in products.values() if "match" in i}
    reviews = sum(i["reviews"] for i in products.values() if "match" in i)
    print(f"отброшено без винодельни каталога в названии: {skipped}")
    print(f"товаров: {len(products)}, сопоставлено: {hit}, "
          f"позиций каталога закрыто: {len(covered)} из {len(wines)}")
    print(f"отзывов у сопоставленных товаров: {reviews}")


# --- шаг 3: сбор фотографий -----------------------------------------------

def harvest_product(site, item: dict, per_product: int, per_review: int,
                    known: dict, photo_pause: float = 0.3) -> int | None:
    """Кадры одного товара, либо None, если страница не открылась.

    Отличать нечего-брать от не-удалось-взять приходится: у снятого
    с площадки товара кадров нет и не будет, а у товара, на котором
    площадка икнула (507), они есть — и помечать такой пройденным
    значит потерять его навсегда.
    """
    url = site.product_url(item)
    try:
        page = fetch(url)
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            # Товар снят с площадки: кадров не будет никогда, и возвращаться
            # к нему каждый запуск незачем — считаем пройденным пустым.
            print(f"  снят с площадки: {url}", flush=True)
            return 0
        print(f"  не открылось: {url} ({error})", flush=True)
        return None
    except Exception as error:
        print(f"  не открылось: {url} ({error})", flush=True)
        return None

    saved = 0
    for key, review, images in site.parse_product(page)[:per_product]:
        for order, image in enumerate(images[:per_review]):
            name = site.frame_name(key, order)
            target = FIELD / name
            if name in known or target.exists():
                continue
            if download(image, target, referer=url):
                known[name] = {"listing": url, "listing_title": item["title"],
                               "review": review, "image": image,
                               "guess": item.get("match", {}).get("slug", "")}
                saved += 1
            time.sleep(photo_pause)
    return saved


def spread(todo: list[dict]) -> list[dict]:
    """Порядок обхода: сначала по одному товару на каждую позицию каталога.

    Сбор длинный и может прерваться — площадкой, сетью, рукой. Если идти
    подряд по уверенности, у одной позиции наберётся десяток товаров,
    а до половины каталога дело не дойдёт вовсе. Поэтому сначала первый
    круг по всем позициям, потом второй.
    """
    groups: dict[str, list[dict]] = {}
    for item in todo:
        groups.setdefault(item["match"]["slug"], []).append(item)
    for items in groups.values():
        items.sort(key=lambda i: (-i["reviews"], -i["match"]["confidence"]))
    order = sorted(groups.values(),
                   key=lambda g: (-g[0]["reviews"], -g[0]["match"]["confidence"]))
    out = []
    for depth in range(max(len(g) for g in order) if order else 0):
        out += [g[depth] for g in order if depth < len(g)]
    return out


def harvest(site, per_product: int, per_review: int, pause: float,
            min_reviews: int, limit: int) -> None:
    products = load_state(site, "products.json")
    todo = [i for i in products.values()
            if "match" in i and i["reviews"] >= min_reviews]
    todo = spread(todo)
    if limit:
        todo = todo[:limit]
    if not todo:
        sys.exit("нечего собирать: сначала discover и match")

    FIELD.mkdir(parents=True, exist_ok=True)
    index_path = FIELD / "sources.json"
    known = (json.loads(index_path.read_text(encoding="utf-8"))
             if index_path.exists() else {})
    done = set(load_state(site, "harvested.json").get("done", []))

    saved = misses = 0
    for number, item in enumerate(todo, 1):
        if item["slug"] in done:
            continue
        if misses >= 3:
            # Отказ подряд — площадке плохо, и перебор остатка очереди
            # только добавляет ей нагрузки. Выходим: непройденное
            # останется в очереди до следующего запуска.
            print("площадка отказывает подряд — останавливаюсь, "
                  "повторить запуск позже", flush=True)
            break
        got = harvest_product(site, item, per_product, per_review, known,
                              photo_pause=min(pause / 4, 0.8))
        saved += got or 0
        print(f"[{number}/{len(todo)}] {item['title'][:58]:58} "
              f"-> {item['match']['slug'][:30]:30} "
              f"{'не открылся' if got is None else f'кадров {got:3}'}", flush=True)
        if got is None:
            misses += 1
            time.sleep(pause)
            continue          # не помечаем пройденным: вернёмся следующим запуском
        misses = 0
        done.add(item["slug"])
        index_path.write_text(json.dumps(known, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        save_state(site, "harvested.json", {"done": sorted(done)})
        time.sleep(pause)

    print(f"\nсохранено новых фотографий: {saved}, всего в наборе: {len(known)}")
    print(f"папка: {FIELD}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["discover", "match", "harvest"])
    ap.add_argument("--site", choices=sorted(SITES), default="otzovik")
    ap.add_argument("--pause", type=float, default=3.0, help="пауза между запросами, с")
    ap.add_argument("--max-pages", type=int, default=400,
                    help="предел страниц на рубрику")
    ap.add_argument("--budget", type=int, default=600,
                    help="предел обращений за один запуск discover")
    ap.add_argument("--floor", type=float, default=0.30,
                    help="порог уверенности текстового матчера на шаге match")
    ap.add_argument("--reviews-per-product", type=int, default=8)
    ap.add_argument("--photos-per-review", type=int, default=3)
    ap.add_argument("--min-reviews", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="сколько товаров обойти")
    args = ap.parse_args()

    site = SITES[args.site]()
    if args.stage == "discover":
        discover(site, args.pause, args.max_pages, args.budget)
    elif args.stage == "match":
        match(site, args.floor)
    else:
        harvest(site, args.reviews_per_product, args.photos_per_review,
                args.pause, args.min_reviews, args.limit)


if __name__ == "__main__":
    main()
