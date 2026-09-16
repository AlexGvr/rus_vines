#!/usr/bin/env python3
"""Разметка полевого набора: какое вино на фотографии и годится ли кадр.

Привязку нельзя доверить автомату. Название страницы отзывов совпадает
с каталогом приблизительно, а каталог полон позиций, различающихся одним
словом: «Мускатель белый» против «Мускатель розовый». Ошибка разметки
здесь дороже, чем в синтетике, — набор маленький, и каждая неверная метка
сдвигает пороги.

Поэтому скрипт только готовит материал для решения человеком:

  --sheets   для каждой страницы отзывов рисует лист: снимки покупателей
             рядом с эталонами каталога, которые предложил текстовый матчер
  --build    собирает манифест по файлу решений

Файл решений — простой текст, по строке на страницу отзывов:

    <ключ страницы> <slug каталога|-> [исключить: файл файл ...]

Прочерк вместо slug означает, что вина нет в каталоге: такие кадры идут
в набор как отрицательные примеры, они нужны для калибровки отказа не
меньше положительных.

Из кадров отбрасываются контрэтикетки, пробки и всё, где лицевой этикетки
не видно: сервис заявлен как поиск по лицевой этикетке, и мерить его на
обороте бутылки нечестно.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from matcher import Matcher  # noqa: E402

FIELD = ROOT / "data" / "field"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def series_of(name: str) -> str:
    """Идентификатор фотосерии — все кадры одного отзыва.

    Кадры одного отзыва снимает один человек, одну бутылку, за один раз.
    Держать их в разных частях выборки нельзя: настройка увидит ту же
    бутылку, что и финальный тест, и качество окажется завышенным.
    """
    stem = Path(name).stem
    if stem.startswith("ir_"):
        return stem[3:].rsplit("_", 1)[0]
    head = stem.rsplit("_", 1)[0]
    return head or stem


def split_of(wine_or_series: str, wines: list[str], series: list[str]) -> str:
    """Часть выборки: настройка или финальный тест.

    Деление идёт по позициям каталога, а не по кадрам: иначе одно вино
    попадёт в обе части, и тест перестанет отвечать на вопрос о переносе
    на другие вина. Отрицательные кадры делятся по фотосерии — позиции
    у них нет.

    Обучающей части у полевого набора нет намеренно: сорока семи кадров
    не хватит, чтобы обучать на них веса, и они обучаются на синтетике.
    Здесь выбираются только пороги, и для этого нужны две части.
    """
    pool = wines if wine_or_series in wines else series
    return "tune" if pool.index(wine_or_series) % 2 == 0 else "test"


def key_of(title: str) -> str:
    """Короткий ключ товара из его названия — по нему идёт разметка."""
    slug = re.sub(r"[^a-zа-яё0-9]+", "-", title.lower()).strip("-")
    return slug[:48] or "bez-nazvaniya"


def groups() -> dict[str, dict]:
    """Фотографии, сгруппированные по товару.

    Группировать по адресу страницы нельзя: страница бренда отдаёт отзывы
    сразу на десяток разных вин, и они слиплись бы в одну группу.
    """
    sources = json.loads((FIELD / "sources.json").read_text(encoding="utf-8"))
    out: dict[str, dict] = defaultdict(lambda: {"title": "", "files": []})
    for name, meta in sorted(sources.items()):
        if not (FIELD / name).exists():
            continue
        title = meta["listing_title"]
        key = key_of(title)
        out[key]["title"] = title
        out[key]["files"].append(name)
    return dict(out)


def sheets(out_dir: Path, per_group: int = 12) -> None:
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    matcher = Matcher(wines)
    by_slug = {w["slug"]: w for w in wines}
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 13)
    small = ImageFont.truetype(FONT, 11)

    for key, group in groups().items():
        found = matcher.search(group["title"], top=5)
        files = group["files"][:per_group]
        cell = 200
        cols = max(len(files), len(found), 1)
        sheet = Image.new("RGB", (cols * cell, 2 * (cell + 46) + 24), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 4), f"{key}   «{group['title'][:80]}»", fill="black", font=font)

        for i, name in enumerate(files):
            image = Image.open(FIELD / name).convert("RGB")
            image.thumbnail((cell - 8, cell - 8))
            draw.text((i * cell + 4, 24), name, fill="#333", font=small)
            sheet.paste(image, (i * cell + (cell - image.width) // 2, 40))

        base = cell + 70
        for i, hit in enumerate(found):
            wine = by_slug[hit["slug"]]
            draw.text((i * cell + 4, base - 16),
                      f"{i + 1}) {hit['confidence']:.2f} {wine['manufacturer'][:20]}",
                      fill="#a00", font=small)
            if wine.get("photo") and (ROOT / wine["photo"]).exists():
                image = Image.open(ROOT / wine["photo"]).convert("RGB")
                image.thumbnail((cell - 8, cell - 8))
                sheet.paste(image, (i * cell + (cell - image.width) // 2, base))
            draw.text((i * cell + 4, base + cell + 2), wine["title"][:28],
                      fill="#333", font=small)
            draw.text((i * cell + 4, base + cell + 16), hit["slug"][:30],
                      fill="#777", font=small)
        sheet.save(out_dir / f"{key}.png")
    print(f"листы разметки: {out_dir}")


def triage(out_dir: Path, per_sheet: int = 8) -> None:
    """Листы разметки: снимок рядом с тем, что находит сам поиск.

    Сопоставление по названию товара оказалось слабым: у части вин на
    снимках покупателей этикетка другого поколения, чем в каталоге, и
    название совпадает, а изображение — нет. Такой кадр положительным
    примером считать нельзя, и увидеть это можно только глазами.

    Поэтому листы строятся от изображения: прогоняем полный конвейер,
    показываем три первых кандидата с их эталонами и решаем.
    """
    import numpy as np
    sys.path.insert(0, str(ROOT / "pipeline"))
    import searchcore as core
    from embed import embed_images
    from imageprep import normalize_batch
    from ocr import read_words
    from rerank import PrecomputedReranker
    from text_match import TextChannel

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    index_dir = ROOT / "data" / "index"
    vectors = np.load(index_dir / "clean_mv.npy")
    with (index_dir / "clean_mv.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(index_dir / "sift")
    channel = TextChannel(wines)

    files = sorted(p for p in FIELD.glob("*.jpg"))
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 12)
    found: dict[str, list] = {}

    for start in range(0, len(files), per_sheet):
        chunk = files[start:start + per_sheet]
        cell = 210
        sheet = Image.new("RGB", (cell * 4, len(chunk) * (cell + 34)), "white")
        draw = ImageDraw.Draw(sheet)
        for row, path in enumerate(chunk):
            image = Image.open(path).convert("RGB")
            views = {steps: normalize_batch([image], steps=steps)[0]
                     for steps in ((), ("detect",))}
            query = embed_images(list(views.values()))
            scores = (vectors @ query.T).max(axis=1)
            candidates = core.shortlist(scores, index_slugs, 20)
            core.rerank(views[("detect",)], candidates, reranker)
            core.resolve(candidates, views[("detect",)], channel, read_words,
                         window=0.80, min_conf=0.60)
            top = candidates[:3]
            found[path.name] = [{"slug": c.slug, "inliers": c.inliers,
                                 "cv": round(c.cv, 4)} for c in top]

            y = row * (cell + 34)
            preview = image.copy(); preview.thumbnail((cell - 8, cell - 8))
            sheet.paste(preview, ((cell - preview.width) // 2, y + 4))
            draw.text((4, y + cell + 6), path.name[:30], fill="black", font=font)
            for i, candidate in enumerate(top):
                wine = by_slug.get(candidate.slug, {})
                x = (i + 1) * cell
                if wine.get("photo") and (ROOT / wine["photo"]).exists():
                    reference = Image.open(ROOT / wine["photo"]).convert("RGB")
                    reference.thumbnail((cell - 8, cell - 8))
                    sheet.paste(reference, (x + (cell - reference.width) // 2, y + 4))
                draw.text((x + 4, y + cell + 6),
                          f"{i + 1}) инл {candidate.inliers} {wine.get('title', '')[:26]}",
                          fill="#a00", font=font)
                draw.text((x + 4, y + cell + 20), candidate.slug[:34],
                          fill="#777", font=font)
        sheet.save(out_dir / f"triage_{start // per_sheet:02d}.png")
        print(f"  лист {start // per_sheet}", flush=True)

    (FIELD / "candidates.json").write_text(
        json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"листы: {out_dir}, кандидаты: {FIELD / 'candidates.json'}")


def build(decisions_path: Path, out_path: Path) -> None:
    """Манифест по решениям, принятым глазами. Строка на кадр:

        <файл> <slug каталога | - | x> [field|studio]   # комментарий

    slug — позиция каталога, изображённая на кадре (положительный пример);
    «-»  — такой позиции в каталоге нет, любая карточка для неё ошибка;
    «x»  — кадр не годится: контрэтикетка, пробка, лицевой этикетки не видно.

    Правильный ответ определяется позицией, а не дизайном этикетки. Другое
    поколение упаковки отсутствием позиции не является: у «Абрау Купаж
    тёмный» в каталоге эталон с рисунком машинки, покупатели снимают
    бутылки с пейзажем, но вино то же самое. Если засчитать такой кадр
    как отсутствующее вино, ненайденная существующая позиция превратится
    в «правильный отказ», и метрика отказов станет ложью.

    Третье поле отделяет съёмку от предметной карточки товара: снимок
    на белом фоне устойчивость к съёмке телефоном не проверяет.
    """
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    known = {w["slug"] for w in catalog}
    sources = json.loads((FIELD / "sources.json").read_text(encoding="utf-8"))

    rows, dropped = [], 0
    for line in decisions_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        name, label = parts[0], parts[1]
        kind = parts[2] if len(parts) > 2 else "field"
        if not (FIELD / name).exists() and not (ROOT / "dataset" / "eval" / "queries" / name).exists():
            sys.exit(f"нет такого кадра: {name}")
        if label == "x":
            dropped += 1
            continue
        if label != "-" and label not in known:
            sys.exit(f"нет такого slug в каталоге: {label}")
        if kind not in ("field", "studio"):
            sys.exit(f"{name}: третье поле должно быть field или studio, не «{kind}»")
        rows.append({"image": name, "slug": "" if label == "-" else label,
                     "kind": kind, "series": series_of(name),
                     "source": sources.get(name, {}).get("review", "публичный набор кейса"),
                     "title": sources.get(name, {}).get("listing_title", "")})

    # Деление на части: по вину для положительных, по фотосерии для
    # отрицательных. Порядок фиксирован сортировкой, поэтому повторный
    # сбор манифеста даёт то же самое.
    wines_order = sorted({r["slug"] for r in rows if r["slug"]})
    series_order = sorted({r["series"] for r in rows if not r["slug"]})
    for row in rows:
        row["split"] = split_of(row["slug"] or row["series"], wines_order, series_order)

    crossing = {s for s in {r["series"] for r in rows}
                if len({r["split"] for r in rows if r["series"] == s}) > 1}
    if crossing:
        sys.exit(f"фотосерии попали в разные части: {sorted(crossing)[:5]}")

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["image", "slug", "kind", "split",
                                                "series", "source", "title"])
        writer.writeheader()
        writer.writerows(rows)
    positive = sum(1 for r in rows if r["slug"])
    shot = sum(1 for r in rows if r["kind"] == "field")
    print(f"манифест: {out_path}")
    print(f"кадров {len(rows)}: вино есть в каталоге {positive}, "
          f"вина в каталоге нет {len(rows) - positive}; отброшено кадров {dropped}")
    print(f"из них съёмка: {shot}, предметная карточка товара: {len(rows) - shot}")
    for part in ("tune", "test"):
        chunk = [r for r in rows if r["split"] == part]
        print(f"  {part}: кадров {len(chunk)}, позиций "
              f"{len({r['slug'] for r in chunk if r['slug']})}, "
              f"без вина в каталоге {sum(1 for r in chunk if not r['slug'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", action="store_true", help="листы по названию товара")
    ap.add_argument("--triage", action="store_true",
                    help="листы от изображения: снимок рядом с находками поиска")
    ap.add_argument("--build", metavar="ФАЙЛ", help="собрать манифест по решениям")
    ap.add_argument("--out", default=str(FIELD / "manifest.csv"))
    ap.add_argument("--sheet-dir",
                    default=str(ROOT / "data" / "field" / "sheets"))
    args = ap.parse_args()
    if args.sheets:
        sheets(Path(args.sheet_dir))
    if args.triage:
        triage(Path(args.sheet_dir))
    if args.build:
        build(Path(args.build), Path(args.out))
    if not (args.sheets or args.triage or args.build):
        ap.error("укажите --sheets, --triage или --build")


if __name__ == "__main__":
    main()
