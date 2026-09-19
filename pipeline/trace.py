#!/usr/bin/env python3
"""Цепочка решений по одному снимку: от кропа до статуса выдачи.

Нужен для разбора конкретных подмен внутри линейки. Общие метрики говорят,
что ошибок столько-то, но не говорят, на каком шаге всё пошло не так,
а шагов пять: кадрирование, отбор кандидатов, геометрия, чтение текста,
решение о выдаче. Без развёрнутой цепочки причину приходится угадывать.

Печатает по шагам и, если попросить, складывает кроп и разметку строк OCR
картинкой — видно, что именно попало в область чтения.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import read_words  # noqa: E402
from rerank import MAX_SIDE, PrecomputedReranker  # noqa: E402
from text_match import (TextChannel, card_features, central_words,  # noqa: E402
                        discriminating, extract_attributes)

INDEX_DIR = ROOT / "data" / "index"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def reference_image(slug: str, by_slug: dict) -> Image.Image | None:
    """Эталонное фото позиции: из дампа или из дополнительных эталонов."""
    from build_index import extra_references
    photo = (by_slug.get(slug) or {}).get("photo")
    if photo:
        path = next((ROOT / "dataset" / "uploads_root").rglob("uploads")) / Path(photo).name
        if path.exists():
            return Image.open(path).convert("RGB")
    for extra_path, extra_slug in extra_references():
        if extra_slug == slug:
            return Image.open(extra_path).convert("RGB")
    return None


def pair_picture(crop: Image.Image, slugs: list[str], by_slug: dict,
                 out: Path) -> None:
    """Совпавшие точки запроса и эталона, по строке на кандидата.

    Нужно, чтобы увидеть глазами, чем именно набраны инлаеры. Проценты
    говорят, что лидер выиграл по числу точек, но не говорят, что точки
    эти легли на общую для всей линейки гравюру дворца, а не на название
    сорта.
    """
    from rerank import descriptors, inlier_pairs

    query_kp, query_desc = descriptors(crop)
    rows = []
    for slug in dict.fromkeys(slugs):
        ref = reference_image(slug, by_slug)
        if ref is None:
            continue
        ref_kp, ref_desc = descriptors(ref)
        # descriptors масштабирует картинку; для рисования нужен тот же размер.
        scaled = ref.copy()
        scaled.thumbnail((MAX_SIDE, MAX_SIDE))
        rows.append((slug, scaled, inlier_pairs(query_kp, query_desc,
                                                ref_kp, ref_desc)))
    if not rows:
        return
    width = max(crop.width + ref.width for _, ref, _ in rows)
    height = sum(max(crop.height, ref.height) + 26 for _, ref, _ in rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for slug, ref, pairs in rows:
        sheet.paste(crop, (0, y + 26))
        sheet.paste(ref, (crop.width, y + 26))
        draw.text((4, y + 6), f"{by_slug[slug]['title'][:60]} — точек {len(pairs)}",
                  fill="black")
        for (qx, qy), (rx, ry) in pairs:
            draw.line((qx, y + 26 + qy, crop.width + rx, y + 26 + ry),
                      fill="#c02020", width=1)
            draw.ellipse((qx - 2, y + 24 + qy, qx + 2, y + 28 + qy), outline="#1060c0")
        y += max(crop.height, ref.height) + 26
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--gold", default="", help="правильный slug, если известен")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--picture", default="", help="куда сохранить кроп с разметкой OCR")
    ap.add_argument("--matches", default="",
                    help="куда сохранить картинку совпавших точек: лидер против "
                         "правильного кандидата, точка к точке")
    args = ap.parse_args()

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    weights = json.loads((INDEX_DIR / "confidence.json")
                         .read_text(encoding="utf-8"))["outcome_weights"]
    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    channel = TextChannel(wines)

    image = Image.open(args.image).convert("RGB")
    views = {steps: normalize_batch([image], steps=steps)[0]
             for steps in ((), ("detect",))}
    crop = views[("detect",)]
    print(f"1. кадрирование: исходник {image.size[0]}×{image.size[1]}, "
          f"кроп по бутылке {crop.size[0]}×{crop.size[1]}")

    query = embed_images(list(views.values()))
    scores = (vectors @ query.T).max(axis=1)
    candidates = core.shortlist(scores, index_slugs, args.topk)
    place = [c.slug for c in candidates].index(args.gold) if args.gold in [
        c.slug for c in candidates] else -1
    print(f"2. отбор: {len(candidates)} кандидатов; правильный на месте "
          f"{place if place >= 0 else 'вне шортлиста'}")

    core.rerank(crop, candidates, reranker)
    order = [c.slug for c in candidates]
    place = order.index(args.gold) if args.gold in order else -1
    print(f"3. геометрия: лидер {by_slug[order[0]]['title'][:40]} "
          f"({candidates[0].inliers} инлаеров); правильный на месте "
          f"{place if place >= 0 else 'вне шортлиста'}"
          + (f", инлаеров {candidates[place].inliers}" if place >= 0 else ""))
    for candidate in candidates[:5]:
        mark = " <- правильный" if candidate.slug == args.gold else ""
        print(f"     {candidate.inliers:>4} инл  {by_slug[candidate.slug]['title'][:44]}{mark}")

    words = read_words(crop)
    central = central_words(words, 0.60)
    print(f"4. чтение: строк всего {len(words)}, уверенных в центре "
          f"{len(central)}")
    for word in words:
        inside = "центр" if word in central else "     "
        print(f"     {inside} conf {word.conf:.2f}  bbox {word.box}  «{word.text}»")

    attrs = extract_attributes(" ".join(w.text for w in central))
    print(f"5. признаки из текста: цвет={attrs.color} сладость={attrs.sweetness} "
          f"год={sorted(attrs.years) or '—'} крепость={attrs.alcohol}")
    leader = card_features(order[0], channel)
    print(f"   карточка лидера: цвет={leader.color} сладость={leader.sweetness} "
          f"сорт={sorted(leader.grapes)} год={sorted(leader.years) or '—'}")
    marks = discriminating(order[:5], channel)
    for slug in order[:5]:
        print(f"   различает {by_slug[slug]['title'][:34]:34} {sorted(marks[slug])[:6]}")

    core.resolve(candidates, crop, channel, read_words, window=0.80, min_conf=0.60)
    conflicts = core.check_leader(candidates, crop, channel, read_words, 0.60, 0.80)
    feats = core.features(candidates)
    probability, probability5 = core.outcomes(feats, weights)
    print(f"6. текст против лидера: улик {conflicts} "
          f"{candidates[0].conflicts[:2] if candidates[0].conflicts else ''}")
    print(f"7. решение: лидер {by_slug[candidates[0].slug]['title'][:40]}, "
          f"уверенность {probability:.3f} (в пятёрке {probability5:.3f}), "
          f"признаки { {k: round(v, 3) for k, v in feats.items()} }")

    if args.matches and args.gold:
        pair_picture(crop, [candidates[0].slug, args.gold], by_slug,
                     Path(args.matches))
        print(f"\nсовпавшие точки: {args.matches} "
              f"(слева запрос, справа эталон; линия — пара точек)")

    if args.picture:
        marked = crop.copy()
        draw = ImageDraw.Draw(marked)
        for word in words:
            color = "#0a0" if word in central else "#a00"
            draw.rectangle(word.box, outline=color, width=2)
        Path(args.picture).parent.mkdir(parents=True, exist_ok=True)
        marked.save(args.picture)
        print(f"\nкроп с разметкой строк: {args.picture} "
              f"(зелёным — что пошло в разбор)")


if __name__ == "__main__":
    main()
