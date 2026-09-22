#!/usr/bin/env python3
"""Сколько промахов исправит идеальное чтение этикетки.

Диагностика, а не приём. На кадрах с транскрипцией, снятой глазами
(`data/field/transcripts_<split>.json`), текст OCR заменяется на неё,
а визуальные кандидаты, выбор кропа и геометрия остаются те же, что в
сервисе. Разница двух прогонов делит промахи на части:

  исправлено идеальным чтением — упирается в распознавание текста;
  осталось                     — упирается в сопоставление с каталогом,
                                 отбор кандидатов, эталон или разметку;
  сломано                      — правило текста вредит даже на верном
                                 чтении.

Конвейер повторяет `eval_field.py` и `service/main.py` шаг в шаг: те же
виды запроса, тот же выбор кропа по косинусу, тот же шортлист и геометрия,
те же `settle`/`check_leader` с теми же параметрами. Отличие одно —
читатель. Транскрипция — верхняя граница: человек читает всё, что видно,
распознаватель столько не прочитает.

Формат транскрипции: {"image": [{"text": "...", "x": 0.5, "y": 0.7,
"w": 0.4, "unreadable": false, "note": "..."}]}. Координаты — доли ширины
и высоты кропа целевой бутылки (центр строки и ширина); слова с пометкой
unreadable не подставляются, они нужны, чтобы видеть, где на этикетке
текст есть, но прочитать его нельзя даже глазами. Слова из каталога, которых
на снимке не видно, в транскрипцию не дописываются.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from eval_field import SCORABLE, load_equivalents, locate  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import Word, read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel, card_features  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
FIELD = ROOT / "data" / "field"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


@dataclass
class Typed:
    """Читатель, отдающий транскрипцию вместо OCR, в координатах кропа."""
    lines: list[dict]

    def __call__(self, crop: Image.Image) -> list[Word]:
        width, height = crop.size
        out = []
        for line in self.lines:
            if line.get("unreadable") or not line.get("text", "").strip():
                continue
            cx, cy = float(line.get("x", 0.5)) * width, float(line.get("y", 0.7)) * height
            half = float(line.get("w", 0.4)) * width / 2
            out.append(Word(text=line["text"].strip(), conf=float(line.get("conf", 1.0)),
                            box=(int(cx - half), int(cy - 8), int(cx + half), int(cy + 8))))
        return out


def run_once(shot, candidates_source, reader, channel, by_slug, reranker, window, min_conf):
    candidates = [core.Candidate(c.slug, c.cv, 0, 0.0) for c in candidates_source]
    core.rerank(shot, candidates, reranker)
    after_geometry = [c.slug for c in candidates]
    label = core.LabelText(shot, reader)
    core.settle(candidates, shot, channel, label, window=window, min_conf=min_conf,
                by_slug=by_slug)
    core.check_leader(candidates, shot, channel, label, min_conf, window, by_slug)
    words = label(shot)
    return {
        "order": [c.slug for c in candidates],
        "after_geometry": after_geometry,
        "inliers": {c.slug: c.inliers for c in candidates},
        "contradictions": {c.slug: list(getattr(c, "contradictions", []) or [])
                           for c in candidates if getattr(c, "contradictions", None)},
        "hard": {c.slug: list(getattr(c, "hard", []) or []) for c in candidates
                 if getattr(c, "hard", None)},
        "conflicts": list(candidates[0].conflicts) if candidates else [],
        "confirmed": list(candidates[0].confirmed) if candidates else [],
        "words": [{"text": w.text, "conf": round(w.conf, 2), "box": list(w.box)} for w in words],
    }


def diagnose(gold: str, before: dict, after: dict, channel: TextChannel,
             findable: set[str]) -> str:
    """Куда упирается промах после идеального чтения."""
    if gold not in findable:
        return "эталон отсутствует"
    if gold not in after["order"]:
        return "кандидат не в шортлисте"
    if after["order"][0] == gold:
        return "исправлено чтением"
    top = after["order"][0]
    g, t = card_features(gold, channel), card_features(top, channel)
    diffs = []
    if g.sweetness != t.sweetness:
        diffs.append("сладость")
    if g.color != t.color:
        diffs.append("цвет")
    if g.name != t.name:
        diffs.append("имя")
    if g.grapes != t.grapes:
        diffs.append("сорт")
    if g.years != t.years:
        diffs.append("год")
    if not diffs:
        return "карточки не различимы текстом (дубль или разметка)"
    if after["after_geometry"][0] != gold and after["order"][0] == after["after_geometry"][0]:
        return "текст не переставил (различие: " + ", ".join(diffs) + ")"
    return "текстовое сравнение (различие: " + ", ".join(diffs) + ")"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="tune")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--views", default="raw,detect")
    ap.add_argument("--window", type=float, default=0.80)
    ap.add_argument("--min-conf", type=float, default=0.60)
    ap.add_argument("--transcripts", default="")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    transcripts_path = Path(args.transcripts or (FIELD / f"transcripts_{args.split}.json"))
    typed = json.loads(transcripts_path.read_text(encoding="utf-8"))

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    findable = {s for s in index_slugs if s}
    reranker = PrecomputedReranker(os.environ.get("SIFT_DIR", str(INDEX_DIR / "sift")))
    channel = TextChannel(wines)
    equivalents = load_equivalents(by_slug)
    query_views = [tuple(x for x in spec.split("+") if x and x != "raw")
                   for spec in args.views.split(",")]

    rows = [r for r in csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8"))
            if r["split"] == args.split and r["image"] in typed]
    if not rows:
        sys.exit("нет кадров с транскрипцией в этой части")

    records = []
    for row in rows:
        path = locate(row["image"])
        if path is None:
            sys.exit(f"нет кадра {row['image']}")
        image = Image.open(path).convert("RGB")
        views = {steps: normalize_batch([image], steps=steps)[0] for steps in query_views}
        query = embed_images(list(views.values()))
        similarity = vectors @ query.T
        scores = similarity.max(axis=1)
        shortlist = core.shortlist(scores, index_slugs, args.topk)
        shot = list(views.values())[core.pick_view(list(views.values()), similarity)]
        before = run_once(shot, shortlist, read_words, channel, by_slug, reranker,
                          args.window, args.min_conf)
        after = run_once(shot, shortlist, Typed(typed[row["image"]]), channel, by_slug,
                         reranker, args.window, args.min_conf)
        gold = row["slug"]
        records.append({
            "image": row["image"], "gold": gold, "frame": row["frame"],
            "findable": bool(gold) and gold in findable and row["frame"] in SCORABLE,
            "top1_ocr": before["order"][0] if before["order"] else "",
            "top1_typed": after["order"][0] if after["order"] else "",
            "top5_ocr": gold in before["order"][:5], "top5_typed": gold in after["order"][:5],
            "rank_geometry": (before["after_geometry"].index(gold)
                              if gold in before["after_geometry"] else -1),
            "inliers_gold": before["inliers"].get(gold), "inliers_top1": before["inliers"].get(before["order"][0]) if before["order"] else None,
            "ocr_words": before["words"], "typed_words": after["words"],
            "unreadable": [l for l in typed[row["image"]] if l.get("unreadable")],
            "reasons_typed": {k: v for k, v in after["contradictions"].items()
                              if k in (gold, after["order"][0])},
            "hard_typed": after["hard"],
            "diagnosis": diagnose(gold, before, after, channel, findable) if gold else "вина нет в каталоге",
        })
        title = lambda s: by_slug.get(s, {}).get("title", s or "—")[:34]
        print(f"  {row['image'][:30]:30} OCR: {title(records[-1]['top1_ocr']):34} "
              f"→ транскрипция: {title(records[-1]['top1_typed']):34} | {records[-1]['diagnosis']}",
              flush=True)

    scored = [r for r in records if r["gold"] and r["findable"]]
    fixed = [r for r in scored if r["top1_ocr"] != r["gold"] and r["top1_typed"] == r["gold"]]
    broken = [r for r in scored if r["top1_ocr"] == r["gold"] and r["top1_typed"] != r["gold"]]
    stayed = [r for r in scored if r["top1_ocr"] != r["gold"] and r["top1_typed"] != r["gold"]]
    print(f"\nкадров с транскрипцией: {len(records)} (часть {args.split}), "
          f"с эталоном и лицевой этикеткой {len(scored)}")
    print(f"  верных по OCR {sum(1 for r in scored if r['top1_ocr'] == r['gold'])}, "
          f"по транскрипции {sum(1 for r in scored if r['top1_typed'] == r['gold'])}")
    print(f"  исправлено идеальным чтением {len(fixed)}, сломано {len(broken)}, осталось {len(stayed)}")
    counts: dict[str, int] = {}
    for r in stayed:
        counts[r["diagnosis"]] = counts.get(r["diagnosis"], 0) + 1
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {name:60} {n}")

    def digest(path: Path) -> str:
        with path.open("rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    out = Path(args.output or (ROOT / "data" / "validation" / f"transcript_{args.split}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "provenance": {"args": vars(args),
                       "sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  (transcripts_path, FIELD / "manifest.csv", CATALOG,
                                   INDEX_DIR / f"{args.index}.csv",
                                   ROOT / "pipeline" / "searchcore.py",
                                   ROOT / "pipeline" / "text_match.py",
                                   ROOT / "pipeline" / "eval_transcript.py")},
                       "environment": {k: os.environ.get(k) for k in
                                       ("TEXT_ALIASES", "OCR_JOIN", "NAME_FIRST", "LINE_BAND",
                                        "PLATFORM_SWEETNESS", "KIN_GROUP", "BRAND_FIRST",
                                        "LABEL_BAND", "SIFT_MAX_SIDE")},
                       "diagnostic_only": True},
        "summary": {"n": len(records), "scored": len(scored),
                    "correct_ocr": sum(1 for r in scored if r["top1_ocr"] == r["gold"]),
                    "correct_typed": sum(1 for r in scored if r["top1_typed"] == r["gold"]),
                    "fixed": [r["image"] for r in fixed], "broken": [r["image"] for r in broken],
                    "stayed": {r["image"]: r["diagnosis"] for r in stayed}},
        "records": records,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
