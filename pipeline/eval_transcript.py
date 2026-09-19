#!/usr/bin/env python3
"""Сколько промахов исправит идеальное чтение этикетки.

Диагностика, а не приём. На промахах тестовой части текст, прочитанный
OCR, заменяется на транскрипцию, снятую глазами (data/field/transcripts.csv),
и конвейер прогоняется заново. Разница между двумя прогонами делит
оставшиеся ошибки на две части:

  исправлено идеальным чтением — упирается в распознавание текста;
  осталось                     — упирается в сопоставление с каталогом
                                 или в отбор кандидатов, и лучшим OCR
                                 не лечится.

Без этого деления непонятно, куда вкладываться: в чтение или в поиск.
Транскрипция — верхняя граница: человек читает всё, что вообще видно
на кадре, распознаватель столько не прочитает никогда.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from build_index import extra_references  # noqa: E402
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import Word, read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


@dataclass
class Typed:
    """Читатель, отдающий заранее снятую транскрипцию вместо OCR.

    Слова раскладываются столбиком по центру кадра: отбор по месту
    отсеивает надписи с краёв, и транскрипция нужного вина должна этот
    отбор проходить — она и снята с нужной бутылки.
    """
    text: str

    def __call__(self, crop: Image.Image) -> list[Word]:
        tokens = [t for t in self.text.split() if t]
        if not tokens:
            return []
        width, height = crop.size
        step = max(height // (len(tokens) + 1), 1)
        left, right = int(width * 0.3), int(width * 0.7)
        return [Word(text=token, conf=1.0,
                     box=(left, step * (i + 1) - 6, right, step * (i + 1) + 6))
                for i, token in enumerate(tokens)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--transcripts", default=str(FIELD / "transcripts.csv"))
    args = ap.parse_args()

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    findable = ({s for s, w in by_slug.items() if w.get("photo")}
                | {s for _, s in extra_references()})
    typed = {row["image"]: row["text"] for row in csv.DictReader(
        Path(args.transcripts).open(encoding="utf-8"))}

    pairs = []
    for row in csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8")):
        if row["split"] != args.split or not row["slug"]:
            continue
        if row["slug"] not in findable or row["frame"] not in ("front", "collage",
                                                               "multi"):
            continue
        if row["image"] not in typed:
            continue
        path = FIELD / row["image"]
        if not path.exists():
            path = PUBLIC / row["image"]
        if path.exists():
            pairs.append((path, row["slug"]))
    if not pairs:
        sys.exit("нет кадров с транскрипцией в этой части")

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    channel = TextChannel(wines)

    paths = [str(p) for p, _ in pairs]
    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in ((), ("detect",))]
    kept = per_view[0][1]
    row_of = {p: i for i, p in enumerate(kept)}
    blocks = [qv @ vectors.T for qv, _ in per_view]

    rows = []
    for path, gold in pairs:
        i = row_of.get(str(path))
        if i is None:
            continue
        with Image.open(path) as raw:
            source = raw.convert("RGB")
        shots = [normalize_batch([source], steps=steps)[0]
                 for steps in ((), ("detect",))]
        similarity = np.stack([block[i] for block in blocks], axis=1)
        shot = shots[core.pick_view(shots, similarity)]

        answers = {}
        for name, reader in (("OCR", read_words),
                             ("транскрипция", Typed(typed[path.name]))):
            candidates = core.shortlist(similarity.max(axis=1), index_slugs,
                                        args.topk)
            core.rerank(shot, candidates, reranker)
            label = core.LabelText(shot, reader)
            core.settle(candidates, shot, channel, label, window=0.80,
                        min_conf=0.60)
            core.check_leader(candidates, shot, channel, label, 0.60, 0.80,
                              by_slug)
            order = [c.slug for c in candidates]
            answers[name] = (order[0] if order else "", gold in order[:5])
        rows.append((path.name, gold, answers))

    fixed = [r for r in rows if r[2]["OCR"][0] != r[1]
             and r[2]["транскрипция"][0] == r[1]]
    broken = [r for r in rows if r[2]["OCR"][0] == r[1]
              and r[2]["транскрипция"][0] != r[1]]
    stayed = [r for r in rows if r[2]["OCR"][0] != r[1]
              and r[2]["транскрипция"][0] != r[1]]

    print(f"\nкадров с транскрипцией: {len(rows)} (часть {args.split})")
    print(f"  исправлено идеальным чтением {len(fixed)}")
    print(f"  сломано идеальным чтением    {len(broken)}")
    print(f"  осталось промахом            {len(stayed)}")
    if fixed:
        print("\nисправились — упирается в чтение:")
        for name, gold, _ in fixed:
            print(f"  {name[:28]:28} {by_slug[gold]['title'][:40]}")
    if stayed:
        print("\nостались — упирается в сопоставление или отбор:")
        for name, gold, answers in stayed:
            in_five = "в пятёрке" if answers["транскрипция"][1] else "вне пятёрки"
            print(f"  {name[:28]:28} {in_five:11} "
                  f"выдано {by_slug.get(answers['транскрипция'][0], {}).get('title', '?')[:32]}")
    if broken:
        print("\nсломались — правило текста вредит даже на верном чтении:")
        for name, gold, answers in broken:
            print(f"  {name[:28]:28} стало "
                  f"{by_slug.get(answers['транскрипция'][0], {}).get('title', '?')[:36]}")

    out = INDEX_DIR / f"transcript_{args.split}.json"
    out.write_text(json.dumps(
        {"split": args.split, "n": len(rows), "fixed": [r[0] for r in fixed],
         "broken": [r[0] for r in broken], "stayed": [r[0] for r in stayed]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
