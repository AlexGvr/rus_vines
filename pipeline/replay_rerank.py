#!/usr/bin/env python3
"""Переигрывание ранжирования по сохранённому прогону eval_field.py.

Прогон с --candidate-features хранит для каждого кадра все двадцать
кандидатов (косинус, инлаеры, покрытие) и строки, прочитанные OCR. Этого
достаточно, чтобы заново пройти rerank -> settle -> check_leader ->
признаки уверенности без видеокарты: эмбеддинги, SIFT и OCR не
пересчитываются, меняются только правила порядка, заданные окружением.

Сначала прогон переигрывается как есть и сверяется с сохранённым кадр
в кадр: лидер и уверенность должны совпасть. Если не совпадают — replay
не описывает конвейер, и сравнение правил по нему ничего не значит.

    GEOMETRY_MARGIN=0.25 python pipeline/replay_rerank.py data/validation/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from ocr import Word  # noqa: E402
from rerank import MatchStats  # noqa: E402
from text_match import TextChannel  # noqa: E402

CATALOG = ROOT / "data" / "catalog" / "catalog.json"


class StoredGeometry:
    """Реранкер, отдающий сохранённые инлаеры и покрытие."""

    def __init__(self, stored: list[dict]) -> None:
        self.by_slug = {c["slug"]: c for c in stored}

    def stats(self, _crop, slugs: list[str]) -> dict[str, MatchStats]:
        return {slug: MatchStats(int(self.by_slug[slug]["inliers"]), 0,
                                 float(self.by_slug[slug]["coverage"]), 0.0)
                for slug in slugs if slug in self.by_slug}


class StoredText:
    """LabelText с сохранёнными строками вместо OCR."""

    def __init__(self, words: list[dict]) -> None:
        self.words = [Word(w["text"], float(w["conf"]), tuple(w["box"])) for w in words]

    def __call__(self, _crop=None) -> list:
        return self.words


def replay(row: dict, channel, by_slug: dict, weights, index_order: dict,
           geometry=None) -> dict:
    """Решение по сохранённым кандидатам; geometry — другой источник инлаеров."""
    stored = row["candidates"]
    # Порядок шортлиста до геометрии: по убыванию косинуса, ничья — порядком
    # строк индекса, как в core.shortlist.
    candidates = [core.Candidate(slug=c["slug"], cv=float(c["cv"]))
                  for c in sorted(stored, key=lambda c: (-c["cv"], index_order.get(c["slug"], 0)))]
    core.rerank(None, candidates, geometry or StoredGeometry(stored))
    label = StoredText(row.get("words") or [])
    core.settle(candidates, None, channel, label, window=0.80, min_conf=0.60, by_slug=by_slug)
    core.check_leader(candidates, None, channel, label, 0.60, 0.80, by_slug)
    feats = core.features(candidates)
    top = candidates[0].slug if candidates else ""
    return {"top1": top,
            "probability": core.outcomes(feats, weights)[0],
            "correct": bool(row["gold"]) and top == row["gold"],
            "in_top5": bool(row["gold"]) and row["gold"] in [c.slug for c in candidates[:5]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="отчёты eval_field.py с --candidate-features")
    ap.add_argument("--weights", default=str(ROOT / "data" / "index" / "confidence.json"))
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--output", help="куда записать переигранные результаты (JSON)")
    args = ap.parse_args()

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    channel = TextChannel(wines)
    weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))["outcome_weights"]
    import csv
    index_order: dict[str, int] = {}
    with (ROOT / "data" / "index" / f"{args.index}.csv").open(encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            index_order.setdefault(r["slug"], i)

    print("правила:", {k: v for k, v in core.switches().items() if v is not None})
    out = {}
    for path in args.runs:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = [r for r in run["results"] if r.get("candidates")]
        changed = []
        replayed = []
        for r in rows:
            new = replay(r, channel, by_slug, weights, index_order)
            replayed.append({"image": r["image"], "gold": r["gold"], "findable": r["findable"],
                             "stored_top1": r["top1"], "stored_probability": r["probability"],
                             **new})
            if new["top1"] != r["top1"] or abs(new["probability"] - r["probability"]) > 1e-6:
                changed.append((r, new))
        pos = [x for x in replayed if x["findable"]]
        was = sum(1 for r in rows if r["findable"] and r["correct"])
        now = sum(1 for x in pos if x["correct"])
        print(f"\n{path}: кадров {len(rows)}, находимых {len(pos)}; "
              f"top-1 было {was}, стало {now}; изменений {len(changed)}")
        for r, new in changed:
            mark = ("+" if new["correct"] and not r["correct"] else
                    "-" if r["correct"] and not new["correct"] else " ")
            print(f"  {mark} {r['image']:22} {str(r['gold'])[:34]:34} "
                  f"{r['top1'][:30]:30} p={r['probability']:.3f} -> "
                  f"{new['top1'][:30]:30} p={new['probability']:.3f}")
        out[path] = replayed
    if args.output:
        Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
