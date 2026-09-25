#!/usr/bin/env python3
"""Перебор θ правила TEXT_VOTE по сохранённым прогонам (без видеокарты).

Для каждого θ решение переигрывается (replay_rerank.replay) по кандидатам
и строкам OCR из прогонов eval_field.py. θ выбирается по настроечной
части — наибольший top-1 по находимым кадрам организатора, при равенстве
по всей настроечной части, при равенстве меньший θ, — и один раз
проверяется на регрессионной.

    python pipeline/experiment_text_vote.py [--output out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_rerank import replay  # noqa: E402
from stats import mcnemar_exact  # noqa: E402
from text_match import TextChannel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("data/validation/dsonly_cal2_tune.json", "data/validation/dsonly_cal_test.json")
GRID = (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
SHOW_P = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    args = ap.parse_args()
    wines = json.loads((ROOT / "data/catalog/catalog.json").read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    channel = TextChannel(wines)
    weights = json.loads((ROOT / "data/index/confidence.json").read_text())["outcome_weights"]
    index_order: dict[str, int] = {}
    with (ROOT / "data/index/clean_mv.csv").open(encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            index_order.setdefault(r["slug"], i)
    rows = []
    for path in RUNS:
        split = "tune" if "tune" in path else "test"
        rows += [dict(r, split=split) for r in
                 json.loads((ROOT / path).read_text(encoding="utf-8"))["results"]
                 if r.get("candidates")]

    def run(theta: float | None) -> dict:
        if theta is None:
            os.environ.pop("TEXT_VOTE", None)
        else:
            os.environ["TEXT_VOTE"] = str(theta)
        out = {r["image"]: replay(r, channel, by_slug, weights, index_order) for r in rows}
        os.environ.pop("TEXT_VOTE", None)
        return out

    base = run(None)
    stored_same = sum(base[r["image"]]["top1"] == r["top1"] for r in rows)
    print(f"без правила: лидер совпал с прогоном у {stored_same} из {len(rows)}")
    organizer = lambda r: r["image"].startswith("org")  # noqa: E731
    summary = {}
    for theta in GRID:
        res = run(theta)
        cells = {}
        for split in ("tune", "test"):
            for part, keep in (("org", organizer), ("other", lambda r: not organizer(r))):
                chunk = [r for r in rows if r["split"] == split and keep(r)]
                pos = [r for r in chunk if r["findable"]]
                absent = [r for r in chunk if not r["gold"]]
                gain = sum(1 for r in pos if res[r["image"]]["correct"]
                           and not base[r["image"]]["correct"])
                loss = sum(1 for r in pos if base[r["image"]]["correct"]
                           and not res[r["image"]]["correct"])
                shown = (sum(res[r["image"]]["probability"] >= SHOW_P for r in absent)
                         - sum(base[r["image"]]["probability"] >= SHOW_P for r in absent))
                cells[f"{split}/{part}"] = {"gain": gain, "loss": loss, "absent_shown": shown,
                                            "top1": sum(res[r["image"]]["correct"] for r in pos),
                                            "n": len(pos)}
        changed = [(r["split"], r["image"], r["gold"], base[r["image"]]["top1"],
                    res[r["image"]]["top1"]) for r in rows
                   if r["findable"] and res[r["image"]]["correct"] != base[r["image"]]["correct"]]
        summary[theta] = {"cells": cells, "changed": changed}
        print(f"θ={theta:<5} " + " | ".join(
            f"{k} +{c['gain']}/-{c['loss']} (p={mcnemar_exact(c['gain'], c['loss']):.2f}) "
            f"ложн.{c['absent_shown']:+d}" for k, c in cells.items()))
    tune_key = lambda t: (summary[t]["cells"]["tune/org"]["top1"],  # noqa: E731
                          summary[t]["cells"]["tune/org"]["top1"]
                          + summary[t]["cells"]["tune/other"]["top1"], -t)
    pick = max(GRID, key=tune_key)
    print(f"\nвыбран по tune: θ={pick}")
    for split, image, gold, a, b in summary[pick]["changed"]:
        print(f"  {'+' if b == gold else '-'} {split} {image[:26]:26} gold={gold[:30]:30} {a[:28]:28} -> {b[:28]}")
    if args.output:
        Path(args.output).write_text(json.dumps({"pick": pick, "grid": summary},
                                                ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
