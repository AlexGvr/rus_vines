#!/usr/bin/env python3
"""Замер SIFT_ONE_TO_ONE и SIFT_UNIQUE_INLIERS против текущей геометрии.

Парный замер по сохранённым прогонам eval_field.py (--candidate-features).
Для каждого кадра заново строится кроп для геометрии (тот же выбор вида,
что в сервисе) и пересчитываются инлаеры тех же двадцати кандидатов
в трёх вариантах: как сейчас, один к одному до RANSAC (o2o) и подсчёт
уникальных точек эталона среди инлаеров (uniq). Косинус
и строки OCR берутся из сохранённого прогона, поэтому оба варианта
отличаются только геометрией. Решение проходит через replay_rerank.replay —
тот же rerank -> settle -> check_leader, что в сервисе.

Инлаеры варианта «как сейчас» сверяются с сохранёнными: совпадение
значит, что кроп тот же, что был в исходном прогоне. Видеокарта не нужна:
CUDA_VISIBLE_DEVICES= гонит эмбеддинг и детектор на процессоре.

    CUDA_VISIBLE_DEVICES= python pipeline/experiment_one_to_one.py \\
        data/validation/tune.json data/validation/test.json --output out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import rerank  # noqa: E402
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from eval_field import locate  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from replay_rerank import replay  # noqa: E402
from text_match import TextChannel  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


class Computed:
    def __init__(self, stats: dict) -> None:
        self.by_slug = stats

    def stats(self, _crop, slugs):
        return {s: self.by_slug[s] for s in slugs if s in self.by_slug}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--weights", default=str(INDEX_DIR / "confidence.json"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    channel = TextChannel(wines)
    weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))["outcome_weights"]
    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    index_order: dict[str, int] = {}
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            index_order.setdefault(r["slug"], i)
    reranker = rerank.PrecomputedReranker(INDEX_DIR / "sift")

    out = []
    t0 = time.time()
    for path in args.runs:
        split = "tune" if "tune" in Path(path).name else "test"
        rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))["results"]
                if r.get("candidates")]
        for n, row in enumerate(rows, 1):
            image = Image.open(locate(row["image"])).convert("RGB")
            views = [normalize_batch([image], steps=s)[0] for s in [(), ("detect",)]]
            similarity = vectors @ embed_images(views).T
            shot = views[core.pick_view(views, similarity)]
            slugs = [c["slug"] for c in row["candidates"]]
            result = {"image": row["image"], "gold": row["gold"], "split": split,
                      "findable": row["findable"], "stored_top1": row["top1"],
                      "stored_correct": row["correct"]}
            for name, flag, unique in (("base", False, False), ("o2o", True, False),
                                       ("uniq", False, True)):
                rerank.ONE_TO_ONE, rerank.UNIQUE_INLIERS = flag, unique
                stats = reranker.stats(shot, slugs)
                result[name] = replay(row, channel, by_slug, weights, index_order,
                                      geometry=Computed(stats))
                result[name]["inliers"] = {s: int(stats[s].inliers) for s in slugs}
            stored = {c["slug"]: c["inliers"] for c in row["candidates"]}
            result["crop_matches"] = stored == result["base"]["inliers"]
            out.append(result)
            if n % 25 == 0:
                print(f"  {split} {n}/{len(rows)}  {time.time() - t0:.0f} с", flush=True)
    rerank.ONE_TO_ONE = rerank.UNIQUE_INLIERS = False
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"готово: {len(out)} кадров, кроп совпал у "
          f"{sum(r['crop_matches'] for r in out)}")


if __name__ == "__main__":
    main()
