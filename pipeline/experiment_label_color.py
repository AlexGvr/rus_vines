#!/usr/bin/env python3
"""Цвет этикетки (label_color.color_distance) для 20 кандидатов каждого кадра.

Кроп — тот же выбор вида, что в сервисе; кандидаты — из сохранённых
прогонов eval_field.py. Результат читает experiment_rerank_rules.py
(семейство colortie).

    python pipeline/experiment_label_color.py data/validation/tune.json \\
        data/validation/test.json data/validation/label_color.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from eval_field import locate  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from label_color import References, color_distance, scaled_rgb  # noqa: E402
from rerank import PrecomputedReranker, descriptors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    *runs, output = sys.argv[1:]
    wines = {w["slug"]: w for w in json.loads(
        (ROOT / "data/catalog/catalog.json").read_text(encoding="utf-8"))["wines"]}
    refs = References(wines)
    reranker = PrecomputedReranker(ROOT / "data/index/sift")
    vectors = np.load(ROOT / "data/index/clean_mv.npy")
    out, t0 = {}, time.time()
    for path in runs:
        rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))["results"]
                if r.get("candidates")]
        for n, row in enumerate(rows, 1):
            image = Image.open(locate(row["image"])).convert("RGB")
            views = [normalize_batch([image], steps=s)[0] for s in [(), ("detect",)]]
            shot = views[core.pick_view(views, vectors @ embed_images(views).T)]
            qkp, qd = descriptors(shot)
            qrgb = scaled_rgb(shot)
            dist = {}
            for c in row["candidates"]:
                ref = refs.get(c["slug"])
                entries = reranker.positions.get(c["slug"], [])
                if ref is None or not entries:
                    dist[c["slug"]] = None
                    continue
                kp, desc, _ = reranker.entry(entries[0])
                dist[c["slug"]] = color_distance(qrgb, qkp, qd, ref, kp, desc)
            out[row["image"]] = dist
            if n % 50 == 0:
                print(f"  {Path(path).name} {n}/{len(rows)} {time.time() - t0:.0f} с", flush=True)
    Path(output).write_text(json.dumps(out), encoding="utf-8")
    print(f"готово: {len(out)} кадров")


if __name__ == "__main__":
    main()
