#!/usr/bin/env python3
"""S1.6 — вклад каждого шага нормализации, замер в одном процессе.

Для честного сравнения индекс и запросы проходят одну и ту же обработку:
если эталон обрезан до этикетки, а запрос нет, сравниваются разные сущности.
Поэтому каждый вариант — это пересборка индекса и пересчёт запросов.

Модель и детектор грузятся один раз на все варианты, иначе половина времени
уходит на инициализацию.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_index import collect  # noqa: E402
from embed import embed_paths  # noqa: E402
from eval_index import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"

# chain — шаги накапливаются, как в конвейере; isolate — каждый поверх детекции
# отдельно, чтобы отличить «плохая цепочка» от «плохой конкретный шаг».
PRESETS = {
    "chain": [
        ("без нормализации", ()),
        ("+детекция", ("detect",)),
        ("+этикетка", ("detect", "label")),
        ("+разворот", ("detect", "label", "dewarp")),
        ("+блик и свет", ("detect", "label", "dewarp", "glare", "balance")),
    ],
    "isolate": [
        ("детекция", ("detect",)),
        ("детекция+этикетка", ("detect", "label")),
        ("детекция+разворот", ("detect", "dewarp")),
        ("детекция+блик", ("detect", "glare")),
        ("детекция+свет", ("detect", "balance")),
    ],
}


def build(name: str, variant: str, steps: tuple[str, ...]) -> None:
    paths, slugs = collect(variant)
    vectors, kept = embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
    kept_slugs = slugs if len(kept) == len(paths) else [slugs[paths.index(p)] for p in kept]
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / f"{name}.npy", vectors)
    with (INDEX_DIR / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "slug"])
        for path, slug in zip(kept, kept_slugs):
            writer.writerow([str(Path(path).relative_to(ROOT)), slug or ""])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="hard")
    ap.add_argument("--variant", default="clean", choices=["clean", "dirty"])
    ap.add_argument("--preset", default="chain", choices=sorted(PRESETS))
    args = ap.parse_args()
    variants = PRESETS[args.preset]

    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    qpaths = [str(QUERIES / args.subset / row["image"]) for row in manifest]
    truth_by_path = {p: row["slug"] for p, row in zip(qpaths, manifest)}

    rows = []
    for label, steps in variants:
        name = f"{args.variant}_{'_'.join(steps) if steps else 'raw'}"
        t0 = time.time()
        build(name, args.variant, steps)
        qvec, kept = embed_paths(qpaths, batch_size=32, progress_every=0, steps=steps)
        truth = [truth_by_path[p] for p in kept]
        result = evaluate(name, qvec, truth)
        result.update(label=label, steps=",".join(steps) or "-",
                      seconds=round(time.time() - t0, 1))
        rows.append(result)
        print(f"  {label:18} top-1 {result['top1'] * 100:5.1f}%  "
              f"top-5 {result['top5'] * 100:5.1f}%  ({result['seconds']} с)", flush=True)

    print(f"\nнабор {args.subset}, индекс {args.variant}, запросов {len(truth)}")
    print(f"{'вариант':20} {'top-1':>8} {'top-5':>8} {'отрыв':>9} {'Δtop-1':>9}")
    base = rows[0]["top1"]
    for row in rows:
        print(f"{row['label']:20} {row['top1'] * 100:>7.1f}% {row['top5'] * 100:>7.1f}% "
              f"{row['gap_median']:>9.4f} {(row['top1'] - base) * 100:>+8.1f}")

    out = INDEX_DIR / f"ablation_{args.preset}_{args.subset}_{args.variant}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
