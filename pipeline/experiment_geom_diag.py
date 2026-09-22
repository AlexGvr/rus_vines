#!/usr/bin/env python3
"""Диагностика локальных соответствий: на чём совпадают верный и ошибочный кандидаты.

Для каждого кадра с вином каталога повторяется визуальная часть сервиса
(два вида, выбор кропа по косинусу, шортлист 20), затем для всех кандидатов
считаются инлаеры с номерами точек запроса (`PrecomputedReranker.stats_points`).
Сохраняется:
  - число точек запроса и координаты;
  - по каждой позиции: лучший эталон, его инлаеры и покрытие, объединение
    точек по всем эталонам позиции (повторы одного вина не раздувают
    распространённость);
  - кратность каждой точки — со сколькими разными позициями она совпала;
  - доля «общих» инлаеров (кратность ≥ 2) у верного ответа и у лидера;
  - взвешенные оценки при alpha 0.5 и 1.0 — что дал бы порядок по ним.
Правильные и ошибочные ответы сравниваются по этим величинам; примеры
рисуются: точки-инлаеры верного и лидера, уникальные зелёным, общие красным.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from eval_field import SCORABLE, locate  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import MAX_SIDE, PrecomputedReranker  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
FIELD = ROOT / "data" / "field"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def draw_points(crop: Image.Image, keypoints, points: set[int], multiplicity: dict[int, int],
                title: str) -> Image.Image:
    arr = crop.convert("RGB")
    scale = min(1.0, MAX_SIDE / max(arr.size))
    im = arr.resize((int(arr.width * scale), int(arr.height * scale)))
    d = ImageDraw.Draw(im)
    for p in points:
        x, y = keypoints[p]
        colour = (0, 200, 0) if multiplicity.get(p, 1) == 1 else (230, 30, 30)
        d.ellipse([x - 3, y - 3, x + 3, y + 3], outline=colour, width=2)
    d.rectangle([0, 0, im.width, 18], fill=(255, 255, 255))
    d.text((3, 3), title[:60], fill=(0, 0, 0))
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "data/validation/final3_tune.json"),
                    help="базовый прогон: из него берутся лидер и правильный ответ")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--output", default=str(ROOT / "data/validation/geom_diag_tune.json"))
    ap.add_argument("--examples-dir", default=str(ROOT / "data/validation/geom_diag_examples"))
    args = ap.parse_args()
    base = json.loads(Path(args.report).read_text(encoding="utf-8"))
    records = {r["image"]: r for r in base["results"] if r["gold"] and r["findable"]}
    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]}
    title = lambda s: wines.get(s, {}).get("title", s or "—")
    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    rows = [r for r in csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8"))
            if r["image"] in records]
    out, timing = [], {"stats_ms": [], "stats_points_ms": []}
    Path(args.examples_dir).mkdir(parents=True, exist_ok=True)
    drawn = 0
    for row in rows:
        rec = records[row["image"]]
        image = Image.open(locate(row["image"])).convert("RGB")
        views = {steps: normalize_batch([image], steps=steps)[0] for steps in ((), ("detect",))}
        query = embed_images(list(views.values()))
        similarity = vectors @ query.T
        scores = similarity.max(axis=1)
        shortlist = core.shortlist(scores, index_slugs, args.topk)
        shot = list(views.values())[core.pick_view(list(views.values()), similarity)]
        slugs = [c.slug for c in shortlist]
        t0 = time.perf_counter(); reranker.stats(shot, slugs); t1 = time.perf_counter()
        stats = reranker.stats_points(shot, slugs); t2 = time.perf_counter()
        timing["stats_ms"].append((t1 - t0) * 1000); timing["stats_points_ms"].append((t2 - t1) * 1000)
        multiplicity: dict[int, int] = {}
        for s in slugs:
            for p in stats[s]["union"]:
                multiplicity[p] = multiplicity.get(p, 0) + 1
        weighted = {a: core.weighted_scores(stats, slugs, a)[0] for a in (0.5, 1.0)}
        gold, leader = rec["gold"], rec["top1"]

        def summary(slug):
            st = stats.get(slug)
            if not st:
                return None
            pts = st["points"]
            shared = sum(1 for p in pts if multiplicity.get(p, 1) >= 2)
            return {"inliers": st["inliers"], "coverage": round(st["coverage"], 3),
                    "unique": len(pts) - shared, "shared": shared,
                    "shared_share": round(shared / len(pts), 3) if pts else None,
                    "entries": [(e["entry"], e["inliers"]) for e in st["entries"]],
                    "weighted_0.5": round(weighted[0.5].get(slug, 0.0), 2),
                    "weighted_1.0": round(weighted[1.0].get(slug, 0.0), 2)}

        def order_by(scores_dict):
            ranked = sorted(slugs, key=lambda s: (-scores_dict.get(s, 0.0), -next(c.cv for c in shortlist if c.slug == s)))
            top = scores_dict.get(ranked[0], 0.0); second = scores_dict.get(ranked[1], 0.0) if len(ranked) > 1 else 0.0
            decisive = top >= second * (1.0 + core.GEOMETRY_MARGIN)
            return (ranked if decisive else slugs)[0], decisive

        raw_scores = {s: float(stats[s]["inliers"]) for s in slugs}
        item = {"image": row["image"], "series": row["series"], "gold": gold, "leader": leader,
                "correct": rec["correct"], "query_points": stats["__query__"]["n"],
                "shortlist": slugs, "gold_in_shortlist": gold in slugs,
                "multiplicity_hist": {str(k): v for k, v in sorted(
                    __import__("collections").Counter(multiplicity.values()).items())},
                "gold": gold, "gold_stats": summary(gold), "leader_stats": summary(leader),
                "geometry_leader_raw": order_by(raw_scores),
                "geometry_leader_w05": order_by(weighted[0.5]),
                "geometry_leader_w10": order_by(weighted[1.0])}
        out.append(item)
        g, l = item["gold_stats"], item["leader_stats"]
        print(f"  {row['image'][:26]:26} {'ВЕРНО' if rec['correct'] else 'ошибка':6} "
              f"верно «{title(gold)[:24]}» inl {g['inliers'] if g else '—'} общих {g['shared_share'] if g else '—'} | "
              f"лидер «{title(leader)[:24]}» inl {l['inliers'] if l else '—'} общих {l['shared_share'] if l else '—'} | "
              f"взвеш. лидер a=1: «{title(item['geometry_leader_w10'][0])[:22]}»", flush=True)
        if not rec["correct"] and drawn < args.examples and g and l and gold in slugs:
            kps = stats["__query__"]["keypoints"]
            a = draw_points(shot, kps, set(stats[gold]["points"]), multiplicity,
                            f"ВЕРНО {title(gold)[:34]} inl {g['inliers']} общих {g['shared']}")
            b = draw_points(shot, kps, set(stats[leader]["points"]), multiplicity,
                            f"ЛИДЕР {title(leader)[:34]} inl {l['inliers']} общих {l['shared']}")
            sheet = Image.new("RGB", (a.width + b.width + 10, max(a.height, b.height)), "white")
            sheet.paste(a, (0, 0)); sheet.paste(b, (a.width + 10, 0))
            sheet.save(Path(args.examples_dir) / f"{row['image'][:-4]}.jpg", quality=88)
            drawn += 1

    correct = [i for i in out if i["correct"] and i["leader_stats"]]
    wrong = [i for i in out if not i["correct"] and i["gold_stats"] and i["leader_stats"]]
    def med(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.median(vals), 3) if vals else None
    agg = {
        "n_frames": len(out), "n_correct": len(correct), "n_wrong": len(wrong),
        "leader_shared_share_median_correct": med([i["leader_stats"]["shared_share"] for i in correct]),
        "leader_shared_share_median_wrong": med([i["leader_stats"]["shared_share"] for i in wrong]),
        "gold_shared_share_median_wrong": med([i["gold_stats"]["shared_share"] for i in wrong]),
        "gold_unique_median_wrong": med([i["gold_stats"]["unique"] for i in wrong]),
        "leader_unique_median_wrong": med([i["leader_stats"]["unique"] for i in wrong]),
        "wrong_gold_more_unique": sum(1 for i in wrong if i["gold_stats"]["unique"] > i["leader_stats"]["unique"]),
        "wrong_gold_more_inliers": sum(1 for i in wrong if i["gold_stats"]["inliers"] > i["leader_stats"]["inliers"]),
        "offline_geometry_leader_correct": {
            "raw": sum(1 for i in out if i["geometry_leader_raw"][0] == i["gold"]),
            "w0.5": sum(1 for i in out if i["geometry_leader_w05"][0] == i["gold"]),
            "w1.0": sum(1 for i in out if i["geometry_leader_w10"][0] == i["gold"])},
        "timing_ms_median": {k: round(statistics.median(v), 1) for k, v in timing.items() if v},
    }
    print("\n" + json.dumps(agg, ensure_ascii=False, indent=1))
    Path(args.output).write_text(json.dumps({"aggregate": agg, "items": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {args.output}; примеры: {args.examples_dir}")


if __name__ == "__main__":
    main()
