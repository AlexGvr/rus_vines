#!/usr/bin/env python3
"""Правила порядка кандидатов, проверенные на промахе Victor Dravigny.

Переигрывает решение по сохранённым прогонам (replay_rerank.replay) и
статистике цвета (experiment_color.py) в трёх семействах вариантов:

  colw, veto<t>  — цвет в совпавших точках: инлаеры, умноженные на согласие
                   цвета, или обнулённые при согласии ниже t;
  floor<s>       — когда геометрия не решила, косинус выбирает только среди
                   кандидатов с инлаерами не ниже доли s от лучшего;
  colortie<d>    — когда геометрия не решила, из близких по точкам
                   кандидатов выбывают те, чей цвет этикетки
                   (label_color.color_distance, средний ΔE) хуже лучшего
                   больше чем на d; оставшиеся идут первыми по косинусу.
                   Нужен второй файл — experiment_label_color.py.

Печатает изменение строгого top-1 по частям (кадры организатора отдельно)
с точным McNemar и ложные показы вин вне каталога на пороге 0.10.

    python pipeline/experiment_rerank_rules.py data/validation/color_stats.json \
        [data/validation/label_color.json]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402
from replay_rerank import replay  # noqa: E402
from rerank import MatchStats  # noqa: E402
from stats import mcnemar_exact  # noqa: E402
from text_match import TextChannel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("data/validation/dsonly_cal2_tune.json", "data/validation/dsonly_cal_test.json")
SHOW_P = 0.10


class Geometry:
    """Инлаеры из статистики цвета, пропущенные через правило f."""

    def __init__(self, geom: dict, f) -> None:
        self.geom, self.f = geom, f

    def stats(self, _crop, slugs):
        return {s: MatchStats(self.f(*self.geom[s]), 0, self.geom[s][1], 0.0, (), self.geom[s][2])
                for s in slugs if s in self.geom}


def floor_rerank(share: float):
    original = core.rerank

    def rerank(crop, candidates, reranker, margin=core.GEOMETRY_MARGIN):
        original(crop, candidates, reranker, margin)
        best = max((c.inliers for c in candidates), default=0)
        ranked = sorted(candidates, key=lambda c: (-c.inliers, -c.cv))
        decisive = len(ranked) < 2 or ranked[0].inliers >= ranked[1].inliers * (1 + margin)
        if not decisive and best > 0:
            candidates.sort(key=lambda c: (c.inliers < share * best, -c.cv))
        return candidates
    return rerank


def colortie_rerank(delta: float, current: dict):
    original = core.rerank

    def rerank(crop, candidates, reranker, margin=core.GEOMETRY_MARGIN):
        original(crop, candidates, reranker, margin)
        ranked = sorted(candidates, key=lambda c: (-c.inliers, -c.cv))
        if (len(ranked) < 2 or ranked[0].inliers <= 0
                or ranked[0].inliers >= ranked[1].inliers * (1 + margin)):
            return candidates
        group = [c for c in candidates if c.inliers >= ranked[0].inliers / (1 + margin)]
        dist = {c.slug: (current["dist"].get(c.slug) or {}).get("mean") for c in group}
        known = [v for v in dist.values() if v is not None]
        if len(known) < 2:
            return candidates
        keep = {s for s, v in dist.items() if v is None or v <= min(known) + delta}
        if len(keep) < len(group):
            candidates.sort(key=lambda c: (c.slug not in keep, -c.cv))
        return candidates
    return rerank


def main() -> None:
    color = {r["image"]: r["geom"] for r in json.loads(Path(sys.argv[1]).read_text())}
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
                 json.loads((ROOT / path).read_text(encoding="utf-8"))["results"] if r.get("candidates")]

    geometry = {
        "colw": lambda i, cov, col: int(round(i * col)),
        "veto0.2": lambda i, cov, col: i if col >= 0.2 else 0,
        "veto0.3": lambda i, cov, col: i if col >= 0.3 else 0,
        "veto0.5": lambda i, cov, col: i if col >= 0.5 else 0,
    }
    original = core.rerank
    res = {"base": {r["image"]: replay(r, channel, by_slug, weights, index_order,
                                       Geometry(color[r["image"]], lambda i, cov, col: i))
                    for r in rows}}
    for name, f in geometry.items():
        res[name] = {r["image"]: replay(r, channel, by_slug, weights, index_order,
                                        Geometry(color[r["image"]], f)) for r in rows}
    for share in (0.3, 0.5, 0.7):
        core.rerank = floor_rerank(share)
        res[f"floor{share}"] = {r["image"]: replay(r, channel, by_slug, weights, index_order,
                                                   Geometry(color[r["image"]], lambda i, cov, col: i))
                                for r in rows}
        core.rerank = original

    if len(sys.argv) > 2:
        label = json.loads(Path(sys.argv[2]).read_text())
        current: dict = {}
        for delta in (5, 10, 20):
            core.rerank = colortie_rerank(delta, current)
            res[f"colortie{delta}"] = {}
            for r in rows:
                current["dist"] = label[r["image"]]
                res[f"colortie{delta}"][r["image"]] = replay(
                    r, channel, by_slug, weights, index_order,
                    Geometry(color[r["image"]], lambda i, cov, col: i))
            core.rerank = original

    organizer = lambda r: r["image"].startswith("org")  # noqa: E731
    for name in res:
        if name == "base":
            continue
        print(f"=== {name}")
        for split in ("tune", "test"):
            for part, keep in (("org", organizer), ("other", lambda r: not organizer(r))):
                chunk = [r for r in rows if r["split"] == split and keep(r)]
                pos = [r for r in chunk if r["findable"]]
                absent = [r for r in chunk if not r["gold"]]
                was = sum(res["base"][r["image"]]["correct"] for r in pos)
                now = sum(res[name][r["image"]]["correct"] for r in pos)
                gain = sum(1 for r in pos if res[name][r["image"]]["correct"]
                           and not res["base"][r["image"]]["correct"])
                loss = sum(1 for r in pos if res["base"][r["image"]]["correct"]
                           and not res[name][r["image"]]["correct"])
                shown = lambda k: sum(res[k][r["image"]]["probability"] >= SHOW_P for r in absent)  # noqa: E731
                print(f"  {split:4} {part:5} top-1 {was} -> {now} (+{gain}/-{loss}, "
                      f"p={mcnemar_exact(gain, loss):.2f}); вне каталога показано "
                      f"{shown('base')} -> {shown(name)} из {len(absent)}")


if __name__ == "__main__":
    main()
