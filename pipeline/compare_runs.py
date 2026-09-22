#!/usr/bin/env python3
"""Парное сравнение двух прогонов eval_field.py на одном наборе кадров.

Один эксперимент — одна пара файлов. Печатает не только проценты, но и
кадры, которые изменение исправило и которые испортило: без этого выигрыш
в среднем может прятать регрессию на конкретных винах.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_metrics import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(path: str) -> tuple[dict, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data["results"]
    show_p = (data.get("metrics") or {}).get("show_p")
    if show_p is None:
        show_p = json.loads((ROOT / "data/index/confidence.json").read_text())["thresholds"]["show_p"]
    return summarize(results, show_p), results


def stage(r: dict) -> str:
    if not r["findable"]:
        return "эталона нет"
    if r["rank_shortlist"] < 0:
        return "не в шортлисте"
    if r["rank_geometry"] != 0:
        return "геометрия"
    if r["rank_text"] != 0:
        return "текст"
    return "первый"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--titles", action="store_true", help="печатать названия вместо slug")
    ap.add_argument("--by-group", action="store_true",
                    help="разбивка исправлений и поломок по позициям и фотосериям манифеста")
    args = ap.parse_args()
    ma, ra = load(args.before)
    mb, rb = load(args.after)
    by_a = {r["image"]: r for r in ra}
    by_b = {r["image"]: r for r in rb}
    if set(by_a) != set(by_b):
        sys.exit(f"наборы кадров различаются: только в первом {len(set(by_a) - set(by_b))}, "
                 f"только во втором {len(set(by_b) - set(by_a))}")
    titles = {}
    if args.titles:
        wines = json.loads((ROOT / "data/catalog/catalog.json").read_text(encoding="utf-8"))["wines"]
        titles = {w["slug"]: w["title"] for w in wines}
    name = lambda s: (titles.get(s) or s or "—")[:40]

    def row(label, fa, fb):
        print(f"  {label:38} {fa:>10} {fb:>10}")

    def pct(v):
        return "—" if v is None else f"{v * 100:.1f}%"

    print(f"{'':40} {'до':>10} {'после':>10}")
    pa, pb = ma["all_positives"], mb["all_positives"]
    row(f"положительных кадров", pa["n"], pb["n"])
    for k in ("top1", "top3", "top5"):
        row(f"строгий {k} по всем положительным", pct(pa[k]), pct(pb[k]))
    fa, fb = ma["findable_diagnostic"], mb["findable_diagnostic"]
    row("top1 среди кадров с эталоном", f"{int(round(fa['top1'] * fa['n']))}/{fa['n']}",
        f"{int(round(fb['top1'] * fb['n']))}/{fb['n']}")
    for k in ("20", "50", "100"):
        row(f"recall шортлиста @{k}", pct(pa["retrieval_recall"][k]), pct(pb["retrieval_recall"][k]))
    qa, qb = ma["with_rejection_all_queries"], mb["with_rejection_all_queries"]
    for k in ("1", "5"):
        row(f"F1 top-{k} с отказом (порог {ma['show_p']})", pct(qa[k]["f1"]), pct(qb[k]["f1"]))
        row(f"  ложных показов top-{k} (FP)", qa[k]["fp"], qb[k]["fp"])
    row("ложные показы на винах вне каталога", pct(ma["false_accept_rate"]), pct(mb["false_accept_rate"]))
    refused = lambda rs, p: sum(1 for r in rs if r["gold"] and r["correct"] and r["probability"] < p)
    row("верных, но ниже порога показа", refused(ra, ma["show_p"]), refused(rb, mb["show_p"]))

    fixed = [i for i in by_a if by_a[i]["gold"] and not by_a[i]["correct"] and by_b[i]["correct"]]
    broken = [i for i in by_a if by_a[i]["gold"] and by_a[i]["correct"] and not by_b[i]["correct"]]
    print(f"\nисправлено top-1: {len(fixed)}, испорчено: {len(broken)}")
    for label, group in (("исправлено", fixed), ("испорчено", broken)):
        for i in sorted(group):
            a, b = by_a[i], by_b[i]
            print(f"  {label:10} {i[:40]:40} {name(a['gold']):40} было {name(a['top1']):32} стало {name(b['top1']):32} [{stage(a)} → {stage(b)}]")
    moved5 = [(i, by_a[i]["in_top5"], by_b[i]["in_top5"]) for i in by_a
              if by_a[i]["gold"] and by_a[i]["in_top5"] != by_b[i]["in_top5"]]
    if moved5:
        print(f"\nизменения в пятёрке: вошло {sum(1 for _, a, b in moved5 if b)}, "
              f"выпало {sum(1 for _, a, b in moved5 if a)}")
    if args.by_group:
        # Прирост не должен объясняться одной серией: считаем, сколько разных
        # позиций и фотосерий затронуто, и печатаем каждую.
        import csv as _csv
        series = {r["image"]: r["series"] for r in _csv.DictReader(
            (ROOT / "data/field/manifest.csv").open(encoding="utf-8"))}
        for label, group in (("исправлено", fixed), ("испорчено", broken)):
            by_slug, by_series = {}, {}
            for i in group:
                by_slug.setdefault(by_a[i]["gold"], []).append(i)
                by_series.setdefault(series.get(i, "?"), []).append(i)
            print(f"\n{label}: позиций {len(by_slug)}, фотосерий {len(by_series)}")
            for slug, imgs in sorted(by_slug.items(), key=lambda kv: -len(kv[1])):
                print(f"  {name(slug):40} кадров {len(imgs):>2}  серий {len({series.get(i) for i in imgs})}")
    neg_a = [i for i in by_a if not by_a[i]["gold"]]
    fp_new = [i for i in neg_a if by_a[i]["probability"] < ma["show_p"] <= by_b[i]["probability"]]
    fp_gone = [i for i in neg_a if by_b[i]["probability"] < mb["show_p"] <= by_a[i]["probability"]]
    if fp_new or fp_gone:
        print(f"\nвина вне каталога: новых ложных показов {len(fp_new)}, исчезнувших {len(fp_gone)}")
        for i in fp_new:
            print(f"  новый ложный показ {i[:40]:40} -> {name(by_b[i]['top1'])} p={by_b[i]['probability']:.3f}")


if __name__ == "__main__":
    main()
