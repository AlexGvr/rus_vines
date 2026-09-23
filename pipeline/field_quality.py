#!/usr/bin/env python3
"""Блок качества на полевом наборе для /health и README.

Берёт сохранённый отчёт eval_field.py и, если есть, отчёт api_check.py,
и пишет data/index/field_quality.json с явными знаменателями и пометкой,
что часть test — регрессионная, уже использованная при разборе ошибок,
а не независимая приёмка. Ничего не пересчитывает заново.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_metrics import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="отчёт eval_field.py по регрессионной части")
    ap.add_argument("--api", default="", help="отчёт api_check.py (латентность, /v1/search)")
    ap.add_argument("--output", default=str(ROOT / "data" / "index" / "field_quality.json"))
    args = ap.parse_args()
    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    show_p = json.loads((ROOT / "data/index/confidence.json").read_text())["thresholds"]["show_p"]
    m = summarize(data["results"], show_p)
    p, f, q = m["all_positives"], m["findable_diagnostic"], m["with_rejection_all_queries"]["1"]
    block = {
        "set": "полевой набор, регрессионная часть (development, не независимая приёмка)",
        "independent_holdout": False,
        "report": str(Path(args.report).relative_to(ROOT)) if Path(args.report).is_absolute() else args.report,
        "report_sha256": hashlib.sha256(Path(args.report).read_bytes()).hexdigest(),
        "positives": p["n"], "top1_all_positives": round(p["top1"], 4),
        "top3_all_positives": round(p["top3"], 4), "top5_all_positives": round(p["top5"], 4),
        "with_reference": f["n"], "top1_with_reference": round(f["top1"], 4),
        "show_p": show_p, "f1_top1_with_rejection": round(q["f1"], 4),
        "false_shows": q["fp"], "absent_n": m["absent_n"],
        "absent_shown_rate": round(m["false_accept_rate"], 4),
    }
    # Кадры организатора — распределение, ближайшее к закрытой проверке;
    # отзывы (600×800) — другое, и среднее их смешивает.
    manifest = {r["image"]: r for r in csv.DictReader(
        (ROOT / "data/field/manifest.csv").open(encoding="utf-8"))}
    org = [r for r in data["results"]
           if manifest.get(r["image"], {}).get("source") == "публичный набор кейса"]
    if org:
        pos = [r for r in org if r["gold"]]
        neg = [r for r in org if not r["gold"]]
        block["organizer"] = {
            "frames": len(org), "positives": len(pos),
            "top1_positives": round(sum(r["correct"] for r in pos) / len(pos), 4) if pos else None,
            "absent_n": len(neg),
            "absent_shown_rate": (round(sum(r["probability"] >= show_p for r in neg) / len(neg), 4)
                                  if neg else None),
        }
    if args.api:
        api = json.loads(Path(args.api).read_text(encoding="utf-8"))["summary"]
        block["latency_ms"] = api["latency_ms"]
        if "search" in api:
            block["search"] = api["search"]
    out = Path(args.output)
    out.write_text(json.dumps(block, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(block, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
