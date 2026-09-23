#!/usr/bin/env python3
"""Порог показа по сохранённому прогону полевого набора.

Правило то же, что записано в confidence.json: наибольший F1 top-1 по всем
запросам среди порогов, у которых ложное принятие (карточка для вина вне
каталога) ниже 30%. Считается по настроечной части и только после
фиксации ранжирования: порог решает, показывать ли ответ, и не должен
подменять выбор ответа. Отложенная часть — только для проверки уже
выбранного порога, не для перебора.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_metrics import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID = [round(0.02 * i, 2) for i in range(1, 30)]


def table(results: list[dict], grid: list[float]) -> list[dict]:
    rows = []
    for p in grid:
        m = summarize(results, p)
        q1, q5 = m["with_rejection_all_queries"]["1"], m["with_rejection_all_queries"]["5"]
        refused = sum(1 for r in results if r["gold"] and r["correct"] and r["probability"] < p)
        rows.append({"threshold": p, "f1_top1": q1["f1"], "precision": q1["precision"],
                     "recall": q1["recall"], "fp": q1["fp"], "f1_top5": q5["f1"],
                     "false_accept": m["false_accept_rate"], "refused_correct": refused})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="отчёт eval_field.py по настроечной части")
    ap.add_argument("--check", help="отчёт по отложенной части: только проверка выбранного порога")
    ap.add_argument("--max-false-accept", type=float, default=0.30)
    ap.add_argument("--write", action="store_true", help="записать show_p в confidence.json")
    ap.add_argument("--source", choices=["all", "organizer"], default="all",
                    help="organizer — только кадры публичного набора кейса")
    args = ap.parse_args()
    results = json.loads(Path(args.report).read_text(encoding="utf-8"))["results"]
    manifest = {r["image"]: r for r in csv.DictReader(
        (ROOT / "data/field/manifest.csv").open(encoding="utf-8"))}
    keep = (lambda r: manifest.get(r["image"], {}).get("source") == "публичный набор кейса") \
        if args.source == "organizer" else (lambda r: True)
    results = [r for r in results if keep(r)]
    rows = table(results, GRID)
    print(f"{'порог':>6} {'F1 top-1':>9} {'точн':>6} {'полн':>6} {'FP':>4} {'F1 top-5':>9} "
          f"{'ложн.прин':>10} {'зря отказ':>10}")
    for r in rows:
        print(f"{r['threshold']:>6.2f} {r['f1_top1'] * 100:>8.1f}% {r['precision'] * 100:>5.0f}% "
              f"{r['recall'] * 100:>5.0f}% {r['fp']:>4} {r['f1_top5'] * 100:>8.1f}% "
              f"{r['false_accept'] * 100:>9.1f}% {r['refused_correct']:>10}")
    allowed = [r for r in rows if r["false_accept"] < args.max_false_accept]
    best = max(allowed or rows, key=lambda r: (r["f1_top1"], -r["threshold"]))
    print(f"\nвыбор по правилу (ложное принятие < {args.max_false_accept:.0%}): "
          f"порог {best['threshold']:.2f}, F1 top-1 {best['f1_top1'] * 100:.1f}%, "
          f"ложное принятие {best['false_accept'] * 100:.1f}%, зря отказано {best['refused_correct']}")
    if args.check:
        held = [r for r in json.loads(Path(args.check).read_text(encoding="utf-8"))["results"]
                if keep(r)]
        for p in sorted({best["threshold"], 0.11, 0.15, 0.20}):
            r = table(held, [p])[0]
            print(f"  отложенная часть при {p:.2f}: F1 top-1 {r['f1_top1'] * 100:.1f}%, "
                  f"top-5 {r['f1_top5'] * 100:.1f}%, ложное принятие {r['false_accept'] * 100:.1f}%, "
                  f"зря отказано {r['refused_correct']}, лишних показов {r['fp']}")
    if args.write:
        path = ROOT / "data" / "index" / "confidence.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["thresholds"]["show_p"] = best["threshold"]
        artifact["thresholds_note"] = (
            f"Порог показа {best['threshold']:.2f} выбран threshold_sweep.py по {Path(args.report).name}"
            f"{' (только кадры организатора)' if args.source == 'organizer' else ''}: "
            f"наибольший F1 top-1 среди порогов с ложным принятием ниже {args.max_false_accept:.0%} "
            f"(F1 {best['f1_top1'] * 100:.1f}%, ложное принятие {best['false_accept'] * 100:.1f}%, "
            f"верных отказано {best['refused_correct']}). Считается по всем запросам настроечной части, "
            f"включая вина вне каталога; веса модели уверенности не менялись.")
        path.write_text(json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"записано: {path}")


if __name__ == "__main__":
    main()
