#!/usr/bin/env python3
"""Сводная таблица по нескольким отчётам eval_field.py.

Для журнала экспериментов: одна строка на прогон, строгие top-1/3/5 по всем
положительным, top-1 среди кадров с эталоном, F1 с отказом при текущем
пороге, число ложных показов и напрасных отказов. Все числа читаются из
сохранённых отчётов, ничего не пересчитывается заново.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_metrics import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    show_p = json.loads((ROOT / "data/index/confidence.json").read_text())["thresholds"]["show_p"]
    print(f"{'прогон':22} {'n+':>4} {'top1':>6} {'top3':>6} {'top5':>6} {'с этал.':>9} "
          f"{'F1@' + str(show_p):>8} {'FP':>4} {'зря отк':>7} {'ложн.пр':>8} {'OCR отк':>7}")
    for path in sys.argv[1:]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        rs = data["results"]
        m = summarize(rs, show_p)
        p, f, q = m["all_positives"], m["findable_diagnostic"], m["with_rejection_all_queries"]["1"]
        refused = sum(1 for r in rs if r["gold"] and r["correct"] and r["probability"] < show_p)
        fails = (data.get("provenance") or {}).get("ocr_failures", "—")
        print(f"{Path(path).stem:22} {p['n']:>4} {p['top1'] * 100:>5.1f}% {p['top3'] * 100:>5.1f}% "
              f"{p['top5'] * 100:>5.1f}% {int(round(f['top1'] * f['n'])):>4}/{f['n']:<4} "
              f"{q['f1'] * 100:>7.1f}% {q['fp']:>4} {refused:>7} {m['false_accept_rate'] * 100:>7.1f}% {fails:>7}")


if __name__ == "__main__":
    main()
