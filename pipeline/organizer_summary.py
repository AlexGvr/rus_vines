#!/usr/bin/env python3
"""Сводка по фотографиям организатора — таблица README.

Берёт сохранённые прогоны eval_field.py (настроечная и регрессионная
части) и ответы VLM из пилота (scores.jsonl) и считает для каждой части:
строгий top-1 по винам каталога, top-3/top-5, F1 с отказом на пороге
показа, сколько вин вне каталога получили карточку, и что засчитает
`/v1/eval/predict` в режиме по умолчанию (отказ ниже порога показа, если
ответ не подтвердил VLM) при двух правилах подсчёта.

Набор по умолчанию — 100 снимков организатора («Реальные фото», кадры
org*); --set public добавляет примеры из материалов кейса (eval.zip).

    python pipeline/organizer_summary.py data/validation/tune.json \\
        data/validation/test.json --vlm data/validation/vlm_pilot/scores.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from field_metrics import summarize  # noqa: E402
from stats import fmt_share  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "field" / "manifest.csv"
PUBLIC = "публичный набор кейса"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tune")
    ap.add_argument("test")
    ap.add_argument("--vlm", required=True, help="scores.jsonl пилота experiment_vlm.py")
    ap.add_argument("--vlm-p", type=float, default=0.8, help="порог подтверждения (VLM_CONFIRM_P)")
    ap.add_argument("--set", choices=["photos", "public"], default="photos")
    ap.add_argument("--output")
    args = ap.parse_args()

    manifest = {r["image"]: r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))}
    show_p = json.loads((ROOT / "data/index/confidence.json").read_text())["thresholds"]["show_p"]
    vlm = {}
    for line in Path(args.vlm).open(encoding="utf-8"):
        row = json.loads(line)
        vlm[row["image"]] = {c["slug"]: c["p_yes"] for c in row["candidates"]}

    def keep(image: str) -> bool:
        if manifest.get(image, {}).get("source") != PUBLIC:
            return False
        return args.set == "public" or image.startswith("org")

    summary = {"set": args.set, "show_p": show_p, "vlm_p": args.vlm_p, "parts": {}}
    missing_vlm = []
    for part, path in (("tune", args.tune), ("test", args.test), ("all", None)):
        if path:
            rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))["results"]
                    if keep(r["image"])]
            summary["parts"][part] = {"rows": rows}
        else:
            rows = summary["parts"]["tune"]["rows"] + summary["parts"]["test"]["rows"]
        pos = [r for r in rows if r["gold"]]
        absent = [r for r in rows if not r["gold"]]
        metrics = summarize(rows, show_p)

        def answered(r: dict) -> bool:
            # Режим /v1/eval/predict по умолчанию: ответ при уверенности не ниже
            # порога показа или при подтверждении VLM.
            if r["probability"] >= show_p:
                return True
            p_yes = vlm.get(r["image"], {}).get(r["top1"])
            if p_yes is None:
                missing_vlm.append(r["image"])
                return False
            return p_yes >= args.vlm_p

        right = sum(1 for r in pos if r["correct"] and answered(r))
        nulls = sum(1 for r in absent if not answered(r))
        cell = {
            "positives": len(pos), "absent": len(absent),
            "top1": sum(r["correct"] for r in pos),
            "top3": sum(r["in_top3"] for r in pos),
            "top5": sum(r["in_top5"] for r in pos),
            "f1_top1": metrics["with_rejection_all_queries"]["1"]["f1"],
            "f1_top5": metrics["with_rejection_all_queries"]["5"]["f1"],
            "absent_shown": sum(r["probability"] >= show_p for r in absent),
            "predict_null_ok": [right + nulls, len(rows)],
            "predict_strict": [right, len(rows)],
            "predict_positives": [right, len(pos)],
            "always_answer": [sum(r["correct"] for r in pos), len(rows)],
        }
        summary["parts"][part] = {"metrics": cell, **({"rows": rows} if part != "all" else {})}
        print(f"{part:5} вин каталога {len(pos):3}, вне {len(absent):3}; "
              f"top-1 {fmt_share(cell['top1'], len(pos))}; "
              f"top-3 {cell['top3']}, top-5 {cell['top5']}; "
              f"F1 top-1 {cell['f1_top1']:.1%}, top-5 {cell['f1_top5']:.1%}; "
              f"вне каталога показано {cell['absent_shown']}; "
              f"predict null_ok {right + nulls}/{len(rows)}, strict {right}/{len(rows)}")
    if missing_vlm:
        print("нет ответа VLM для лидера:", sorted(set(missing_vlm)))
    for part in ("tune", "test"):
        summary["parts"][part].pop("rows", None)
    summary["missing_vlm"] = sorted(set(missing_vlm))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        print(f"записано: {args.output}")


if __name__ == "__main__":
    main()
