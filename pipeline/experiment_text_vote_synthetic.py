#!/usr/bin/env python3
"""TEXT_VOTE на синтетике из эталонов дампа: тот же прогон, что у калибровки.

calibrate.collect проходит полный конвейер (шортлист, геометрия, OCR,
settle) по синтетическим запросам — каждый дважды: с вином в каталоге и
с вычеркнутой позицией. Здесь он запускается без правила и с правилом,
и по одним и тем же весам уверенности считаются top-1, top-5 и F1 при
пороге показа.

    python pipeline/experiment_text_vote_synthetic.py --theta 0.2 [--subset sharp]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402
from calibrate import collect  # noqa: E402
from stats import mcnemar_exact  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def summary(cases, weights, show_p) -> dict:
    present = [c for c in cases if c.present]
    absent = [c for c in cases if not c.present]
    prob = [core.outcomes(c.feats, weights)[0] for c in cases]
    shown = [p >= show_p for p in prob]
    out = {"present": len(present), "absent": len(absent),
           "top1": sum(c.correct for c in present), "top5": sum(c.in_top5 for c in present)}
    for k, key in ((1, "correct"), (5, "in_top5")):
        tp = sum(1 for c, s in zip(cases, shown) if s and c.present and getattr(c, key))
        fp = sum(shown) - tp
        fn = len(present) - tp
        out[f"f1_top{k}_strict"] = 2 * tp / (2 * tp + fp + fn)
        tp_c = sum(1 for c, s in zip(cases, shown) if s and c.present and getattr(c, key))
        fp_c = sum(1 for c, s in zip(cases, shown) if s and c.present and not getattr(c, key))
        out[f"f1_top{k}_catalog"] = 2 * tp_c / (2 * tp_c + fp_c + fn)
    out["absent_shown"] = sum(1 for c, s in zip(cases, shown) if s and not c.present)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.2)
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--output")
    args = ap.parse_args()
    artifact = json.loads((ROOT / "data/index/confidence.json").read_text())
    weights, show_p = artifact["outcome_weights"], artifact["thresholds"]["show_p"]
    runs = {}
    for name, theta in (("без правила", None), (f"TEXT_VOTE={args.theta}", args.theta)):
        if theta is None:
            os.environ.pop("TEXT_VOTE", None)
        else:
            os.environ["TEXT_VOTE"] = str(theta)
        runs[name] = collect(args.subset, "clean_mv", [(), ("detect",)], 20, 0.80, 0.60, 0)
        os.environ.pop("TEXT_VOTE", None)
    names = list(runs)
    base, rule = runs[names[0]], runs[names[1]]
    gain = sum(1 for a, b in zip(base, rule) if a.present and b.correct and not a.correct)
    loss = sum(1 for a, b in zip(base, rule) if a.present and a.correct and not b.correct)
    result = {name: summary(cases, weights, show_p) for name, cases in runs.items()}
    result["paired"] = {"gain": gain, "loss": loss, "mcnemar_p": mcnemar_exact(gain, loss)}
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
