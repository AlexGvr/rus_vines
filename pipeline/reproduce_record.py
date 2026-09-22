#!/usr/bin/env python3
"""Запись для воспроизведения: хеши данных, индексов, кода, версии моделей и библиотек.

Пишет data/validation/reproduce_<дата>.json. Нужна, чтобы любой отчёт можно
было привязать к точному состоянию входов: манифест, каталог, привязки,
дополнительные эталоны, индексы, веса уверенности, код конвейера.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

FILES = ["data/field/manifest.csv", "data/field/manifest.splits.json", "data/field/decisions.txt",
         "data/field/transcripts_tune.json", "data/catalog/catalog.json", "data/catalog/extra_refs.csv",
         "data/datapack/photo_map.csv", "data/datapack/photo_overrides.csv",
         "data/datapack/verified_attributes.json", "data/datapack/wines.json", "data/freshness.csv",
         "data/index/confidence.json", "data/index/clean_mv.csv", "data/index/clean_mv.npy",
         "data/index/sift/desc.npy", "data/index/sift/slugs.csv", "data/index/field_quality.json",
         "pipeline/searchcore.py", "pipeline/text_match.py", "pipeline/rerank.py", "pipeline/imageprep.py",
         "pipeline/ocr.py", "pipeline/embed.py", "pipeline/eval_field.py", "pipeline/eval_transcript.py",
         "pipeline/field_metrics.py", "pipeline/build_index.py", "pipeline/build_catalog.py",
         "service/main.py", "yolo11m.pt"]
SWITCH_DEFAULTS = {"TEXT_ALIASES": "1", "OCR_JOIN": "1", "NAME_FIRST": "1", "LINE_BAND": "1",
                   "PLATFORM_SWEETNESS": "1", "NAME_STRONG": "1", "NAME_TRANSLIT": "1",
                   "GROUP_LEADER": "0", "DEMOTE_GEOMETRY": "0", "DEMOTE_DEPTH": "0",
                   "LABEL_BAND": "0", "KIN_GROUP": "0", "BRAND_FIRST": "1", "SIFT_MAX_SIDE": "700"}


def sha(path: str) -> str | None:
    p = ROOT / path
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-09-22")
    ap.add_argument("--reports", nargs="*", default=[])
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    import embed
    versions = {}
    for mod in ("torch", "transformers", "easyocr", "ultralytics", "cv2", "numpy", "PIL"):
        try:
            versions[mod] = getattr(importlib.import_module(mod), "__version__", None)
        except Exception as error:
            versions[mod] = f"n/a ({type(error).__name__})"
    refs_dir = ROOT / "data" / "refs"
    record = {
        "date": args.date,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "working_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()),
        "models": {"siglip": embed.MODEL_ID, "yolo_weights": "yolo11m.pt", "easyocr_languages": ["ru", "en"]},
        "versions": versions,
        "code_defaults": SWITCH_DEFAULTS,
        "environment_overrides": {k: v for k, v in os.environ.items() if k in SWITCH_DEFAULTS or k in ("SHOW_P", "CONFIDENT_P")},
        "thresholds": json.loads((ROOT / "data/index/confidence.json").read_text())["thresholds"],
        "sha256": {f: sha(f) for f in FILES},
        "extra_refs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(refs_dir.iterdir()) if p.is_file()},
        "reports": args.reports,
        "note": args.note,
    }
    out = ROOT / "data" / "validation" / f"reproduce_{args.date.replace('-', '')}.json"
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"записано: {out}")


if __name__ == "__main__":
    main()
