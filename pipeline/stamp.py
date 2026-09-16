#!/usr/bin/env python3
"""Отпечаток прогона: что за код, индекс, модель уверенности и пороги.

Отчёт без этого — число без контекста. Через неделю нельзя сказать, на каком
индексе он получен и с какими порогами, а значит нельзя ни повторить, ни
сравнить с соседним отчётом. Печатается в начале harness.sh и кладётся
в артефакты замеров.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        revision = out.stdout.strip() or "не определена"
    except Exception:
        return "не определена"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10).stdout.strip()
    return revision + (" + несохранённые правки" if dirty else "")


def service_constants() -> dict:
    """Пороги и веса читаются из сервиса — там, где они действуют."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    text = (ROOT / "service" / "main.py").read_text(encoding="utf-8")
    scope: dict = {}
    for line in text.splitlines():
        if line.startswith(("CONF_WEIGHTS", "CONFIDENT_P", "SHOW_P", "CLOSE_WINDOW",
                            "OCR_MIN_CONF", "CONFLICT_GATE", "RERANK_TOPK",
                            "INDEX_VARIANT")):
            try:
                exec(line, {"os": __import__("os"), "float": float, "int": int}, scope)
            except Exception:
                continue
    return scope


def describe() -> dict:
    constants = service_constants()
    index = constants.get("INDEX_VARIANT", "clean_mv")
    rows = 0
    path = INDEX_DIR / f"{index}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            rows = sum(1 for _ in csv.DictReader(fh))
    weights = constants.get("CONF_WEIGHTS", {})
    digest = hashlib.sha256(
        json.dumps(weights, sort_keys=True).encode()).hexdigest()[:10]
    sift = INDEX_DIR / "sift" / "desc.npy"
    return {
        "код": git_revision(),
        "индекс": f"{index}, строк {rows}",
        "признаки эталонов": (f"{sift.stat().st_size / 1e6:.0f} МБ"
                              if sift.exists() else "нет"),
        "модель уверенности": f"{len(weights)} весов, отпечаток {digest}",
        "порог показа": constants.get("SHOW_P"),
        "порог без оговорок": constants.get("CONFIDENT_P"),
        "шортлист": constants.get("RERANK_TOPK"),
    }


def main() -> None:
    stamp = describe()
    for key, value in stamp.items():
        print(f"  {key:22} {value}")
    (INDEX_DIR / "stamp.json").write_text(
        json.dumps(stamp, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
