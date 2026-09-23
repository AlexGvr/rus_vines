#!/usr/bin/env python3
"""Сколько засчитает кейсодержатель при разных правилах подсчёта.

Скрипт оценки отправляет кадр и пишет один slug; как закрытая таблица
оценивает вино, которого нет в каталоге, организатор не сообщил. А в его
публичном наборе такие кадры — треть (31 из 93), и два из трёх примеров
в eval.zip именно такие. От правила зависит, должен ли /v1/eval/predict
отказываться отвечать (`EVAL_ABSTAIN=1`), поэтому оба варианта считаются
заранее по сохранённым прогонам eval_field.py:

  strict     — каждый кадр в знаменателе; вино вне каталога засчитать
               нельзя никаким ответом, отказ только теряет верные ответы;
  null_ok    — для вина вне каталога верен пустой slug (`null`);
  positives  — вина вне каталога в счёт не идут (как главная метрика README).

Порог отказа выбирается по настроечной части и один раз проверяется
на регрессионной: подбирать его по той же выборке, по которой отчитываешься,
значит завышать число.

    python pipeline/eval_scoring.py data/validation/final3_tune.json \
        data/validation/final3_test.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import fmt_share  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "field" / "manifest.csv"
GRID = [0.0] + [round(0.01 * i, 2) for i in range(1, 31)]
RULES = ("strict", "null_ok", "positives")
ORGANIZER = "публичный набор кейса"


def source_of(row: dict) -> str:
    src = row.get("source", "")
    if src == ORGANIZER:
        return "organizer"
    return re.sub(r"https?://([^/]+)/.*", r"\1", src) or "?"


def credited(r: dict, threshold: float, rule: str) -> int | None:
    """1 — засчитано, 0 — нет, None — кадр в счёт не идёт."""
    answered = r["probability"] >= threshold
    if r["gold"]:
        return int(answered and (r["correct"] or r.get("duplicate", False)))
    if rule == "positives":
        return None
    if rule == "null_ok":
        return int(not answered)
    return 0


def score(results: list[dict], threshold: float, rule: str) -> tuple[int, int]:
    marks = [credited(r, threshold, rule) for r in results]
    marks = [m for m in marks if m is not None]
    return sum(marks), len(marks)


def best_threshold(results: list[dict], rule: str) -> float:
    # При равенстве — меньший порог: отказ дешевле не включать без выигрыша.
    return max(GRID, key=lambda t: (score(results, t, rule)[0], -t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tune", help="отчёт eval_field.py по настроечной части")
    ap.add_argument("test", nargs="?", help="отчёт по регрессионной части")
    ap.add_argument("--output", help="сохранить сводку в JSON")
    args = ap.parse_args()

    manifest = {r["image"]: r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))}
    parts = {"tune": json.loads(Path(args.tune).read_text(encoding="utf-8"))["results"]}
    if args.test:
        parts["test"] = json.loads(Path(args.test).read_text(encoding="utf-8"))["results"]
    for rows in parts.values():
        for r in rows:
            r["source"] = source_of(manifest.get(r["image"], {}))

    subsets = {
        "organizer": lambda r: r["source"] == "organizer",
        "reviews": lambda r: r["source"] != "organizer",
        "all": lambda r: True,
    }
    show_p = json.loads((ROOT / "data/index/confidence.json").read_text())["thresholds"]["show_p"]
    summary: dict = {"show_p": show_p, "rules": {}}

    for rule in RULES:
        print(f"\n=== правило {rule} ===")
        # Порог выбирается на настроечной части кадров организатора: это
        # распределение ближе всего к закрытой проверке. Выбор по всей
        # настроечной части печатается рядом для сравнения.
        tune = parts["tune"]
        t_org = best_threshold([r for r in tune if subsets["organizer"](r)], rule)
        t_all = best_threshold(tune, rule)
        summary["rules"][rule] = {"threshold_organizer_tune": t_org,
                                  "threshold_all_tune": t_all, "cells": {}}
        print(f"порог по tune/organizer: {t_org:.2f}; по всей tune: {t_all:.2f}; "
              f"порог показа сервиса: {show_p:.2f}")
        print(f"{'часть':6} {'подмножество':11} {'без отказа':>26} "
              f"{'отказ ' + format(t_org, '.2f'):>26} {'отказ ' + format(show_p, '.2f'):>26}")
        for part, rows in parts.items():
            for name, keep in subsets.items():
                chunk = [r for r in rows if keep(r)]
                cells = {}
                for label, t in (("none", 0.0), ("tuned", t_org), ("show_p", show_p)):
                    k, n = score(chunk, t, rule)
                    cells[label] = {"threshold": t, "credited": k, "n": n}
                summary["rules"][rule]["cells"][f"{part}/{name}"] = cells
                print(f"{part:6} {name:11} "
                      + " ".join(f"{fmt_share(c['credited'], c['n']):>26}"
                                 for c in cells.values()))

    # Что отказ стоит на винах каталога и что даёт на отсутствующих — по
    # кадрам организатора обеих частей вместе, при пороге показа сервиса.
    org = [r for rows in parts.values() for r in rows if r["source"] == "organizer"]
    pos = [r for r in org if r["gold"]]
    neg = [r for r in org if not r["gold"]]
    lost = [r["image"] for r in pos if r["correct"] and r["probability"] < show_p]
    kept_fp = [r["image"] for r in neg if r["probability"] >= show_p]
    print(f"\nкадры организатора, порог {show_p:.2f}: вина каталога {len(pos)}, "
          f"верных без отказа {sum(r['correct'] for r in pos)}, из них ушли бы в отказ "
          f"{len(lost)}; вне каталога {len(neg)}, ответ всё равно дан {len(kept_fp)}")
    summary["organizer_cost"] = {"threshold": show_p, "positives": len(pos),
                                 "correct_lost": lost, "absent": len(neg),
                                 "absent_answered": kept_fp}
    if args.output:
        Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        print(f"записано: {args.output}")


if __name__ == "__main__":
    main()
