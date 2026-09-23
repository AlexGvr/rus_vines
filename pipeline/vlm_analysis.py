"""Разбор пилота VLM-верификатора (pipeline/experiment_vlm.py).

Правила решения поверх замороженной пятёрки:
  rerank(d)   — лидером становится кандидат с наибольшим P(yes), если он
                обходит текущего лидера больше чем на d;
  gate(g)     — перестановка только при уверенности конвейера ниже g;
  abstain     — для правила подсчёта null_ok: отказ, если уверенность
                конвейера ниже порога показа и P(yes) лидера ниже t.
Параметры выбираются по tune, test только проверяет выбранное.
"""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

from stats import fmt_share, mcnemar_exact

ROOT = Path(__file__).resolve().parent.parent
ORGANIZER = "публичный набор кейса"


def load(out: Path) -> list[dict]:
    manifest = {r["image"]: r for r in csv.DictReader(
        (ROOT / "data/field/manifest.csv").open(encoding="utf-8"))}
    rows = [json.loads(line) for line in (out / "scores.jsonl").open(encoding="utf-8")]
    for r in rows:
        r["source"] = ("organizer" if manifest[r["image"]]["source"] == ORGANIZER
                       else "reviews")
    return rows


def decide(r: dict, d: float, g: float) -> tuple[str, float]:
    """Ответ и P(yes) ответа при правиле rerank(d) с воротами g."""
    cands = r["candidates"]
    if not cands:
        return r["top1"], 0.0
    lead = cands[0]
    if r["probability"] < g:
        best = max(cands, key=lambda c: c["p_yes"])
        if best["p_yes"] - lead["p_yes"] > d:
            return best["slug"], best["p_yes"]
    return lead["slug"], lead["p_yes"]


def evaluate(rows: list[dict], d: float, g: float, show_p: float | None,
             t: float | None) -> dict:
    """Строгий top-1 по положительным и счёт null_ok по всем кадрам.

    show_p/t = None — без отказа. Отказ: уверенность конвейера ниже show_p
    и P(yes) ответа ниже t (t = 1.01 — отказ только по конвейеру,
    show_p = 1.01 — только по VLM).
    """
    pos_ok = null_ok = 0
    marks = {}
    for r in rows:
        slug, py = decide(r, d, g)
        abstain = (show_p is not None and r["probability"] < show_p and py < t)
        right = bool(r["gold"]) and not abstain and slug == r["gold"]
        marks[r["image"]] = right
        pos_ok += right
        null_ok += right if r["gold"] else abstain
    n_pos = sum(1 for r in rows if r["gold"])
    return {"pos": (pos_ok, n_pos), "null_ok": (null_ok, len(rows)), "marks": marks}


D_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
G_GRID = [0.05, 0.12, 0.2, 0.3, 0.5, 1.01]
T_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
S_GRID = [0.0, 0.05, 0.08, 0.12, 0.15, 0.2, 0.3, 1.01]


def analyze(out: Path) -> None:
    rows = load(out)
    tune = [r for r in rows if r["part"] == "tune"]
    test = [r for r in rows if r["part"] == "test"]
    summary: dict = {"n": {"tune": len(tune), "test": len(test)}}

    # Потолок: правильный ответ в пятёрке, и VLM ставит его выше всех.
    for part, chunk in (("tune", tune), ("test", test)):
        pos = [r for r in chunk if r["gold"]]
        in5 = [r for r in pos if r["gold"] in [c["slug"] for c in r["candidates"]]]
        vlm_best = sum(1 for r in in5
                       if max(r["candidates"], key=lambda c: c["p_yes"])["slug"] == r["gold"])
        print(f"{part}: положительных {len(pos)}, верный в пятёрке {len(in5)}, "
              f"VLM ставит верный выше всех {vlm_best}")

    # 1. Перестановка по VLM без отказа: строгий top-1 по положительным.
    base = {p: evaluate(c, 1.01, 0, None, None) for p, c in (("tune", tune), ("test", test))}
    best = max(itertools.product(D_GRID, G_GRID),
               key=lambda dg: (evaluate(tune, dg[0], dg[1], None, None)["pos"][0],
                               dg[0], -dg[1]))
    d, g = best
    print(f"\nперестановка: выбрано по tune d={d}, g={g}")
    summary["rerank"] = {"d": d, "g": g}
    for part, chunk in (("tune", tune), ("test", test)):
        new = evaluate(chunk, d, g, None, None)
        fixed = [i for i, v in new["marks"].items() if v and not base[part]["marks"][i]]
        broken = [i for i, v in new["marks"].items() if not v and base[part]["marks"][i]]
        line = (f"  {part}: строгий top-1 {fmt_share(*base[part]['pos'])} -> "
                f"{fmt_share(*new['pos'])}  +{len(fixed)}/-{len(broken)} "
                f"p={mcnemar_exact(len(fixed), len(broken)):.3f}")
        print(line)
        for src in ("organizer", "reviews"):
            sub = [r for r in chunk if r["source"] == src]
            b = evaluate(sub, 1.01, 0, None, None)["pos"]
            a = evaluate(sub, d, g, None, None)["pos"]
            print(f"    {src:10} {fmt_share(*b):>24} -> {fmt_share(*a):>24}")
        summary["rerank"][part] = {"before": base[part]["pos"], "after": new["pos"],
                                   "fixed": fixed, "broken": broken}
        for i in fixed:
            print(f"      исправлено {i}")
        for i in broken:
            print(f"      испорчено  {i}")

    # 2. Отказ для правила null_ok: сетка по порогу конвейера и по P(yes).
    print("\nотказ (правило null_ok), перестановка выбранная выше:")
    grid = list(itertools.product(S_GRID, T_GRID))
    pick = max(grid, key=lambda st: (evaluate(tune, d, g, st[0], st[1])["null_ok"][0],
                                     -st[0], -st[1]))
    s, t = pick
    summary["abstain"] = {"show_p": s, "t": t}
    variants = (("без отказа", None, None), ("только конвейер 0.12", 0.12, 1.01),
                (f"выбрано: конвейер {s} и P(yes) {t}", s, t))
    for label, sp, tt in variants:
        cells = []
        for part, chunk in (("tune", tune), ("test", test)):
            for src in ("organizer", "all"):
                sub = [r for r in chunk if src == "all" or r["source"] == src]
                base_rows = evaluate(sub, 1.01, 0, sp, tt)["null_ok"]
                new_rows = evaluate(sub, d, g, sp, tt)["null_ok"]
                cells.append(f"{part}/{src}: {fmt_share(*base_rows)} | с VLM {fmt_share(*new_rows)}")
                summary.setdefault("abstain_table", {}).setdefault(label, {})[f"{part}/{src}"] = {
                    "pipeline": base_rows, "with_vlm": new_rows}
        print(f"  {label}")
        for c in cells:
            print(f"    {c}")
    ms = sorted(r["ms"] for r in rows)
    print(f"\nвремя VLM на кадр (5 кандидатов): медиана {ms[len(ms) // 2]} мс, "
          f"p95 {ms[int(len(ms) * 0.95)]} мс")
    summary["latency_ms"] = {"median": ms[len(ms) // 2], "p95": ms[int(len(ms) * 0.95)]}
    (out / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"записано: {out / 'analysis.json'}")
