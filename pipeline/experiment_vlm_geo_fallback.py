#!/usr/bin/env python3
"""VLM_GEO_FALLBACK на 402 кадрах по кэшу VLM, без видеокарты.

Путь отказа сервиса (уверенность ниже порога показа) переигрывается по
сохранённым прогонам eval_field.py и P(yes) из пилота VLM (пятёрка
кандидатов каждого кадра): подтверждение лидера, проверка ещё двух
кандидатов (VLM_FALLBACK=2) и второй проход — кандидат с P(yes) > 0.5,
за которого геометрия (searchcore.geometry_backs), если VLM отвергла
лидера. Кадры с уверенностью выше порога правило не трогает.

    python pipeline/experiment_vlm_geo_fallback.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchcore import Candidate, geometry_backs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ("data/validation/dsonly_cal2_tune.json", "data/validation/dsonly_cal_test.json")
VLM = ROOT / "data/validation/vlm_pilot_dsonly_20260923/scores.jsonl"
SHOW_P, CONFIRM_P, REJECT_P, FALLBACK = 0.10, 0.8, 0.5, 2


def answer(row: dict, p_yes: dict, geo: bool) -> tuple[str | None, str]:
    """Ответ /v1/eval/predict в пути отказа и какой шаг его дал."""
    cands = [Candidate(slug=c["slug"], cv=c["cv"], inliers=c["inliers"])
             for c in row["candidates"]]
    leader = cands[0]
    if p_yes.get(leader.slug, 0.0) >= CONFIRM_P:
        return leader.slug, "лидер подтверждён"
    pool = cands[1:1 + FALLBACK]
    for c in pool:
        if p_yes.get(c.slug, 0.0) >= CONFIRM_P:
            return c.slug, "проверка следующих"
    if geo and p_yes.get(leader.slug, 1.0) < REJECT_P:
        backed = [c for c in pool if p_yes.get(c.slug, 0.0) > REJECT_P
                  and geometry_backs(c, leader, cands)]
        if backed:
            return max(backed, key=lambda c: p_yes[c.slug]).slug, "второй проход"
    return None, "отказ"


def main() -> None:
    vlm = {}
    for line in VLM.open(encoding="utf-8"):
        r = json.loads(line)
        vlm[r["image"]] = {c["slug"]: c["p_yes"] for c in r["candidates"]}
    rows = []
    for path in RUNS:
        split = "tune" if "tune" in path else "test"
        rows += [dict(r, split=split) for r in
                 json.loads((ROOT / path).read_text(encoding="utf-8"))["results"]
                 if r.get("candidates") and r["probability"] < SHOW_P]
    ok = lambda r, a: (a == r["gold"]) if r["gold"] else (a is None)  # noqa: E731
    changed = []
    for r in rows:
        before, _ = answer(r, vlm.get(r["image"], {}), geo=False)
        after, how = answer(r, vlm.get(r["image"], {}), geo=True)
        if before != after:
            changed.append((r, before, after, how))
    print(f"кадров в пути отказа: {len(rows)}; изменений: {len(changed)}")
    for r, before, after, how in changed:
        mark = "+" if ok(r, after) and not ok(r, before) else "-" if ok(r, before) and not ok(r, after) else "="
        print(f"  {mark} {r['split']} {r['image']:26} gold={r['gold'] or '-'} {before} -> {after} ({how})")
    borderline = [(r["image"], s, round(p, 3)) for r in rows
                  for s, p in vlm.get(r["image"], {}).items()
                  if 0.45 <= p <= 0.55 and s != r["top1"]]
    print(f"кандидаты с P(yes) у самой границы 0.5: {borderline}")


if __name__ == "__main__":
    main()
