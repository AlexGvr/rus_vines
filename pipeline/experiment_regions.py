#!/usr/bin/env python3
"""Диагностика: сравнение различающих областей этикетки среди близких кандидатов.

Общий рисунок и логотип дают геометрии много совпадений, но не определяют
SKU внутри линейки: у «Русского Игристого» пять карточек с одной эмблемой
и разной строкой под названием. Здесь проверяется, помогает ли отдельная
оценка вручную указанной области эталона (строка названия, полоса цвета).

Механика: по полному совпадению SIFT между кропом запроса и эталоном
считается гомография, кроп запроса переносится в координаты эталона, и
в указанной области сравниваются пиксели: нормированная корреляция по
яркости и расстояние по цветовой гистограмме. Область задаётся долями
ширины/высоты эталона в data/validation/label_regions.json. Кандидаты и
их порядок берутся из сохранённого отчёта eval_field.py — визуальная
часть не пересчитывается.

Это диагностика с ручными областями. Автоматизация имеет смысл, только
если на промахах область меняет ответ в пользу верного чаще, чем портит.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from eval_field import locate  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import RATIO, descriptors, detector  # noqa: E402

CATALOG = ROOT / "data" / "catalog" / "catalog.json"
REGIONS = ROOT / "data" / "validation" / "label_regions.json"


def homography(query: Image.Image, ref: Image.Image):
    """Гомография запрос → эталон по SIFT (масштаб как в rerank.descriptors)."""
    qk, qd = descriptors(query)
    rk, rd = descriptors(ref)
    if qd is None or rd is None or len(qd) < 8 or len(rd) < 8:
        return None, 0
    _, bf = detector()
    matches = bf.knnMatch(qd, rd, k=2)
    good = [m for m, n in (pair for pair in matches if len(pair) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return None, len(good)
    src = np.float32([qk[m.queryIdx].pt for m in good])
    dst = np.float32([rk[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H, int(mask.sum()) if mask is not None else 0


def scaled(image: Image.Image) -> np.ndarray:
    """Тот же масштаб, что у descriptors(): иначе гомография не совпадёт."""
    from rerank import MAX_SIDE
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1.0:
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return arr


def region_scores(query: Image.Image, ref: Image.Image, box: list[float]) -> dict | None:
    H, inl = homography(query, ref)
    if H is None:
        return None
    ref_arr = scaled(ref)
    q_arr = scaled(query)
    h, w = ref_arr.shape[:2]
    warped = cv2.warpPerspective(q_arr, H, (w, h))
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    a, b = ref_arr[y1:y2, x1:x2], warped[y1:y2, x1:x2]
    # непокрытые запросом пиксели после переноса чёрные — исключаем
    covered = (b.sum(axis=2) > 0)
    if covered.mean() < 0.5:
        return {"inliers": inl, "ncc": None, "hist": None, "coverage": float(covered.mean())}
    ga, gb = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    ga, gb = ga[covered], gb[covered]
    ncc = float(np.corrcoef(ga - ga.mean(), gb - gb.mean())[0, 1]) if ga.std() > 1 and gb.std() > 1 else 0.0
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_RGB2HSV)], [0, 1], covered.astype(np.uint8), [18, 8], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_RGB2HSV)], [0, 1], covered.astype(np.uint8), [18, 8], [0, 180, 0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    return {"inliers": inl, "ncc": round(ncc, 3), "hist": round(hist, 3), "coverage": round(float(covered.mean()), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", help="отчёт eval_field.py с --candidate-features")
    ap.add_argument("--images", nargs="*", default=[], help="кадры; по умолчанию все промахи с областями")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--output", default=str(ROOT / "data/validation/regions_experiment.json"))
    args = ap.parse_args()
    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]}
    regions = json.loads(REGIONS.read_text(encoding="utf-8"))
    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = [r for r in data["results"] if r["gold"] and (r["image"] in args.images if args.images else not r["correct"])]
    out, changed, fixed, broken = [], 0, 0, 0
    for r in rows:
        cands = [c["slug"] for c in r["candidates"][:args.top]]
        boxes = {s: regions.get(s) or regions.get(wines[s]["manufacturer"]) for s in cands}
        if not boxes.get(r["gold"]) or r["gold"] not in cands:
            continue
        crop = normalize_batch([Image.open(locate(r["image"])).convert("RGB")], steps=("detect",))[0]
        scores = {}
        for s in cands:
            if not boxes.get(s) or not wines[s].get("photo"):
                continue
            sc = region_scores(crop, Image.open(ROOT / wines[s]["photo"]).convert("RGB"), boxes[s])
            if sc:
                scores[s] = sc
        usable = {s: v for s, v in scores.items() if v.get("ncc") is not None}
        best = max(usable, key=lambda s: usable[s]["ncc"] + 0.5 * usable[s]["hist"]) if usable else None
        out.append({"image": r["image"], "gold": r["gold"], "top1": r["top1"], "region_best": best,
                    "scores": scores})
        if best and best != r["top1"]:
            changed += 1
            if best == r["gold"]:
                fixed += 1
            elif r["top1"] == r["gold"]:
                broken += 1
        t = lambda s: wines.get(s, {}).get("title", s or "—")[:28]
        line = ", ".join(f"{t(s)}: ncc {v['ncc']} hist {v['hist']} inl {v['inliers']}" for s, v in scores.items())
        print(f"{r['image'][:26]:26} верно «{t(r['gold'])}» лидер «{t(r['top1'])}» → область «{t(best)}» | {line}")
    print(f"\nкадров {len(out)}: область меняет ответ в {changed}, из них исправляет {fixed}, портит {broken}")
    Path(args.output).write_text(json.dumps({"n": len(out), "changed": changed, "fixed": fixed, "broken": broken, "items": out},
                                            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
