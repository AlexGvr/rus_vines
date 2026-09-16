#!/usr/bin/env python3
"""Замер на полевом наборе: реальные снимки вместо синтетики.

Полевой набор устроен иначе, чем синтетический, и отвечает на другой вопрос.
Синтетика умеет много положительных примеров — запрос делается из эталона,
поэтому правильный ответ известен всегда. Но такой запрос легче настоящего,
и пороги, обученные на нём, на съёмке отказывают.

Реальные снимки из отзывов покупателей дают обратное распределение: почти
все они — вина, которых в каталоге нет или у которых другое поколение
этикетки. Положительных мало. Зато они впервые позволяют измерить главную
ошибку отказа не на выдуманных, а на настоящих кадрах: как часто сервис
показывает карточку вину, которого у него нет.

Поэтому отчёт делится честно:
  отрицательные — измеряются на полевых данных, их достаточно;
  положительные — перечисляются поимённо, их слишком мало для метрики.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel  # noqa: E402

FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
INDEX_DIR = ROOT / "data" / "index"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def locate(name: str) -> Path | None:
    for base in (FIELD, PUBLIC):
        if (base / name).exists():
            return base / name
    return None


def resolve_all(rows: list[dict]) -> list[tuple[dict, Path]]:
    """Все кадры манифеста или отказ считать.

    Фотографии в репозиторий не коммитятся, поэтому на чужой машине их
    может не быть. Молча пропускать такие строки нельзя: отчёт получится
    по огрызку набора, а выглядеть будет как полноценный замер. Лучше
    остановиться и сказать, чего не хватает.
    """
    found, missing = [], []
    for row in rows:
        path = locate(row["image"])
        (found.append((row, path)) if path else missing.append(row["image"]))
    if missing:
        print(f"нет {len(missing)} кадров из {len(rows)}, например: "
              f"{', '.join(missing[:5])}")
        sys.exit("полевой набор неполон — соберите его pipeline/collect_field.py "
                 "или укажите другой манифест")
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(FIELD / "manifest.csv"))
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--weights", default=str(INDEX_DIR / "calibration_sharp.json"))
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    weights = json.loads(Path(args.weights).read_text(encoding="utf-8"))["weights"]
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    reranker = PrecomputedReranker(INDEX_DIR / "sift")
    channel = TextChannel(wines)

    results = []
    for row, path in resolve_all(rows):
        image = Image.open(path).convert("RGB")
        views = {steps: normalize_batch([image], steps=steps)[0]
                 for steps in ((), ("detect",))}
        query = embed_images(list(views.values()))
        scores = (vectors @ query.T).max(axis=1)
        candidates = core.shortlist(scores, index_slugs, args.topk)
        core.rerank(views[("detect",)], candidates, reranker)
        core.resolve(candidates, views[("detect",)], channel, read_words,
                     window=0.80, min_conf=0.60)
        core.check_leader(candidates, views[("detect",)], channel, read_words, 0.60)
        feats = core.features(candidates)
        results.append({
            "image": row["image"],
            "gold": row["slug"],
            "kind": row.get("kind", "field"),
            "top1": candidates[0].slug if candidates else "",
            "inliers": candidates[0].inliers if candidates else 0,
            "probability": core.probability(feats, weights),
            "conflicts": list(candidates[0].conflicts) if candidates else [],
            "correct": bool(row["slug"]) and bool(candidates)
                       and candidates[0].slug == row["slug"],
            "in_top5": bool(row["slug"])
                       and row["slug"] in [c.slug for c in candidates[:5]],
        })
        print(f"  {row['image'][:44]:44} p={results[-1]['probability']:.3f} "
              f"инл {results[-1]['inliers']:>4} "
              f"{'ВЕРНО' if results[-1]['correct'] else ''}", flush=True)

    positives = [r for r in results if r["gold"]]
    negatives = [r for r in results if not r["gold"]]
    shot = [r for r in results if r["kind"] == "field"]
    print(f"\nполевой набор: {len(results)} кадров — "
          f"{len(positives)} с вином из каталога, {len(negatives)} без; "
          f"съёмка {len(shot)}, предметная карточка {len(results) - len(shot)}")

    if negatives:
        probability = np.array([r["probability"] for r in negatives])
        inliers = np.array([r["inliers"] for r in negatives])
        print(f"\nвина в каталоге нет ({len(negatives)} кадров):")
        print(f"  инлаеры лидера: медиана {np.median(inliers):.0f}, "
              f"90-й процентиль {np.percentile(inliers, 90):.0f}, "
              f"максимум {inliers.max()}")
        print(f"  уверенность: медиана {np.median(probability):.3f}, "
              f"90-й процентиль {np.percentile(probability, 90):.3f}, "
              f"максимум {probability.max():.3f}")
        print(f"\n{'порог':>7} {'карточка показана':>19}")
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70):
            share = float((probability >= threshold).mean())
            print(f"{threshold:>7.2f} {share * 100:>18.1f}%")

    if positives:
        right = sum(1 for r in positives if r["correct"])
        print(f"\nвино есть в каталоге ({len(positives)} кадров): "
              f"лидер верен в {right}, top-5 содержит верный в "
              f"{sum(1 for r in positives if r['in_top5'])}")
        for r in positives:
            title = by_slug.get(r["top1"], {}).get("title", "—")
            print(f"  {r['image'][:42]:42} {r['kind']:6} p={r['probability']:.3f} "
                  f"инл {r['inliers']:>4} "
                  f"{'верно' if r['correct'] else 'ошибка: ' + title[:30]}")

    for kind in ("field", "studio"):
        part = [r for r in results if r["kind"] == kind]
        if not part:
            continue
        good = [r for r in part if r["gold"]]
        bad = [r for r in part if not r["gold"]]
        name = "съёмка" if kind == "field" else "предметная карточка"
        line = (f"\n{name}: {len(part)} кадров")
        if good:
            line += (f", лидер верен в {sum(1 for r in good if r['correct'])}"
                     f" из {len(good)}")
        if bad:
            shown = sum(1 for r in bad if r["probability"] >= 0.06)
            line += (f", карточка для отсутствующего вина в {shown}"
                     f" из {len(bad)}")
        print(line)

    out = INDEX_DIR / "field_report.json"
    out.write_text(json.dumps({
        "n": len(results), "positives": len(positives), "negatives": len(negatives),
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
