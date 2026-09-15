#!/usr/bin/env python3
"""S5.7 — LightGlue против SIFT на том же шортлисте.

SIFT сопоставляет дескрипторы попарно и независимо: каждая точка ищет себе
ближайшую, а согласованность проверяется потом гомографией. LightGlue решает
задачу целиком — сеть смотрит на оба набора точек сразу и выдаёт соответствия
с учётом взаимного расположения. На стандартных наборах это заметно лучше,
но этикетки вин — не стандартный набор: плоская поверхность, повторяющийся
макет внутри серии, много мелкого текста. Выигрыш нужно подтвердить здесь.

Сравнение честное: один и тот же шортлист от двухвидового индекса, один
и тот же кроп, меняется только способ сопоставления. Признаки эталонов
LightGlue считает на лету с кэшем — предрасчёт имеет смысл только после
того, как выигрыш подтвердится.

Результат (data/index/lightglue_*.json): не подтвердился. На общем наборе
90.0% против 91.0% у SIFT, на почти-дублях 94.7% против 96.8%, и вчетверо
дороже по времени. Оговорка честности ради: запросы синтетические и получены
из тех же файлов, что и эталоны, а это условия, выгодные попарному
сопоставлению дескрипторов. На полевой пересъёмке расклад может отличаться,
и проверку стоит повторить, когда такой набор появится.

Зависимость ставится отдельно и в требования сервиса не входит:
  pip install git+https://github.com/cvg/LightGlue.git
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed import embed_paths  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from rerank import PrecomputedReranker, descriptors, inlier_count  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
MAX_SIDE = 700          # тот же масштаб, что у SIFT, иначе сравнение нечестное


class Glue:
    """SuperPoint + LightGlue с кэшем признаков эталонов."""

    def __init__(self, max_keypoints: int = 1024, capacity: int = 2400) -> None:
        from lightglue import LightGlue, SuperPoint

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.capacity = capacity

    def features(self, image: Image.Image) -> dict:
        arr = image.convert("L")
        scale = MAX_SIDE / max(arr.size)
        if scale < 1.0:
            arr = arr.resize((max(int(arr.width * scale), 1),
                              max(int(arr.height * scale), 1)), Image.LANCZOS)
        tensor = torch.from_numpy(np.array(arr, dtype=np.float32) / 255.0)
        tensor = tensor[None, None].to(self.device)
        with torch.inference_mode():
            return self.extractor.extract(tensor)

    def reference(self, slug: str, path: Path) -> dict | None:
        if slug in self.cache:
            self.cache.move_to_end(slug)
            return self.cache[slug]
        try:
            with Image.open(path) as im:
                value = self.features(im)
        except Exception:
            value = None
        self.cache[slug] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def matches(self, query: dict, reference: dict | None) -> int:
        if reference is None:
            return 0
        with torch.inference_mode():
            out = self.matcher({"image0": query, "image1": reference})
        return int(out["matches0"].gt(-1).sum().item())

    def verified(self, query: dict, reference: dict | None) -> int:
        """Соответствия LightGlue, дополнительно проверенные гомографией.

        Сеть уже учитывает взаимное расположение, но проверка нужна для
        сопоставимости: у SIFT в числе стоят именно инлаеры RANSAC.
        """
        if reference is None:
            return 0
        import cv2
        with torch.inference_mode():
            out = self.matcher({"image0": query, "image1": reference})
        pairs = out["matches"][0].cpu().numpy() if "matches" in out else None
        if pairs is None or len(pairs) < 8:
            return 0 if pairs is None else len(pairs)
        kp0 = query["keypoints"][0].cpu().numpy()[pairs[:, 0]]
        kp1 = reference["keypoints"][0].cpu().numpy()[pairs[:, 1]]
        _, mask = cv2.findHomography(kp0.reshape(-1, 1, 2), kp1.reshape(-1, 1, 2),
                                     cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--views", default="raw,detect")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    views = [tuple(s for s in spec.split("+") if s and s != "raw")
             for spec in args.views.split(",")]

    wines = {w["slug"]: w for w in json.loads(CATALOG.read_text())["wines"]}
    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    if args.limit:
        manifest = manifest[:args.limit]
    paths, gold = [], {}
    for row in manifest:
        path = QUERIES / args.subset / row["image"]
        if path.exists():
            paths.append(str(path))
            gold[str(path)] = row["slug"]

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]

    per_view = [embed_paths(paths, batch_size=32, progress_every=0, steps=steps)
                for steps in views]
    kept = per_view[0][1]
    truth = [gold[p] for p in kept]
    scores = np.maximum.reduce([qv @ vectors.T for qv, _ in per_view])
    order = np.argsort(-scores, axis=1)[:, :200]

    sift = PrecomputedReranker(INDEX_DIR / "sift")
    glue = Glue()
    hits = {"sift": 0, "lightglue": 0, "lightglue+ransac": 0}
    top5 = {name: 0 for name in hits}
    spent = {name: 0.0 for name in hits}

    for i, path in enumerate(kept):
        with Image.open(path) as raw:
            crop = normalize_batch([raw.convert("RGB")], steps=("detect",))[0]

        shortlist, seen = [], set()
        for j in order[i]:
            slug = index_slugs[j]
            if slug is None or slug in seen:
                continue
            seen.add(slug)
            shortlist.append(slug)
            if len(shortlist) == args.topk:
                break

        t0 = time.perf_counter()
        query_kp, query_desc = descriptors(crop)
        by_sift = {slug: inlier_count(query_kp, query_desc, *sift.reference(slug))
                   for slug in shortlist}
        spent["sift"] += time.perf_counter() - t0

        t1 = time.perf_counter()
        query_features = glue.features(crop)
        refs = {slug: glue.reference(slug, ROOT / wines[slug]["photo"])
                for slug in shortlist if wines.get(slug, {}).get("photo")}
        by_glue = {slug: glue.matches(query_features, ref) for slug, ref in refs.items()}
        spent["lightglue"] += time.perf_counter() - t1

        t2 = time.perf_counter()
        by_verified = {slug: glue.verified(query_features, ref)
                       for slug, ref in refs.items()}
        spent["lightglue+ransac"] += time.perf_counter() - t2

        for name, table in (("sift", by_sift), ("lightglue", by_glue),
                            ("lightglue+ransac", by_verified)):
            ranked = sorted(table, key=lambda s: -table[s])
            hits[name] += ranked[:1] == [truth[i]]
            top5[name] += truth[i] in ranked[:5]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(kept)}", flush=True)

    n = max(len(kept), 1)
    print(f"\nнабор {args.subset}, запросов {n}, шортлист {args.topk}")
    print(f"{'способ':20} {'top-1':>8} {'top-5':>8} {'мс/запрос':>11}")
    for name in hits:
        print(f"{name:20} {hits[name] / n * 100:>7.1f}% {top5[name] / n * 100:>7.1f}% "
              f"{spent[name] / n * 1000:>11.0f}")

    out = INDEX_DIR / f"lightglue_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "n": n, "topk": args.topk,
        "top1": {k: v / n for k, v in hits.items()},
        "top5": {k: v / n for k, v in top5.items()},
        "ms_per_query": {k: v / n * 1000 for k, v in spent.items()},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
