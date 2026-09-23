#!/usr/bin/env python3
"""DINOv2 как второй канал отбора кандидатов — замер на полевом наборе.

SigLIP 2 обучен сопоставлять картинку с текстом и хорошо отвечает на
вопрос «похожая этикетка»; самоконтролируемые признаки DINOv2 обычно
сильнее в поиске конкретного экземпляра. В проекте они не проверялись.
Замер отвечает на два вопроса, не трогая конвейер:

  1. отбор: recall@20 и top-1 по косинусу у SigLIP, у DINOv2 и у суммы
     косинусов (виды raw + detect, свёртка по slug максимумом — как в индексе);
  2. соседи по линейке: внутри итоговой пятёрки конвейера (отчёты --tune/--test) ставит
     ли DINOv2 верный ответ первым чаще, чем нынешний порядок.

Вход DINOv2 — квадрат 448 без центрального кропа: стандартный кроп
процессора отрезает верх и низ бутылки.

    python pipeline/experiment_dino.py --out data/validation/dino_20260923
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from stats import fmt_share  # noqa: E402

INDEX = ROOT / "data" / "index"
FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
MODEL = "facebook/dinov2-large"
SIDE = 448
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Dino:
    def __init__(self, model_id: str = MODEL):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()

    def embed(self, images: list[Image.Image], batch: int = 16) -> dict[str, np.ndarray]:
        cls_out, gem_out = [], []
        for i in range(0, len(images), batch):
            arr = np.stack([(np.asarray(im.convert("RGB").resize((SIDE, SIDE), Image.BICUBIC),
                                        dtype=np.float32) / 255 - MEAN) / STD
                            for im in images[i:i + batch]])
            px = torch.from_numpy(arr).permute(0, 3, 1, 2).to("cuda", torch.float16)
            with torch.inference_mode():
                hidden = self.model(pixel_values=px).last_hidden_state.float()
            cls = hidden[:, 0]
            patches = hidden[:, 1:].clamp(min=1e-6)
            gem = patches.pow(3).mean(1).pow(1 / 3)
            cls_out.append(torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
            gem_out.append(torch.nn.functional.normalize(gem, dim=-1).cpu().numpy())
        return {"cls": np.concatenate(cls_out), "gem": np.concatenate(gem_out)}


def locate(name: str) -> Path:
    for base in (FIELD, PUBLIC):
        if (base / name).exists():
            return base / name
    raise FileNotFoundError(name)


def by_slug(scores: np.ndarray, slugs: list[str | None]) -> dict[str, float]:
    best: dict[str, float] = {}
    for s, v in zip(slugs, scores):
        if s and v > best.get(s, -9):
            best[s] = float(v)
    return best


def rank_of(gold: str, ranked: list[str]) -> int:
    return ranked.index(gold) if gold in ranked else -1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--crops", default=str(ROOT / "data/validation/vlm_pilot_20260923/crops"))
    ap.add_argument("--tune", default=str(ROOT / "data/validation/final3_tune.json"))
    ap.add_argument("--test", default=str(ROOT / "data/validation/final3_test.json"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((INDEX / "clean_mv.csv").open(encoding="utf-8")))
    slugs = [r["slug"] or None for r in rows]
    siglip = np.load(INDEX / "clean_mv.npy")
    cache = out / "index_dino.npz"
    if cache.exists():
        z = np.load(cache)
        ref = {"cls": z["cls"], "gem": z["gem"]}
    else:
        from imageprep import normalize_batch
        dino = Dino()
        ref = {"cls": [], "gem": []}
        for i in range(0, len(rows), 32):
            chunk = rows[i:i + 32]
            ims = [Image.open(ROOT / r["path"]).convert("RGB") for r in chunk]
            det = [j for j, r in enumerate(chunk) if r["view"] == "detect"]
            if det:
                crops = normalize_batch([ims[j] for j in det], steps=("detect",))
                for j, c in zip(det, crops):
                    ims[j] = c
            e = dino.embed(ims)
            ref["cls"].append(e["cls"])
            ref["gem"].append(e["gem"])
            if i % 640 == 0:
                print(f"  index {i}/{len(rows)}", flush=True)
        ref = {k: np.concatenate(v) for k, v in ref.items()}
        np.savez(cache, **ref)
        del dino
        torch.cuda.empty_cache()

    manifest = {r["image"]: r for r in csv.DictReader((FIELD / "manifest.csv").open(encoding="utf-8"))}
    reports = {p: json.loads(Path(path).read_text(encoding="utf-8"))["results"]
               for p, path in (("tune", args.tune), ("test", args.test))}
    frames = [(p, r) for p, rs in reports.items() for r in rs]
    qcache = out / "queries.npz"
    if qcache.exists():
        z = np.load(qcache)
        q = {k: z[k] for k in z.files}
    else:
        from embed import embed_images
        dino = Dino()
        raw = [Image.open(locate(r["image"])).convert("RGB") for _, r in frames]
        det = [Image.open(Path(args.crops) / (Path(r["image"]).stem + ".jpg")).convert("RGB")
               for _, r in frames]
        q = {}
        for view, ims in (("raw", raw), ("det", det)):
            e = dino.embed(ims)
            q[f"dino_cls_{view}"], q[f"dino_gem_{view}"] = e["cls"], e["gem"]
        del dino
        torch.cuda.empty_cache()
        for view, ims in (("raw", raw), ("det", det)):
            q[f"siglip_{view}"] = embed_images(ims)
        np.savez(qcache, **q)

    def scores(kind: str) -> np.ndarray:
        """[кадр, строка индекса] — максимум по двум видам запроса."""
        if kind == "siglip":
            a, b = q["siglip_raw"] @ siglip.T, q["siglip_det"] @ siglip.T
        else:
            a, b = q[f"dino_{kind}_raw"] @ ref[kind].T, q[f"dino_{kind}_det"] @ ref[kind].T
        return np.maximum(a, b)

    mats = {k: scores(k) for k in ("siglip", "cls", "gem")}
    variants = {"siglip": mats["siglip"], "dino_cls": mats["cls"], "dino_gem": mats["gem"]}
    for w in (0.5, 1.0):
        variants[f"siglip+{w}*cls"] = mats["siglip"] + w * mats["cls"]
        variants[f"siglip+{w}*gem"] = mats["siglip"] + w * mats["gem"]

    summary: dict = {}
    for name, mat in variants.items():
        cells = {}
        for part in ("tune", "test"):
            idx = [i for i, (p, r) in enumerate(frames) if p == part and r["gold"]]
            r1 = r20 = 0
            for i in idx:
                ranked = [s for s, _ in sorted(by_slug(mat[i], slugs).items(),
                                               key=lambda kv: -kv[1])]
                k = rank_of(frames[i][1]["gold"], ranked)
                r1 += k == 0
                r20 += 0 <= k < 20
            cells[part] = {"top1": (r1, len(idx)), "recall20": (r20, len(idx))}
        summary[name] = cells
        print(f"{name:16} " + "  ".join(
            f"{p}: top1 {fmt_share(*c['top1']):>24} r@20 {fmt_share(*c['recall20']):>24}"
            for p, c in cells.items()))

    # Соседи по линейке: порядок внутри пятёрки final3.
    print("\nвнутри пятёрки конвейера (верный есть в пятёрке):")
    within: dict = {}
    for name in ("current", "siglip", "dino_cls", "dino_gem"):
        for part in ("tune", "test"):
            hit = n = 0
            for i, (p, r) in enumerate(frames):
                if p != part or not r["gold"]:
                    continue
                five = [c["slug"] for c in r["candidates"][:5]]
                if r["gold"] not in five:
                    continue
                n += 1
                if name == "current":
                    hit += five[0] == r["gold"]
                else:
                    s = by_slug(variants[name][i], slugs)
                    hit += max(five, key=lambda x: s.get(x, -9)) == r["gold"]
            within.setdefault(name, {})[part] = (hit, n)
        print(f"  {name:10} " + "  ".join(f"{p}: {fmt_share(*v):>24}"
                                          for p, v in within[name].items()))
    summary["within_top5"] = within
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(f"записано: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
