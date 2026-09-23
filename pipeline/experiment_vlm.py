#!/usr/bin/env python3
"""Пилот: VLM проверяет кандидатов пятёрки попарно «тот же товар?».

OCR-ветка исчерпана (EasyOCR, PaddleOCR, полоски, развёртка, апскейл —
всё около нуля), а оставшиеся промахи — соседи по линейке, которых
различают мелкое слово и цвет, и вина вне каталога. VLM читает мелкий
стилизованный шрифт и видит цвет, поэтому проверяется как верификатор
поверх замороженного конвейера:

  кадр (кроп целевой бутылки тем же YOLO) + эталон кандидата + поля
  карточки -> «это тот же товар? yes/no», P(yes) из логитов двух токенов.

Кандидаты и их порядок берутся из сохранённого прогона eval_field.py
(`--candidate-features`), поэтому сравнение парное: меняется только
решение поверх той же пятёрки. Правила решения выбираются по tune
и один раз проверяются на test (`--analyze`).

    python pipeline/experiment_vlm.py --reports data/validation/final3_tune.json \
        data/validation/final3_test.json --out data/validation/vlm_pilot_20260923
    python pipeline/experiment_vlm.py --analyze --out data/validation/vlm_pilot_20260923
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from vlm_verify import MODEL, QUERY_SIDE, REF_SIDE, Verifier, describe, shrink  # noqa: E402

FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
INDEX_CSV = ROOT / "data" / "index" / "clean_mv.csv"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def locate(name: str) -> Path | None:
    for base in (FIELD, PUBLIC):
        if (base / name).exists():
            return base / name
    return None


def prepare_crops(frames: list[str], cache: Path) -> dict[str, Path]:
    """Кроп целевой бутылки тем же детектором, что в сервисе; кэш на диске."""
    from imageprep import normalize_batch
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    todo = []
    for name in frames:
        path = cache / (Path(name).stem + ".jpg")
        out[name] = path
        if not path.exists():
            todo.append(name)
    for i in range(0, len(todo), 8):
        chunk = todo[i:i + 8]
        images = [Image.open(locate(n)).convert("RGB") for n in chunk]
        crops = normalize_batch(images, steps=("detect",))
        for name, crop in zip(chunk, crops):
            shrink(crop, QUERY_SIDE).save(out[name], quality=92)
    return out


def run(args) -> None:
    from text_match import TextChannel
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    channel = TextChannel(wines)
    refs: dict[str, list[str]] = {}
    for row in csv.DictReader(INDEX_CSV.open(encoding="utf-8")):
        if row["slug"] and row["view"] == "raw":
            refs.setdefault(row["slug"], []).append(row["path"])

    records = []
    for report in args.reports:
        part = "tune" if "tune" in Path(report).name else "test"
        for r in json.loads(Path(report).read_text(encoding="utf-8"))["results"]:
            records.append((part, r))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.jsonl"
    done = set()
    if scores_path.exists():
        done = {json.loads(line)["image"] for line in scores_path.open(encoding="utf-8")}
    records = [(p, r) for p, r in records if r["image"] not in done]
    if args.limit:
        records = records[:args.limit]
    crops = prepare_crops([r["image"] for _, r in records], out / "crops")
    import torch
    torch.cuda.empty_cache()

    verifier = Verifier(args.model)
    ref_cache: dict[str, Image.Image] = {}
    t_all = time.perf_counter()
    with scores_path.open("a", encoding="utf-8") as fh:
        for n, (part, r) in enumerate(records, 1):
            query = Image.open(crops[r["image"]]).convert("RGB")
            t0 = time.perf_counter()
            cands = []
            for c in r["candidates"][:args.topk]:
                slug = c["slug"]
                desc = describe(slug, by_slug.get(slug, {}), channel)
                best = 0.0
                for path in refs.get(slug, [])[:args.max_refs]:
                    if path not in ref_cache:
                        ref_cache[path] = shrink(Image.open(ROOT / path), REF_SIDE)
                    best = max(best, verifier.p_yes(query, ref_cache[path], desc))
                cands.append({"slug": slug, "p_yes": round(best, 4)})
            ms = (time.perf_counter() - t0) * 1000
            fh.write(json.dumps({"image": r["image"], "part": part, "gold": r["gold"],
                                 "top1": r["top1"], "probability": r["probability"],
                                 "candidates": cands, "ms": round(ms)},
                                ensure_ascii=False) + "\n")
            fh.flush()
            print(f"Start {n}/{len(records)} {r['image'][:40]:40} "
                  f"{' '.join(f'{c['p_yes']:.2f}' for c in cands)} {ms:.0f} ms", flush=True)
    print(f"chain done in {(time.perf_counter() - t_all) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-refs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        from vlm_analysis import analyze
        analyze(Path(args.out))
    else:
        run(args)


if __name__ == "__main__":
    main()
