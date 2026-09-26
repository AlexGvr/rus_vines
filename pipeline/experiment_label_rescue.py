#!/usr/bin/env python3
"""Спасение отказов по этикетке: пилот на кадрах, где /v1/eval/predict молчит.

Для /v1/analogs VLM читает с этикетки винодельню и название, и среди вин
этой винодельни нужное вино часто оказывается первым по сходству названия.
Здесь проверяется, можно ли так же отвечать скрипту оценки вместо null.

Для каждого кадра с отказом в прогоне pipeline/api_check.py сохраняется:
  - чтение этикетки полным промптом (как /v1/analogs) и коротким — только
    винодельня и название, ради времени ответа;
  - три ближайших по названию вина той же винодельни и лучшее по названию
    во всём каталоге, со сходством;
  - P(yes) верификатора (pipeline/vlm_verify.py) для первых двух вин
    винодельни и для лучшего по каталогу.
Правила решения выбираются потом по кэшу (--analyze), без видеокарты.

Сервис во время прогона должен быть остановлен: второй экземпляр VLM
в 16 ГБ не помещается.

    python pipeline/experiment_label_rescue.py --run data/validation/api_check.json
    python pipeline/experiment_label_rescue.py --analyze
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "service"))

OUT = ROOT / "data" / "validation" / "label_rescue_20260926.jsonl"
MANIFEST = ROOT / "data" / "field" / "manifest.csv"
ORGANIZER = "публичный набор кейса"


def closeness(name: str, title: str) -> float:
    from recommend import name_closeness
    return name_closeness(name, title)


def run(report: Path) -> None:
    from PIL import Image
    import vlm_label
    import vlm_verify
    from api_check import locate
    from imageprep import normalize_batch
    from recommend import Recommender
    from text_match import TextChannel

    wines = json.loads((ROOT / "data/catalog/catalog.json").read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    channel = TextChannel(wines)
    rec = Recommender(wines, {s: a.sweetness for s, a in channel.attributes_of.items()})
    manifest = {r["image"]: r for r in csv.DictReader(MANIFEST.open(encoding="utf-8"))}
    rows = [r for r in json.loads(report.read_text(encoding="utf-8"))["results"]
            if r["predicted"] is None]
    verifier = vlm_verify.load()
    done = set()
    if OUT.exists():
        done = {json.loads(line)["image"] for line in OUT.open(encoding="utf-8")}

    short_prompt = (
        "Read the label of the central, fully visible wine bottle in the photo. "
        "Answer with one JSON object and nothing else: "
        '{"producer": producer as written on the label or null, '
        '"producer_ru": producer name in Cyrillic or null, "name": wine name or null}. '
        "Write only what the label shows; use null when it is not visible.")

    def generate(crop, prompt: str, tokens: int) -> tuple[str, float]:
        torch = verifier.torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": vlm_verify.shrink(crop, vlm_label.SIDE)},
            {"type": "text", "text": prompt}]}]
        inputs = verifier.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        started = time.perf_counter()
        with torch.no_grad():
            out = verifier.model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
        raw = verifier.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return raw, (time.perf_counter() - started) * 1000

    with OUT.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(rows, 1):
            if row["image"] in done:
                continue
            path = locate(row["image"])
            crop = normalize_batch([Image.open(path).convert("RGB")], steps=("detect",))[0]
            raw_full, ms_full = generate(crop, vlm_label.PROMPT, vlm_label.MAX_NEW_TOKENS)
            raw_short, ms_short = generate(crop, short_prompt, 60)
            full = vlm_label.parse(raw_full) or {}
            short = vlm_label.parse(raw_short) or {}

            entry = {"image": row["image"], "gold": row["gold"],
                     "split": manifest[row["image"]]["split"],
                     "source": "organizer" if manifest[row["image"]]["source"] == ORGANIZER
                     else "reviews",
                     "ms_full": round(ms_full), "ms_short": round(ms_short),
                     "full": full, "short": short}
            for variant, label in (("full", full), ("short", short)):
                maker = rec.maker_of(label)
                name = label.get("name") or ""
                own = [w for w in wines if rec.facets[w["slug"]].manufacturer == maker] if maker else []
                ranked = sorted(((closeness(name, w["title"]), w["slug"]) for w in own),
                                key=lambda x: -x[0])[:3]
                best = max(((closeness(name, w["title"]), w["slug"]) for w in wines),
                           key=lambda x: x[0], default=(0.0, None))
                entry[variant + "_maker"] = maker
                entry[variant + "_ranked"] = [{"slug": s, "close": round(c, 3)} for c, s in ranked]
                entry[variant + "_global"] = {"slug": best[1], "close": round(best[0], 3)}
            # P(yes) — одинаковый для обоих вариантов, если slug совпал.
            checked: dict[str, float] = {}
            for variant in ("full", "short"):
                slugs = [r["slug"] for r in entry[variant + "_ranked"][:2]]
                if entry[variant + "_global"]["slug"]:
                    slugs.append(entry[variant + "_global"]["slug"])
                for slug in slugs:
                    if slug not in checked:
                        checked[slug] = round(vlm_verify.p_yes(
                            crop, slug, by_slug.get(slug, {}), channel), 4)
            entry["p_yes"] = checked
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{n}/{len(rows)} {row['image']} gold={row['gold']} "
                  f"maker={entry['full_maker']} top={entry['full_ranked'][:1]} "
                  f"{ms_full:.0f}/{ms_short:.0f} мс", flush=True)


FAST = ROOT / "data" / "validation" / "label_rescue_fast_20260926.jsonl"


def run_fast(report: Path) -> None:
    """Дешёвые варианты: короткое чтение с меньшего кадра и кандидат по OCR.

    Слова этикетки сервис уже прочитал (searchcore.LabelText на выбранном
    виде), так что кандидат по тексту через TextChannel.scores не стоит
    ни одной генерации — только проверку P(yes).
    """
    import csv as _csv
    import time as _time

    from PIL import Image
    import searchcore as core
    import vlm_label
    import vlm_verify
    from api_check import locate
    from embed import embed_images, load as load_model
    from imageprep import normalize_batch
    from ocr import load as load_ocr, read_words
    from recommend import Recommender
    from text_match import TextChannel
    from vector_store import open_store

    wines = json.loads((ROOT / "data/catalog/catalog.json").read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}
    channel = TextChannel(wines)
    rec = Recommender(wines, {s: a.sweetness for s, a in channel.attributes_of.items()})
    store = open_store("clean_mv")
    with (ROOT / "data/index/clean_mv.csv").open(encoding="utf-8") as fh:
        index_slugs = core.scope_index([row["slug"] or None for row in _csv.DictReader(fh)])
    load_model()
    load_ocr()
    verifier = vlm_verify.load()
    cached = {json.loads(line)["image"]: json.loads(line) for line in OUT.open(encoding="utf-8")}
    rows = [r for r in json.loads(report.read_text(encoding="utf-8"))["results"]
            if r["predicted"] is None]
    prompt = (
        "Read the label of the central, fully visible wine bottle. Answer with one JSON "
        'object and nothing else: {"producer": producer as written or null, '
        '"producer_ru": producer in Cyrillic or null, "name": wine name or null}.')

    def generate(crop, side: int) -> tuple[dict, float]:
        torch = verifier.torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": vlm_verify.shrink(crop, side)},
            {"type": "text", "text": prompt}]}]
        inputs = verifier.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        started = _time.perf_counter()
        with torch.no_grad():
            out = verifier.model.generate(**inputs, max_new_tokens=48, do_sample=False)
        raw = verifier.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return vlm_label.parse(raw) or {}, (_time.perf_counter() - started) * 1000

    with FAST.open("w", encoding="utf-8") as fh:
        for n, row in enumerate(rows, 1):
            image = Image.open(locate(row["image"])).convert("RGB")
            views = {steps: normalize_batch([image], steps=steps)[0] for steps in ((), ("detect",))}
            similarity = store.similarity(embed_images(list(views.values())))
            prepared = list(views.values())[core.pick_view(list(views.values()), similarity)]
            crop = views[("detect",)]
            words = core.read_label(prepared, read_words)
            text = " ".join(w.text for w in words)
            ranked = sorted(channel.scores(text).items(), key=lambda kv: -kv[1])[:3]
            entry = {"image": row["image"], "gold": row["gold"],
                     "split": cached[row["image"]]["split"],
                     "source": cached[row["image"]]["source"],
                     "ocr_text": text[:300],
                     "ocr_top": [{"slug": s, "score": round(v, 3)} for s, v in ranked]}
            for side in (768, 512):
                label, ms = generate(crop, side)
                maker = rec.maker_of(label)
                name = label.get("name") or ""
                own = [w for w in wines if rec.facets[w["slug"]].manufacturer == maker] if maker else []
                pool = own or wines
                best = max(((closeness(name, w["title"]), w["slug"]) for w in pool),
                           key=lambda x: x[0], default=(0.0, None))
                entry[f"vlm{side}"] = {"label": label, "ms": round(ms), "maker": maker,
                                       "slug": best[1], "close": round(best[0], 3)}
            p_yes = dict(cached[row["image"]]["p_yes"])
            wanted = [c["slug"] for c in entry["ocr_top"][:1]]
            wanted += [entry[f"vlm{s}"]["slug"] for s in (768, 512) if entry[f"vlm{s}"]["slug"]]
            for slug in wanted:
                if slug not in p_yes:
                    p_yes[slug] = round(vlm_verify.p_yes(crop, slug, by_slug.get(slug, {}),
                                                         channel), 4)
            entry["p_yes"] = p_yes
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{n}/{len(rows)} {row['image']} ocr={entry['ocr_top'][:1]} "
                  f"v768={entry['vlm768']['slug']}:{entry['vlm768']['ms']}мс "
                  f"v512={entry['vlm512']['slug']}:{entry['vlm512']['ms']}мс", flush=True)


def decide(entry: dict, variant: str, close: float, p: float, scope: str) -> str | None:
    """Ответ правила или None (отказ остаётся)."""
    pool = list(entry[variant + "_ranked"][:1])
    if scope == "global" and not pool:
        pool = [entry[variant + "_global"]]
    for cand in pool:
        if cand["slug"] and cand["close"] >= close and entry["p_yes"].get(cand["slug"], 0.0) >= p:
            return cand["slug"]
    return None


def analyze() -> None:
    entries = [json.loads(line) for line in OUT.open(encoding="utf-8")]
    print(f"кадров с отказом: {len(entries)}")
    for source in ("organizer", "reviews"):
        for split in ("tune", "test"):
            rows = [e for e in entries if e["source"] == source and e["split"] == split]
            pos = sum(1 for e in rows if e["gold"])
            print(f"  {source}/{split}: {len(rows)} (в каталоге {pos}, вне {len(rows) - pos})")
    ms = sorted(e["ms_full"] for e in entries)
    ms_s = sorted(e["ms_short"] for e in entries)
    print(f"чтение: полный промпт медиана {ms[len(ms)//2]} мс, p95 {ms[int(len(ms)*.95)]}; "
          f"короткий медиана {ms_s[len(ms_s)//2]} мс, p95 {ms_s[int(len(ms_s)*.95)]}")

    print("\nправило: вариант, сходство названия >=, P(yes) >=, область | "
          "organizer tune: +верных / ложных вне каталога / неверных в каталоге | test | reviews")
    for variant in ("full", "short"):
        for scope in ("maker", "global"):
            for close in (0.5, 0.6, 0.7, 0.8):
                for p in (0.0, 0.5, 0.8):
                    line = []
                    for source, split in (("organizer", "tune"), ("organizer", "test"),
                                          ("reviews", None)):
                        rows = [e for e in entries if e["source"] == source
                                and (split is None or e["split"] == split)]
                        good = absent = wrong = 0
                        for e in rows:
                            slug = decide(e, variant, close, p, scope)
                            if slug is None:
                                continue
                            if not e["gold"]:
                                absent += 1
                            elif slug == e["gold"]:
                                good += 1
                            else:
                                wrong += 1
                        line.append(f"+{good}/{absent}/{wrong}")
                    print(f"  {variant:5} {close:.1f} {p:.1f} {scope:6} | " + " | ".join(line))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, help="отчёт api_check.py с отказами")
    ap.add_argument("--run-fast", type=Path, help="то же, дешёвые варианты")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.run:
        run(args.run)
    if args.run_fast:
        run_fast(args.run_fast)
    if args.analyze:
        analyze()


if __name__ == "__main__":
    main()
