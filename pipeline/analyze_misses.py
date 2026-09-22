#!/usr/bin/env python3
"""Разбор промахов выбора лидера по категориям причин.

Вход — отчёт eval_field.py, собранный с --candidate-features: в нём есть
кандидаты с признаками и прочитанные OCR строки. Нулевое текстовое
подтверждение не значит, что OCR ничего не прочитал: слово может стоять
у нескольких кандидатов сразу. Поэтому для каждого промаха печатаются
прочитанные строки и реальные различия карточек правильного ответа и
выданного лидера — по ним и назначается категория.

Категории:
  эталон отсутствует/неверен — у правильной позиции нет эталона в индексе;
  потеря кандидата            — правильный ответ не попал в шортлист;
  ошибка OCR                  — лидер выбран геометрией, а различающего
                                слова (сладость, цвет, сорт, имя) OCR не
                                прочитал вовсе;
  ошибка текстового сравнения — различающее слово прочитано, но текст
                                лидера не исправил (или сам испортил);
  ошибка геометрии            — карточки различаются только тем, чего на
                                этикетке нет (год, крепость) либо эталон
                                другой упаковки;
  целевая бутылка / разметка  — кадр с несколькими бутылками или коллаж:
                                нужен взгляд, автоматически не решается.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_match import (CATEGORY_WORDS, GENERIC_WORDS, STOPWORDS, TextChannel,  # noqa: E402
                        alphabet_variants, attributes_from_slug, card_features,
                        extract_attributes)
from matcher import tokens_match  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def differences(gold: str, top1: str, channel: TextChannel) -> dict[str, tuple]:
    """Чем карточка правильного ответа отличается от карточки лидера."""
    g, t = card_features(gold, channel), card_features(top1, channel)
    out = {}
    if g.sweetness != t.sweetness:
        out["сладость"] = (g.sweetness, t.sweetness)
    if g.color != t.color:
        out["цвет"] = (g.color, t.color)
    if g.grapes != t.grapes:
        out["сорт"] = (sorted(g.grapes - t.grapes), sorted(t.grapes - g.grapes))
    if g.name != t.name:
        out["имя"] = (sorted(g.name - t.name), sorted(t.name - g.name))
    if g.producer != t.producer:
        out["винодельня"] = (sorted(g.producer), sorted(t.producer))
    if g.years != t.years:
        out["год"] = (sorted(g.years), sorted(t.years))
    return out


def read_hits(words: list[dict], diffs: dict, channel: TextChannel) -> list[str]:
    """Какие из различающих признаков OCR действительно прочитал."""
    text = " ".join(w["text"] for w in words if w["conf"] >= 0.6)
    attrs = extract_attributes(text)
    hits = []
    if "сладость" in diffs and attrs.sweetness:
        hits.append(f"сладость={attrs.sweetness}")
    if "цвет" in diffs and attrs.color:
        hits.append(f"цвет={attrs.color}")
    tokens = [t for w in words if w["conf"] >= 0.6
              for t in re.split(r"[^a-zа-я0-9]+", w["text"].lower()) if len(t) >= 3]
    for key in ("сорт", "имя"):
        if key not in diffs:
            continue
        gold_only = set(diffs[key][0])
        for tok in tokens:
            if any(tokens_match(v, m) >= 0.7 for v in alphabet_variants(tok) for m in gold_only):
                hits.append(f"{key}:«{tok}»")
                break
    return hits


def categorize(r: dict, channel: TextChannel) -> tuple[str, str]:
    if not r["findable"]:
        return "эталон отсутствует/неверен", ""
    if r["rank_shortlist"] < 0:
        return "потеря кандидата", f"в первой сотне на месте {r['rank_retrieval100'] + 1 if r['rank_retrieval100'] >= 0 else '—'}"
    diffs = differences(r["gold"], r["top1"], channel)
    hits = read_hits(r.get("words", []), diffs, channel)
    label_visible = {k for k in diffs if k in ("сладость", "цвет", "сорт", "имя", "винодельня")}
    if r["frame"] in ("multi", "collage"):
        base = "целевая бутылка / разметка (проверить)"
    elif r["rank_geometry"] == 0 and r["rank_text"] != 0:
        base = "ошибка текстового сравнения (текст испортил)"
    elif not label_visible:
        base = "ошибка геометрии (различий на лицевой этикетке нет)"
    elif hits:
        base = "ошибка текстового сравнения (слово прочитано, не сработало)"
    else:
        base = "ошибка OCR (различающее слово не прочитано)"
    detail = "; ".join(f"{k}: {v[0]} ≠ {v[1]}" for k, v in diffs.items())
    if hits:
        detail += " | прочитано: " + ", ".join(hits)
    return base, detail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--verbose", action="store_true", help="печатать прочитанные строки")
    args = ap.parse_args()
    data = json.loads(Path(args.report).read_text(encoding="utf-8"))
    wines = json.loads((ROOT / "data/catalog/catalog.json").read_text(encoding="utf-8"))["wines"]
    channel = TextChannel(wines)
    titles = {w["slug"]: w["title"] for w in wines}
    misses = [r for r in data["results"] if r["gold"] and not r["correct"]]
    counts: dict[str, list] = {}
    for r in misses:
        cat, detail = categorize(r, channel)
        counts.setdefault(cat, []).append((r, detail))
    print(f"промахов top-1 среди положительных: {len(misses)} из "
          f"{sum(1 for r in data['results'] if r['gold'])}\n")
    for cat, items in sorted(counts.items(), key=lambda kv: -len(kv[1])):
        print(f"== {cat}: {len(items)}")
        for r, detail in items:
            print(f"  {r['image'][:36]:36} верно «{titles.get(r['gold'], r['gold'])[:34]}» "
                  f"→ «{titles.get(r['top1'], r['top1'])[:34]}» "
                  f"[шортлист {r['rank_shortlist'] + 1 if r['rank_shortlist'] >= 0 else '—'}, "
                  f"геом {r['rank_geometry'] + 1 if r['rank_geometry'] >= 0 else '—'}, "
                  f"текст {r['rank_text'] + 1 if r['rank_text'] >= 0 else '—'}, инл {r['inliers']}]")
            if detail:
                print(f"      {detail}")
            if args.verbose and r.get("words"):
                print("      OCR: " + " | ".join(f"{w['text']}({w['conf']:.2f})" for w in r["words"][:14]))
        print()


if __name__ == "__main__":
    main()
