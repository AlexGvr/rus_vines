#!/usr/bin/env python3
"""Матчер: текст с этикетки (OCR) → карточки вин, ранжированные по скору.

Алгоритм (зеркалится в app/js/matcher.js — менять синхронно):
  1. Токены запроса и документов в каноническом виде (см. normalize).
  2. Совпадение токена: точное / префикс (>=4) / Левенштейн <=1 (len>=5)
     или <=2 (len>=8) — OCR путает буквы.
  3. Вес токена = поле (title 3.0, manufacturer 2.0, grapes 1.0,
     region+category 0.5) × IDF-фактор (редкие токены важнее).
  4. Год на этикетке, совпавший с годом в названии, — бонус.
  5. Скор нормируется на сумму весов запроса → confidence 0..1.
"""
import json
import math
from pathlib import Path

from normalize import tokenize, years, trigrams

FIELD_WEIGHTS = (("title", 3.0), ("manufacturer", 2.0), ("grapes", 1.0), ("extra", 0.5))


def levenshtein_le(a: str, b: str, maxd: int) -> bool:
    if abs(len(a) - len(b)) > maxd:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        best = i
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
            best = min(best, cur[-1])
        if best > maxd:
            return False
        prev = cur
    return prev[-1] <= maxd


def tokens_match(q: str, d: str) -> float:
    """1.0 точное, 0.85 префикс, 0.7 фаззи, 0 нет."""
    if q == d:
        return 1.0
    if len(q) >= 4 and (d.startswith(q) or q.startswith(d)) and abs(len(q) - len(d)) <= 3:
        return 0.85
    n = max(len(q), len(d))
    if n >= 8 and levenshtein_le(q, d, 2):
        return 0.7
    if n >= 5 and levenshtein_le(q, d, 1):
        return 0.7
    return 0.0


class Matcher:
    def __init__(self, wines: list[dict]):
        self.wines = wines
        self.docs = []
        df: dict[str, int] = {}
        for w in wines:
            fields = {
                "title": tokenize(w.get("title") or "", drop_stop=True),
                "manufacturer": tokenize(w.get("manufacturer") or "", drop_stop=True),
                "grapes": tokenize(" ".join(w.get("grapes") or []), drop_stop=True),
                "extra": tokenize(f"{w.get('region') or ''} {w.get('category') or ''}"),
            }
            doc_tokens = {}
            for field, weight in FIELD_WEIGHTS:
                for tok in fields[field]:
                    doc_tokens[tok] = max(doc_tokens.get(tok, 0.0), weight)
            self.docs.append({
                "tokens": doc_tokens,
                "years": years(w.get("title") or "") | years(w.get("slug") or ""),
            })
            for tok in doc_tokens:
                df[tok] = df.get(tok, 0) + 1
        n = max(len(wines), 1)
        self.idf = {tok: 0.5 + math.log(n / c) / math.log(n) for tok, c in df.items()}

    def search(self, ocr_text: str, top: int = 5) -> list[dict]:
        q_tokens = [t for t in set(tokenize(ocr_text, drop_stop=True)) if not t.isdigit()]
        q_years = years(ocr_text)
        if not q_tokens:
            return []
        results = []
        for widx, doc in enumerate(self.docs):
            score = 0.0
            denom = 0.0
            for qt in q_tokens:
                q_idf = self.idf.get(qt, 1.0)
                denom += q_idf
                best = 0.0
                for dt, w in doc["tokens"].items():
                    m = tokens_match(qt, dt)
                    if m:
                        best = max(best, m * w * q_idf)
                score += best
            if denom == 0:
                continue
            conf = score / (denom * 3.0)  # 3.0 — максимальный вес поля
            if q_years and doc["years"] & q_years:
                conf = min(1.0, conf + 0.1)
            if conf > 0.02:
                results.append({"index": widx, "confidence": round(conf, 4)})
        results.sort(key=lambda r: (-r["confidence"],
                                    -(self.wines[r["index"]].get("rating") or 0)))
        out = []
        for r in results[:top]:
            w = self.wines[r["index"]]
            out.append({"slug": w["slug"], "title": w["title"],
                        "manufacturer": w.get("manufacturer"), "confidence": r["confidence"]})
        return out


def load_matcher(datapack_json: Path) -> Matcher:
    data = json.loads(datapack_json.read_text())
    return Matcher(data["wines"])
