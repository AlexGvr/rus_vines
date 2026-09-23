"""Подтверждение лидера VLM: «на снимке тот же товар, что на эталоне?».

Пилот (`pipeline/experiment_vlm.py`, data/validation/vlm_pilot_20260923)
показал две разные роли. Перестановка пятёрки по VLM значимого выигрыша
не дала: внутри пятёрки Qwen3-VL-4B выбирает верный хуже конвейера.
А вот подтверждение лидера там, где уверенность конвейера ниже порога
показа, возвращает верные ответы, которые иначе ушли бы в отказ:
геометрия спорит с косинусом, и уверенность низкая, хотя ответ верен.

Поэтому модуль делает одно: P(yes) для лидера. Модель грузится лениво
при первом вызове; сервис вызывает её только при VLM_CONFIRM=1 и только
для ответов ниже порога показа.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
QUERY_SIDE = 1280
REF_SIDE = 768

PROMPT = (
    "Photo 1 is a customer photo; the target is the central, fully visible bottle. "
    "Photo 2 is the catalog reference photo of this product: {desc}.\n"
    "Is the target bottle in Photo 1 exactly this catalog product: the same producer, "
    "the same wine name or line, the same color and the same sweetness? Wines of one "
    "producer's line with a similar label but a different name, color or sweetness are "
    "different products. The label design may be an older or newer version of the same "
    "product. Answer only yes or no."
)

INDEX_CSV = ROOT / "data" / "index" / "clean_mv.csv"
MAX_REFS = 3

_state: dict = {}


def shrink(im: Image.Image, side: int) -> Image.Image:
    im = im.convert("RGB")
    scale = side / max(im.size)
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return im


def describe(slug: str, wine: dict, channel) -> str:
    attrs = channel.attributes_of.get(slug)
    parts = [f"«{wine.get('title') or slug}»"]
    if wine.get("manufacturer"):
        parts.append(f"producer {wine['manufacturer']}")
    if attrs is not None and attrs.color:
        parts.append(f"color {attrs.color}")
    if attrs is not None and attrs.sweetness:
        parts.append(f"sweetness {attrs.sweetness}")
    return ", ".join(parts)


class Verifier:
    def __init__(self, model_id: str = MODEL):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16).to("cuda").eval()
        tok = self.processor.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")})

    def p_yes(self, query: Image.Image, ref: Image.Image, desc: str) -> float:
        torch = self.torch
        messages = [{"role": "user", "content": [
            {"type": "image", "image": query},
            {"type": "image", "image": ref},
            {"type": "text", "text": PROMPT.format(desc=desc)},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        y = torch.logsumexp(logits[self.yes], 0)
        n = torch.logsumexp(logits[self.no], 0)
        return float(torch.sigmoid(y - n))


def _refs() -> dict[str, list[str]]:
    if "refs" not in _state:
        refs: dict[str, list[str]] = {}
        with INDEX_CSV.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["slug"] and row["view"] == "raw":
                    refs.setdefault(row["slug"], []).append(row["path"])
        _state["refs"] = refs
        _state["images"] = {}
    return _state["refs"]


def load():
    if "verifier" not in _state:
        _state["verifier"] = Verifier(MODEL)
    return _state["verifier"]


def p_yes(crop: Image.Image, slug: str, wine: dict, channel) -> float:
    """Наибольшее P(yes) по эталонам позиции (до трёх); 0 — эталона нет."""
    verifier = load()
    paths = _refs().get(slug, [])[:MAX_REFS]
    if not paths:
        return 0.0
    query = shrink(crop, QUERY_SIDE)
    desc = describe(slug, wine, channel)
    best = 0.0
    for path in paths:
        images = _state["images"]
        if path not in images:
            images[path] = shrink(Image.open(ROOT / path), REF_SIDE)
        best = max(best, verifier.p_yes(query, images[path], desc))
    return best
