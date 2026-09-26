"""Чтение этикетки VLM для вина, которого сервис не нашёл в каталоге.

Когда поиск отказался от ответа, пользователю нужна не выдача «похожего
по виду» (для вина вне каталога это случайные бутылки), а аналоги: вина
того же стиля из других виноделен. Для этого нужно знать стиль, и его
читает та же Qwen3-VL, что подтверждает лидера (pipeline/vlm_verify.py):
модель уже в памяти, второй экземпляр не грузится.

Модель только читает этикетку; позиции выбирает каталог
(service/recommend.py). Полное чтение (read) вызывается отдельным запросом
после ответа поиска и на распознавание не влияет. Короткое (read_name) —
часть поиска, но только при LABEL_RESCUE=1 (service/main.py).
"""
from __future__ import annotations

import json
import re

from PIL import Image

import vlm_verify
from text_match import extract_attributes

SIDE = 1024
MAX_NEW_TOKENS = 200

PROMPT = (
    "Read the label of the central, fully visible wine bottle in the photo. "
    "Answer with one JSON object and nothing else:\n"
    '{"producer": producer as written on the label or null, '
    '"producer_ru": producer name in Cyrillic or null, '
    '"name": wine name or null, '
    '"style_text": the exact words on the label that state colour and sweetness, '
    'e.g. "красное полусладкое" or "Brut", or null, '
    '"color": "red" | "white" | "rose" | "orange" | null, '
    '"sweetness": "dry" | "semi-dry" | "semi-sweet" | "sweet" | "brut" | "extra brut" | null, '
    '"sparkling": true | false | null, '
    '"grapes": [grape varieties only, in Russian, e.g. "Каберне Совиньон", "Саперави"], '
    '"region": wine region in Russian or null, "country": country in Russian or null}\n'
    "Write only what the label or the bottle shows; use null when it is not visible. "
    "Do not guess."
)

COLORS = {
    "red": "красное", "white": "белое", "rose": "розовое", "rosé": "розовое",
    "orange": "оранжевое",
    "красное": "красное", "белое": "белое", "розовое": "розовое", "оранжевое": "оранжевое",
}
SWEETNESS = {
    "dry": "сухое", "semi-dry": "полусухое", "semi dry": "полусухое", "medium dry": "полусухое",
    "semi-sweet": "полусладкое", "semi sweet": "полусладкое", "medium sweet": "полусладкое",
    "sweet": "сладкое", "dessert": "сладкое",
    "brut": "брют", "extra brut": "экстра брют", "brut nature": "экстра брют",
    "сухое": "сухое", "полусухое": "полусухое", "полусладкое": "полусладкое",
    "сладкое": "сладкое", "брют": "брют", "экстра брют": "экстра брют",
}
SPARKLING = {"брют", "экстра брют"}


def text_or_none(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = re.sub(r"\s+", " ", value).strip()
    return value if value and value.lower() not in {"null", "none", "unknown", "n/a"} else None


def parse(raw: str) -> dict | None:
    """Ответ модели в словарь со значениями из словаря каталога.

    Модель иногда оборачивает JSON в ```json …``` или добавляет пояснение;
    берётся первый объект. Значения вне допустимых становятся None: лучше
    не знать сладость, чем подобрать аналоги к выдуманной.
    """
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    color = COLORS.get((text_or_none(data.get("color")) or "").lower())
    sweetness = SWEETNESS.get((text_or_none(data.get("sweetness")) or "").lower())
    grapes = data.get("grapes") if isinstance(data.get("grapes"), list) else []
    grapes = [g for g in (text_or_none(x) for x in grapes) if g]
    # Слова, переписанные с этикетки, надёжнее классификации моделью: на
    # «Алазанской долине» она написала sweetness «dry», а «Красное
    # полусладкое» с этикетки положила в сорта. Читает их тот же разбор,
    # что и OCR распознавания.
    # Цвет по сортам не читается: «Мускат Белый» бывает и в розовом.
    style = " ".join([text_or_none(data.get("style_text")) or "",
                      text_or_none(data.get("name")) or ""])
    color = extract_attributes(style).color or color
    sweetness = extract_attributes(" ".join([style, *grapes])).sweetness or sweetness
    grapes = [g for g in grapes if not extract_attributes(g).sweetness][:4]
    sparkling = data.get("sparkling")
    sparkling = sparkling if isinstance(sparkling, bool) else None
    if sweetness in SPARKLING:
        sparkling = True
    return {
        "producer": text_or_none(data.get("producer")),
        "producer_ru": text_or_none(data.get("producer_ru")),
        "name": text_or_none(data.get("name")),
        "color": color,
        "sweetness": sweetness,
        "sparkling": sparkling,
        "grapes": grapes,
        "region": text_or_none(data.get("region")),
        "country": text_or_none(data.get("country")),
    }


def generate(crop: Image.Image, prompt: str, side: int, tokens: int) -> str:
    verifier = vlm_verify.load()
    torch = verifier.torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": vlm_verify.shrink(crop, side)},
        {"type": "text", "text": prompt},
    ]}]
    inputs = verifier.processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = verifier.model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
    return verifier.processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def read(crop: Image.Image) -> dict | None:
    """Прочитать этикетку; None — модель ответила не по формату."""
    return parse(generate(crop, PROMPT, SIDE, MAX_NEW_TOKENS))


# Только винодельня и название — для спасения отказа в пути поиска, где
# важна каждая секунда. Генерация здесь упирается в число токенов, а не
# в размер кадра: 512 и 1024 px читаются одинаково, около 1.1 с
# (data/validation/label_rescue_fast_20260926.jsonl), полный промпт — 2.9 с.
NAME_PROMPT = (
    "Read the label of the central, fully visible wine bottle. Answer with one JSON "
    'object and nothing else: {"producer": producer as written or null, '
    '"producer_ru": producer in Cyrillic or null, "name": wine name or null}.'
)
NAME_SIDE = 768
NAME_TOKENS = 48


def read_name(crop: Image.Image) -> dict | None:
    return parse(generate(crop, NAME_PROMPT, NAME_SIDE, NAME_TOKENS))
