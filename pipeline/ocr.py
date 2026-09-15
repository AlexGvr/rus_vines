#!/usr/bin/env python3
"""S2.1 — распознавание текста этикетки.

Tesseract из прототипа на полевых фото выдавал шум, поэтому канал переведён
на EasyOCR с русской и английской моделями: этикетки российских вин почти
всегда кириллические, а бренд часто продублирован латиницей.

OCR работает по кропу бутылки от детектора: так текст занимает большую часть
кадра, распознаётся точнее и требует меньше видеопамяти.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

MAX_SIDE = 1280      # больше — не точнее, но заметно дороже по памяти
MIN_CONF = 0.30      # ниже этого порога EasyOCR обычно выдаёт мусорные строки
_state: dict = {}


def load(languages: tuple[str, ...] = ("ru", "en"), gpu: bool = True):
    if "reader" not in _state:
        import easyocr
        _state["reader"] = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
    return _state["reader"]


def _prepare(image: Image.Image) -> np.ndarray:
    im = image.convert("RGB")
    if max(im.size) > MAX_SIDE:
        im = im.copy()
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return np.array(im)


def read_text(image: Image.Image, min_conf: float = MIN_CONF) -> str:
    """Весь распознанный текст одной строкой, порядок — как вернул детектор строк."""
    reader = load()
    try:
        blocks = reader.readtext(_prepare(image), detail=1, paragraph=False)
    except Exception:
        return ""
    return " ".join(text for _, text, conf in blocks if conf >= min_conf)


def read_batch(images: list[Image.Image], min_conf: float = MIN_CONF) -> list[str]:
    """EasyOCR не батчится по изображениям — идём последовательно, модель одна."""
    load()
    return [read_text(im, min_conf) for im in images]
