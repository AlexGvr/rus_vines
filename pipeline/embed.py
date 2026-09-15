#!/usr/bin/env python3
"""Эмбеддинги изображений SigLIP 2 — общий слой для индекса и запросов.

Используется и при построении индекса каталога, и при обработке запроса
в сервисе, поэтому препроцессинг здесь один на оба пути.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageFile
from transformers import AutoModel, AutoProcessor

ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_ID = "google/siglip2-so400m-patch14-384"
_state: dict = {}


def load(model_id: str = MODEL_ID, device: str | None = None):
    """Ленивая загрузка: модель живёт одна на процесс."""
    if _state.get("id") == model_id:
        return _state["proc"], _state["model"], _state["device"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=dtype).vision_model.to(device).eval()
    _state.update(id=model_id, proc=proc, model=model, device=device, dtype=dtype)
    return proc, model, device


def embed_images(images: list[Image.Image], batch_size: int = 16,
                 model_id: str = MODEL_ID) -> np.ndarray:
    """L2-нормированные векторы, float32 [N, D]. Косинус = скалярное произведение."""
    proc, model, device = load(model_id)
    dtype = _state["dtype"]
    out = []
    with torch.inference_mode():
        for i in range(0, len(images), batch_size):
            chunk = [im.convert("RGB") for im in images[i:i + batch_size]]
            px = proc(images=chunk, return_tensors="pt")["pixel_values"].to(device, dtype)
            vec = model(pixel_values=px).pooler_output.float()
            out.append(torch.nn.functional.normalize(vec, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_paths(paths: list[str], batch_size: int = 16, progress_every: int = 500,
                model_id: str = MODEL_ID,
                steps: tuple[str, ...] | None = None) -> tuple[np.ndarray, list[str]]:
    """Эмбеддинги по путям. Битые файлы пропускаются, возвращаются только живые.

    steps — шаги нормализации (см. pipeline/normalize.py). Применяются к тем же
    пачкам, что уходят в модель: детекция работает батчем, это её дешёвый режим.
    """
    vecs, kept, buf, buf_paths = [], [], [], []
    normalize_batch = None
    if steps:
        from normalize import normalize_batch  # импорт здесь: YOLO нужен не всегда

    def flush():
        if buf:
            batch = normalize_batch(buf, steps=steps) if normalize_batch else buf
            vecs.append(embed_images(batch, batch_size, model_id))
            kept.extend(buf_paths)
            buf.clear()
            buf_paths.clear()

    for n, path in enumerate(paths, 1):
        try:
            with Image.open(path) as im:
                im.load()
                buf.append(im.convert("RGB"))
                buf_paths.append(path)
        except Exception:
            continue
        if len(buf) >= batch_size:
            flush()
        if progress_every and n % progress_every == 0:
            print(f"  {n}/{len(paths)}", flush=True)
    flush()
    if not vecs:
        return np.zeros((0, 1152), dtype=np.float32), []
    return np.concatenate(vecs), kept
