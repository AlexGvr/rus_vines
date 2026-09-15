#!/usr/bin/env python3
"""S0.4 — сервис распознавания вина по фотографии этикетки.

Эндпоинты:
  POST /v1/eval/predict  — контракт скрипта оценки кейсодержателя:
                           multipart-поле image, ответ {"slug": "..."}
  POST /v1/search        — полный ответ для фронтенда: карточка, уверенность
                           top-1 и top-5, запасные варианты
  GET  /health           — готовность модели и индекса

Индекс и модель поднимаются один раз при старте: холодный прогон модели
стоит секунды, а SLA на запрос — три.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from embed import embed_images, load as load_model  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

INDEX_VARIANT = "clean"
CATALOG_PATH = ROOT / "data" / "catalog" / "catalog.json"
INDEX_DIR = ROOT / "data" / "index"

# Порог показа единственной карточки. Значение предварительное: калибруется
# в спринте 2 по валидационному набору, сейчас служит заглушкой контракта.
SINGLE_CARD_THRESHOLD = 0.75

state: dict = {}


def build_state() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["wines"]
    state["by_slug"] = {w["slug"]: w for w in catalog}
    state["vectors"] = np.load(INDEX_DIR / f"{INDEX_VARIANT}.npy")
    with (INDEX_DIR / f"{INDEX_VARIANT}.csv").open(encoding="utf-8") as fh:
        state["slugs"] = [row["slug"] or None for row in csv.DictReader(fh)]
    load_model()
    # Прогрев: первый прогон модели инициализирует ядра CUDA и стоит секунды.
    embed_images([Image.new("RGB", (384, 384), "white")])


@asynccontextmanager
async def lifespan(_: FastAPI):
    build_state()
    yield
    state.clear()


app = FastAPI(title="Сканер вин «Своё Вино»", lifespan=lifespan)


def rank(image: Image.Image, top: int = 5) -> tuple[list[dict], float]:
    """Кандидаты по убыванию близости и время поиска в миллисекундах."""
    t0 = time.perf_counter()
    query = embed_images([image])[0]
    scores = state["vectors"] @ query
    order = np.argsort(-scores)[:60]

    ranked, seen = [], set()
    for j in order:
        slug = state["slugs"][j]
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        ranked.append({"slug": slug, "score": float(scores[j])})
        if len(ranked) == top:
            break
    return ranked, (time.perf_counter() - t0) * 1000


def confidence(ranked: list[dict]) -> dict:
    """Уверенность top-1 и отрыв от второго кандидата.

    Косинус SigLIP лежит в узком диапазоне, поэтому как «уверенность»
    отдаём softmax по кандидатам — он показывает не абсолютное сходство,
    а насколько первый оторвался от остальных. Калибровка — спринт 2.
    """
    if not ranked:
        return {"top1": 0.0, "top5": 0.0, "gap": 0.0}
    scores = np.array([c["score"] for c in ranked], dtype=np.float64)
    weights = np.exp((scores - scores.max()) / 0.01)
    probs = weights / weights.sum()
    return {
        "top1": round(float(probs[0]), 4),
        "top5": round(float(probs.sum()), 4),
        "gap": round(float(scores[0] - scores[1]), 4) if len(scores) > 1 else 0.0,
    }


async def read_image(upload: UploadFile) -> Image.Image | None:
    try:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "vectors" in state else "loading",
        "index": INDEX_VARIANT,
        "vectors": int(state["vectors"].shape[0]) if "vectors" in state else 0,
        "catalog": len(state.get("by_slug", {})),
    }


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> JSONResponse:
    """Контракт скрипта оценки: ровно один slug плоским объектом."""
    img = await read_image(image)
    if img is None:
        return JSONResponse({"slug": None}, status_code=200)
    ranked, _ = rank(img, top=1)
    return JSONResponse({"slug": ranked[0]["slug"] if ranked else None})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)) -> JSONResponse:
    img = await read_image(image)
    if img is None:
        return JSONResponse({"error": "не удалось прочитать изображение"}, status_code=400)

    ranked, search_ms = rank(img, top=5)
    conf = confidence(ranked)
    cards = [{**state["by_slug"].get(c["slug"], {}), "score": round(c["score"], 4)}
             for c in ranked]
    return JSONResponse({
        "match": cards[0] if cards and conf["top1"] >= SINGLE_CARD_THRESHOLD else None,
        "alternatives": cards if not cards or conf["top1"] < SINGLE_CARD_THRESHOLD else [],
        "confidence": conf,
        "latency_ms": round(search_ms, 1),
    })
