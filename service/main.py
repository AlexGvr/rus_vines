#!/usr/bin/env python3
"""Сервис распознавания вина по фотографии этикетки.

Эндпоинты:
  POST /v1/eval/predict  — контракт скрипта оценки кейсодержателя:
                           multipart-поле image, ответ {"slug": "..."}
  POST /v1/search        — полный ответ для фронтенда: карточка, уверенность
                           top-1 и top-5, запасные варианты
  GET  /health           — готовность модели и индекса

Поиск двухступенчатый:
  1) детектор вырезает бутылку, SigLIP отбирает двадцатку кандидатов —
     дёшево, но не различает позиции одной серии;
  2) локальные признаки считают, сколько точек этикетки совпало с каждым
     кандидатом, и переупорядочивают выдачу.

Вторая ступень поднимает top-1 с 71.5% до 85.2% на общем наборе и с 62.1%
до 94.3% на почти-дублях (data/index/rerank_*.json). Она же даёт ответ
«такого вина нет»: мало совпавших точек — значит похожего в каталоге нет.

Модель, детектор и индекс поднимаются один раз при старте: холодный прогон
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
from fastapi import Body, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageFile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from embed import embed_images, load as load_model  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import load as load_ocr, read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel, resolve_close  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recommend import QUESTIONS, Recommender  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Позиция описывается двумя векторами, и запрос идёт теми же двумя видами:
# кадром целиком и кропом по найденной бутылке. Одного кропа мало — когда
# детектор берёт соседнюю бутылку, вектор описывает чужую этикетку, и нужная
# позиция улетает ниже сотого места. Второй вид даёт ей независимый шанс:
# recall@20 91.2% → 96.8% при нуле потерянных (data/index/multiview_sharp.json).
# Остальные шаги конвейера на замере не оправдались и отключены.
QUERY_VIEWS: tuple[tuple[str, ...], ...] = ((), ("detect",))
GEOMETRY_VIEW = ("detect",)     # SIFT работает по плотному кропу, не по кадру
INDEX_VARIANT = "clean_mv"
CATALOG_PATH = ROOT / "data" / "catalog" / "catalog.json"
INDEX_DIR = ROOT / "data" / "index"

RERANK_TOPK = 20         # потолок точности = recall@20, дальше растёт только время

# Два порога вместо одного (калибровка в data/index/rerank_sharp.json).
# Абсолютный: меньше MIN_INLIERS совпавших точек — похожего в каталоге нет,
# такое правило отсекает 46% неверных ответов, теряя 0.9% верных.
# Относительный: доминирование лидера над вторым кандидатом. Порог 0.65
# оставляет всего 6.8% неверных, но и 58% верных — слишком строго, чтобы
# по нему прятать карточку, зато в самый раз, чтобы отличить уверенный
# ответ от спорного.
MIN_INLIERS = 15
CONFIDENT_DOMINANCE = 0.65

# Разрешение противоречий текстом. Запускается только когда геометрия не
# развела кандидатов: инлаеров у соседа не меньше CLOSE_WINDOW от лидера.
# Правило одностороннее — уверенно прочитанное слово, принадлежащее ровно
# одному кандидату, опускает остальных; неуверенное чтение не делает ничего.
# На общем наборе это +0.8 п.п. top-1 (4 исправления против 1 поломки),
# OCR при этом считается для 15% запросов (data/index/tiebreak_sharp.json).
CLOSE_WINDOW = 0.80
OCR_MIN_CONF = 0.60

state: dict = {}


def build_state() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["wines"]
    state["by_slug"] = {w["slug"]: w for w in catalog}
    state["vectors"] = np.load(INDEX_DIR / f"{INDEX_VARIANT}.npy")
    with (INDEX_DIR / f"{INDEX_VARIANT}.csv").open(encoding="utf-8") as fh:
        state["slugs"] = [row["slug"] or None for row in csv.DictReader(fh)]
    state["reranker"] = PrecomputedReranker(INDEX_DIR / "sift")
    state["recommender"] = Recommender(catalog)
    state["text"] = TextChannel(catalog)
    load_model()
    load_ocr()
    # Прогрев: первый прогон модели и детектора инициализирует ядра CUDA
    # и стоит секунды — на запросе такой задержки быть не должно.
    warm = [Image.new("RGB", (384, 384), "white")]
    for steps in QUERY_VIEWS:
        embed_images(normalize_batch(warm, steps=steps))
    read_words(warm[0])


@asynccontextmanager
async def lifespan(_: FastAPI):
    build_state()
    yield
    state.clear()


app = FastAPI(title="Сканер вин «Своё Вино»", lifespan=lifespan)

# Фронтенд поднимается своим dev-сервером, поэтому обращается с другого порта.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def search_candidates(image: Image.Image, top: int = 5) -> tuple[list[dict], dict]:
    """Двухступенчатый поиск. Возвращает кандидатов и тайминги по ступеням."""
    t0 = time.perf_counter()
    views = {steps: normalize_batch([image], steps=steps)[0] for steps in QUERY_VIEWS}
    queries = embed_images(list(views.values()))
    # Строка индекса получает лучший косинус по видам запроса: видам не нужно
    # совпасть всем сразу, достаточно одной пары «вид запроса — вид эталона».
    scores = (state["vectors"] @ queries.T).max(axis=1)
    prepared = views[GEOMETRY_VIEW]
    order = np.argsort(-scores)[:200]

    shortlist, seen = [], set()
    for j in order:
        slug = state["slugs"][j]
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        shortlist.append({"slug": slug, "cv_score": float(scores[j])})
        if len(shortlist) == RERANK_TOPK:
            break
    t_cv = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    inliers = state["reranker"].scores(prepared, [c["slug"] for c in shortlist])
    for candidate in shortlist:
        candidate["inliers"] = int(inliers.get(candidate["slug"], 0))
    shortlist.sort(key=lambda c: (-c["inliers"], -c["cv_score"]))
    t_rerank = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    reasons = resolve_ties(shortlist, prepared)
    t_text = (time.perf_counter() - t2) * 1000

    return shortlist[:top], {"cv_ms": round(t_cv, 1), "rerank_ms": round(t_rerank, 1),
                             "text_ms": round(t_text, 1), "resolved": bool(reasons)}


def resolve_ties(shortlist: list[dict], prepared: Image.Image) -> dict[str, list[str]]:
    """Переставляет близких кандидатов по прочитанному тексту, на месте.

    Считать OCR на каждом запросе незачем: когда у лидера инлаеров на порядок
    больше, текст ничего не решит, а время ответа вырастет на сотню миллисекунд.
    """
    best = shortlist[0]["inliers"] if shortlist else 0
    if best <= 0 or sum(1 for c in shortlist if c["inliers"] >= best * CLOSE_WINDOW) < 2:
        return {}
    pairs = [(c["slug"], float(c["inliers"])) for c in shortlist]
    ranked, reasons = resolve_close(pairs, read_words(prepared), state["text"],
                                    window=CLOSE_WINDOW, min_conf=OCR_MIN_CONF)
    if not any(reasons.values()):
        return {}
    position = {slug: i for i, (slug, _) in enumerate(ranked)}
    shortlist.sort(key=lambda c: position.get(c["slug"], len(position)))
    for candidate in shortlist:
        if reasons.get(candidate["slug"]):
            candidate["contradictions"] = reasons[candidate["slug"]]
    return reasons


def confidence(candidates: list[dict]) -> dict:
    """Уверенность по инлаерам: величина абсолютная, в отличие от косинуса.

    top1 — насколько лидер оторвался от второго кандидата; top5 — доля
    инлаеров, собранная всей пятёркой, относительно шума.
    """
    if not candidates:
        return {"top1": 0.0, "top5": 0.0, "inliers": 0, "gap": 0}
    best = candidates[0]["inliers"]
    second = candidates[1]["inliers"] if len(candidates) > 1 else 0
    total = sum(c["inliers"] for c in candidates) or 1
    return {
        "top1": round(best / (best + second), 4) if best + second else 0.0,
        "top5": round(min(total / (total + MIN_INLIERS), 1.0), 4),
        "inliers": best,
        "gap": best - second,
    }


async def read_image(upload: UploadFile) -> Image.Image | None:
    try:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def card(candidate: dict) -> dict:
    wine = dict(state["by_slug"].get(candidate["slug"], {}))
    wine["inliers"] = candidate["inliers"]
    wine["cv_score"] = round(candidate["cv_score"], 4)
    return wine


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "vectors" in state else "loading",
        "index": INDEX_VARIANT,
        "vectors": int(state["vectors"].shape[0]) if "vectors" in state else 0,
        "catalog": len(state.get("by_slug", {})),
        "rerank_topk": RERANK_TOPK,
    }


@app.get("/v1/wines/{slug}")
def wine_card(slug: str) -> JSONResponse:
    wine = state["by_slug"].get(slug)
    if wine is None:
        return JSONResponse({"error": "позиция не найдена"}, status_code=404)
    return JSONResponse({
        "wine": wine,
        "pairings": state["recommender"].pairings(wine),
        "similar": state["recommender"].similar(slug),
    })


@app.get("/v1/sommelier/questions")
def sommelier_questions() -> JSONResponse:
    return JSONResponse({"questions": QUESTIONS})


@app.post("/v1/sommelier")
def sommelier(answers: dict = Body(default_factory=dict)) -> JSONResponse:
    """Подбор под ответы на наводящие вопросы.

    Если под все условия сразу ничего не нашлось, мягко отпускаем блюдо:
    пустой ответ на экране хуже, чем близкая по вкусу позиция.
    """
    picks = state["recommender"].sommelier(answers)
    relaxed = False
    if not picks and answers.get("dish"):
        picks = state["recommender"].sommelier({**answers, "dish": ""})
        relaxed = bool(picks)
    return JSONResponse({"recommendations": picks, "relaxed": relaxed})


@app.get("/v1/photo/{slug}")
def wine_photo(slug: str):
    """Фото эталона из дампа каталога — фронтенду нужен один адрес на позицию."""
    wine = state["by_slug"].get(slug) or {}
    photo = wine.get("photo")
    if not photo:
        return JSONResponse({"error": "нет фото"}, status_code=404)
    path = ROOT / photo
    if not path.exists():
        return JSONResponse({"error": "файл отсутствует"}, status_code=404)
    return FileResponse(path, media_type="image/webp")


@app.post("/v1/eval/predict")
async def eval_predict(image: UploadFile = File(...)) -> JSONResponse:
    """Контракт скрипта оценки: ровно один slug плоским объектом.

    Порог отсечки здесь не применяется: скрипт кейсодержателя ждёт лучший
    ответ, а пустой slug засчитывается как промах в любом случае.
    """
    img = await read_image(image)
    if img is None:
        return JSONResponse({"slug": None})
    candidates, _ = search_candidates(img, top=1)
    return JSONResponse({"slug": candidates[0]["slug"] if candidates else None})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)) -> JSONResponse:
    img = await read_image(image)
    if img is None:
        return JSONResponse({"error": "не удалось прочитать изображение"}, status_code=400)

    candidates, timings = search_candidates(img, top=5)
    conf = confidence(candidates)
    found = bool(candidates) and candidates[0]["inliers"] >= MIN_INLIERS

    # Три состояния вместо «нашли или нет». Бинарное решение по одному порогу
    # либо прячет треть верных ответов, либо пропускает половину неверных —
    # середина честнее: уверенный ответ показываем одной карточкой, спорный
    # тоже показываем, но с признанием сомнения и ближайшими вариантами.
    if not found:
        status = "not_found"
    elif conf["top1"] >= CONFIDENT_DOMINANCE:
        status = "confident"
    else:
        status = "uncertain"

    return JSONResponse({
        "status": status,
        "match": card(candidates[0]) if found else None,
        "alternatives": [card(c) for c in candidates[1:]] if status == "confident"
                        else [card(c) for c in candidates],
        "confidence": conf,
        "latency_ms": {**timings,
                       "total": round(timings["cv_ms"] + timings["rerank_ms"]
                                      + timings["text_ms"], 1)},
    })
