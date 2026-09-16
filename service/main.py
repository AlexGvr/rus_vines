#!/usr/bin/env python3
"""Сервис распознавания вина по фотографии этикетки.

Эндпоинты:
  POST /v1/eval/predict  — контракт скрипта оценки кейсодержателя:
                           multipart-поле image, ответ {"slug": "..."}
  POST /v1/search        — полный ответ для фронтенда: карточка, уверенность
                           top-1 и top-5, запасные варианты
  GET  /health           — готовность модели и индекса

Поиск трёхступенчатый:
  1) кадр идёт двумя видами — целиком и кропом по найденной бутылке; SigLIP
     отбирает двадцатку кандидатов, выдача сворачивается по slug;
  2) локальные признаки считают, сколько точек этикетки совпало с каждым
     кандидатом, и переупорядочивают выдачу;
  3) если геометрия не развела лидеров, этикетка читается: уверенно
     прочитанное слово, принадлежащее одному кандидату, опускает остальных.

Вклад ступеней на общем наборе (data/index/*_sharp.json): второй вид
поднимает top-1 с 87.8% до 90.2%, разрешение противоречий — до 91.0%.
Показывать ли карточку, решает совместная оценка уверенности, а не порог
по одному числу.

Модель, детектор и индекс поднимаются один раз при старте: холодный прогон
стоит секунды, а SLA на запрос — три.
"""
from __future__ import annotations

import csv
import io
import json
import os
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
import searchcore as core  # noqa: E402
from ocr import load as load_ocr, read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel  # noqa: E402

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

# Уверенность считается по пяти признакам сразу, а не по одному числу
# инлаеров. Веса — логистическая регрессия, обученная на половине запросов
# и проверенная на другой (pipeline/calibrate.py, data/index/calibration_*.json).
# Отрицательные примеры получены вычёркиванием правильной позиции из индекса:
# тот же снимок становится запросом к вину, которого в каталоге нет.
#
# Признаки считает pipeline/searchcore.py — тот же модуль, которым идёт
# калибровка. Разъехавшиеся определения уже стоили ошибки: сервис брал
# наибольший косинус в шортлисте вместо косинуса лидера и показывал
# ошибочную карточку Katharon как уверенную на 80% там, где по формуле
# обучения выходило 30%. Теперь считать порознь попросту негде.
#
# Признаки относительные: сравнивают лидера с остальным шортлистом, а не
# с абсолютной шкалой. Абсолютные не переносятся с синтетики на съёмку —
# у запроса, полученного из файла эталона, косинус около 0.9, а у снимка
# с полки 0.75.
#
# Пороги подтверждены на двух наборах сразу.
#
# Показ карточки, p >= 0.06. На синтетике F1 топ-1 держится на максимуме
# 93.0% в полосе 0.05-0.10, а топ-5 на 0.06 чуть выше (96.1%). На полевом
# наборе (data/index/field_report.json, реальные снимки из отзывов) на этом
# пороге проходят все три положительных кадра, а карточку для вина, которого
# в каталоге нет, сервис показывает в 6.7% случаев.
#
# Синтетика оценивает ложное принятие куда мрачнее — 62.5%, — но её
# отрицательные примеры сделаны вычёркиванием позиции из индекса, и соседи
# по серии остаются на месте. Реальный снимок чужого вина обычно не похож
# на каталог вовсе, и вся разница видна в распределении: у полевых
# отрицательных медиана уверенности 0.007, и только два кадра из тридцати
# поднимаются выше — оба это варианты упаковки, которых в каталоге нет
# (BRÛLÉ Cuvée Rosé Demi-Sec против Brut, ZB MY ROSE против MY SAUVIGNON).
#
# Порог «без оговорок» оставлен синтетическим, 0.70. Полевых положительных
# кадров три, их уверенность 0.07-0.32, и подбирать по ним порог значило бы
# подгонять решение под три снимка. Следствие названо честно: на реальной
# съёмке состояние «одна карточка без оговорок» почти не срабатывает.
CONF_WEIGHTS = {"dominance": 4.4068, "log_inliers": 0.5282, "cv_margin": 8.1869, "cv_lift": 14.4323, "coverage": 4.5929, "text_support": -0.4506, "bias": -9.5306}
# Пороги вынесены в окружение: они откалиброваны на синтетическом наборе,
# а он систематически проще съёмки — у реального кадра и совпавших точек
# меньше, и покрывают они этикетку хуже. Поднять или опустить порог после
# сбора полевых фотографий должно быть можно без пересборки образа.
CONFIDENT_P = float(os.environ.get("CONFIDENT_P", "0.70"))   # карточка без оговорок
SHOW_P = float(os.environ.get("SHOW_P", "0.06"))             # ниже — не показываем

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
    # ТЗ требует показывать метрику уверенности (F1) для топ-1 и топ-5.
    # Уверенность конкретного ответа сервис считает сам, а F1 — свойство
    # конвейера целиком, поэтому берётся из отчёта замеров и отдаётся вместе
    # с ответом: пользователю важно знать, чего стоит показанная карточка.
    report = INDEX_DIR / "report_sharp.json"
    state["quality"] = (json.loads(report.read_text(encoding="utf-8"))
                        if report.exists() else {})
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


def search_candidates(image: Image.Image,
                      top: int = 5) -> tuple[list[dict], dict, dict]:
    """Поиск целиком: кандидаты, уверенность и тайминги по ступеням.

    Все шаги берутся из pipeline/searchcore.py — того же модуля, которым
    считает калибровка. Иначе пороги подбираются для одного конвейера,
    а работает другой.
    """
    t0 = time.perf_counter()
    views = {steps: normalize_batch([image], steps=steps)[0] for steps in QUERY_VIEWS}
    queries = embed_images(list(views.values()))
    # Строка индекса получает лучший косинус по видам запроса: видам не нужно
    # совпасть всем сразу, достаточно одной пары «вид запроса — вид эталона».
    scores = (state["vectors"] @ queries.T).max(axis=1)
    prepared = views[GEOMETRY_VIEW]
    candidates = core.shortlist(scores, state["slugs"], RERANK_TOPK)
    t_cv = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    core.rerank(prepared, candidates, state["reranker"])
    t_rerank = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    core.resolve(candidates, prepared, state["text"], read_words,
                 window=CLOSE_WINDOW, min_conf=OCR_MIN_CONF)
    t_text = (time.perf_counter() - t2) * 1000

    # Уверенность считается по всему шортлисту, а не по видимой пятёрке:
    # признак «превышение над шумом» смотрит на его хвост, и на усечённом
    # списке значил бы другое, чем при калибровке.
    conf = confidence(candidates)
    resolved = any(c.contradictions for c in candidates)
    return ([c.as_dict() for c in candidates[:top]], conf,
            {"cv_ms": round(t_cv, 1), "rerank_ms": round(t_rerank, 1),
             "text_ms": round(t_text, 1), "resolved": resolved})


def confidence(candidates: list) -> dict:
    """Уверенность в лидере и вероятность, что нужное вино есть в пятёрке.

    ТЗ требует показывать метрику уверенности и для топ-1, и для топ-5:
    от неё зависит, нужен ли пользователю экран вариантов вообще.
    """
    feats = core.features(candidates)
    leader = candidates[0] if candidates else None
    rival = max((c.inliers for c in candidates[1:]), default=0) if candidates else 0
    return {
        "probability": round(core.probability(feats, CONF_WEIGHTS), 4),
        "probability_top5": round(core.top5_probability(candidates, CONF_WEIGHTS), 4),
        "dominance": round(feats["dominance"], 4),
        "inliers": leader.inliers if leader else 0,
        "gap": (leader.inliers - rival) if leader else 0,
        "cv_margin": round(feats["cv_margin"], 4),
    }


async def read_image(upload: UploadFile) -> Image.Image | None:
    try:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None


def card(candidate: dict) -> dict:
    wine = dict(state["by_slug"].get(candidate["slug"], {}))
    wine.update({k: v for k, v in candidate.items() if k != "slug"})
    wine["slug"] = candidate["slug"]
    return wine


def quality_block() -> dict:
    """Измеренное качество конвейера: F1 топ-1 и топ-5 на валидационном наборе."""
    measured = (state.get("quality") or {}).get("f1", {}).get("по каталогу", {})
    if not measured:
        return {}
    return {
        "f1_top1": round(measured["top1"]["f1"], 4),
        "f1_top5": round(measured["top5"]["f1"], 4),
        "subset": (state.get("quality") or {}).get("subset"),
        "n": (state.get("quality") or {}).get("answerable"),
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "vectors" in state else "loading",
        "index": INDEX_VARIANT,
        "vectors": int(state["vectors"].shape[0]) if "vectors" in state else 0,
        "catalog": len(state.get("by_slug", {})),
        "quality": quality_block(),
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
    candidates, _, _ = search_candidates(img, top=1)
    return JSONResponse({"slug": candidates[0]["slug"] if candidates else None})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)) -> JSONResponse:
    img = await read_image(image)
    if img is None:
        return JSONResponse({"error": "не удалось прочитать изображение"}, status_code=400)

    candidates, conf, timings = search_candidates(img, top=5)
    found = bool(candidates) and conf["probability"] >= SHOW_P

    # Три состояния вместо «нашли или нет». Бинарное решение по одному порогу
    # либо прячет треть верных ответов, либо пропускает половину неверных —
    # середина честнее: уверенный ответ показываем одной карточкой, спорный
    # тоже показываем, но с признанием сомнения и ближайшими вариантами.
    if not found:
        status = "unsure"
    elif conf["probability"] >= CONFIDENT_P:
        status = "confident"
    else:
        status = "uncertain"

    return JSONResponse({
        "status": status,
        "match": card(candidates[0]) if found else None,
        # Когда карточка показана, лидер уже в match, и дублировать его
        # в вариантах незачем. Когда не показана — наоборот, он должен войти
        # в список похожего по виду, иначе лучший кандидат просто пропадает.
        "alternatives": [card(c) for c in (candidates if not found
                                           else candidates[1:])],
        "confidence": conf,
        "quality": quality_block(),
        "latency_ms": {**timings,
                       "total": round(timings["cv_ms"] + timings["rerank_ms"]
                                      + timings["text_ms"], 1)},
    })
