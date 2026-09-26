#!/usr/bin/env python3
"""Сервис распознавания вина по фотографии этикетки.

Эндпоинты:
  POST /v1/eval/predict  — контракт скрипта оценки кейсодержателя:
                           multipart-поле image, ответ {"slug": "..."}
  POST /v1/search        — полный ответ для фронтенда: карточка, уверенность
                           top-1 и top-5, запасные варианты
  POST /v1/analogs       — вино не найдено: стиль с этикетки и аналоги
                           из каталога (вызывается после /v1/search)
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
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
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
from text_match import TextChannel, confident_attributes  # noqa: E402
from vector_store import open_store  # noqa: E402

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
# Веса и пороги лежат одним файлом — его пишет pipeline/calibrate.py.
# Раньше веса были вписаны сюда числами, а пороги жили в окружении, и отчёт
# мог считаться с одними, а сервис отвечать с другими. Теперь источник один:
# и сервис, и замеры читают data/index/confidence.json, а pipeline/stamp.py
# печатает его отпечаток рядом с числами.
#
# Переменные окружения оставлены, чтобы подвинуть порог без пересборки
# образа, но по умолчанию действует то, что записано в артефакте.
CONFIDENCE = json.loads((INDEX_DIR / "confidence.json").read_text(encoding="utf-8"))
OUTCOME_WEIGHTS = CONFIDENCE["outcome_weights"]
CONFIDENT_P = float(os.environ.get("CONFIDENT_P", CONFIDENCE["thresholds"]["confident_p"]))
SHOW_P = float(os.environ.get("SHOW_P", CONFIDENCE["thresholds"]["show_p"]))

# Отказ в оценочном эндпоинте. Как закрытая таблица оценивает вино вне
# каталога, организатор не сообщил, а в его публичном наборе таких кадров
# треть. Если для них верен пустой slug, отказ ниже порога показа вместе
# с подтверждением VLM (ниже) поднимает счёт на всех 93 кадрах организатора
# с 56 до 81; если пустой ответ не засчитывается никогда — 56 -> 55
# (pipeline/eval_scoring.py, эталоны только из датасета). Правило
# организатор не назвал, а закрытая проверка
# идёт после стоп-кода, поэтому включено по умолчанию; EVAL_ABSTAIN=0
# возвращает прежнее поведение «всегда лучший кандидат».
EVAL_ABSTAIN = os.environ.get("EVAL_ABSTAIN", "1") == "1"
EVAL_ABSTAIN_P = float(os.environ.get("EVAL_ABSTAIN_P", SHOW_P))

# Подтверждение лидера VLM (pipeline/vlm_verify.py). Спрашивается только
# ниже порога показа: там верный ответ часто есть, но геометрия спорит
# с косинусом. P(yes) >= VLM_CONFIRM_P оставляет ответ показанным (и не
# даёт отказа в оценочном эндпоинте). Порог 0.8 выбран по настроечной части
# кадров организатора (на 0.5-0.8 одинаково); на всех 93 кадрах по правилу
# «null верен для вина вне каталога» отказ без VLM даёт 79, с VLM — 81
# (data/validation/vlm_pilot_dsonly_20260923).
# Ответ VLM не меняет — только решение показывать ли его. Стоит ~0.35 с на
# вызов и ~9 ГБ видеопамяти. Включено по умолчанию; без видеокарты или если
# модель не загрузилась, сервис работает без подтверждения и пишет об этом
# в лог, а /health показывает фактический режим (settings.vlm_active).
VLM_CONFIRM = os.environ.get("VLM_CONFIRM", "1") == "1"
VLM_CONFIRM_P = float(os.environ.get("VLM_CONFIRM_P", "0.8"))

# VLM как арбитр, а не только подтверждение лидера. Правила текста и цвета,
# переставлявшие лидера сами, ломали соседние кадры: внутри линейки с общей
# этикеткой любое слабое свидетельство чаще ошибается, чем помогает. Здесь
# свидетельство только указывает, где ответ сомнителен, а решает VLM.
#   VLM_FALLBACK=k — лидер не подтверждён, ответ ушёл бы в отказ: VLM
#     проверяет ещё k следующих кандидатов, первый подтверждённый становится
#     ответом (Victor Dravigny Брют: Экстра Брют отвергнут, Брют третий).
#   VLM_CONTRA=1 — ответ показан, но уверенно прочитано слово чужой карточки,
#     которого у лидера нет (text_contradiction): если VLM отвергает лидера
#     (P(yes) < VLM_REJECT_P), проверяются владельцы слова.
VLM_FALLBACK = int(os.environ.get("VLM_FALLBACK", "2"))
VLM_CONTRA = os.environ.get("VLM_CONTRA", "1") == "1"
VLM_REJECT_P = 0.5
# Порог подтверждения в пути противоречия ниже общего: к нему приходят
# с двумя независимыми сигналами — OCR уверенно прочитал слово владельца,
# VLM отвергла лидера. На 402 кадрах у владельцев слова P(yes) либо ниже
# 0.2, либо 0.99, в полосе 0.5-0.8 нет ни одного случая, так что порог
# выбран по разобранному снимку («Алушта», 0.75) и замеру не противоречит.
VLM_CONTRA_P = float(os.environ.get("VLM_CONTRA_P", "0.7"))
# Второй проход при отказе (VLM_GEO_FALLBACK): VLM отвергла лидера
# (P(yes) < VLM_REJECT_P), никто из проверенных не дотянул до 0.8, но
# у кого-то P(yes) > VLM_REJECT_P и геометрия за него
# (searchcore.geometry_backs). Новых вызовов VLM нет — берутся уже
# посчитанные P(yes). Из прошедших — с наибольшим P(yes).
# Замер через HTTP на 402 кадрах: изменился один кадр (отказ -> верный
# ответ), потерь и новых показов вин вне каталога нет, время то же.
VLM_GEO_FALLBACK = os.environ.get("VLM_GEO_FALLBACK", "1") == "1"
# Проверка лидера в полосе «скорее всего» (VLM_BAND_VETO): уверенность
# между SHOW_P и CONFIDENT_P, ответ показан бы без VLM. Если VLM явно
# отвергает лидера (P(yes) < VLM_VETO_P), кадр идёт тем же путём, что
# отказ: проверка следующих, второй проход, спасение по этикетке; никто не
# подтверждён — отказ. Вина вне каталога с оформлением линейки из каталога
# (Дивноморское «Вторая линия» против «Холодного Тумана») показывались в
# этой полосе без проверки. Через HTTP на 402 кадрах: снимки организатора
# 85 -> 88/97 (+3/-0, три вина вне каталога), при строгом подсчёте 56 без
# изменений; ответов вину вне каталога 17 -> 9. На остальных кадрах +7/-3,
# из трёх потерь две — ошибки разметки отзывов (на этикетке Шираз, а не
# Мерло Black Out; Траминер «Планерское», а не «Высокий Берег»). Цена —
# вызов VLM на каждом кадре полосы: медиана на снимках организатора
# 927 -> 1295 мс, дольше 3 с — 28 -> 31 из 97. VLM_BAND_VETO=0 выключает.
VLM_BAND_VETO = os.environ.get("VLM_BAND_VETO", "1") == "1"
VLM_VETO_P = float(os.environ.get("VLM_VETO_P", "0.05"))
# Бюджет времени на дополнительные вызовы VLM (VLM_BUDGET_MS, 0 — без
# ограничения, по умолчанию). На снимке 3024×4032 поиск без них идёт до 2 с,
# а каждая проверка кандидата — ещё 0.3-0.7 с (до трёх эталонов). Замер на
# 402 кадрах: без бюджета +5 верных, p95 2.2 с, 7 запросов дольше 3 с
# (5 из 97 снимков организатора); с бюджетом 2000 мс за SLA не выходит
# ничего, но теряется «Цитрон» org076 — +4. Выбрано качество.
VLM_BUDGET_MS = float(os.environ.get("VLM_BUDGET_MS", "0"))
VLM_CALL_GUESS_MS = 400.0

# Спасение отказа по этикетке (LABEL_RESCUE). Поиск не нашёл вино, VLM не
# подтвердила никого из шортлиста, но с этикетки читаются винодельня и
# название (vlm_label.read_name), и по ним в каталоге находится позиция
# (Recommender.by_label_name: среди вин этой винодельни, иначе во всём
# каталоге; сходство названия >= LABEL_RESCUE_CLOSE). Ответом она становится,
# только если её подтвердил тот же верификатор с тем же порогом
# VLM_CONFIRM_P: без этой проверки правило отвечало бы на половину вин вне
# каталога (15 из 30 кадров организатора). Пилот —
# pipeline/experiment_label_rescue.py, замер через HTTP — README и
# docs/metrics.md: на 97 снимках организатора 82 -> 85 (53 -> 56 при строгом
# подсчёте), ложных ответов нет. Цена — около 1.2 с чтения и 0.4 с проверки
# на каждом отказе: дольше 3 с отвечают 28 снимков из 97 вместо 5. Выбрано
# качество, как и для VLM_BUDGET_MS; LABEL_RESCUE=0 выключает, а
# LABEL_RESCUE_BUDGET_MS > 0 не начинает спасение, если ответ из него не
# уложится в бюджет.
LABEL_RESCUE = os.environ.get("LABEL_RESCUE", "1") == "1"
LABEL_RESCUE_CLOSE = 0.6
LABEL_RESCUE_BUDGET_MS = float(os.environ.get("LABEL_RESCUE_BUDGET_MS", "0"))
LABEL_READ_GUESS_MS = 1300.0

# Разрешение противоречий текстом. Запускается только когда геометрия не
# развела кандидатов: инлаеров у соседа не меньше CLOSE_WINDOW от лидера.
# Правило одностороннее — уверенно прочитанное слово, принадлежащее ровно
# одному кандидату, опускает остальных; неуверенное чтение не делает ничего.
# На общем наборе это +0.8 п.п. top-1 (4 исправления против 1 поломки),
# OCR при этом считается для 15% запросов (data/index/tiebreak_sharp.json).
CLOSE_WINDOW = 0.80
OCR_MIN_CONF = 0.60
# Ниже этой уверенности незачем собирать улики и подтверждения: они меняют
# только оформление выдачи, а такой ответ всё равно не будет показан.
# Сам выбор вина порогом не управляется — он завершается в core.settle.
CONFLICT_GATE = 0.05

state: dict = {}

# Видеокарта одна, и VLM держит состояние между вызовами. Чтение этикетки
# (/v1/analogs) идёт в пуле потоков и не должно пересечься с поиском;
# скрипт оценки аналоги не вызывает, так что замок для него всегда свободен.
GPU_LOCK = threading.Lock()


def build_state() -> None:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["wines"]
    state["by_slug"] = {w["slug"]: w for w in catalog}
    # numpy в памяти по умолчанию; VECTOR_BACKEND=pgvector — PostgreSQL (vector_store.py).
    state["vectors"] = open_store(INDEX_VARIANT)
    with (INDEX_DIR / f"{INDEX_VARIANT}.csv").open(encoding="utf-8") as fh:
        state["slugs"] = core.scope_index([row["slug"] or None for row in csv.DictReader(fh)])
    # Строки индекса по slug: кандидату, найденному по этикетке вне
    # шортлиста, нужен его косинус.
    state["rows_of"] = {}
    for j, slug in enumerate(state["slugs"]):
        if slug:
            state["rows_of"].setdefault(slug, []).append(j)
    state["reranker"] = PrecomputedReranker(INDEX_DIR / "sift")
    # ТЗ требует показывать метрику уверенности (F1) для топ-1 и топ-5.
    # Уверенность конкретного ответа сервис считает сам, а F1 — свойство
    # конвейера целиком, поэтому берётся из отчёта замеров и отдаётся вместе
    # с ответом: пользователю важно знать, чего стоит показанная карточка.
    report = INDEX_DIR / "report_sharp.json"
    state["quality"] = (json.loads(report.read_text(encoding="utf-8"))
                        if report.exists() else {})
    state["text"] = TextChannel(catalog)
    # Сладость для аналогов и сомелье — та же, что знает распознавание.
    state["recommender"] = Recommender(
        catalog, {slug: attrs.sweetness for slug, attrs in state["text"].attributes_of.items()})
    load_model()
    load_ocr()
    state["vlm"] = start_vlm() if VLM_CONFIRM else False
    # Прогрев: первый прогон модели и детектора инициализирует ядра CUDA
    # и стоит секунды — на запросе такой задержки быть не должно.
    warm = [Image.new("RGB", (384, 384), "white")]
    for steps in QUERY_VIEWS:
        embed_images(normalize_batch(warm, steps=steps))
    read_words(warm[0])


def start_vlm() -> bool:
    """Загрузить и прогреть VLM; False — работать без подтверждения.

    На процессоре 4B-модель отвечает секунды на вызов и съедает запас SLA,
    поэтому без CUDA подтверждение не включается. Прогрев нужен по той же
    причине, что у SigLIP: первый вызов инициализирует ядра, и платить за
    это должен старт, а не первый запрос скрипта оценки.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("VLM_CONFIRM: CUDA недоступна — подтверждение VLM выключено", file=sys.stderr)
            return False
        import vlm_verify
        verifier = vlm_verify.load()
        blank = Image.new("RGB", (256, 512), "white")
        verifier.p_yes(blank, blank, "warm-up")
        return True
    except Exception as exc:  # noqa: BLE001 — без VLM сервис остаётся рабочим
        print(f"VLM_CONFIRM: модель не загрузилась ({exc}) — подтверждение VLM выключено",
              file=sys.stderr)
        return False


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
    similarity = state["vectors"].similarity(queries)
    scores = similarity.max(axis=1)
    prepared = list(views.values())[core.pick_view(list(views.values()), similarity)]
    candidates = core.shortlist(scores, state["slugs"], RERANK_TOPK)
    t_cv = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    core.rerank(prepared, candidates, state["reranker"])
    t_rerank = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    label = core.LabelText(prepared, read_words)
    core.settle(candidates, prepared, state["text"], label,
                window=CLOSE_WINDOW, min_conf=OCR_MIN_CONF,
                by_slug=state["by_slug"])
    # Улики и подтверждения собираются только если ответ иначе был бы
    # показан: они меняют оформление выдачи, а не сам ответ, и на заведомо
    # слабом совпадении не нужны. Слова к этому моменту уже прочитаны
    # в settle, так что проверка почти бесплатна.
    if core.outcomes(core.features(candidates), OUTCOME_WEIGHTS)[0] >= CONFLICT_GATE:
        core.check_leader(candidates, prepared, state["text"], label,
                          OCR_MIN_CONF, CLOSE_WINDOW, state["by_slug"])
    t_text = (time.perf_counter() - t2) * 1000

    # Уверенность считается по всему шортлисту, а не по видимой пятёрке:
    # признак «превышение над шумом» смотрит на его хвост, и на усечённом
    # списке значил бы другое, чем при калибровке.
    conf = confidence(candidates)
    resolved = any(c.contradictions for c in candidates)
    t_vlm = 0.0
    conf["vlm_confirmed"] = False
    if state.get("vlm") and candidates:
        import vlm_verify
        t3 = time.perf_counter()

        checked: dict[str, float] = {}
        last_call = [VLM_CALL_GUESS_MS]

        def verify(candidate) -> float:
            if candidate.slug in checked:
                return checked[candidate.slug]
            started = time.perf_counter()
            p_yes = vlm_verify.p_yes(views[("detect",)], candidate.slug,
                                     state["by_slug"].get(candidate.slug, {}), state["text"])
            last_call[0] = (time.perf_counter() - started) * 1000
            checked[candidate.slug] = round(p_yes, 4)
            return p_yes

        def affordable() -> bool:
            """Успеет ли ещё один вызов VLM в бюджет времени ответа."""
            spent = (time.perf_counter() - t0) * 1000
            if VLM_BUDGET_MS <= 0 or spent + last_call[0] <= VLM_BUDGET_MS:
                return True
            conf["vlm_budget_hit"] = True
            return False

        def arbitrate(pool: list, threshold: float) -> dict | None:
            """Первый подтверждённый VLM кандидат из pool становится лидером."""
            for candidate in pool:
                if not affordable():
                    return None
                p_yes = verify(candidate)
                if p_yes >= threshold:
                    previous = candidates[0].slug
                    candidates.remove(candidate)
                    candidates.insert(0, candidate)
                    switched = confidence(candidates)
                    switched.update(vlm_p_yes=round(p_yes, 4), vlm_confirmed=True,
                                    vlm_switched_from=previous)
                    return switched
            return None

        def geo_second_pass() -> dict | None:
            """Проверенный кандидат с P(yes) > VLM_REJECT_P, за которого геометрия."""
            leader = candidates[0]
            if checked.get(leader.slug, 1.0) >= VLM_REJECT_P:
                return None
            backed = [c for c in candidates[1:]
                      if checked.get(c.slug, 0.0) > VLM_REJECT_P
                      and core.geometry_backs(c, leader, candidates, CLOSE_WINDOW)]
            if not backed:
                return None
            best = max(backed, key=lambda c: checked[c.slug])
            candidates.remove(best)
            candidates.insert(0, best)
            switched = confidence(candidates)
            switched.update(vlm_p_yes=checked[best.slug], vlm_confirmed=True,
                            vlm_switched_from=leader.slug, vlm_geo_corroborated=True)
            return switched

        def label_rescue() -> dict | None:
            """Позиция по винодельне и названию с этикетки, если VLM за неё."""
            spent = (time.perf_counter() - t0) * 1000
            if LABEL_RESCUE_BUDGET_MS > 0 and (spent + LABEL_READ_GUESS_MS + VLM_CALL_GUESS_MS
                                               > LABEL_RESCUE_BUDGET_MS):
                conf["label_rescue"] = {"skipped": "budget"}
                return None
            import vlm_label
            read = vlm_label.read_name(views[("detect",)]) or {}
            slug, close = state["recommender"].by_label_name(read)
            info = {"read": {k: read.get(k) for k in ("producer", "producer_ru", "name")},
                    "slug": slug, "close": round(close, 3)}
            conf["label_rescue"] = info
            if slug is None or close < LABEL_RESCUE_CLOSE:
                return None
            candidate = next((c for c in candidates if c.slug == slug), None)
            if candidate is None:
                rows = state["rows_of"].get(slug, [])
                candidate = core.Candidate(
                    slug=slug, cv=float(max((scores[j] for j in rows), default=0.0)))
            p_yes = checked[slug] if slug in checked else verify(candidate)
            info["p_yes"] = round(p_yes, 4)
            if p_yes < VLM_CONFIRM_P:
                return None
            previous = candidates[0].slug
            if candidate in candidates:
                candidates.remove(candidate)
            candidates.insert(0, candidate)
            switched = confidence(candidates)
            switched.update(vlm_p_yes=round(p_yes, 4), vlm_confirmed=True,
                            vlm_switched_from=previous, label_rescue=info)
            return switched

        if conf["probability"] < SHOW_P:
            p_yes = verify(candidates[0])
            conf["vlm_p_yes"] = round(p_yes, 4)
            conf["vlm_confirmed"] = p_yes >= VLM_CONFIRM_P
            if not conf["vlm_confirmed"] and VLM_FALLBACK:
                conf = arbitrate(candidates[1:1 + VLM_FALLBACK], VLM_CONFIRM_P) or conf
            if not conf["vlm_confirmed"] and VLM_GEO_FALLBACK:
                conf = geo_second_pass() or conf
            if not conf["vlm_confirmed"] and LABEL_RESCUE:
                conf = label_rescue() or conf
        elif VLM_CONTRA:
            owners = core.text_contradiction(candidates, label(prepared), state["text"])
            if owners and affordable():
                p_yes = verify(candidates[0])
                conf["vlm_p_yes"] = round(p_yes, 4)
                if p_yes < VLM_REJECT_P:
                    pool = [c for c in candidates if c.slug in owners][:max(VLM_FALLBACK, 2)]
                    conf = arbitrate(pool, VLM_CONTRA_P) or conf
        if (VLM_BAND_VETO and not conf["vlm_confirmed"]
                and SHOW_P <= conf["probability"] < CONFIDENT_P and affordable()):
            p_yes = verify(candidates[0])
            conf["vlm_p_yes"] = round(p_yes, 4)
            if p_yes < VLM_VETO_P:
                conf["vlm_vetoed"] = True
                if VLM_FALLBACK:
                    conf = arbitrate(candidates[1:1 + VLM_FALLBACK], VLM_CONFIRM_P) or conf
                if not conf["vlm_confirmed"] and VLM_GEO_FALLBACK:
                    conf = geo_second_pass() or conf
                if not conf["vlm_confirmed"] and LABEL_RESCUE:
                    conf = label_rescue() or conf
        if checked:
            conf["vlm_checked"] = checked
        t_vlm = (time.perf_counter() - t3) * 1000
    return ([c.as_dict() for c in candidates[:top]], conf,
            {"cv_ms": round(t_cv, 1), "rerank_ms": round(t_rerank, 1),
             "text_ms": round(t_text, 1), "vlm_ms": round(t_vlm, 1),
             "resolved": resolved})


def confidence(candidates: list) -> dict:
    """Уверенность в лидере и вероятность, что нужное вино есть в пятёрке.

    ТЗ требует показывать метрику уверенности и для топ-1, и для топ-5:
    от неё зависит, нужен ли пользователю экран вариантов вообще.
    """
    feats = core.features(candidates)
    top1, top5 = core.outcomes(feats, OUTCOME_WEIGHTS)
    leader = candidates[0] if candidates else None
    rival = max((c.inliers for c in candidates[1:]), default=0) if candidates else 0
    return {
        "probability": round(top1, 4),
        "probability_top5": round(top5, 4),
        "dominance": round(feats["dominance"], 4),
        "inliers": leader.inliers if leader else 0,
        "gap": (leader.inliers - rival) if leader else 0,
        "cv_margin": round(feats["cv_margin"], 4),
        "conflicts": list(leader.conflicts) if leader else [],
        "confirmed": list(leader.confirmed) if leader else [],
        "may_drop_caveat": core.may_drop_caveat(candidates, state["by_slug"],
                                               CLOSE_WINDOW),
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


FIELD_QUALITY = INDEX_DIR / "field_quality.json"


def quality_block() -> dict:
    """Измеренное качество конвейера.

    Числа F1 берутся из report_sharp.json — это синтетический набор: запросы
    сделаны из эталонных фотографий каталога преобразованиями, и абсолютные
    значения он завышает. Чтобы их не приняли за качество на реальной
    съёмке, источник назван явно, а рядом отдаётся последний замер на
    полевом наборе из data/index/field_quality.json, если он есть, — со
    своей пометкой, что это регрессионная часть, а не независимая приёмка.
    """
    measured = (state.get("quality") or {}).get("f1", {}).get("по каталогу", {})
    if not measured:
        return {}
    report_show = (state.get("quality") or {}).get("show_threshold")
    block = {
        "f1_top1": round(measured["top1"]["f1"], 4),
        "f1_top5": round(measured["top5"]["f1"], 4),
        "subset": (state.get("quality") or {}).get("subset"),
        "n": (state.get("quality") or {}).get("answerable"),
        "source": "synthetic",
        "source_note": "синтетические запросы из эталонных фото каталога (report_sharp.json); "
                       "не качество на реальной съёмке",
        # Отчёт пересчитывается отдельно от порога сервиса; расхождение
        # показывается явно, а не молча выдаётся за текущее качество.
        "report_show_p": report_show,
        "matches_service_show_p": report_show is not None and abs(report_show - SHOW_P) < 1e-9,
    }
    if FIELD_QUALITY.exists():
        try:
            block["field"] = json.loads(FIELD_QUALITY.read_text(encoding="utf-8"))
        except Exception:
            pass
    return block


def settings_block() -> dict:
    """Фактические настройки процесса: пороги и переключатели правил.

    Отчёты считаются с параметрами из своих аргументов и окружения, сервис —
    из своего. Без этого блока нельзя проверить, что запущенный сервис и
    последний замер — один и тот же конвейер.
    """
    return {
        "show_p": SHOW_P, "confident_p": CONFIDENT_P,
        "eval_abstain": EVAL_ABSTAIN, "eval_abstain_p": EVAL_ABSTAIN_P,
        "vlm_confirm": VLM_CONFIRM, "vlm_active": bool(state.get("vlm")),
        "vlm_fallback": VLM_FALLBACK, "vlm_contra": VLM_CONTRA, "vlm_contra_p": VLM_CONTRA_P,
        "vlm_geo_fallback": VLM_GEO_FALLBACK,
        "vlm_band_veto": VLM_BAND_VETO, "vlm_veto_p": VLM_VETO_P,
        "label_rescue": LABEL_RESCUE, "label_rescue_budget_ms": LABEL_RESCUE_BUDGET_MS,
        "vlm_budget_ms": VLM_BUDGET_MS,
        "vlm_confirm_p": VLM_CONFIRM_P,
        "thresholds_from_env": {k: os.environ.get(k) for k in ("SHOW_P", "CONFIDENT_P")
                                if os.environ.get(k) is not None},
        "rules_env": core.switches(),
        "rerank_topk": RERANK_TOPK, "close_window": CLOSE_WINDOW, "ocr_min_conf": OCR_MIN_CONF,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "vectors" in state else "loading",
        "index": INDEX_VARIANT,
        "vectors": len(state["vectors"]) if "vectors" in state else 0,
        "vector_backend": os.environ.get("VECTOR_BACKEND", "numpy"),
        "catalog": len(state.get("by_slug", {})),
        "quality": quality_block(),
        "rerank_topk": RERANK_TOPK,
        "settings": settings_block(),
    }


@app.get("/v1/wines/{slug}")
def wine_card(slug: str) -> JSONResponse:
    wine = state["by_slug"].get(slug)
    if wine is None:
        return JSONResponse({"error": "позиция не найдена"}, status_code=404)
    return JSONResponse({
        "wine": wine,
        # Сладость и игристость: в выгрузке отдельных полей нет, они собраны
        # из slug, названия и карточек платформы (recommend.facets_of).
        "style": state["recommender"].style(slug),
        "pairings": state["recommender"].pairings(wine),
        "similar": state["recommender"].similar(slug),
    })


def read_label(img: Image.Image) -> tuple[dict, str]:
    """Стиль вина с этикетки: VLM, а без неё — OCR (цвет и сладость)."""
    with GPU_LOCK:
        crop = normalize_batch([img], steps=("detect",))[0]
        if state.get("vlm"):
            import vlm_label
            try:
                label = vlm_label.read(crop)
                if label is not None:
                    return label, "vlm"
            except Exception as exc:  # noqa: BLE001 — без чтения остаётся OCR
                print(f"/v1/analogs: VLM не прочитала этикетку ({exc})", file=sys.stderr)
        attrs = confident_attributes(read_words(crop), OCR_MIN_CONF)
    return {"producer": None, "producer_ru": None, "name": None, "color": attrs.color,
            "sweetness": attrs.sweetness,
            "sparkling": True if attrs.sweetness in {"брют", "экстра брют"} else None,
            "grapes": [], "region": None, "country": None}, "ocr"


@app.post("/v1/analogs")
async def analogs(image: UploadFile = File(...)) -> JSONResponse:
    """Аналоги для вина, которое поиск не нашёл: стиль читается с этикетки.

    Фронтенд вызывает его после ответа /v1/search со статусом unsure, так
    что на время поиска и на оценочный эндпоинт он не влияет. Чтение идёт
    в пуле потоков, чтобы фото вариантов грузились, пока модель читает,
    а GPU_LOCK не даёт ему пересечься с поиском.

    Стиль читает VLM (pipeline/vlm_label.py). Без неё — OCR: цвет и
    сладость из уверенно прочитанных слов, без сортов и винодельни.
    """
    img = await read_image(image)
    if img is None:
        return JSONResponse({"error": "не удалось прочитать изображение"}, status_code=400)
    t0 = time.perf_counter()
    label, source = await run_in_threadpool(read_label, img)
    recommender = state["recommender"]
    facets = recommender.label_facets(label)
    seed = f"{label.get('producer') or ''}|{label.get('name') or ''}"
    # Без цвета и сорта подбирать не по чему: аналоги были бы случайными.
    found = (recommender.analogs(facets, seed=seed)
             if (facets.color or facets.grapes) else [])
    # Этого вина нет, но винодельня в каталоге есть — её вина тоже подсказка.
    own = (recommender.from_maker(facets.manufacturer, facets.color, label.get("name") or "")
           if facets.manufacturer else [])
    return JSONResponse({
        "label": label,
        "source": source,
        "maker": facets.manufacturer and next(
            (w["wine"]["manufacturer"] for w in own), None),
        "maker_wines": own,
        "analogs": found,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
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

    Ниже EVAL_ABSTAIN_P возвращается {"slug": null} — ответ «вина нет
    в каталоге», если лидера не подтвердил VLM. EVAL_ABSTAIN=0 — всегда
    лучший кандидат.
    """
    img = await read_image(image)
    if img is None:
        return JSONResponse({"slug": None})
    with GPU_LOCK:
        candidates, conf, _ = search_candidates(img, top=1)
    if not candidates or (EVAL_ABSTAIN and not conf["vlm_confirmed"]
                          and (conf["probability"] < EVAL_ABSTAIN_P or conf.get("vlm_vetoed"))):
        return JSONResponse({"slug": None})
    return JSONResponse({"slug": candidates[0]["slug"]})


@app.post("/v1/search")
async def search(image: UploadFile = File(...)) -> JSONResponse:
    img = await read_image(image)
    if img is None:
        return JSONResponse({"error": "не удалось прочитать изображение"}, status_code=400)

    # Десять, а не пять: кейсодержатель просил при неуверенности показывать
    # ближайших кандидатов «как текущая выдача до 10 вин».
    with GPU_LOCK:
        candidates, conf, timings = search_candidates(img, top=10)
    # Подтверждённый VLM ответ показывается, но только как спорный:
    # уверенным его делает лишь совместная оценка конвейера.
    found = bool(candidates) and (conf["vlm_confirmed"] or (conf["probability"] >= SHOW_P
                                                            and not conf.get("vlm_vetoed")))

    # Три состояния вместо «нашли или нет». Бинарное решение по одному порогу
    # либо прячет треть верных ответов, либо пропускает половину неверных —
    # середина честнее: уверенный ответ показываем одной карточкой, спорный
    # тоже показываем, но с признанием сомнения и ближайшими вариантами.
    if not found:
        status = "unsure"
    elif conf["probability"] >= CONFIDENT_P and conf["may_drop_caveat"]:
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
                                      + timings["text_ms"] + timings["vlm_ms"], 1)},
    })
