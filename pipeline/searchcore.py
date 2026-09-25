#!/usr/bin/env python3
"""Ядро поиска: одно определение шагов для сервиса и для замеров.

Зачем отдельный модуль. Раньше сервис и калибровка считали признаки
уверенности порознь, и определения разошлись: обучение брало косинус
выбранного кандидата, а сервис — наибольший косинус в шортлисте, который
может принадлежать другому вину. На карточке Katharon это давало 80% там,
где по формуле обучения выходило 30%. Плюс калибровка вообще не включала
последнюю ступень — перестановку по тексту, то есть подбирала пороги
не для того конвейера, который работает.

Поэтому шаги живут здесь по одному экземпляру, а сервис и скрипты замеров
их вызывают. Разойтись им больше негде.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Все переключатели окружения, от которых зависит ответ конвейера. Один
# список на сервис (/health) и отчёты замеров: раньше каждый держал свой,
# и в обоих не хватало половины правил, так что по отчёту нельзя было
# проверить, что сервис и замер — один и тот же конвейер.
RULE_SWITCHES = (
    "TEXT_ALIASES", "OCR_JOIN", "NAME_FIRST", "LINE_BAND", "PLATFORM_SWEETNESS",
    "NAME_STRONG", "NAME_TRANSLIT", "GROUP_LEADER", "DEMOTE_GEOMETRY", "DEMOTE_DEPTH",
    "LABEL_BAND", "KIN_GROUP", "BRAND_FIRST", "GEOM_WEIGHT", "GEOM_WEIGHT_ALPHA",
    "SIFT_MAX_SIDE", "SIFT_DIR", "YOLO_WEIGHTS", "YOLO_CONF", "YOLO_IMGSZ",
    "PLATFORM_ADDITIONS", "GEOMETRY_MARGIN", "SIFT_ONE_TO_ONE",
    "SIFT_UNIQUE_INLIERS", "TEXT_VOTE",
)

# Позиции, появившиеся на платформе после выгрузки кейса (sync_platform.py).
# По умолчанию ищутся вместе с остальными: организатор снимал и такие вина.
# PLATFORM_ADDITIONS=0 возвращает каталог выгрузки — на случай, если
# закрытая таблица размечена по ней: строки индекса этих позиций становятся
# мусором и в шортлист не попадают, пересборка индекса не нужна.
USE_PLATFORM_ADDITIONS = os.environ.get("PLATFORM_ADDITIONS", "1") != "0"
_ADDITIONS = (Path(__file__).resolve().parent.parent
              / "data" / "datapack" / "platform_additions.json")


def scope_index(slugs: list[str | None]) -> list[str | None]:
    """Slug строк индекса с учётом PLATFORM_ADDITIONS."""
    if USE_PLATFORM_ADDITIONS or not _ADDITIONS.exists():
        return slugs
    import json
    added = {w["slug"] for w in json.loads(_ADDITIONS.read_text(encoding="utf-8"))["wines"]}
    return [None if s in added else s for s in slugs]


def switches() -> dict[str, str | None]:
    """Значения переключателей из окружения; None — действует умолчание кода."""
    return {key: os.environ.get(key) for key in RULE_SWITCHES}


# Порядок признаков уверенности. Совпадает с порядком весов в артефакте
# калибровки, поэтому менять его без пересчёта весов нельзя.
FEATURE_ORDER = ("dominance", "log_inliers", "cv_margin", "cv_lift",
                 "coverage", "text_support", "text_conflict")

# Хвост шортлиста — кандидаты ниже пятого места по косинусу. Это уровень
# «просто похожих бутылок», от которого отсчитывается превышение лидера.
TAIL_FROM = 5


@dataclass
class Candidate:
    slug: str
    cv: float
    inliers: int = 0
    coverage: float = 0.0
    color: float = 1.0
    contradictions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    hard: list[str] = field(default_factory=list)
    weighted: float = 0.0     # взвешенные инлаеры, только при GEOM_WEIGHT

    def as_dict(self) -> dict:
        row = {"slug": self.slug, "cv_score": round(self.cv, 4),
               "inliers": self.inliers, "coverage": round(self.coverage, 4)}
        if self.contradictions:
            row["contradictions"] = self.contradictions
        if self.conflicts:
            row["conflicts"] = self.conflicts
        if self.confirmed:
            row["confirmed"] = self.confirmed
        return row


def shortlist(scores: np.ndarray, index_slugs: list[str | None],
              topk: int, depth: int = 200) -> list[Candidate]:
    """Кандидаты по убыванию косинуса, по одному на slug.

    Свёртка по slug обязательна: позиция описывается несколькими векторами
    (кроп и кадр целиком), и без неё шортлист на двадцать мест занимают
    разные виды одной позиции.
    """
    out: list[Candidate] = []
    seen: set[str] = set()
    # Устойчивая сортировка: у позиций с общим эталоном косинус равен точно,
    # и без неё порядок такой ничьей зависел от остального содержимого
    # массива (у pgvector-варианта, где строки вне первых k равны -1, он
    # выходил другим). Теперь ничья решается порядком строк индекса.
    for j in np.argsort(-scores, kind="stable")[:depth]:
        slug = index_slugs[j]
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        out.append(Candidate(slug=slug, cv=float(scores[j])))
        if len(out) == topk:
            break
    return out


def pick_view(views: list, similarity) -> int:
    """Каким кропом кормить геометрию: тем, который лучше узнал каталог.

    Раньше геометрии всегда доставался кроп по бутылке. На части кадров
    он хуже целого: лежащая бутылка даёт рамку, на три четверти состоящую
    из стола, а на снимке полки детектор не находит ничего и кроп равен
    кадру. Выбор по косинусу ничего не стоит — оба вида уже посчитаны для
    шортлиста, — и на настроечной части полевого набора даёт top-1 20 из 30
    против 18 и top-5 23 против 22.
    """
    best = similarity.max(axis=0)
    return int(best.argmax()) if len(best) else 0


# Решающий отрыв геометрии: порядок по инлаерам принимается, только если
# лидер обходит второго больше чем на эту долю, иначе порядок косинусный.
# Переменная окружения — для парного замера без правки кода.
GEOMETRY_MARGIN = float(os.environ.get("GEOMETRY_MARGIN", "0.50"))


def rerank(crop: Image.Image, candidates: list[Candidate], reranker,
           margin: float = GEOMETRY_MARGIN) -> list[Candidate]:
    """Геометрическая проверка: сколько точек этикетки совпало и где.

    Порядок меняется, только если геометрия ответила уверенно — у лидера
    по совпавшим точкам отрыв от второго не меньше margin. Иначе остаётся
    порядок по косинусу.

    Правило появилось из разбора полевых промахов. У линейки одной
    винодельни общая часть этикетки больше различающей: гравюра дворца
    и зелёная лента «Массандры» совпадают у всех её вин, название сорта
    занимает три слова мелким шрифтом. Инлаеров тогда выходит 46-52
    у пяти кандидатов подряд — разница внутри шума, а порядок она задавала
    целиком, и правильный ответ, стоявший первым по косинусу, уезжал
    на третье место.

    Замерено на обоих наборах (pipeline/eval_rank.py). Полевая настроечная
    часть: top-1 46.7% -> 60.0%, top-5 60.0% -> 70.0%. Синтетика, 400
    запросов: 90.2% -> 91.2% и 94.2% -> 95.2%. Отключать геометрию совсем
    нельзя — на синтетике один косинус даёт 80.8%.

    Порог отрыва пересчитан, когда настроечная часть выросла с 30 кадров
    до 203 и в неё вошла съёмка у полки. На 169 находимых кадрах: один
    косинус 77.5% / 91.7%, отрыв 25% — 77.5% / 88.8%, отрыв 50% — 78.7%
    / 91.1%. При прежнем пороге геометрия не добавляла к косинусу ничего
    по top-1 и отнимала три пункта по top-5: на магазинном снимке этикетка
    видна целиком, и косинус сам справляется, а перестановка по точкам
    чаще мешает. Отсюда 0.50.
    """
    if os.environ.get("GEOM_WEIGHT", "0") != "0" and hasattr(reranker, "stats_points"):
        return rerank_weighted(crop, candidates, reranker, margin)
    stats = reranker.stats(crop, [c.slug for c in candidates])
    for candidate in candidates:
        match = stats.get(candidate.slug)
        if match is not None:
            candidate.inliers = int(match.inliers)
            candidate.coverage = float(match.coverage)
            candidate.color = float(match.color)
    by_geometry = sorted(candidates, key=lambda c: (-c.inliers, -c.cv))
    decisive = (len(by_geometry) < 2
                or by_geometry[0].inliers >= by_geometry[1].inliers * (1.0 + margin))
    candidates.sort(key=(lambda c: (-c.inliers, -c.cv)) if decisive
                    else (lambda c: -c.cv))
    return candidates


def weighted_scores(stats: dict[str, dict], slugs: list[str],
                    alpha: float) -> tuple[dict[str, float], dict[int, int]]:
    """Взвешенные инлаеры: вес точки запроса убывает с числом разных вин.

    Формула. Для точки запроса p множество S(p) — позиции шортлиста, у
    которых p оказалась инлаером хотя бы одного эталона (повторные эталоны
    одной позиции считаются один раз). Вес w(p) = |S(p)|^(-alpha). Оценка
    позиции — сумма весов по инлаерам её лучшего эталона (того, у которого
    инлаеров больше); каждая точка запроса входит в сумму позиции один раз.
    Геометрическая проверка (тест Лоу, RANSAC) остаётся прежней —
    взвешиваются только точки, её прошедшие.
    """
    multiplicity: dict[int, int] = {}
    for slug in slugs:
        for point in stats.get(slug, {}).get("union", ()):
            multiplicity[point] = multiplicity.get(point, 0) + 1
    scores = {}
    for slug in slugs:
        points = stats.get(slug, {}).get("points", ())
        scores[slug] = float(sum(multiplicity.get(p, 1) ** (-alpha) for p in points))
    return scores, multiplicity


def rerank_weighted(crop: Image.Image, candidates: list[Candidate], reranker,
                    margin: float) -> list[Candidate]:
    """Экспериментальный порядок по взвешенным инлаерам (GEOM_WEIGHT=1).

    Порядок и решающий отрыв считаются по взвешенной оценке, а число
    инлаеров и покрытие остаются сырыми — их читают признаки уверенности,
    и менять их смысл в этом эксперименте нельзя. Показатель степени —
    GEOM_WEIGHT_ALPHA (по умолчанию 1.0).
    """
    alpha = float(os.environ.get("GEOM_WEIGHT_ALPHA", "1.0"))
    slugs = [c.slug for c in candidates]
    stats = reranker.stats_points(crop, slugs)
    scores, _ = weighted_scores(stats, slugs, alpha)
    for candidate in candidates:
        match = stats.get(candidate.slug)
        if match is not None:
            candidate.inliers = int(match["inliers"])
            candidate.coverage = float(match["coverage"])
            candidate.weighted = scores.get(candidate.slug, 0.0)
    by_geometry = sorted(candidates, key=lambda c: (-c.weighted, -c.cv))
    decisive = (len(by_geometry) < 2
                or by_geometry[0].weighted >= by_geometry[1].weighted * (1.0 + margin))
    candidates.sort(key=(lambda c: (-c.weighted, -c.cv)) if decisive
                    else (lambda c: -c.cv))
    return candidates


def has_close_group(candidates: list[Candidate], window: float) -> bool:
    """Развела ли геометрия лидеров. Если нет — есть смысл читать этикетку.

    Считается от максимума инлаеров, а не от первого в списке: с тех пор
    как геометрия переупорядочивает только при уверенном отрыве, первым
    может стоять кандидат с меньшим числом совпавших точек.
    """
    if len(candidates) < 2:
        return False
    best = max(c.inliers for c in candidates)
    if best <= 0:
        return False
    return sum(1 for c in candidates if c.inliers >= best * window) >= 2


class LabelText:
    """Строки этикетки, прочитанные один раз на запрос.

    До этого OCR вызывался дважды: сначала перестановкой близких
    кандидатов, потом проверкой лидера, — и каждый раз заново по тому же
    кропу. С развёрткой полосы проходов становилось четыре, и время
    ответа выходило за SLA. Кроп за запрос один, значит и чтение одно.
    """

    def __init__(self, crop: Image.Image, read_words) -> None:
        self.crop = crop
        self.read_words = read_words
        self.words: list | None = None

    def __call__(self, _crop=None) -> list:
        if self.words is None:
            self.words = read_label(self.crop, self.read_words)
        return self.words


def read_label(crop: Image.Image, read_words) -> list:
    """Строки этикетки: с кропа, а при нужде ещё и с развёрнутой полосы.

    Одного прохода по кропу не всегда хватает. Строка, которая отличает
    вино от соседа по линейке, идёт по дуге тонкими разрядёнными буквами,
    и распознаватель отдаёт из неё кашу: «РОЗОВОЕ ПОАУСУХОЕ» вместо
    «РОЗОВОЕ ПОЛУСУХОЕ», «ИГРИстов» вместо «ИГРИСТОЕ». После развёртки
    с цилиндра и увеличения вдвое те же строки читаются целиком.

    Второй проход зовётся выборочно — только когда первый не дал ни цвета,
    ни сладости: если различающий признак уже прочитан, читать заново
    нечего. Выигрыша по времени это почти не дало, и вот почему: цвет
    читается на 4 кадрах из 33, сладость на 6, то есть условие почти
    всегда выполняется и проход всё равно нужен.

    По умолчанию выключено, включается LABEL_BAND=1. Вопрос решается
    не развёрткой, а разрешением входного кадра, и это стоило отдельной
    проверки. На копиях съёмки организаторов, сжатых до 1600 px, развёртка
    давала +1 кадр (47 из 55 против 46). На тех же кадрах в исходных
    3024x4032 разницы нет вовсе: 47 из 55 и 54 в пятёрке с ней и без неё,
    F1 top-1 81.5% против 80.7%, F1 top-5 87.0% против 88.1% — знак
    разный, величина внутри шума. Медиана ответа при этом 1597 мс против
    654. Кроп по бутылке берётся с оригинала, и на исходном разрешении
    у OCR уже есть те пиксели, ради которых полосу увеличивали вдвое.

    Прежний замер на старом наборе из 33 кадров говорил то же самое:
    top-1 34 из 64 против 33 при медиане 1888 мс против 615.

    Координаты строк полосы переводятся обратно в кроп: отбор по месту
    на кадре отсеивает надписи с соседней бутылки, и сравнивать он должен
    в одной системе координат.
    """
    from dataclasses import replace

    from imageprep import label_band_view
    from text_match import extract_attributes

    from text_match import join_spaced

    # Разрядка и разрыв слова склеиваются один раз здесь: все потребители
    # строк — перестановка, отвод лидера, улики — читают уже склеенное.
    words = [replace(w, text=join_spaced(w.text)) for w in read_words(crop)]
    if os.environ.get("LABEL_BAND", "0") != "1":
        return words
    attrs = extract_attributes(" ".join(w.text for w in words))
    if attrs.color and attrs.sweetness:
        return words
    try:
        band, to_crop = label_band_view(crop)
    except Exception:
        return words
    for word in read_words(band):
        box = to_crop(word.box)
        words.append(replace(word, box=tuple(int(v) for v in box)))
    return words


def kin_group(candidates: list[Candidate], by_slug: dict | None,
              words, channel, min_conf: float) -> set[str]:
    """Кандидаты, которых надо сравнить текстом, даже если геометрия их развела.

    Два случая, в обоих число совпавших точек обманывает.

    Однофамильцы лидера. У линейки с общей этикеткой точки считаются по
    общей части: у «Табии» верное вино набирает 10 точек против 42 у
    соседа, хотя различает их одно слово под рисунком. Окно по инлаерам
    такого кандидата в группу не пускает, и текст его не видит.

    Кандидаты прочитанной винодельни. Имя винодельни — самая крупная
    надпись, и читается оно там, где сорт уже не читается. «INKERMAN»
    и «Литавщуков» распознаются целиком, а первым при этом стоит вино
    другой винодельни с похожей по виду этикеткой.

    По умолчанию выключено, включается KIN_GROUP=1. Замерено на
    настроечной части (160 находимых кадров): без правила лидер верен
    в 131, с правилом — в 115, F1 top-1 81.8% против 74.7%. Причина
    в качестве чтения. На снимке у полки надпись крупная и читается,
    а на кадре из отзыва OCR отдаёт кашу вроде «CiOE» и «FOJCTOE», и эта
    каша, попав в расширенную группу, штрафует верного лидера, которого
    геометрия до того выбирала правильно. Расширять группу можно только
    вместе с проверкой качества чтения, а её пока нет.
    """
    from text_match import brand_slugs, name_slugs

    if not candidates:
        return set()
    group: set[str] = set()
    # Однофамильцы и вина прочитанной винодельни — под KIN_GROUP, как и было
    # замерено. Владелец прочитанного имени вина — отдельное правило со
    # своим переключателем (NAME_FIRST внутри name_slugs): оно приводит
    # одного кандидата по точно прочитанному слову, а не всю линейку.
    if os.environ.get("KIN_GROUP", "0") == "1":
        if by_slug:
            brand = (by_slug.get(candidates[0].slug) or {}).get("manufacturer")
            if brand:
                group |= {c.slug for c in candidates
                          if (by_slug.get(c.slug) or {}).get("manufacturer") == brand}
        group |= brand_slugs([c.slug for c in candidates], words, channel, min_conf)
    group |= name_slugs([c.slug for c in candidates], words, channel, min_conf,
                        weight={c.slug: float(c.inliers) for c in candidates})
    return group


def prefer_brand(candidates: list[Candidate], words, channel,
                 min_conf: float) -> list[Candidate]:
    """Вперёд — вина винодельни, чьё имя прочитано на этикетке.

    Отдельный шаг, а не улика внутри группы, потому что правило разрешения
    одностороннее: оно опускает кандидата, у которого прочитанного слова
    нет, но поднять кандидата выше тех, кто в группу не попал, не может.
    А здесь нужно именно это. На снимке у полки крупно читается «INKERMAN»
    или «Литавщуков», при этом первым по геометрии стоит вино другой
    винодельни с похожей этикеткой: «Марселан» Мысхако, «Мерло» Усадьбы
    Перовских. Имя винодельни — самая надёжная надпись на кадре, и если
    она прочитана уверенно и принадлежит ровно одной винодельне шортлиста,
    её вина идут первыми.

    Порядок внутри обеих частей сохраняется: шаг только разделяет своих
    и чужих, выбор конкретного вина остаётся за текстом и геометрией.
    """
    from text_match import brand_slugs

    if len(candidates) < 2 or os.environ.get("BRAND_FIRST", "1") != "1":
        return candidates
    own = brand_slugs([c.slug for c in candidates], words, channel, min_conf)
    if not own or len(own) == len(candidates):
        return candidates
    place = {c.slug: i for i, c in enumerate(candidates)}
    candidates.sort(key=lambda c: (c.slug not in own, place[c.slug]))
    return candidates


def resolve(candidates: list[Candidate], crop: Image.Image, channel,
            read_words, window: float, min_conf: float,
            by_slug: dict | None = None) -> list[Candidate]:
    """Перестановка близких кандидатов по прочитанному тексту.

    OCR считается только когда есть кого сравнивать: близкая по геометрии
    группа либо родня лидера по винодельне. При одиноком лидере текст
    ничего не решит, а время ответа вырастет на сотню миллисекунд.
    """
    from text_match import resolve_close

    words = read_words(crop)
    kin = kin_group(candidates, by_slug, words, channel, min_conf)
    # Группа сравнения — лучший по геометрии плюс приведённые. Сравнивать
    # есть с кем, если приведён хоть один кандидат кроме самого лидера:
    # владелец прочитанного имени приходит один, и прежняя проверка
    # «в родне меньше двух» отправляла его обратно, не сравнив.
    others = kin - ({candidates[0].slug} if candidates else set())
    if not has_close_group(candidates, window) and not others:
        return candidates
    # Сильное имя (NAME_STRONG): все, кто стоит впереди владельца, входят
    # в группу. Правило одностороннее и поднять никого не может — оно
    # только опускает тех, у кого прочитанного слова нет; чтобы владелец
    # точно прочитанного имени вышел вперёд, противоречие должны получить
    # все, кто перед ним.
    if os.environ.get("NAME_STRONG", "1") != "0":
        from text_match import strong_name_slugs
        strong = strong_name_slugs([c.slug for c in candidates], words, channel, min_conf)
        for slug in strong:
            index = next((i for i, c in enumerate(candidates) if c.slug == slug), None)
            if index:
                kin |= {c.slug for c in candidates[:index]} | {slug}
    pairs = [(c.slug, float(c.inliers)) for c in candidates]
    ranked, reasons = resolve_close(pairs, words, channel,
                                    window=window, min_conf=min_conf, extra=kin)
    if not any(reasons.values()):
        return candidates
    position = {slug: i for i, (slug, _) in enumerate(ranked)}
    for candidate in candidates:
        candidate.contradictions = reasons.get(candidate.slug, [])
    candidates.sort(key=lambda c: position.get(c.slug, len(position)))
    return candidates


def series_rivals(candidates: list[Candidate], by_slug: dict,
                  share: float = 0.5) -> list[Candidate]:
    """Соседи лидера по линейке, держащиеся близко по геометрии.

    Окно близких кандидатов настроено на случай, когда геометрия не смогла
    выбрать вовсе. Подмена внутри линейки выглядит иначе: лидер уверенно
    впереди — 94 инлаера против 67, — но соперники это его же однофамильцы
    с почти той же этикеткой, и разница в тридцать точек ничего не решает.
    Считаем их отдельно: та же винодельня и не меньше половины инлаеров.
    """
    if not candidates:
        return []
    leader = candidates[0]
    brand = (by_slug.get(leader.slug) or {}).get("manufacturer")
    best = max(c.inliers for c in candidates)
    if not brand or best <= 0:
        return []
    limit = best * share
    return [c for c in candidates[1:]
            if c.inliers >= limit
            and (by_slug.get(c.slug) or {}).get("manufacturer") == brand]


def features(candidates: list[Candidate]) -> dict[str, float]:
    """Признаки уверенности для лидера — того кандидата, которого покажем.

    Все величины относительные: сравнивают лидера с остальным шортлистом,
    а не с абсолютной шкалой. Абсолютные не переносятся с синтетики на
    съёмку — у запроса, полученного из файла эталона, и косинус, и число
    инлаеров заметно выше, чем у снимка с полки.


    Отрыв по косинусу считается против прямого соперника — того кандидата,
    которого показали бы вместо лидера, — а не против максимума по всему
    шортлисту. Максимум означал бы «согласился ли косинус с геометрией»,
    и штрафовал бы ровно те случаи, ради которых геометрия и добавлена:
    на реальном снимке нужное вино часто не первое по косинусу, иначе
    вторая ступень была бы не нужна. Общую картину шортлиста держит
    отдельный признак cv_lift.

    Исключение из этого правила — кандидаты, отвергнутые по тексту. Если на
    этикетке уверенно прочитано слово, которое принадлежит другому вину,
    такой кандидат перестаёт быть соперником: именно это и решила
    предыдущая ступень. Считать его конкуренцию значило бы наказывать
    ответ за то, что он исправлен, — перестановка по построению поднимает
    кандидата, который зрению понравился меньше.

    Признак text_support при этом остаётся: он отмечает, что решение
    опиралось на текст, и модель сама решает, доверять ли такому решению
    так же, как чисто геометрическому.
    """
    if not candidates:
        return {name: 0.0 for name in FEATURE_ORDER}
    leader, others = candidates[0], candidates[1:]
    rivals = [c for c in others if not c.contradictions] or others
    # Прямой соперник — тот, кого показали бы вместо лидера, то есть
    # следующий в том же порядке. Брать максимум по инлаерам больше нельзя:
    # порядок задаёт геометрия не всегда, и «следующий» может стоять выше
    # кандидата с большим числом совпавших точек.
    rival = rivals[0] if rivals else None
    rival_inliers = rival.inliers if rival else 0
    total = leader.inliers + rival_inliers
    rival_cv = rival.cv if rival else leader.cv
    by_cv = sorted((c.cv for c in candidates), reverse=True)
    tail = by_cv[TAIL_FROM:] or by_cv
    supported = (not leader.contradictions
                 and any(c.contradictions for c in others))
    # Улики против лидера считаются отдельно от перестановки: они работают
    # там, где перестановка бессильна — когда нужного вина в каталоге нет
    # и сосед по серии побеждает без конкурента.
    conflict = min(len(leader.conflicts), 3) / 3.0
    return {
        "dominance": leader.inliers / total if total else 0.0,
        "log_inliers": math.log1p(leader.inliers),
        "cv_margin": leader.cv - rival_cv,
        "cv_lift": leader.cv - sum(tail) / len(tail),
        "coverage": leader.coverage,
        "text_support": 1.0 if supported else 0.0,
        "text_conflict": conflict,
    }


def may_drop_caveat(candidates: list[Candidate], by_slug: dict,
                   window: float = 0.80) -> bool:
    """Можно ли показать карточку без оговорки «скорее всего».

    Живёт здесь, а не в сервисе, потому что то же правило нужно калибровке:
    порог «без оговорок» подбирается по той выдаче, которую увидит человек.
    Раньше правило было только в сервисе, и подбор порога шёл по выдаче без
    него — числа расходились с реальностью.
    """
    leader = candidates[0]
    if leader.conflicts:
        return False
    crowded = (has_close_group(candidates, window)
               or len(series_rivals(candidates, by_slug)) >= 2)
    return bool(leader.confirmed) if crowded else True


def demote_contradicted(candidates: list[Candidate], channel, words,
                        min_conf: float, depth: int = 4,
                        margin: float = GEOMETRY_MARGIN) -> None:
    """Лидер, которому этикетка прямо противоречит, уступает место.

    До этой правки проверка лидера только понижала уверенность, а ответ
    оставляла прежним. На 5881591_0.jpg OCR читал «КРАСНОЕ», карточка
    лидера была белым брютом, противоречие фиксировалось — и лидер всё
    равно уходил пользователю и в оценочное API, которое берёт первого
    кандидата и порогов не знает. Проверка, не меняющая ответа, ничего
    не проверяет.

    Понижение идёт только по структурным полям карточки — цвет, сладость,
    год. Улика вида «на этикетке слово другого кандидата» ответ не меняет:
    на снимке с двумя бутылками читается название соседней.

    Глубина ограничена: если противоречат подряд несколько кандидатов,
    скорее ошибается чтение, а не каталог, и лучше оставить порядок
    зрению.
    """
    # Два независимых признака (цвет и сладость), прочитанных вместе,
    # ошибкой чтения быть почти не могут — тогда глубина не ограничена
    # (DEMOTE_DEPTH). На 6182553_0.jpg «РОЗОВОЕ ПОЛУСУХОЕ» противоречило
    # четырём кандидатам подряд, и на четвёртом отвод останавливался.
    if os.environ.get("DEMOTE_DEPTH", "0") != "0" and candidates:
        from text_match import attributes_from_words, central_words
        attrs = attributes_from_words(central_words(words, min_conf, band=0.5))
        if attrs.color and attrs.sweetness:
            depth = len(candidates)
    demoted = False
    for _ in range(depth):
        leader = candidates[0]
        leader.hard = channel.hard_conflicts(leader.slug, words, min_conf)
        if not leader.hard:
            break
        nxt = next((i for i, c in enumerate(candidates[1:], 1)
                    if not channel.hard_conflicts(c.slug, words, min_conf)), None)
        if nxt is None:
            return
        candidates.insert(nxt, candidates.pop(0))
        demoted = True
    # После отвода геометрия решает заново среди оставшихся (DEMOTE_GEOMETRY).
    # Порядок по косинусу держался потому, что геометрия не была решающей
    # на всём шортлисте; когда противоречащие выбыли, среди оставшихся она
    # может быть решающей: у «Victor Dravigny. Полусладкое» 47 точек против
    # 5 у «Красного полусладкого», а отвод ставил первым просто следующего.
    if demoted and os.environ.get("DEMOTE_GEOMETRY", "0") != "0":
        clean_ones = [c for c in candidates if not channel.hard_conflicts(c.slug, words, min_conf)]
        if len(clean_ones) >= 2:
            ranked = sorted(clean_ones, key=lambda c: (-c.inliers, -c.cv))
            if ranked[0].inliers >= ranked[1].inliers * (1.0 + margin) and ranked[0] is not candidates[0]:
                candidates.remove(ranked[0])
                candidates.insert(0, ranked[0])


# Голосование словами этикетки (TEXT_VOTE=θ, пусто — выключено): лидером
# становится кандидат, в карточке которого нашлось строго больше
# прочитанных слов, чем у текущего лидера, и не меньше VOTE_MIN_WORDS.
# θ — допуск на ошибку чтения, расстояние Дамерау–Левенштейна на длину слова.
VOTE_MIN_WORDS = 2


def vote_by_text(candidates: list[Candidate], words, channel, min_conf: float) -> None:
    """Порог θ из TEXT_VOTE; без него шаг ничего не делает.

    Разбор, откуда правило: «Портвейн белый Алушта» Массандры. Гравюра дворца,
    шапка и год основания у новых этикеток линейки общие, и геометрия отдаёт
    121 точку «Гурзуфу» против 59 у «Алушты». OCR при этом читает ПОРТВЕЙН,
    БЕАЫЙ и АУШТА, но прежние правила слово берут только точным и только
    если оно принадлежит одному кандидату, а «Алушта» есть у белого и
    красного портвейна. Различает их лишь сочетание слов.
    """
    theta = os.environ.get("TEXT_VOTE", "")
    if not theta or len(candidates) < 2:
        return
    from text_match import text_votes
    votes = text_votes([c.slug for c in candidates], words, channel, float(theta), min_conf)
    best = max(votes.values())
    winners = [c for c in candidates if votes[c.slug] == best]
    leader = candidates[0]
    if (len(winners) == 1 and winners[0] is not leader and best >= VOTE_MIN_WORDS
            and best > votes[leader.slug]):
        candidates.remove(winners[0])
        candidates.insert(0, winners[0])


CONTRA_THETA = 0.2      # допуск на ошибку чтения, как у TEXT_VOTE
CONTRA_CONF = 0.85      # слово должно быть прочитано уверенно
CONTRA_LEN = 5


def text_contradiction(candidates: list[Candidate], words, channel) -> list[str]:
    """Кандидаты, чьё слово уверенно прочитано на этикетке, а у лидера его нет.

    Само по себе ответа не меняет — это повод спросить арбитра (VLM_CONTRA
    в сервисе). «Портвейн белый Алушта»: прочитано АУШТА с уверенностью
    0.96, у лидера «Гурзуфа» такого слова нет, у двух «Алушт» — есть.
    Порядок — порядок кандидатов.
    """
    from text_match import (GENERIC_WORDS, STOPWORDS, card_vocabulary, central_words,
                            clean, near)
    if len(candidates) < 2:
        return []
    read = {t for w in central_words(words, 0.60) if w.conf >= CONTRA_CONF
            for t in clean(w.text).split()
            if len(t) >= CONTRA_LEN and t not in GENERIC_WORDS and t not in STOPWORDS}
    leader = card_vocabulary(candidates[0].slug, channel)
    foreign = [t for t in read if not any(near(t, v, CONTRA_THETA) for v in leader)]
    if not foreign:
        return []
    return [c.slug for c in candidates[1:]
            if any(near(t, v, CONTRA_THETA) for t in foreign
                   for v in card_vocabulary(c.slug, channel))]


def settle(candidates: list[Candidate], crop: Image.Image, channel,
           read_words, window: float, min_conf: float,
           by_slug: dict | None = None) -> list[Candidate]:
    """Окончательный выбор вина: перестановка по тексту и отвод лидера.

    Отделено от check_leader намеренно. Раньше отвод лидера по прочитанному
    противоречию жил внутри проверки, а проверка звалась только если
    предварительная уверенность выше CONFLICT_GATE. На 5881591_0.jpg
    уверенность была 0.036, проверка пропускалась, и в оценочное API
    уходило белое брют при «КРАСНОЕ» на этикетке — хотя сама проверка,
    если её выполнить, отвечает правильно.

    Порог уверенности решает, показывать ли ответ, и не должен решать,
    какой это ответ. Поэтому здесь — всё, что меняет ответ, без порогов;
    в check_leader — всё, что меняет только оформление выдачи.
    """
    words = read_words(crop)
    prefer_brand(candidates, words, channel, min_conf)
    resolve(candidates, crop, channel, read_words, window, min_conf, by_slug)
    vote_by_text(candidates, words, channel, min_conf)
    if candidates:
        demote_contradicted(candidates, channel, read_words(crop), min_conf)
    return candidates


def check_leader(candidates: list[Candidate], crop: Image.Image, channel,
                 read_words, min_conf: float, window: float = 0.80,
                 by_slug: dict | None = None) -> int:
    """Улики против лидера и подтверждения за него. Возвращает число улик.

    Меняет только оформление выдачи: улика снижает уверенность и запрещает
    снять оговорку, подтверждение разрешает. Сам ответ к этому моменту уже
    выбран в settle.

    Кроме улик собираются подтверждения — слова, отличающие лидера от
    соперников. Они нужны решению о выдаче: молчание текста согласием
    не является.
    """
    if not candidates:
        return 0
    words = read_words(crop)
    leader = candidates[0]
    leader.conflicts = channel.conflicts_with(
        leader.slug, words, [c.slug for c in candidates[1:]], min_conf)
    # Подтверждения сверяются только с близкими соперниками. По всему
    # шортлисту «brule» выглядит различителем — там есть вина других
    # виноделен, — и имя серии засчитывалось за подтверждение лидера
    # против его же однофамильцев.
    limit = leader.inliers * window
    close = {c.slug for c in candidates[1:] if c.inliers >= limit}
    if by_slug is not None:
        close |= {c.slug for c in series_rivals(candidates, by_slug)}
    leader.confirmed = (channel.confirms(leader.slug, words, sorted(close), min_conf)
                        if close else [])
    return len(leader.conflicts)


def probability(feats: dict[str, float], weights: dict[str, float]) -> float:
    """Уверенность в лидере. Совместимость со старой двоичной моделью."""
    logit = weights.get("bias", 0.0) + sum(weights.get(name, 0.0) * feats[name]
                                           for name in FEATURE_ORDER)
    return 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))


def outcomes(feats: dict[str, float], weights: list[list[float]]) -> tuple[float, float]:
    """Вероятности «лидер верен» и «верный ответ есть в пятёрке».

    Модель одна на три взаимоисключающих исхода: первое место, места со
    второго по пятое, вне пятёрки. Поэтому top-5 не бывает ниже top-1 —
    это сумма первых двух вероятностей, а не отдельно обученное число.
    """
    row = [feats[name] for name in FEATURE_ORDER] + [1.0]
    logits = [sum(value * column[i] for value, column in zip(row, weights))
              for i in range(3)]
    top = max(logits)
    exponent = [math.exp(min(x - top, 30.0)) for x in logits]
    total = sum(exponent) or 1.0
    first, middle = exponent[0] / total, exponent[1] / total
    return first, first + middle


# Уверенность в пятёрке считается отдельной моделью на тех же признаках
# (веса CONF_WEIGHTS_TOP5, обучение в pipeline/calibrate.py). Прежний
# вариант перемножал вероятности по местам, будто они независимы, — а они
# связаны жёстко: если верен первый, остальные неверны по определению,
# и произведение давало не вероятность, а произвольное число.


def states(probability_top1: float, confident_p: float, show_p: float) -> str:
    """Состояние выдачи. Одно место, чтобы сервис и отчёт не разошлись."""
    if probability_top1 < show_p:
        return "unsure"
    return "confident" if probability_top1 >= confident_p else "uncertain"
