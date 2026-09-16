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
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

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
    contradictions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)

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
    for j in np.argsort(-scores)[:depth]:
        slug = index_slugs[j]
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        out.append(Candidate(slug=slug, cv=float(scores[j])))
        if len(out) == topk:
            break
    return out


def rerank(crop: Image.Image, candidates: list[Candidate], reranker) -> list[Candidate]:
    """Геометрическая проверка: сколько точек этикетки совпало и где."""
    stats = reranker.stats(crop, [c.slug for c in candidates])
    for candidate in candidates:
        match = stats.get(candidate.slug)
        if match is not None:
            candidate.inliers = int(match.inliers)
            candidate.coverage = float(match.coverage)
    candidates.sort(key=lambda c: (-c.inliers, -c.cv))
    return candidates


def has_close_group(candidates: list[Candidate], window: float) -> bool:
    """Развела ли геометрия лидеров. Если нет — есть смысл читать этикетку."""
    if len(candidates) < 2 or candidates[0].inliers <= 0:
        return False
    limit = candidates[0].inliers * window
    return sum(1 for c in candidates if c.inliers >= limit) >= 2


def resolve(candidates: list[Candidate], crop: Image.Image, channel,
            read_words, window: float, min_conf: float) -> list[Candidate]:
    """Перестановка близких кандидатов по прочитанному тексту.

    OCR считается только когда группа близких кандидатов есть: при ясном
    лидере текст ничего не решит, а время ответа вырастет на сотню
    миллисекунд.
    """
    from text_match import resolve_close

    if not has_close_group(candidates, window):
        return candidates
    pairs = [(c.slug, float(c.inliers)) for c in candidates]
    ranked, reasons = resolve_close(pairs, read_words(crop), channel,
                                    window=window, min_conf=min_conf)
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
    if not brand or leader.inliers <= 0:
        return []
    limit = leader.inliers * share
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
    # Прямой соперник — лучший из оставшихся по геометрии: именно он занял бы
    # место лидера, если бы того не было.
    rival = max(rivals, key=lambda c: (c.inliers, c.cv), default=None)
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


def check_leader(candidates: list[Candidate], crop: Image.Image, channel,
                 read_words, min_conf: float, window: float = 0.80,
                 by_slug: dict | None = None) -> int:
    """Улики против лидера и подтверждения за него. Возвращает число улик.

    Вызывается только когда ответ иначе был бы показан: OCR стоит около
    сотни миллисекунд, и тратить их на заведомо слабое совпадение незачем.

    Кроме улик собираются подтверждения — слова, отличающие лидера от
    соперников. Они нужны решению о выдаче: молчание текста согласием
    не является.
    """
    if not candidates:
        return 0
    leader = candidates[0]
    words = read_words(crop)
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
