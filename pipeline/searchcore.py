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
    color: float = 1.0
    contradictions: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    confirmed: list[str] = field(default_factory=list)
    hard: list[str] = field(default_factory=list)

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


GEOMETRY_MARGIN = 0.25


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
    """
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

    По умолчанию выключено. С развёрткой медиана ответа на публичных фото
    кейса 1888 мс против 615 без неё. Качество при этом внутри шума:
    на настроечной части top-1 34 из 64 против 33 и уверенных ответов
    10 против 7 (все верны в обоих случаях), на тестовой top-1 тот же
    19 из 57, а F1 top-1 34.9% против 38.6%. Включается LABEL_BAND=1.

    Координаты строк полосы переводятся обратно в кроп: отбор по месту
    на кадре отсеивает надписи с соседней бутылки, и сравнивать он должен
    в одной системе координат.
    """
    from dataclasses import replace

    from imageprep import label_band_view
    from text_match import extract_attributes

    words = list(read_words(crop))
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
                        min_conf: float, depth: int = 4) -> None:
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
    for _ in range(depth):
        leader = candidates[0]
        leader.hard = channel.hard_conflicts(leader.slug, words, min_conf)
        if not leader.hard:
            return
        nxt = next((i for i, c in enumerate(candidates[1:], 1)
                    if not channel.hard_conflicts(c.slug, words, min_conf)), None)
        if nxt is None:
            return
        candidates.insert(nxt, candidates.pop(0))


def settle(candidates: list[Candidate], crop: Image.Image, channel,
           read_words, window: float, min_conf: float) -> list[Candidate]:
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
    resolve(candidates, crop, channel, read_words, window, min_conf)
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
