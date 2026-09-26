#!/usr/bin/env python3
"""S3.3 + S3.5 — что предложить пользователю после того, как вино найдено.

Две функции удержания из ТЗ:
  «Цифровой сомелье» — несколько наводящих вопросов и подбор под ответы;
  аналоги — похожие вина других виноделен, если нужного нет или хочется
  альтернативу.

Подбор детерминированный, по атрибутам каталога, и объясняет сам себя:
к каждой рекомендации прилагается причина. Внешняя языковая модель здесь
не нужна — сама по себе она не знает каталог, а отвечать «из головы»
в сервисе поиска по каталогу недопустимо. Место для неё оставлено:
LLM может переформулировать вопросы и причины, но выбор позиций
остаётся за каталогом.
"""
from __future__ import annotations

import difflib
import re
import zlib
from dataclasses import dataclass

# Слова категории лежат в slug: "...-beloe-polusuhoe-125".
SWEETNESS_TOKENS = [
    ("ekstra-bryut", "экстра брют"), ("ekstra-brut", "экстра брют"),
    ("polusuhoe", "полусухое"), ("polusladkoe", "полусладкое"),
    ("bryut", "брют"), ("brut", "брют"),
    ("suhoe", "сухое"), ("sladkoe", "сладкое"),
]
# Те же категории словами в названии: «Белая Львица белое полусладкое».
# Целыми словами и длинные раньше коротких, иначе «полусладкое»
# прочиталось бы как «сладкое».
SWEETNESS_WORDS = [
    ("экстра брют", "экстра брют"), ("полусухое", "полусухое"),
    ("полусладкое", "полусладкое"), ("брют", "брют"),
    ("сухое", "сухое"), ("сладкое", "сладкое"),
]
SPARKLING = {"брют", "экстра брют"}
# Игристое бывает и сухим, и полусладким, так что по одной сладости его
# не узнать. Признаки берутся из названия и slug.
SPARKLING_MARKERS = (
    "игрист", "шампан", "петнат", "пет нат", "pet nat", "frizzante",
    "фриззанте", "фризанте", "spumante", "спуманте", "method classic",
    "традиционн", "cremant", "креман",
)

# Не больше двух вин одной винодельни в подборке: ценна разнообразием.
MAX_PER_MAKER = 2
# С какого сходства названия вино винодельни помечается «похоже по названию».
NAME_CLOSE = 0.6

# В выгрузке вместо сорта иногда стоит заглушка. Считать её признаком
# сходства нельзя: по ней «похожими» становятся все ассамбляжи подряд.
GENERIC_GRAPES = {"белые сорта винограда", "красные сорта винограда"}

# Вопросы сомелье. Порядок важен: сначала повод, он сужает сильнее всего.
QUESTIONS = [
    {
        "id": "occasion",
        "title": "Что за случай?",
        "options": [
            {"value": "dinner", "label": "Ужин дома"},
            {"value": "party", "label": "Праздник"},
            {"value": "gift", "label": "В подарок"},
            {"value": "solo", "label": "Просто попробовать"},
        ],
    },
    {
        "id": "dish",
        "title": "К чему подбираете?",
        "options": [
            {"value": "Мясо и стейки", "label": "Мясо"},
            {"value": "Рыба и морепродукты", "label": "Рыба"},
            {"value": "Сыры", "label": "Сыры"},
            {"value": "Легкие закуски", "label": "Закуски"},
            {"value": "Десерты", "label": "Десерт"},
            {"value": "", "label": "Без еды"},
        ],
    },
    {
        "id": "taste",
        "title": "Какой вкус ближе?",
        "options": [
            {"value": "сухое", "label": "Сухое"},
            {"value": "полусухое", "label": "Полусухое"},
            {"value": "полусладкое", "label": "Полусладкое"},
            {"value": "брют", "label": "Игристое брют"},
            {"value": "", "label": "Не знаю"},
        ],
    },
]


@dataclass
class Facets:
    sweetness: str | None
    color: str
    grapes: frozenset[str]
    region: str
    manufacturer: str
    rating: float
    # None — неизвестно (так бывает у этикетки, прочитанной VLM).
    sparkling: bool | None = False


def sweetness_of(slug: str, title: str = "") -> str | None:
    for token, value in SWEETNESS_TOKENS:
        if token in slug:
            return value
    words = " ".join(re.findall(r"[а-яё]+", title.lower()))
    for phrase, value in SWEETNESS_WORDS:
        if re.search(rf"\b{phrase}\b", words):
            return value
    return None


def is_sparkling(wine: dict, sweetness: str | None) -> bool:
    if sweetness in SPARKLING:
        return True
    text = f"{wine.get('title') or ''} {wine['slug'].replace('-', ' ')}".lower()
    return any(marker in text for marker in SPARKLING_MARKERS)


def facets_of(wine: dict, sweetness: str | None = None) -> Facets:
    """Фасеты позиции; sweetness — уже известная сладость (из TextChannel)."""
    sweet = sweetness or sweetness_of(wine["slug"], wine.get("title") or "")
    return Facets(
        sweetness=sweet,
        color=(wine.get("category") or "").strip().lower(),
        grapes=frozenset(g.strip().lower() for g in wine.get("grapes") or [])
        - GENERIC_GRAPES,
        region=(wine.get("region") or "").strip().lower(),
        manufacturer=(wine.get("manufacturer") or "").strip().lower(),
        rating=float(wine.get("rating") or 0.0),
        sparkling=is_sparkling(wine, sweet),
    )


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower().replace("ё", "е"))


def name_closeness(name: str, title: str) -> float:
    """Сходство прочитанного названия с названием позиции, от 0 до 1:
    доля слов названия, найденных в заголовке, или посимвольная близость."""
    key = fold(re.sub(r"[^\w\s]", " ", name or ""))
    if not key:
        return 0.0
    other = fold(re.sub(r"[^\w\s]", " ", title or ""))
    words, own = set(key.split()), set(other.split())
    overlap = len(words & own) / max(len(words), 1)
    return max(overlap, difflib.SequenceMatcher(None, key, other).ratio())


def closest(value: str, vocabulary: dict[str, str], cutoff: float) -> str | None:
    """Написание из каталога для прочитанного значения, или None."""
    key = fold(value)
    if not key:
        return None
    if key in vocabulary:
        return vocabulary[key]
    near = difflib.get_close_matches(key, list(vocabulary), n=1, cutoff=cutoff)
    return vocabulary[near[0]] if near else None


class Recommender:
    def __init__(self, wines: list[dict],
                 sweetness: dict[str, str | None] | None = None) -> None:
        """sweetness — сладость по slug из TextChannel распознавания: там она
        собрана из slug, карточек платформы и проверенных этикеток, и
        аналоги с распознаванием так не расходятся в том, что считать
        полусладким. Где её нет, берётся slug и название.
        """
        self.wines = wines
        known = sweetness or {}
        self.facets = {w["slug"]: facets_of(w, known.get(w["slug"])) for w in wines}
        # Словари для сопоставления прочитанного с этикетки с каталогом.
        self.grape_names = {fold(g): g for f in self.facets.values() for g in f.grapes}
        self.regions = {fold(f.region): f.region for f in self.facets.values() if f.region}
        self.manufacturers = {fold(f.manufacturer): f.manufacturer
                              for f in self.facets.values() if f.manufacturer}

    def style(self, slug: str) -> dict:
        facet = self.facets.get(slug)
        if facet is None:
            return {}
        return {"sweetness": facet.sweetness, "sparkling": facet.sparkling}

    # --- сомелье -------------------------------------------------------

    def sommelier(self, answers: dict, limit: int = 6) -> list[dict]:
        """Подбор под ответы; like — slug найденного вина, если сомелье
        спрашивают после сканирования: тогда в плюсе вина того же цвета,
        а само найденное в подборку не входит."""
        dish = (answers.get("dish") or "").strip()
        taste = (answers.get("taste") or "").strip().lower()
        occasion = (answers.get("occasion") or "").strip()
        like_slug = (answers.get("like") or "").strip()
        like = self.facets.get(like_slug)

        scored = []
        for wine in self.wines:
            facet = self.facets[wine["slug"]]
            if wine["slug"] == like_slug:
                continue
            score, reasons = 0.0, []

            if dish:
                if dish in (wine.get("dishes") or []):
                    score += 3.0
                    reasons.append(f"подходит к блюду «{dish.lower()}»")
                else:
                    continue          # к чему подать — самый твёрдый критерий

            if taste:
                if facet.sweetness == taste:
                    score += 2.0
                    reasons.append(f"{taste}, как вы просили")
                else:
                    continue

            if like is not None and like.color and facet.color == like.color:
                score += 0.5
                reasons.append(f"тоже {facet.color}, как найденное")

            if occasion == "party" and facet.sparkling:
                score += 1.5
                reasons.append("игристое — для праздника")
            elif occasion == "gift" and facet.rating >= 4.8:
                score += 1.5
                reasons.append("высокий рейтинг платформы")
            elif occasion == "solo" and facet.rating >= 4.5:
                score += 0.8
                reasons.append("хорошая оценка, чтобы познакомиться")

            score += min(facet.rating, 5.0) / 5.0
            if not reasons and facet.rating:
                reasons.append(f"оценка {facet.rating:g} на платформе")
            scored.append((score, wine, reasons))

        salt = "|".join(f"{k}={answers.get(k) or ''}" for k in sorted(answers))
        return self.pick(scored, limit, salt)

    def pick(self, scored: list[tuple[float, dict, list[str]]], limit: int,
             salt: str) -> list[dict]:
        """Лучшие по счёту, не больше MAX_PER_MAKER вин одной винодельни.

        При равном счёте порядок каталога — это алфавит, и первыми всегда
        шли бы «100 оттенков…» и «A.G. Poezdnik». Ничьи разбиваются
        нейтрально, но воспроизводимо: хешем slug с затравкой от запроса.
        """
        scored = sorted(scored, key=lambda item: (-item[0], zlib.crc32(
            f"{salt}|{item[1]['slug']}".encode())))
        picked, per_maker = [], {}
        for score, wine, reasons in scored:
            maker = self.facets[wine["slug"]].manufacturer
            if per_maker.get(maker, 0) >= MAX_PER_MAKER:
                continue
            per_maker[maker] = per_maker.get(maker, 0) + 1
            picked.append({"wine": wine, "reasons": reasons, "score": round(score, 2)})
            if len(picked) == limit:
                break
        return picked

    # --- аналоги -------------------------------------------------------

    def similar(self, slug: str, limit: int = 6) -> list[dict]:
        """Похожие вина других виноделен.

        Своя же винодельня исключается намеренно: пользователь уже держит
        её бутылку в руках, а ценность подсказки — в открытии нового.
        """
        source = self.facets.get(slug)
        if source is None:
            return []
        return self.analogs(source, exclude=slug, limit=limit)

    def label_facets(self, label: dict) -> Facets:
        """Фасеты вина, которого нет в каталоге, по прочитанной этикетке.

        Сорта, регион и винодельня приводятся к написанию каталога: иначе
        «Каберне-Совиньон» с этикетки не совпал бы с «Каберне Совиньон».
        Не узнанное каталогом просто не участвует в сравнении.
        """
        grapes = {closest(g, self.grape_names, 0.85) for g in label.get("grapes") or []}
        return Facets(
            sweetness=label.get("sweetness"),
            color=label.get("color") or "",
            grapes=frozenset(g for g in grapes if g) - GENERIC_GRAPES,
            region=closest(label.get("region") or "", self.regions, 0.8) or "",
            manufacturer=self.maker_of(label),
            rating=0.0,
            sparkling=label.get("sparkling"),
        )

    def maker_of(self, label: dict) -> str:
        """Винодельня каталога по этикетке: латиницей или кириллицей."""
        for key in ("producer_ru", "producer"):
            found = closest(label.get(key) or "", self.manufacturers, 0.85)
            if found:
                return found
        return ""

    def from_maker(self, maker: str, color: str = "", name: str = "",
                   limit: int = 6) -> list[dict]:
        """Вина винодельни с этикетки, ближайшие по названию — первыми.

        Поиск не нашёл вино уверенно, но винодельню VLM читает почти всегда
        (30 из 31 снимка организатора с известным вином). Если вино в
        каталоге всё же есть, сходство названия ставит его наверх; если нет —
        остаются другие вина той же винодельни. Ответ поиска это не меняет:
        список только показывается.
        """
        own = [(round(name_closeness(name, w["title"]), 1), w) for w in self.wines
               if self.facets[w["slug"]].manufacturer == maker]
        # Слабое сходство названия («Олег» и «Розе») не должно перебивать цвет.
        own.sort(key=lambda item: (-(item[0] if item[0] >= NAME_CLOSE else 0.0),
                                   self.facets[item[1]["slug"]].color != color,
                                   -self.facets[item[1]["slug"]].rating))
        return [{"wine": w, "reasons": ["похоже по названию"] if close >= NAME_CLOSE else []}
                for close, w in own[:limit]]

    def by_label_name(self, label: dict) -> tuple[str | None, float]:
        """Позиция каталога по винодельне и названию с этикетки.

        Среди вин прочитанной винодельни, а если её в каталоге не нашлось —
        по всему каталогу. Возвращает slug и сходство названия; решать,
        достаточно ли этого, должен независимый проверяющий (VLM).
        """
        name = label.get("name") or ""
        if not name:
            return None, 0.0
        maker = self.maker_of(label)
        pool = [w for w in self.wines
                if self.facets[w["slug"]].manufacturer == maker] if maker else self.wines
        best = max(pool, key=lambda w: name_closeness(name, w["title"]), default=None)
        if best is None:
            return None, 0.0
        return best["slug"], name_closeness(name, best["title"])

    def analogs(self, source: Facets, exclude: str | None = None,
                limit: int = 6, seed: str = "") -> list[dict]:
        scored = []
        seen_titles: set[tuple[str, str]] = set()
        for wine in self.wines:
            other = self.facets[wine["slug"]]
            if wine["slug"] == exclude:
                continue
            if source.manufacturer and other.manufacturer == source.manufacturer:
                continue
            if source.color and other.color != source.color:
                continue
            # Аналог совпадает по стилю: полусладкое не заменяют брютом,
            # а тихое вино — игристым. Где стиль неизвестен, фильтр молчит.
            if source.sparkling is not None and other.sparkling != source.sparkling:
                continue
            if source.sweetness and other.sweetness and other.sweetness != source.sweetness:
                continue
            # В каталоге есть позиции-тёзки одной линейки и винтажи одного
            # вина; в подборке аналогов они выглядят как повтор, поэтому от
            # серии берём одну.
            title_key = (other.manufacturer, fold(re.sub(
                r"\b(19|20)\d\d\b|[^\w\s]", " ", wine["title"])))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            reasons = [f"тоже {other.color}"] if source.color and other.color else []
            score = 0.0
            common = source.grapes & other.grapes
            if common:
                union = source.grapes | other.grapes
                score += 3.0 * len(common) / max(len(union), 1)
                reasons.append("те же сорта: " + ", ".join(sorted(common)[:2]))
            if other.sweetness and other.sweetness == source.sweetness:
                score += 1.0
                reasons.append(other.sweetness)
            elif source.sparkling and other.sparkling:
                reasons.append("тоже игристое")
            if source.region and other.region == source.region:
                score += 0.5
                # Регион в фасетах приведён к нижнему регистру для сравнения,
                # а показываем его так, как он записан в каталоге.
                reasons.append(f"тот же регион — {wine.get('region') or other.region}")
            score += min(other.rating, 5.0) / 5.0
            if score <= 1.0:
                continue
            scored.append((score, wine, reasons))

        return self.pick(scored, limit, seed or exclude or "")

    # --- гастропары найденного вина -------------------------------------

    def pairings(self, wine: dict) -> list[str]:
        return list(wine.get("dishes") or [])
