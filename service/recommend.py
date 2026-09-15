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

import re


from dataclasses import dataclass

# Слова категории лежат в slug: "...-beloe-polusuhoe-125".
SWEETNESS_TOKENS = [
    ("ekstra-bryut", "экстра брют"), ("ekstra-brut", "экстра брют"),
    ("polusuhoe", "полусухое"), ("polusladkoe", "полусладкое"),
    ("bryut", "брют"), ("brut", "брют"),
    ("suhoe", "сухое"), ("sladkoe", "сладкое"),
]
SPARKLING = {"брют", "экстра брют"}

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


def sweetness_of(slug: str) -> str | None:
    for token, value in SWEETNESS_TOKENS:
        if token in slug:
            return value
    return None


def facets_of(wine: dict) -> Facets:
    return Facets(
        sweetness=sweetness_of(wine["slug"]),
        color=(wine.get("category") or "").strip().lower(),
        grapes=frozenset(g.strip().lower() for g in wine.get("grapes") or [])
        - GENERIC_GRAPES,
        region=(wine.get("region") or "").strip().lower(),
        manufacturer=(wine.get("manufacturer") or "").strip().lower(),
        rating=float(wine.get("rating") or 0.0),
    )


class Recommender:
    def __init__(self, wines: list[dict]) -> None:
        self.wines = wines
        self.facets = {w["slug"]: facets_of(w) for w in wines}

    # --- сомелье -------------------------------------------------------

    def sommelier(self, answers: dict, limit: int = 6) -> list[dict]:
        dish = (answers.get("dish") or "").strip()
        taste = (answers.get("taste") or "").strip().lower()
        occasion = (answers.get("occasion") or "").strip()

        scored = []
        for wine in self.wines:
            facet = self.facets[wine["slug"]]
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

            if occasion == "party" and facet.sweetness in SPARKLING:
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

        scored.sort(key=lambda item: -item[0])
        return [{"wine": wine, "reasons": reasons, "score": round(score, 2)}
                for score, wine, reasons in scored[:limit]]

    # --- аналоги -------------------------------------------------------

    def similar(self, slug: str, limit: int = 6) -> list[dict]:
        """Похожие вина других виноделен.

        Своя же винодельня исключается намеренно: пользователь уже держит
        её бутылку в руках, а ценность подсказки — в открытии нового.
        """
        source = self.facets.get(slug)
        if source is None:
            return []

        scored = []
        seen_titles: set[tuple[str, str]] = set()
        for wine in self.wines:
            other = self.facets[wine["slug"]]
            if wine["slug"] == slug or other.manufacturer == source.manufacturer:
                continue
            if other.color != source.color:
                continue
            # В каталоге есть позиции-тёзки одной линейки; в подборке аналогов
            # они выглядят как повтор, поэтому от серии берём одну.
            title_key = (other.manufacturer, re.sub(r"\s+", " ",
                                                    wine["title"].strip().lower()))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            reasons = [f"тоже {other.color}"] if other.color else []
            score = 0.0
            common = source.grapes & other.grapes
            if common:
                union = source.grapes | other.grapes
                score += 3.0 * len(common) / max(len(union), 1)
                reasons.append("те же сорта: " + ", ".join(sorted(common)[:2]))
            if other.sweetness and other.sweetness == source.sweetness:
                score += 1.0
                reasons.append(other.sweetness)
            if other.region == source.region:
                score += 0.5
                # Регион в фасетах приведён к нижнему регистру для сравнения,
                # а показываем его так, как он записан в каталоге.
                reasons.append(f"тот же регион — {wine.get('region') or other.region}")
            score += min(other.rating, 5.0) / 5.0
            if score <= 1.0:
                continue
            scored.append((score, wine, reasons))

        scored.sort(key=lambda item: -item[0])
        return [{"wine": wine, "reasons": reasons, "score": round(score, 2)}
                for score, wine, reasons in scored[:limit]]

    # --- гастропары найденного вина -------------------------------------

    def pairings(self, wine: dict) -> list[str]:
        return list(wine.get("dishes") or [])
