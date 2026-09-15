#!/usr/bin/env python3
"""S2.2 + S2.3 — текстовый канал поиска и слияние с визуальным.

Зрение находит серию: бутылку той же винодельни, того же силуэта и макета.
Внутри серии позиции различаются словами на этикетке — год, категория,
цвет, сорт, — и развести их может только текст. Отсюда разделение ролей:
CV даёт кандидатов, текст выбирает среди них.

Текстовый матчер переиспользует ядро прототипа (core/matcher.py): те же
веса полей, IDF и нечёткое сравнение токенов, устойчивое к ошибкам OCR.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from matcher import Matcher, tokens_match  # noqa: E402
from normalize import clean  # noqa: E402

CATALOG = ROOT / "data" / "catalog" / "catalog.json"

# Категория и цвет пишутся на контрэтикетке почти всегда — это самые
# надёжные различители внутри серии одной винодельни.
SWEETNESS = {
    "экстра брют": "экстра брют", "extra brut": "экстра брют",
    "брют": "брют", "brut": "брют",
    "полусухое": "полусухое", "полусладкое": "полусладкое",
    "сладкое": "сладкое", "сухое": "сухое",
}
COLORS = {
    "белое": "белое", "белый": "белое", "blanc": "белое", "bianco": "белое",
    "красное": "красное", "красный": "красное", "rouge": "красное", "rosso": "красное",
    "розовое": "розовое", "rose": "розовое", "роза": "розовое",
    "оранжевое": "оранжевое", "orange": "оранжевое",
}


@dataclass
class Attributes:
    """Что удалось прочитать с этикетки помимо названия."""
    years: set[str] = field(default_factory=set)
    sweetness: str | None = None
    color: str | None = None
    alcohol: float | None = None


def extract_attributes(text: str) -> Attributes:
    low = clean(text)
    attrs = Attributes()
    attrs.years = set(re.findall(r"\b(19[89]\d|20[0-3]\d)\b", low))

    # Длинные варианты идут первыми: «экстра брют» не должен схлопнуться в «брют»
    for phrase, value in SWEETNESS.items():
        if phrase in low:
            attrs.sweetness = value
            break
    for word, value in COLORS.items():
        if re.search(rf"\b{word}\b", low):
            attrs.color = value
            break

    alc = re.search(r"\b(\d{1,2})[.,](\d)\s*%", text) or re.search(r"\b(\d{1,2})\s*%", text)
    if alc:
        try:
            attrs.alcohol = float(alc.group(0).replace("%", "").replace(",", ".").strip())
        except ValueError:
            attrs.alcohol = None
    return attrs


# Категория сладости и крепость в выгрузке отдельными полями не приехали,
# но они зашиты в slug: "...-beloe-polusuhoe-125" — белое полусухое 12.5%.
SLUG_SWEETNESS = [
    ("ekstra-bryut", "экстра брют"), ("ekstra-brut", "экстра брют"),
    ("polusuhoe", "полусухое"), ("polusladkoe", "полусладкое"),
    ("bryut", "брют"), ("brut", "брют"),
    ("suhoe", "сухое"), ("sladkoe", "сладкое"),
]
SLUG_COLORS = [("beloe", "белое"), ("krasnoe", "красное"),
               ("rozovoe", "розовое"), ("oranzhevoe", "оранжевое")]


def attributes_from_slug(slug: str, category: str) -> Attributes:
    attrs = Attributes()
    for token, value in SLUG_SWEETNESS:
        if token in slug:
            attrs.sweetness = value
            break
    # Цвет надёжнее брать из поля «Категория»: в slug он может отсутствовать,
    # а поле «Цвет» в выгрузке описательное («Светло-соломенный»).
    attrs.color = clean(category) or next(
        (v for t, v in SLUG_COLORS if t in slug), None)
    tail = re.search(r"-(\d{2,3})$", slug)
    if tail:
        value = int(tail.group(1))
        if len(tail.group(1)) == 3 and 90 <= value <= 200:
            attrs.alcohol = value / 10
        elif len(tail.group(1)) == 2 and 9 <= value <= 20:
            attrs.alcohol = float(value)
    return attrs


class TextChannel:
    """Оценка позиций каталога по распознанному тексту этикетки."""

    def __init__(self, wines: list[dict]):
        self.wines = wines
        self.matcher = Matcher(wines)
        self.by_slug = {w["slug"]: w for w in wines}
        self.attributes_of = {
            w["slug"]: attributes_from_slug(w["slug"], w.get("category") or "")
            for w in wines
        }

    @classmethod
    def from_catalog(cls, path: Path = CATALOG) -> "TextChannel":
        return cls(json.loads(path.read_text(encoding="utf-8"))["wines"])

    def scores(self, text: str) -> dict[str, float]:
        if not text.strip():
            return {}
        found = self.matcher.search(text, top=len(self.wines))
        return {r["slug"]: r["confidence"] for r in found}

    def attribute_score(self, slug: str, attrs: Attributes) -> float:
        """Согласие прочитанных атрибутов с карточкой, от -1 до +1.

        Ключевое отличие от простого бонуса — штраф за противоречие. Внутри
        серии одной винодельни зрение бессильно: позиции различаются только
        словами «сухое» против «полусухое» и крепостью. Поощрять совпадение
        мало, нужно уметь опустить кандидата, который прочитанному противоречит.
        """
        card = self.attributes_of.get(slug)
        if card is None:
            return 0.0
        votes = []
        if attrs.sweetness and card.sweetness:
            votes.append(1.0 if attrs.sweetness == card.sweetness else -1.0)
        if attrs.color and card.color:
            votes.append(1.0 if attrs.color == card.color else -1.0)
        if attrs.alcohol and card.alcohol:
            votes.append(1.0 if abs(attrs.alcohol - card.alcohol) < 0.35 else -0.5)
        if attrs.years:
            title = clean(self.by_slug[slug].get("title") or "")
            if any(y in title for y in attrs.years):
                votes.append(1.0)
        return sum(votes) / len(votes) if votes else 0.0


    def contradictions(self, slug: str, attrs: Attributes) -> list[str]:
        """Прочитанные атрибуты, которым карточка прямо противоречит.

        Отличие от attribute_score — здесь только противоречия, без наград.
        Награда за совпадение поднимает кандидата, у которого совпало
        случайное слово, а противоречие — факт: на этикетке написано
        «розовое», а в карточке белое вино, значит это не оно.
        """
        card = self.attributes_of.get(slug)
        if card is None:
            return []
        out = []
        if attrs.sweetness and card.sweetness and attrs.sweetness != card.sweetness:
            out.append(f"сладость {attrs.sweetness} против {card.sweetness}")
        if attrs.color and card.color and attrs.color not in card.color:
            out.append(f"цвет {attrs.color} против «{card.color}»")
        if attrs.years:
            title = clean(self.by_slug[slug].get("title") or "")
            in_title = re.findall(r"\b(19[89]\d|20[0-3]\d)\b", title)
            # Противоречие только если в названии год есть и он другой:
            # у большинства позиций года в названии нет вовсе.
            if in_title and not (attrs.years & set(in_title)):
                out.append(f"год {sorted(attrs.years)} против {in_title}")
        return out


def confident_attributes(words, min_conf: float = 0.60) -> Attributes:
    """Атрибуты из уверенно прочитанных строк.

    Порог выше рабочего порога OCR намеренно: неуверенное чтение не должно
    ничего решать. Ошибочно прочитанное «розовое» вместо «белое» опустило бы
    правильный ответ, а пропуск слова всего лишь оставляет решение зрению.
    """
    text = " ".join(w.text for w in words if w.conf >= min_conf)
    return extract_attributes(text)


# Слова, которые стоят на каждой второй этикетке и потому ничего не различают.
# Цвет и сладость сюда не входят намеренно: у «Мускателя белого» и «Мускателя
# розового» это единственное отличие. Общие для группы слова и так вычитаются
# в discriminating(), а список нужен только против мусора вроде «вино России».
STOPWORDS = {"вино", "wine", "вина", "россия", "russia", "крым", "кубань",
             "выдержанное", "коллекционное", "натуральное", "года", "год"}

# Порог нечёткого совпадения зависит от уверенности чтения. На этикетке
# «Массандры» OCR читает БЕЛЫЙ как БЕАЫЙ с уверенностью 0.98 — шрифт
# сливает Л и А. Ошибка в одну букву при такой уверенности — это всё ещё
# уверенно прочитанное слово, а при чтении наугад одной буквы мало.
SURE_CONF = 0.85
SURE_MATCH = 0.70
VAGUE_MATCH = 0.85


def central_words(words, min_conf: float = 0.60, band: float = 0.72):
    """Уверенные строки из центральной части кропа.

    Кроп по бутылке почти всегда прихватывает край соседней: на почти-дубле
    «Велвет Сизон Рислинг» OCR уверенно читает CABERNET с соседа и уводит
    решение в чужую сторону. Отсюда отбор по координатам — именно для этого
    слова и хранят место на кадре.
    """
    picked = [w for w in words if w.conf >= min_conf]
    if not picked:
        return []
    left = min(w.box[0] for w in picked)
    right = max(w.box[2] for w in picked)
    span = (right - left) or 1
    center = (left + right) / 2
    return [w for w in picked if abs(w.center_x - center) <= span * band / 2]


def discriminating(candidates: list[str], channel: "TextChannel") -> dict[str, set[str]]:
    """Токены, которыми кандидаты различаются между собой.

    Общие слова серии («Велвет Сизон», «Фанагория») стоят у всех и решают
    ничего не могут. Различает ровно то, что есть у одних и нет у других:
    сорт винограда, год, категория. Их и ищем в прочитанном.
    """
    own = {}
    for slug in candidates:
        wine = channel.by_slug.get(slug) or {}
        words = set()
        for field_value in (wine.get("title") or "", " ".join(wine.get("grapes") or [])):
            words |= {t for t in clean(field_value).split() if len(t) >= 3}
        own[slug] = words - STOPWORDS
    shared = set.intersection(*own.values()) if own else set()
    return {slug: words - shared for slug, words in own.items()}


def resolve_close(candidates: list[tuple[str, float]], words,
                  channel: "TextChannel", window: float = 0.80,
                  min_conf: float = 0.60,
                  ) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
    """Переупорядочивает только группу близких кандидатов.

    Геометрия отвечает на вопрос «та же ли это этикетка» и обычно отвечает
    уверенно: у верного кандидата инлаеров на порядок больше. Вмешиваться
    туда текстом незачем. Работа текста — там, где геометрия сомневается:
    позиции одной серии с близким числом инлаеров, различающиеся словом.

    Правило одностороннее. Если уверенно прочитанное слово совпало с
    различающим токеном ровно одного кандидата, остальные получают
    противоречие. Совпадение никого не поднимает: слово могло попасть
    в кадр случайно, а вот его отсутствие у кандидата — это факт о нём.
    """
    if len(candidates) < 2:
        return candidates, {}
    best = candidates[0][1]
    if best <= 0:
        return candidates, {}
    group = [c for c in candidates if c[1] >= best * window]
    if len(group) < 2:
        return candidates, {}

    marks = discriminating([slug for slug, _ in group], channel)
    read = [(t, w.conf) for w in central_words(words, min_conf)
            for t in clean(w.text).split() if len(t) >= 3]

    found: dict[str, list[str]] = {slug: [] for slug, _ in group}
    for token, conf in read:
        need = SURE_MATCH if conf >= SURE_CONF else VAGUE_MATCH
        owners = [slug for slug, marked in marks.items()
                  if any(tokens_match(token, m) >= need for m in marked)]
        # Слово, подходящее всем или никому, ничего не различает.
        if len(owners) != 1:
            continue
        for slug in found:
            if slug != owners[0]:
                found[slug].append(f"на этикетке «{token}» — это {owners[0]}")
    if not any(found.values()):
        return candidates, found
    reordered = sorted(group, key=lambda c: (len(found[c[0]]), -c[1]))
    return reordered + candidates[len(group):], found


def fuse(cv_candidates: list[tuple[str, float]], text_scores: dict[str, float],
         attrs: Attributes | None = None, channel: TextChannel | None = None,
         weight_text: float = 0.5, top: int = 5,
         weight_attr: float = 0.0) -> list[tuple[str, float]]:
    """Слияние каналов.

    Косинус SigLIP лежит в узком диапазоне, поэтому визуальная часть
    приводится к 0..1 внутри набора кандидатов — иначе вес текста пришлось бы
    подбирать под абсолютный масштаб, разный от запроса к запросу.
    """
    if not cv_candidates:
        ranked = sorted(text_scores.items(), key=lambda kv: -kv[1])[:top]
        return ranked

    cv_values = [score for _, score in cv_candidates]
    lo, hi = min(cv_values), max(cv_values)
    span = (hi - lo) or 1.0

    fused = []
    for slug, cv_score in cv_candidates:
        cv_norm = (cv_score - lo) / span
        text = text_scores.get(slug, 0.0)
        attr = channel.attribute_score(slug, attrs) if (channel and attrs) else 0.0
        fused.append((slug, (1 - weight_text) * cv_norm
                      + weight_text * text
                      + weight_attr * attr))
    fused.sort(key=lambda kv: -kv[1])
    return fused[:top]
