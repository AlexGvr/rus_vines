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
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
from matcher import Matcher, tokens_match  # noqa: E402
from normalize import clean  # noqa: E402

CATALOG = ROOT / "data" / "catalog" / "catalog.json"
VERIFIED_ATTRIBUTES = ROOT / "data" / "datapack" / "verified_attributes.json"
# Карточки платформы из обхода: у них есть поле сладости, которого нет в CSV
# кейса. Из 1658 позиций, где сладость есть и в slug, и в карточке, они
# расходятся в 11 (брют против экстра брют и подобное); у 386 позиций slug
# сладости не содержит, и для 355 из них её знает карточка.
PLATFORM_CARDS = ROOT / "data" / "datapack" / "wines.json"
# Позиции, появившиеся на платформе после выгрузки кейса (sync_platform.py):
# в CSV их нет вовсе, и сладость знает только карточка.
PLATFORM_ADDITIONS = ROOT / "data" / "datapack" / "platform_additions.json"
USE_PLATFORM_SWEETNESS = os.environ.get("PLATFORM_SWEETNESS", "1") != "0"

# Категория и цвет пишутся на контрэтикетке почти всегда — это самые
# надёжные различители внутри серии одной винодельни.
SWEETNESS = {
    "экстра брют": "экстра брют", "extra brut": "экстра брют",
    "брют натюр": "экстра брют", "brut nature": "экстра брют",
    "брют": "брют", "brut": "брют",
    # Латиница нужна не для красоты: на этикетках серий её пишут вместо
    # кириллицы, и без неё «BRÛLÉ Rosé Demi-Sec» неотличим от «Cuvée Brut».
    "полусухое": "полусухое", "demi sec": "полусухое", "demi-sec": "полусухое",
    "semi dry": "полусухое", "semi-dry": "полусухое",
    "полусладкое": "полусладкое", "semi sweet": "полусладкое",
    "semi-sweet": "полусладкое", "demi doux": "полусладкое",
    "сладкое": "сладкое", "dolce": "сладкое", "sweet": "сладкое",
    "сухое": "сухое", "secco": "сухое",
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


def read_sweetness(low: str) -> str | None:
    """Категория сладости из строки OCR — или ничего, если чтение спорное.

    Подстрокой это искать нельзя, и проверено на живых кадрах. OCR теряет
    первую букву («ОЛУСУХОЕ»), путает П с латинской N («NOЛУСЛАДКОЕ»)
    и разрывает слово пробелом («ПОЛУ СЛАДКОЕ»). Поиск подстрокой давал
    на всех трёх «сухое» и «сладкое» — противоположность написанному,
    и дальше это шло в улику против правильной карточки.

    Правила три. Пробел после «полу» склеивается. Составные варианты
    («экстра брют», «demi sec») ищутся по границам слов, длинные раньше
    коротких. Однословные засчитываются, только если токен равен слову
    целиком: токен, который на «сладкое» лишь заканчивается, — это
    обрезанное «полусладкое», и читать его как «сладкое» хуже, чем
    не читать вовсе.
    """
    glued = re.sub(r"\bполу\s+", "полу", low)
    phrases = sorted(SWEETNESS.items(), key=lambda kv: -len(kv[0]))
    for phrase, value in phrases:
        if " " in phrase and re.search(rf"\b{re.escape(phrase)}\b", glued):
            return value
    tokens = set(glued.split())
    for phrase, value in phrases:
        if " " not in phrase and phrase in tokens:
            return value
    return None


def extract_attributes(text: str) -> Attributes:
    low = clean(text)
    # Второй проход по строке, где латиница свёрнута в кириллицу. OCR
    # сплошь и рядом отдаёт «KPACHOЕ» вместо «КРАСНОЕ» — буквы K, P, A, C, H
    # в двух алфавитах выглядят одинаково. Без свёртки цвет просто не
    # извлекался, и улика против белого кандидата не возникала: на
    # 5881591_0.jpg лидером оставалось белое брют при «красном» на этикетке.
    # Сначала исходная строка: в ней настоящая латиница — brut, demi sec,
    # rose, — которую свёртка бы испортила.
    folded = low.translate(HOMOGLYPHS)
    attrs = Attributes()
    attrs.years = set(re.findall(r"\b(19[89]\d|20[0-3]\d)\b", low))
    attrs.sweetness = read_sweetness(low) or read_sweetness(folded)
    for source in (low, folded):
        for word, value in COLORS.items():
            if re.search(rf"\b{word}\b", source):
                attrs.color = value
                break
        if attrs.color:
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
    ("extra-brut", "экстра брют"),
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

    def __init__(self, wines: list[dict], verified_attributes: Path | None = VERIFIED_ATTRIBUTES,
                 correct_sweetness: bool = True, platform_cards: Path | None = PLATFORM_CARDS):
        self.wines = wines
        self.correct_sweetness = correct_sweetness
        self.matcher = Matcher(wines)
        self.by_slug = {w["slug"]: w for w in wines}
        self.attributes_of = {
            w["slug"]: attributes_from_slug(w["slug"], w.get("category") or "")
            for w in wines
        }
        # Сладость из карточки платформы — только там, где slug её не несёт:
        # slug остаётся главным источником, карточка добирает пропуски.
        # Проверенные по этикетке факты (ниже) перекрывают и то, и другое.
        if USE_PLATFORM_SWEETNESS and platform_cards is not None and platform_cards.exists():
            cards = json.loads(platform_cards.read_text(encoding="utf-8"))["wines"]
            if PLATFORM_ADDITIONS.exists():
                cards = cards + json.loads(
                    PLATFORM_ADDITIONS.read_text(encoding="utf-8"))["wines"]
            for card in cards:
                attrs = self.attributes_of.get(card.get("slug"))
                value = clean(card.get("sweetness") or "")
                if attrs is not None and attrs.sweetness is None and value in set(SWEETNESS.values()):
                    attrs.sweetness = value
        # The CSV omits sweetness for some otherwise identical SKUs. Use only
        # manually verified reference-label facts, never guessed tasting notes.
        if verified_attributes is not None and verified_attributes.exists():
            verified = json.loads(verified_attributes.read_text(encoding="utf-8"))
            if verified.get("version") != 1:
                raise ValueError("unsupported verified-attributes version")
            for slug, values in verified["wines"].items():
                if slug not in self.attributes_of:
                    continue
                sweetness = values["sweetness"]
                if sweetness not in set(SWEETNESS.values()):
                    raise ValueError(f"invalid verified sweetness for {slug}: {sweetness}")
                self.attributes_of[slug].sweetness = sweetness

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


    def conflicts_with(self, slug: str, words, rivals: list[str],
                       min_conf: float = 0.60) -> list[str]:
        """Улики против лидера: что на этикетке противоречит его карточке.

        Отличие от resolve_close в том, когда это работает. Перестановка
        внутри группы близких кандидатов помогает, только если нужное вино
        в каталоге есть и стоит рядом. А когда его нет вовсе, сосед по серии
        побеждает без конкурента: отрыв большой, инлаеров много, и ответ
        выглядит уверенным. Именно так «BRÛLÉ Rosé Demi-Sec» выдавался за
        «Cuvée Brut» с уверенностью 0.85.

        Здесь лидер проверяется сам по себе: прочитанная сладость или год,
        противоречащие его карточке, и прочитанное слово, принадлежащее
        другому кандидату шортлиста, а не ему.
        """
        # Полоса уже, чем у перестановки: ложная улика стоит правильного
        # ответа, а на полке в кадр попадает соседняя бутылка. На снимке
        # «Мускателя белого» так прочиталось «КРАСНОГО» с соседнего
        # «Муската белого Красного камня».
        picked = central_words(words, min_conf, band=0.5)
        if not picked:
            return []
        hard = self.contradictions(slug, attributes_from_words(picked, self.correct_sweetness))
        out = list(hard)

        marks = discriminating([slug] + [r for r in rivals if r != slug], self)
        # Сверяться надо со всеми словами карточки лидера, а не только с
        # различающими. discriminating вычёркивает общие слова, и «chateau»
        # с «tamagne» выпадали из набора лидера «Шато Тамань. Каберне
        # Совиньон» — после чего засчитывались уликой против него же,
        # хотя написаны на его собственной этикетке.
        own = card_features(slug, self)
        mine = marks.get(slug, set()) | own.name | own.grapes | own.producer
        for word in picked:
            for token in clean(word.text).split():
                if len(token) < 3 or token in CATEGORY_WORDS:
                    continue
                # Пороги намеренно разные. Слово засчитывается лидеру
                # снисходительно, а сопернику — строго: правило одностороннее,
                # и наказывать можно только когда слово явно не лидера и явно
                # чьё-то ещё. С одинаковыми порогами «МУСКАТЕАЬ» не признавался
                # за «мускатель» лидера, но признавался за «мускат» соседа.
                if best_match(token, mine, SURE_MATCH):
                    continue
                # Владелец слова должен быть чужой винодельни. Внутри
                # одной слово серии принадлежит всем её винам сразу, и
                # засчитывать его уликой нельзя: «chateau» и «tamagne»
                # приписывались «Chateau Tamagne Signature» против лидера
                # «Шато Тамань. Каберне Совиньон» — та же винодельня, та же
                # надпись на этикетке, только в карточке лидера она
                # кириллицей. Разводить соседей по линейке — работа
                # resolve_close, у неё для этого своя группа близких.
                brand = (self.by_slug.get(slug) or {}).get("manufacturer")
                owners = [rival for rival in rivals
                          if rival != slug
                          and (self.by_slug.get(rival) or {}).get("manufacturer") != brand
                          and best_match(token, marks.get(rival, set()), 1.0)]
                # Владелец должен быть ровно один. Слово, на которое
                # претендуют несколько кандидатов, — общее, а не чьё-то:
                # «chateau» стоит на карточках «Chateau André», «Chateau
                # de Talu» и «Шато Тамань» сразу, и уликой против любого
                # из них быть не может.
                if len(owners) == 1:
                    out.append(f"на этикетке «{token}», это {owners[0]}")
                    break
        return out

    def hard_conflicts(self, slug: str, words, min_conf: float = 0.60) -> list[str]:
        """Только противоречия по структурным полям карточки.

        Отделено от conflicts_with намеренно. Улика «на этикетке слово,
        которое принадлежит другому кандидату» бывает ложной: на снимке
        с двумя бутылками читается название соседней, и на 11751350_0.jpg
        так прочиталось «красностоп» рядом с верным ответом. Такая улика
        имеет право снизить уверенность, но не имеет права менять ответ.

        Прочитанный цвет или сладость, прямо противоречащие карточке, —
        другое дело: «красное» на этикетке при белом вине в карточке
        означает, что это не оно, и лидера надо менять.
        """
        picked = central_words(words, min_conf, band=0.5)
        if not picked:
            return []
        return self.contradictions(slug, attributes_from_words(picked, self.correct_sweetness))

    def confirms(self, slug: str, words, rivals: list[str],
                 min_conf: float = 0.60) -> list[str]:
        """Слова этикетки, которые подтверждают лидера против соперников.

        Нужно ровно для одного решения: убирать ли оговорку с карточки.
        Когда геометрия не развела кандидатов, молчание текста — это не
        согласие, а отсутствие проверки. На «BRÛLÉ Rosé Demi-Sec» OCR читает
        только имя серии, общее для всех десяти её вин, и карточка «Cuvée
        Brut» уходила пользователю без оговорок. Подтверждением считается
        лишь слово, которое отличает лидера от соперников.
        """
        mine = distinguishing(slug, [r for r in rivals if r != slug], self)
        if not mine:
            return []
        out = []
        for word in central_words(words, min_conf):
            need = SURE_MATCH if word.conf >= SURE_CONF else VAGUE_MATCH
            for token in clean(word.text).split():
                if len(token) >= 3 and best_match(token, mine, need):
                    out.append(token)
        return out

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
             "выдержанное", "коллекционное", "натуральное", "года", "год",
             # Тип вина может быть опущен в заголовке карточки. Его отсутствие
             # не противоречит надписи и не отличает собственное имя вина.
             "игристое"}

# Порог нечёткого совпадения зависит от уверенности чтения. На этикетке
# «Массандры» OCR читает БЕЛЫЙ как БЕАЫЙ с уверенностью 0.98 — шрифт
# сливает Л и А. Ошибка в одну букву при такой уверенности — это всё ещё
# уверенно прочитанное слово, а при чтении наугад одной буквы мало.
SURE_CONF = 0.85
SURE_MATCH = 0.70
VAGUE_MATCH = 0.85

# Гомоглифы: буквы, которые в кириллице и латинице выглядят одинаково.
# EasyOCR с двумя моделями свободно смешивает алфавиты внутри слова —
# «КРАСНОЕ» на этикетке «Русского Игристого» вернулось как «KPACHOЕ»,
# где шесть букв латинские. Для человека это то же слово, для сравнения
# строк — другое, и различитель пропадал.
HOMOGLYPHS = str.maketrans({
    "a": "а", "b": "ь", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м",
    "o": "о", "p": "р", "t": "т", "u": "и", "x": "х", "y": "у",
})


# Проверенные соответствия латиницы и кириллицы. Карточки каталога пишут
# одно и то же по-разному: «Chateau Tamagne Signature. Каберне» и «Шато
# Тамань. Каберне Совиньон» — одна линейка одной винодельни, а в названиях
# сортов латиница стоит в заголовке («Golubitskoe Estate Chardonnay»), тогда
# как поле сортов всегда кириллическое («Шардоне»). Без сведения слово
# «tamagne», прочитанное с этикетки, считалось различителем: оно есть у
# латинских карточек и «нет» у кириллических, и правильный ответ получал
# улику против себя. Побуквенная транслитерация тут не годится:
# chardonnay → «чардоннаи», sauvignon → «саувигнон».
#
# Список только из пар, подтверждённых каталогом: латинский токен заголовка
# встречается на карточках, у которых в поле сортов стоит кириллический
# (совпадение в ≥80% карточек, не меньше трёх), плюс имя линейки Тамань.
# Нечёткое сравнение при этом не расширяется: сводится только точный токен.
ALIASES = {
    "tamagne": "тамань",
    "sauvignon": "совиньон", "cabernet": "каберне", "pinot": "пино",
    "merlot": "мерло", "noir": "нуар", "chardonnay": "шардоне",
    "riesling": "рислинг", "franc": "фран", "muscat": "мускат",
    "moscato": "мускат", "syrah": "сира", "shiraz": "шираз",
    "saperavi": "саперави", "rkatsiteli": "ркацители", "malbec": "мальбек",
    "aligote": "алиготе", "grigio": "гриджио", "gris": "гри",
    "ottonel": "оттонель", "chenin": "шенен", "traminer": "траминер",
    "gewurztraminer": "гевюрцтраминер", "vermentino": "верментино",
    "viognier": "вионье", "marselan": "марселан", "krasnostop": "красностоп",
    "kokur": "кокур", "tempranillo": "темпранильо", "sangiovese": "санджовезе",
    "semillon": "семильон", "meunier": "менье", "verdot": "вердо",
    "bastardo": "бастардо", "mourvedre": "мурведр", "zinfandel": "зинфандель",
    "magaracha": "магарача", "bianca": "бианка",
    # имя стиля/линейки в обоих алфавитах у одной винодельни («ZB Frizzante
    # Dry» и «Сухое белое ЗБ вайн Фриззанте»): прочитанное FRIZZANTE не
    # должно быть уликой против кириллической карточки
    "frizzante": "фриззанте",
}


# Переключатель нужен только для парного замера «до/после» на одном наборе;
# в сервисе словарь включён.
USE_ALIASES = os.environ.get("TEXT_ALIASES", "1") != "0"


# Побуквенная транслитерация латинских имён (NAME_TRANSLIT): «Di Caspico»
# и «Ди Каспико», «Golubitskoe» и «Голубицкое» — одно имя, а словарь знает
# только сорта и «Тамань». Для русских имён собственных, записанных
# латиницей, побуквенный перевод точен; французские сорта остаются за
# словарём, который проверяется раньше. Категорийные и родовые слова
# сюда не попадают — они вычитаются из имени до сравнения.
USE_NAME_TRANSLIT = os.environ.get("NAME_TRANSLIT", "1") != "0"


def canonical(token: str) -> str:
    """Токен в канонической записи: латинский сорт или линейка — кириллицей."""
    if not USE_ALIASES:
        return token
    if token in ALIASES:
        return ALIASES[token]
    if USE_NAME_TRANSLIT and len(token) >= 5 and re.fullmatch(r"[a-z]+", token) \
            and token not in CATEGORY_WORDS and token not in GENERIC_WORDS:
        from normalize import translit_token
        return translit_token(token)
    return token


def alphabet_variants(token: str) -> tuple[str, ...]:
    """Токен как есть, он же в кириллице по гомоглифам и по словарю соответствий.

    Формы нужны все, потому что смешение бывает в любую сторону: русское
    слово читается латиницей, латинский бренд — кириллицей, а сорт на
    этикетке написан по-французски. Сравнивать по максимуму дешевле, чем
    угадывать алфавит.
    """
    out = [token]
    cyrillic = token.translate(HOMOGLYPHS)
    if cyrillic != token:
        out.append(cyrillic)
    for form in (token, cyrillic):
        alias = canonical(form)
        if alias not in out:
            out.append(alias)
    return tuple(out)


# Переключатели экспериментов: парный замер «до/после» на одном наборе.
USE_JOIN = os.environ.get("OCR_JOIN", "1") != "0"


def join_spaced(text: str) -> str:
    """Склейка разрядки и разрыва слова в строке OCR.

    Две устойчивые ошибки чтения, обе на живых кадрах. Разрядка: буквы
    названия набраны с большими просветами, и распознаватель отдаёт
    «Б А Л А K Л А В А», «B RUT», «Б Р ЮТ», «R Е $ Е R V Е» — по одной-две
    буквы на токен. Разрыв: слово цвета или сладости рассечено пробелом,
    «КРАСН OE», «ПОЛУ СЛАДКОЕ». В обоих случаях слово целиком стоит на
    этикетке, а сравнение видит набор обрывков и ничего не находит.

    Правило узкое. Строка склеивается целиком, только если большинство её
    частей — одиночные символы (разрядка). Иначе склеиваются лишь соседние
    части, которые вместе дают слово категории (цвет, сладость) — точно
    или с одной заменой буквы у длинного слова. Произвольные слова так не
    склеиваются: «Шато Тамань» должно остаться двумя словами.
    """
    if not USE_JOIN or " " not in text.strip():
        return text
    parts = text.split()
    singles = sum(1 for part in parts if len(part) == 1)
    if len(parts) >= 3 and singles >= 0.6 * len(parts):
        return "".join(parts)
    low = clean(text).split()
    if len(low) < 2:
        return text
    terms = {t for phrase in list(SWEETNESS) + list(COLORS) for t in phrase.split()
             if " " not in phrase and len(t) >= 4}
    out, i = [], 0
    raw = text.split()
    while i < len(raw):
        if i + 1 < len(raw):
            pair = clean(raw[i] + raw[i + 1]).replace(" ", "")
            for form in (pair, pair.translate(HOMOGLYPHS)):
                hit = form in terms or (len(form) >= 7 and any(
                    len(t) == len(form) and sum(a != b for a, b in zip(t, form)) <= 1
                    for t in terms))
                if hit:
                    break
            if hit:
                out.append(raw[i] + raw[i + 1])
                i += 2
                continue
        out.append(raw[i])
        i += 1
    return " ".join(out)


def attributes_from_words(words, correct_sweetness: bool = True) -> Attributes:
    """Correct one substitution in a long sweetness word only at high OCR confidence.

    No substring/deletion matching: confusing semi-dry with dry is worse
    than abstaining. Multiple possible sweetness values also mean abstention.
    """
    attrs = extract_attributes(" ".join(w.text for w in words))
    if attrs.sweetness or not correct_sweetness:
        return attrs
    possible = set()
    for word in words:
        if word.conf < SURE_CONF:
            continue
        for token in clean(word.text).split():
            if len(token) < 7:
                continue
            for variant in alphabet_variants(token):
                for term, value in SWEETNESS.items():
                    if len(term) == len(variant) and " " not in term:
                        if sum(a != b for a, b in zip(term, variant)) <= 1:
                            possible.add(value)
    if len(possible) == 1:
        attrs.sweetness = possible.pop()
    return attrs


def best_match(token: str, marked: set[str], need: float) -> bool:
    """Совпал ли прочитанный токен хоть с одним каталожным, с учётом алфавита."""
    return any(tokens_match(variant, m) >= need
               for variant in alphabet_variants(token) for m in marked)


USE_LINE_BAND = os.environ.get("LINE_BAND", "1") != "0"


def text_lines(words) -> list[list]:
    """Слова, стоящие в одной строке этикетки, вплотную друг к другу.

    Одна строка — пересечение по вертикали больше половины меньшей высоты
    и просвет по горизонтали не шире высоты строки. Текст соседней бутылки
    отделён краем бутылки, то есть просветом, и в строку не попадает.
    """
    lines: list[list] = []
    for w in sorted(words, key=lambda w: (w.box[1], w.box[0])):
        for line in lines:
            last = line[-1]
            height = min(last.box[3] - last.box[1], w.box[3] - w.box[1]) or 1
            overlap = min(last.box[3], w.box[3]) - max(last.box[1], w.box[1])
            gap = w.box[0] - max(x.box[2] for x in line)
            if overlap > 0.5 * height and -height <= gap <= height:
                line.append(w)
                break
        else:
            lines.append([w])
    return lines


def central_words(words, min_conf: float = 0.60, band: float = 0.72):
    """Уверенные строки из центральной части кропа.

    Кроп по бутылке почти всегда прихватывает край соседней: на почти-дубле
    «Велвет Сизон Рислинг» OCR уверенно читает CABERNET с соседа и уводит
    решение в чужую сторону. Отсюда отбор по координатам — именно для этого
    слова и хранят место на кадре.

    Отбор идёт по строкам, а не по словам (LINE_BAND). Надпись «ПОЛУСЛАДКОЕ
    РОЗОВОЕ» распознаётся двумя словами, и второе, стоящее правее центра,
    отсеивалось как чужое: на org079_0.jpg прочитанное «РОЗОВОЕ» не
    доходило до проверки лидера, и белое вино той же линейки оставалось
    первым. Слова одной строки, стоящие вплотную, принадлежат одной
    этикетке, и судьба у них общая.
    """
    picked = [w for w in words if w.conf >= min_conf]
    if not picked:
        return []
    left = min(w.box[0] for w in picked)
    right = max(w.box[2] for w in picked)
    span = (right - left) or 1
    center = (left + right) / 2
    if not USE_LINE_BAND:
        return [w for w in picked if abs(w.center_x - center) <= span * band / 2]
    out = []
    for line in text_lines(picked):
        line_center = (min(w.box[0] for w in line) + max(w.box[2] for w in line)) / 2
        if abs(line_center - center) <= span * band / 2:
            out.extend(line)
    return [w for w in picked if w in out]


@dataclass
class CardFeatures:
    """Слова карточки, разложенные по признакам.

    Разложение нужно, чтобы не путать разные вещи. Имя винодельни внутри
    одной винодельни ничего не различает: у «Кокур сухое» в названии нет
    слова «Массандра», а у «Массандра Кокур» есть, и прочитанное с этикетки
    «МАССАНДРА» превращалось в улику против первого — притом что на бутылке
    написана как раз она. Между разными винодельнями то же слово, наоборот,
    различает отлично.

    Сорт, цвет, сладость и год — признаки со значением: для них важно не
    отсутствие слова у кандидата, а несовместимость. Прочитанное «розовое»
    противоречит белому вину, а прочитанное «мерло» — вину из другого сорта.
    """
    producer: set[str]
    name: set[str]
    grapes: set[str]
    color: str | None
    sweetness: str | None
    years: set[str]


# Слова категории и цвета в любом алфавите. Из различителей названия они
# вычёркиваются: категорию сравнивает отдельный путь, где BRUT и «брют»
# сводятся к одному значению. Пока они считались произвольными токенами,
# правильная «Фанагория. Брют белое» получала улику «на этикетке brut,
# а у него нет» — потому что в её карточке слово написано кириллицей,
# а в карточке соперника «Extra Brut Rose» латиницей.
# Слова, называющие не вино, а вообще винодельческое предприятие. Они
# стоят и на этикетке («Семейная винодельня Литавщуков», «сделано в Винной
# деревне»), и в названии доброй половины виноделен каталога, поэтому
# различать ими нельзя: прочитанное «винодельня» принадлежит сразу десятку
# кандидатов, а тем, у кого его нет, идёт в улику. Так «Мерло Литавщук»
# уехал с седьмого места на пятнадцатое.
GENERIC_WORDS = {
    "вино", "вина", "винный", "винная", "винной", "винодельня", "винодельни",
    "виноделие", "винодельческий", "винодельческое", "усадьба", "имение",
    "поместье", "хозяйство", "семейная", "семейное", "агрофирма", "компания",
    "кооператив", "завод", "шато", "chateau", "winery", "wine", "wines",
    "estate", "vineyards", "вайнери",
}

CATEGORY_WORDS = {token
                  for phrase in list(SWEETNESS) + list(SWEETNESS.values())
                  + list(COLORS) + list(COLORS.values())
                  for token in re.split(r"[\s-]+", phrase) if token}


def card_features(slug: str, channel: "TextChannel") -> CardFeatures:
    wine = channel.by_slug.get(slug) or {}
    # Слова карточки сводятся к канонической записи: «Chardonnay» в
    # заголовке и «Шардоне» в сортах — одно слово, и различать им нельзя.
    producer = {canonical(t) for t in clean(wine.get("manufacturer") or "").split()
                if len(t) >= 4}
    # Имя винодельни в заголовке может стоять в другом алфавите, чем в поле
    # производителя: «Fanagoria Extra Brut Rose» при производителе
    # «Фанагория». Такое слово — не имя вина, и различать им нельзя:
    # прочитанное FANAGORIA становилось уликой против «Розовое полусладкое»
    # той же Фанагории.
    producer_forms = producer | set(brand_tokens(wine.get("manufacturer") or ""))
    grapes = {canonical(t) for t in clean(" ".join(wine.get("grapes") or [])).split()
              if len(t) >= 3}
    title = clean(wine.get("title") or "")
    attrs = channel.attributes_of.get(slug)

    def is_producer(token: str) -> bool:
        # Побуквенная транслитерация даёт «фанагориа» для Fanagoria — на одну
        # букву мимо «фанагория», поэтому сравнение нечёткое (SURE_MATCH):
        # имена виноделен длинные, и ложное совпадение с ними маловероятно.
        forms = {token, canonical(token)}
        if re.fullmatch(r"[a-z]+", token):
            from normalize import translit_token
            forms.add(translit_token(token))
        return any(tokens_match(form, known) >= SURE_MATCH
                   for form in forms for known in producer_forms if len(known) >= 5) \
            or bool(forms & producer_forms)

    return CardFeatures(
        producer=producer,
        name=({canonical(t) for t in title.split() if len(t) >= 3 and not is_producer(t)}
              - producer - grapes - STOPWORDS - CATEGORY_WORDS),
        grapes=grapes,
        color=(attrs.color if attrs else None),
        sweetness=(attrs.sweetness if attrs else None),
        years=set(re.findall(r"\b(19[89]\d|20[0-3]\d)\b", title)),
    )


def discriminating(candidates: list[str], channel: "TextChannel") -> dict[str, set[str]]:
    """Токены, которыми кандидаты различаются между собой.

    Общие слова серии («Велвет Сизон», «Фанагория») стоят у всех и решить
    ничего не могут. Различает то, что есть у одних и нет у других.

    Имя винодельни в расчёт идёт только если кандидаты из разных виноделен.
    Внутри одной оно попадало в различители из-за того, что у части позиций
    продублировано в названии, и работало наоборот — против правильного
    ответа.
    """
    features = {slug: card_features(slug, channel) for slug in candidates}
    one_producer = len({frozenset(f.producer) for f in features.values()}) <= 1
    own = {slug: (f.name | f.grapes | f.years
                  | (set() if one_producer else f.producer))
           for slug, f in features.items()}
    shared = set.intersection(*own.values()) if own else set()
    return {slug: words - shared for slug, words in own.items()}


def distinguishing(leader: str, rivals: list[str],
                   channel: "TextChannel") -> set[str]:
    """Слова, которые есть у лидера и нет ни у одного из соперников.

    Отличие от discriminating существенное. Там общими считаются слова,
    стоящие у всех кандидатов сразу, и «кокур» пережил отсев: он есть
    у «Массандра Кокур», у «Кокур сухое» и у «Портвейн Белый Гурзуф»
    (сорт), но нет у «Массандра Мускат». Пересечение по всему шортлисту
    его не убрало, и слово пошло в подтверждение выбора между двумя
    карточками, на каждой из которых оно написано.

    Для снятия оговорки нужна не разница со всем шортлистом, а разница
    с теми, кого лидер обошёл: слово подтверждает, только если ни у одного
    близкого соперника его нет.
    """
    features = {slug: card_features(slug, channel) for slug in [leader] + rivals}
    one_producer = len({frozenset(f.producer) for f in features.values()}) <= 1
    own = {slug: (f.name | f.grapes | f.years
                  | (set() if one_producer else f.producer))
           for slug, f in features.items()}
    return own[leader] - set().union(*(own[r] for r in rivals if r != leader)) \
        if len(rivals) else own[leader]


def brand_slugs(slugs: list[str], words, channel: "TextChannel",
                min_conf: float = 0.60) -> set[str]:
    """Кандидаты той винодельни, чьё имя уверенно прочитано на этикетке.

    Имя винодельни — самая крупная надпись на этикетке и читается там, где
    название вина уже не читается. На снимке у полки «INKERMAN» и «Семейная
    винодельня Литавщуков» распознаются целиком, а сорт мелким шрифтом —
    нет. Геометрия при этом ставила первым «Марселан» Мысхако и «Мерло»
    Усадьбы Перовских: другая винодельня, похожая по виду этикетка.

    Возвращается множество кандидатов прочитанной винодельни. Само по себе
    оно ничего не решает — оно лишь заводит этих кандидатов в группу
    сравнения, а дальше действует обычное правило: слово, которое есть
    у части группы и нет у остальных, остальным добавляет противоречие.
    """
    brands: dict[str, set[str]] = {}
    owners: dict[str, set[str]] = {}
    for slug in slugs:
        name = (channel.by_slug.get(slug) or {}).get("manufacturer") or ""
        for token in brand_tokens(name):
            if len(token) >= 5:
                brands.setdefault(token, set()).add(slug)
                owners.setdefault(token, set()).add(brand_key(name))
    # Токен годится, только если он называет одну винодельню из шортлиста.
    # «Винодельня» и «усадьба» отсеяны выше, но остаются совпадения вроде
    # общего слова у двух разных хозяйств — такое слово ничего не выделяет.
    brands = {t: v for t, v in brands.items() if len(owners[t]) == 1}
    if not brands:
        return set()
    from normalize import translit_token

    found: set[str] = set()
    for word in central_words(words, min_conf):
        for token in clean(word.text).split():
            if len(token) < 5:
                continue
            need = SURE_MATCH if word.conf >= SURE_CONF else VAGUE_MATCH
            # Прочитанное латиницей имя сверяется и в побуквенной кириллице:
            # ключи винодельни выше приведены к ней же.
            forms = set(alphabet_variants(token))
            if USE_ALIASES and re.fullmatch(r"[a-z0-9]+", token) and not token.isdigit():
                forms.add(translit_token(token))
            for brand, owners in brands.items():
                if any(tokens_match(form, brand) >= need for form in forms):
                    found |= owners
    # Несколько прочитанных производителей не указывают целевую бутылку:
    # на полке это могут быть соседи. Уникальность отдельного токена выше
    # не гарантирует уникальность производителя по всему результату OCR.
    matched_brands = {brand_key((channel.by_slug.get(slug) or {}).get("manufacturer") or "")
                      for slug in found}
    if len(matched_brands) != 1:
        return set()
    return found


def brand_tokens(name: str) -> list[str]:
    """Слова имени винодельни в канонической записи.

    Одна винодельня в каталоге пишется по-разному: «Поместье Голубицкое»
    в одних карточках и «Golubitskoe Estate» в других. Прочитанное с
    этикетки «GOLUBITSKOE» находило только латинские карточки, и правило
    «своя винодельня вперёд» отодвигало правильный ответ с кириллической
    карточкой за все латинские (org031_0.jpg). Латинское слово имени
    переводится в кириллицу побуквенно — для русских имён собственных это
    работает, в отличие от французских сортов, где нужен словарь.
    Родовые слова («винодельня», «estate») в имя не входят.
    """
    from normalize import translit_token

    out = []
    for token in clean(name).split():
        if len(token) < 4 or token in GENERIC_WORDS:
            continue
        token = canonical(token)
        if USE_ALIASES and re.fullmatch(r"[a-z0-9]+", token) and not token.isdigit():
            token = translit_token(token)
        out.append(token)
    return out


def brand_key(name: str) -> frozenset[str] | str:
    """Идентичность винодельни: множество канонических слов имени.

    Две записи с одинаковым ключом считаются одной винодельней. Имя без
    значимых слов («Le K2», «В2Р») ключом не сводится и остаётся само собой:
    иначе все такие имена слились бы в одно.
    """
    tokens = frozenset(brand_tokens(name))
    return tokens if tokens else clean(name)


USE_NAME_FIRST = os.environ.get("NAME_FIRST", "1") != "0"
USE_GROUP_LEADER = os.environ.get("GROUP_LEADER", "0") != "0"
USE_NAME_STRONG = os.environ.get("NAME_STRONG", "1") != "0"
STRONG_NAME_LEN = 7


def name_slugs(slugs: list[str], words, channel: "TextChannel",
               min_conf: float = 0.60, weight: dict[str, float] | None = None,
               floor: float = 0.4) -> set[str]:
    """Кандидаты, чьё собственное имя уверенно прочитано на этикетке.

    Правило перестановки работает внутри группы близких по геометрии, и
    правильный ответ туда не попадает, если инлаеров у него мало: на
    «Портвейне белом крымском» OCR уверенно читает «ПОРТВЕЙН КРЫМСКИЙ»,
    но лидер «Портвейн красный Ливадия» держит 46 точек против десятка,
    и слово «крымский» некому предъявить. Здесь читаемое имя заводит
    его владельца в группу — дальше действует обычное одностороннее
    правило: у кого прочитанного слова нет, тот получает противоречие.

    Слово должно быть именем (не сортом, не категорией, не винодельней),
    не короче пяти букв, прочитано уверенно (SURE_CONF), совпадать точно
    и принадлежать ровно одному кандидату шортлиста — иначе оно ничего
    не выделяет. Нечёткое совпадение здесь запрещено: «ЧЕСЕРТНОЕ»
    (десертное) приводило «Траминер десертное» с шестью точками против
    58 у верного «Кагора». По той же причине владелец с числом совпавших
    точек ниже доли floor от лучшего не приводится: слово с чужой
    бутылки или из описания читается так же уверенно, как своё.
    """
    if not USE_NAME_FIRST or len(slugs) < 2:
        return set()
    owners: dict[str, set[str]] = {}
    for slug in slugs:
        for token in card_features(slug, channel).name:
            if len(token) >= 5 and token not in GENERIC_WORDS and token not in STOPWORDS:
                owners.setdefault(token, set()).add(slug)
    unique = {token: next(iter(v)) for token, v in owners.items() if len(v) == 1}
    if not unique:
        return set()
    found: set[str] = set()
    for word in central_words(words, min_conf):
        if word.conf < SURE_CONF:
            continue
        for token in clean(word.text).split():
            if len(token) < 5 or token in CATEGORY_WORDS:
                continue
            # Лучшее совпадение решает: «КРЫМСКИЙ» точно у «Портвейна
            # крымского» и с одной правкой у «Мадеры Крымской» — точное
            # совпадение выделяет владельца, нечёткий сосед его не блокирует.
            hits = {slug for name, slug in unique.items()
                    if any(v == name for v in alphabet_variants(token))}
            if len(hits) == 1:
                found |= hits
    if weight:
        best = max(weight.values()) if weight else 0.0
        strong = strong_name_slugs(slugs, words, channel, min_conf) if USE_NAME_STRONG else set()
        found = {s for s in found if weight.get(s, 0.0) >= floor * best or s in strong}
    return found


def strong_name_slugs(slugs: list[str], words, channel: "TextChannel",
                      min_conf: float = 0.60) -> set[str]:
    """Владельцы сильного имени: точное слово не короче STRONG_NAME_LEN букв.

    «КРЫМСКИЙ» (8 букв, прочитано уверенно, совпало точно) — улика, которой
    порог по точкам не нужен: у «Портвейна белого крымского» 15 точек против
    65 у «Кагора», и по порогу он не приводится, хотя слово прочитано
    целиком. Короткие и нечёткие совпадения сюда не проходят: они и давали
    ложные приводы («ЧЕСЕРТНОЕ», «долины»).
    """
    if not USE_NAME_STRONG or len(slugs) < 2:
        return set()
    owners: dict[str, set[str]] = {}
    for slug in slugs:
        for token in card_features(slug, channel).name:
            if len(token) >= STRONG_NAME_LEN and token not in GENERIC_WORDS and token not in STOPWORDS:
                owners.setdefault(token, set()).add(slug)
    unique = {token: next(iter(v)) for token, v in owners.items() if len(v) == 1}
    found: set[str] = set()
    for word in central_words(words, min_conf):
        if word.conf < SURE_CONF:
            continue
        for token in clean(word.text).split():
            if len(token) < STRONG_NAME_LEN or token in CATEGORY_WORDS:
                continue
            hits = {slug for name, slug in unique.items()
                    if any(v == name for v in alphabet_variants(token))}
            if len(hits) == 1:
                found |= hits
    return found


def osa_distance(a: str, b: str) -> int:
    """Расстояние Дамерау–Левенштейна (оптимальное выравнивание строк).

    Кроме вставки, удаления и замены одной правкой считается перестановка
    соседних букв — типичная ошибка чтения мелкого шрифта.
    """
    prev2: list[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = ca != cb
            value = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                value = min(value, prev2[j - 2] + 1)
            cur.append(value)
        prev2, prev = prev, cur
    return prev[-1]


def near(token: str, word: str, theta: float) -> bool:
    """Прочитанный токен совпал со словом карточки с точностью θ.

    Расстояние нормируется на длину более длинного слова: одна ошибка
    в шестибуквенном «алушта» и две в десятибуквенном «российского» — одно
    и то же качество чтения.
    """
    return any(osa_distance(variant, word) <= theta * max(len(variant), len(word))
               for variant in alphabet_variants(token))


VOTE_MIN_LEN = 4


def card_vocabulary(slug: str, channel: "TextChannel") -> set[str]:
    """Слова карточки, которыми её можно узнать: название, сорта, винодельня."""
    wine = channel.by_slug.get(slug) or {}
    text = " ".join([wine.get("title") or "", " ".join(wine.get("grapes") or []),
                     wine.get("manufacturer") or ""])
    return {t for t in clean(text).split()
            if len(t) >= VOTE_MIN_LEN and t not in GENERIC_WORDS and t not in STOPWORDS}


def text_votes(slugs: list[str], words, channel: "TextChannel", theta: float,
               min_conf: float = 0.60) -> dict[str, int]:
    """Сколько уверенно прочитанных слов этикетки нашлось в карточке кандидата.

    Каждое прочитанное слово даёт кандидату не больше одного очка. Слова
    цвета и сладости тоже считаются: «белый» отличает белый портвейн от
    красного той же линейки так же, как название — «Алушту» от «Гурзуфа».
    """
    read = {t for w in central_words(words, min_conf) for t in clean(w.text).split()
            if len(t) >= VOTE_MIN_LEN and t not in GENERIC_WORDS and t not in STOPWORDS}
    votes = {}
    for slug in slugs:
        vocab = card_vocabulary(slug, channel)
        votes[slug] = sum(1 for t in read if any(near(t, v, theta) for v in vocab))
    return votes


def resolve_close(candidates: list[tuple[str, float]], words,
                  channel: "TextChannel", window: float = 0.80,
                  min_conf: float = 0.60, extra: set[str] | None = None,
                  ) -> tuple[list[tuple[str, float]], dict[str, list[str]]]:
    """Переупорядочивает только группу близких кандидатов.

    Геометрия отвечает на вопрос «та же ли это этикетка» и обычно отвечает
    уверенно: у верного кандидата инлаеров на порядок больше. Вмешиваться
    туда текстом незачем. Работа текста — там, где геометрия сомневается:
    позиции одной серии с близким числом инлаеров, различающиеся словом.

    Правило одностороннее. Уверенно прочитанное слово, которое есть у части
    кандидатов группы и нет у остальных, добавляет остальным противоречие.
    Совпадение никого не поднимает: слово могло попасть в кадр случайно,
    а вот его отсутствие у кандидата — это факт о нём.

    Требовать ровно одного владельца нельзя, хотя так было сначала. Внутри
    линейки Абрау-Дюрсо в группу попадают сразу два красных игристых, и
    прочитанное «КРАСНОЕ» принадлежит обоим. По строгому правилу оно не
    различало ничего, хотя отсекало четырёх кандидатов из шести.
    """
    if len(candidates) < 2:
        return candidates, {}
    # Группа считается от максимума совпавших точек, а не от первого
    # в списке: порядок приходит уже не обязательно геометрический.
    best = max(score for _, score in candidates)
    if best <= 0:
        return candidates, {}
    # Группа — близкие по геометрии плюс те, кого привёл вызывающий:
    # однофамильцы лидера и кандидаты прочитанной винодельни. У линейки
    # с общей этикеткой число совпавших точек внутри линейки ничего не
    # значит («Табия»: у верного вина 10 точек против 42 у соседа), и окно
    # по инлаерам верного кандидата в группу не пускало вовсе.
    extra = extra or set()
    # Текущий лидер входит в группу всегда (GROUP_LEADER). Когда геометрия
    # не решила и порядок задаёт косинус, лидер по косинусу может держать
    # семь точек при двенадцати у другого кандидата — окно по точкам
    # строится от максимума, и лидер в него не попадает. Тогда приведённый
    # владелец прочитанного имени сравнивается не с тем, кто отвечает, и
    # ответ не меняется (13795944_0.jpg: «КРЫМСКИЙ» прочитано, «Портвейн
    # красный Алушта» с семью точками остаётся первым).
    leader = {candidates[0][0]} if USE_GROUP_LEADER else set()
    group = [c for c in candidates if c[1] >= best * window or c[0] in extra or c[0] in leader]
    if len(group) < 2:
        return candidates, {}

    marks = discriminating([slug for slug, _ in group], channel)
    read = [(t, w.conf) for w in central_words(words, min_conf)
            for t in clean(w.text).split()
            if len(t) >= 3 and t not in CATEGORY_WORDS
            and t not in GENERIC_WORDS]

    found: dict[str, list[str]] = {slug: [] for slug, _ in group}
    for token, conf in read:
        need = SURE_MATCH if conf >= SURE_CONF else VAGUE_MATCH
        owners = {slug for slug, marked in marks.items()
                  if best_match(token, marked, need)}
        # Слово, подходящее всем или никому, ничего не различает.
        if not owners or len(owners) == len(found):
            continue
        for slug in found:
            if slug not in owners:
                found[slug].append(f"на этикетке «{token}», а у него нет")
    if not any(found.values()):
        return candidates, found
    # Улика опускает кандидата, всё остальное сохраняет входящий порядок.
    # Раньше группа считалась началом списка и возвращалась отдельным
    # куском; теперь она может быть разбросана по списку, и склейка
    # «группа + хвост» перемешала бы порядок тем, кого улика не касается.
    place = {slug: i for i, (slug, _) in enumerate(candidates)}
    return sorted(candidates,
                  key=lambda c: (len(found.get(c[0], [])), place[c[0]])), found


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
