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
        hard = self.contradictions(slug, extract_attributes(
            " ".join(w.text for w in picked)))
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
                if len(token) < 3:
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
        return self.contradictions(slug, extract_attributes(
            " ".join(w.text for w in picked)))

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
             "выдержанное", "коллекционное", "натуральное", "года", "год"}

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


def alphabet_variants(token: str) -> tuple[str, ...]:
    """Токен как есть и он же, переложенный в кириллицу.

    Обе формы нужны потому, что смешение бывает в любую сторону: русское
    слово читается латиницей, латинский бренд — кириллицей. Сравнивать
    по максимуму дешевле, чем угадывать алфавит.
    """
    cyrillic = token.translate(HOMOGLYPHS)
    return (token,) if cyrillic == token else (token, cyrillic)


def best_match(token: str, marked: set[str], need: float) -> bool:
    """Совпал ли прочитанный токен хоть с одним каталожным, с учётом алфавита."""
    return any(tokens_match(variant, m) >= need
               for variant in alphabet_variants(token) for m in marked)


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


def card_features(slug: str, channel: "TextChannel") -> CardFeatures:
    wine = channel.by_slug.get(slug) or {}
    producer = {t for t in clean(wine.get("manufacturer") or "").split() if len(t) >= 4}
    grapes = {t for t in clean(" ".join(wine.get("grapes") or [])).split() if len(t) >= 3}
    title = clean(wine.get("title") or "")
    attrs = channel.attributes_of.get(slug)
    return CardFeatures(
        producer=producer,
        name={t for t in title.split() if len(t) >= 3} - producer - grapes - STOPWORDS,
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


def resolve_close(candidates: list[tuple[str, float]], words,
                  channel: "TextChannel", window: float = 0.80,
                  min_conf: float = 0.60,
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
    group = [c for c in candidates if c[1] >= best * window]
    if len(group) < 2:
        return candidates, {}

    marks = discriminating([slug for slug, _ in group], channel)
    read = [(t, w.conf) for w in central_words(words, min_conf)
            for t in clean(w.text).split() if len(t) >= 3]

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
