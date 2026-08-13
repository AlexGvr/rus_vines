#!/usr/bin/env python3
"""Нормализация текста для матчинга этикетка ↔ база.

Каноническое пространство — кириллица в нижнем регистре.
Латинские токены (этикетки часто на латинице: Aristov, Fanagoria)
транслитерируются жадно по диграфам.

Модуль зеркалится в app/js/normalize.js — менять синхронно.
"""
import re

# Порядок важен: сначала длинные диграфы
LAT2CYR = [
    ("shch", "щ"), ("sch", "щ"),
    ("zh", "ж"), ("kh", "х"), ("ts", "ц"), ("ch", "ч"), ("sh", "ш"),
    ("yu", "ю"), ("ju", "ю"), ("ya", "я"), ("ja", "я"), ("yo", "е"), ("jo", "е"),
    ("ye", "е"), ("je", "е"), ("ck", "к"), ("qu", "кв"), ("ph", "ф"), ("th", "т"),
    ("a", "а"), ("b", "б"), ("c", "к"), ("d", "д"), ("e", "е"), ("f", "ф"),
    ("g", "г"), ("h", "х"), ("i", "и"), ("j", "й"), ("k", "к"), ("l", "л"),
    ("m", "м"), ("n", "н"), ("o", "о"), ("p", "п"), ("q", "к"), ("r", "р"),
    ("s", "с"), ("t", "т"), ("u", "у"), ("v", "в"), ("w", "в"), ("x", "кс"),
    ("y", "и"), ("z", "з"),
]

# Только шумовые слова. Категорийные (брют/белое/сухое...) НЕ стоп-слова:
# они матчатся полем категории с весом 0.5 и различают вина одной линейки.
STOPWORDS = {
    "вино", "wine", "россия", "russia", "оф", "of", "и", "in",
    "the", "de", "ооо", "зао", "ао", "гост", "литр", "объем", "алк", "alc",
    "vol", "год", "урожай",
}


def translit_token(tok: str) -> str:
    """Латинский токен → кириллица (жадно по диграфам)."""
    out = []
    i = 0
    low = tok.lower()
    while i < len(low):
        for src, dst in LAT2CYR:
            if low.startswith(src, i):
                out.append(dst)
                i += len(src)
                break
        else:
            out.append(low[i])
            i += 1
    return "".join(out)


def clean(text: str) -> str:
    text = text.lower().replace("ё", "е").replace("«", " ").replace("»", " ")
    text = re.sub(r"[^a-zа-я0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, drop_stop: bool = False) -> list[str]:
    """Токены в каноническом (кириллическом) виде."""
    toks = []
    for tok in clean(text).split():
        if len(tok) < 2:
            continue
        if re.fullmatch(r"[a-z0-9]+", tok) and not tok.isdigit():
            tok = translit_token(tok)
        if drop_stop and tok in STOPWORDS:
            continue
        toks.append(tok)
    return toks


def years(text: str) -> set[str]:
    return set(re.findall(r"\b(19[89]\d|20[0-3]\d)\b", text))


def trigrams(tok: str) -> set[str]:
    padded = f"  {tok} "
    return {padded[i:i + 3] for i in range(len(padded) - 2)}
