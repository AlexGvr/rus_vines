"""Интервалы и парные тесты для малых полевых выборок.

На 150 кадрах один кадр — 0.65 п.п., а 95% интервал доли около ±5 п.п.
Без интервала разница в два кадра читается как улучшение; эти функции
печатаются рядом с каждым процентом в отчётах сравнения.
"""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% интервал Уилсона для доли k/n; при n = 0 — (0, 1)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Двусторонний точный тест Мак-Немара по несогласным парам.

    fixed — кадры, которые изменение исправило, broken — которые испортило.
    Возвращает p-value биномиального теста b ~ Bin(b + c, 0.5).
    """
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt_share(k: int, n: int) -> str:
    """«k/n = p% [lo–hi]» для таблиц."""
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n * 100:.1f}% [{lo * 100:.0f}–{hi * 100:.0f}]"
