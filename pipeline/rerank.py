#!/usr/bin/env python3
"""S1.5 — геометрическая проверка кандидатов локальными признаками.

Глобальный эмбеддинг описывает макет этикетки целиком и потому не различает
позиции одной серии: у них совпадает всё, кроме мелкого шрифта. Локальные
признаки работают иначе — считают, сколько конкретных точек изображения
совпало и легли ли они на одно преобразование. Это и есть проверка «та же
самая этикетка», а не «похожая».

Применяется к десятке кандидатов от CV: полный перебор каталога так дорог,
что не уложится в SLA.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import os

# Масштаб, к которому приводится кадр перед поиском точек.
#
# Удвоение проверялось (data/index/zones_*.json): на 1400 точек вдвое
# больше, синтетика подрастает — 90.2% -> 91.5% на общем наборе и
# 96.8% -> 97.9% на почти-дублях, — но на полевом наборе top-1 наоборот
# падает, 15 верных из 47 против 12, зато top-5 растёт, 25 против 31.
# Ни одна из этих разниц не выходит за шум выборки, а сопоставление
# дорожает со 129 до 361 мс на запрос и индекс признаков удваивается.
# Поэтому оставлен дешёвый вариант; переключается переменной окружения,
# и вместе с ней нужно пересобрать признаки эталонов.
MAX_SIDE = int(os.environ.get("SIFT_MAX_SIDE", "700"))
RATIO = 0.75            # тест Лоу: отсечь неоднозначные соответствия
MIN_MATCHES = 8
_sift: dict = {}


def detector():
    if "sift" not in _sift:
        _sift["sift"] = cv2.SIFT_create(nfeatures=1200)
        _sift["bf"] = cv2.BFMatcher(cv2.NORM_L2)
    return _sift["sift"], _sift["bf"]


def descriptors(image: Image.Image):
    sift, _ = detector()
    arr = np.array(image.convert("L"))
    height, width = arr.shape[:2]
    scale = MAX_SIDE / max(height, width)
    if scale < 1.0:
        arr = cv2.resize(arr, (int(width * scale), int(height * scale)),
                         interpolation=cv2.INTER_AREA)
    keypoints, desc = sift.detectAndCompute(arr, None)
    return keypoints, desc


def inlier_count(query_kp, query_desc, ref_kp, ref_desc) -> int:
    """Число соответствий, переживших тест Лоу и проверку гомографией."""
    if query_desc is None or ref_desc is None:
        return 0
    if len(query_desc) < 2 or len(ref_desc) < 2:
        return 0
    _, bf = detector()
    pairs = bf.knnMatch(query_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return len(good)
    src = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if mask is not None else 0


@dataclass(frozen=True)
class MatchStats:
    """Развёрнутый результат сопоставления пары «запрос — эталон».

    Одно число инлаеров описывает совпадение бедно. Сто точек, слипшихся
    на логотипе винодельни, и сто точек, разложенных по всей этикетке, —
    разные события: первое означает «та же серия», второе «то же вино».
    """
    inliers: int
    good: int            # прошедшие тест Лоу, до проверки гомографией
    coverage: float      # доля ячеек сетки эталона, где есть инлаеры
    spread: float        # площадь охвата инлаеров к площади всех точек эталона
    points: tuple = ()   # координаты инлаеров на эталоне, 0..1 по обеим осям

    @property
    def precision(self) -> float:
        """Доля совпавших точек, переживших проверку гомографией."""
        return self.inliers / self.good if self.good else 0.0


def _layout(points, mask, ref_kp, grid: int = 4):
    """Как инлаеры разложены по эталону: занятость сетки, охват, координаты.

    Координаты нормируются по облаку точек эталона, а не по кадру: рамка
    кадра зависит от того, как сработал детектор, а облако точек — это
    сама бутылка с этикеткой. Ноль по вертикали — верх, единица — низ.
    """
    if mask is None or not len(points):
        return 0.0, 0.0, ()
    chosen = [p for p, keep in zip(points, mask.ravel()) if keep]
    if len(chosen) < 3:
        return 0.0, 0.0, ()
    all_x = [kp.pt[0] for kp in ref_kp]
    all_y = [kp.pt[1] for kp in ref_kp]
    left, top = min(all_x), min(all_y)
    width = (max(all_x) - left) or 1.0
    height = (max(all_y) - top) or 1.0
    cells = {(min(int((x - left) / width * grid), grid - 1),
              min(int((y - top) / height * grid), grid - 1))
             for x, y in chosen}
    xs = [x for x, _ in chosen]
    ys = [y for _, y in chosen]
    spread = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / (width * height)
    normalized = tuple(((x - left) / width, (y - top) / height) for x, y in chosen)
    return len(cells) / (grid * grid), min(spread, 1.0), normalized


def match_stats(query_kp, query_desc, ref_kp, ref_desc) -> MatchStats:
    """inlier_count плюс геометрия совпадения. Стоит столько же."""
    empty = MatchStats(0, 0, 0.0, 0.0)
    if query_desc is None or ref_desc is None:
        return empty
    if len(query_desc) < 2 or len(ref_desc) < 2:
        return empty
    _, bf = detector()
    pairs = bf.knnMatch(query_desc, ref_desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return MatchStats(len(good), len(good), 0.0, 0.0)
    src = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return MatchStats(0, len(good), 0.0, 0.0)
    ref_points = [ref_kp[m.trainIdx].pt for m in good]
    coverage, spread, normalized = _layout(ref_points, mask, ref_kp)
    return MatchStats(int(mask.sum()), len(good), coverage, spread, normalized)


class Reranker:
    """Кэширует признаки эталонов: они не меняются между запросами.

    Кэш ограничен по размеру: дескрипторы всего каталога заняли бы порядка
    гигабайта, а в выдачу попадает небольшая часть позиций.
    """

    def __init__(self, capacity: int = 800) -> None:
        self.cache: OrderedDict[str, tuple] = OrderedDict()
        self.capacity = capacity

    def reference(self, path: str):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        try:
            with Image.open(path) as im:
                value = descriptors(im)
        except Exception:
            value = (None, None)
        self.cache[path] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value

    def scores(self, query: Image.Image, candidates: list[tuple[str, str]]) -> dict[str, int]:
        """candidates — пары (slug, путь к эталонному фото)."""
        query_kp, query_desc = descriptors(query)
        out = {}
        for slug, path in candidates:
            ref_kp, ref_desc = self.reference(path)
            out[slug] = inlier_count(query_kp, query_desc, ref_kp, ref_desc)
        return out


class Point:
    """Минимальная замена cv2.KeyPoint: для проверки гомографии нужны только
    координаты, а восстанавливать полноценные keypoint-объекты дорого."""

    __slots__ = ("pt",)

    def __init__(self, x: float, y: float) -> None:
        self.pt = (x, y)


class PrecomputedReranker:
    """Реранкер поверх заранее посчитанных признаков каталога.

    Массивы открываются через memory-map: файл дескрипторов занимает сотни
    мегабайт, держать его целиком в памяти процесса ни к чему.
    """

    def __init__(self, index_dir) -> None:
        import csv as _csv
        from pathlib import Path as _Path

        index_dir = _Path(index_dir)
        self.desc = np.load(index_dir / "desc.npy", mmap_mode="r")
        self.kpts = np.load(index_dir / "kpts.npy", mmap_mode="r")
        self.offsets = np.load(index_dir / "offsets.npy")
        # У позиции может быть несколько эталонов — основной из каталога
        # и независимые снимки из data/catalog/extra_refs.csv. Храним все
        # и берём лучший по числу совпавших точек.
        self.positions: dict[str, list[int]] = {}
        with (index_dir / "slugs.csv").open(encoding="utf-8") as fh:
            for i, row in enumerate(_csv.DictReader(fh)):
                self.positions.setdefault(row["slug"], []).append(i)

    def entry(self, idx: int):
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        if end <= start:
            return None, None
        desc = np.asarray(self.desc[start:end], dtype=np.float32)
        points = [Point(float(x), float(y)) for x, y in self.kpts[start:end]]
        return points, desc

    def reference(self, slug: str):
        """Первый эталон позиции. Для совместимости со старым вызовом."""
        entries = self.positions.get(slug) or []
        return self.entry(entries[0]) if entries else (None, None)

    def scores(self, query: Image.Image, slugs: list[str]) -> dict[str, int]:
        return {slug: stats.inliers
                for slug, stats in self.stats(query, slugs).items()}

    def stats(self, query: Image.Image, slugs: list[str]) -> dict[str, MatchStats]:
        """Инлаеры вместе с раскладкой совпадений — цена та же.

        Раскладка нужна решению о выдаче: сто точек, слипшихся на логотипе
        винодельни, и сто точек по всей этикетке — разная уверенность.
        """
        query_kp, query_desc = descriptors(query)
        out = {}
        for slug in slugs:
            best = MatchStats(0, 0, 0.0, 0.0)
            for idx in self.positions.get(slug, []):
                stats = match_stats(query_kp, query_desc, *self.entry(idx))
                if stats.inliers > best.inliers:
                    best = stats
            out[slug] = best
        return out
