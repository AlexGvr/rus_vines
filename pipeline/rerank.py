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

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 700          # выше — дороже, а прирост инлаеров уже незаметен
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
        with (index_dir / "slugs.csv").open(encoding="utf-8") as fh:
            self.position = {row["slug"]: i
                             for i, row in enumerate(_csv.DictReader(fh))}

    def reference(self, slug: str):
        idx = self.position.get(slug)
        if idx is None:
            return None, None
        start, end = int(self.offsets[idx]), int(self.offsets[idx + 1])
        if end <= start:
            return None, None
        desc = np.asarray(self.desc[start:end], dtype=np.float32)
        points = [Point(float(x), float(y)) for x, y in self.kpts[start:end]]
        return points, desc

    def scores(self, query: Image.Image, slugs: list[str]) -> dict[str, int]:
        query_kp, query_desc = descriptors(query)
        out = {}
        for slug in slugs:
            ref_kp, ref_desc = self.reference(slug)
            out[slug] = inlier_count(query_kp, query_desc, ref_kp, ref_desc)
        return out
