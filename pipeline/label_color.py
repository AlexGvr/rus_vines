"""Цвет этикетки целиком: кадр совмещается с эталоном и сравнивается по площади.

Согласие цвета в совпавших точках (rerank.keypoint_colors) не различает
соседей по линейке: точки ложатся на общие для всех элементы — золото,
герб, год основания, — а различает их то, на чём точек нет, например
заливка ромба Victor Dravigny: светло-голубая у Брюта, тёмно-синяя у
Экстра Брюта. Здесь кадр переносится на эталон той же гомографией, что
считает геометрия, и цвет сравнивается по всей области этикетки вокруг
совпавших точек.

Освещение в магазине меняет яркость сильнее, чем отличаются многие
этикетки, поэтому яркость кадра приводится к эталону по светлым участкам
вокруг совпавших точек — бумаге и золоту этикетки. Тёмная заливка после
этого остаётся тёмной, а общий сдвиг от лампы уходит.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from rerank import MAX_SIDE, MIN_MATCHES, detector, good_matches

ROOT = Path(__file__).resolve().parent.parent
WHITE_PCT = 90          # перцентиль яркости, который считается белым бумаги
BLUR = 3.0              # сглаживание перед сравнением: гасит неточность совмещения
EXPAND = 0.5            # на сколько раздвигается рамка совпавших точек с каждой стороны


def scaled_rgb(image: Image.Image) -> np.ndarray:
    """Изображение в том же масштабе, в котором считались точки SIFT."""
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1.0:
        arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return arr


class References:
    """Эталоны в масштабе SIFT, с кэшем: у соседних кадров общие кандидаты."""

    def __init__(self, by_slug: dict, capacity: int = 400) -> None:
        self.by_slug = by_slug
        self.cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self.capacity = capacity

    def get(self, slug: str) -> np.ndarray | None:
        if slug in self.cache:
            self.cache.move_to_end(slug)
            return self.cache[slug]
        photo = (self.by_slug.get(slug) or {}).get("photo")
        arr = None
        if photo and (ROOT / photo).exists():
            with Image.open(ROOT / photo) as im:
                arr = scaled_rgb(im)
        self.cache[slug] = arr
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return arr


def lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def color_distance(query_rgb: np.ndarray, query_kp, query_desc,
                   ref_rgb: np.ndarray, ref_kp, ref_desc) -> dict | None:
    """Расхождение цвета этикетки кадра и эталона после совмещения.

    Сравнивается только область, накрытая совпавшими точками (выпуклая
    оболочка, чуть расширенная): за её пределами гомография этикетки уже
    не описывает снимок — там стекло с отражениями полки и фон эталона.
    Яркость кадра приводится к эталону по светлым участкам рамки вокруг
    точек (бумага, золото); оттенок не выравнивается: приведение к белому
    сделало бы одинаковыми однотонные голубой и розовый ромбы.

    None — совместить нельзя: мало соответствий или гомография вырождена.
    """
    if query_desc is None or ref_desc is None or len(query_desc) < 2 or len(ref_desc) < 2:
        return None
    _, bf = detector()
    good = good_matches(bf.knnMatch(query_desc, ref_desc, k=2))
    if len(good) < MIN_MATCHES:
        return None
    src = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None or int(mask.sum()) < MIN_MATCHES:
        return None
    det = float(np.linalg.det(H[:2, :2]))
    if not 0.002 < abs(det) < 500:
        return None
    points = dst.reshape(-1, 2)[mask.ravel().astype(bool)]
    unique = np.unique(np.round(points), axis=0)
    if len(unique) < MIN_MATCHES:
        return None
    rh, rw = ref_rgb.shape[:2]
    warped = cv2.warpPerspective(query_rgb, H, (rw, rh))
    valid = cv2.warpPerspective(np.ones(query_rgb.shape[:2], np.uint8), H, (rw, rh)) > 0
    hull = np.zeros((rh, rw), np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(unique.astype(np.int32)), 1)
    hull = cv2.dilate(hull, np.ones((9, 9), np.uint8)).astype(bool) & valid
    if hull.sum() < 200:
        return None
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    dx, dy = (x1 - x0) * EXPAND, (y1 - y0) * EXPAND
    frame = np.zeros((rh, rw), bool)
    frame[max(int(y0 - dy), 0):min(int(y1 + dy) + 1, rh),
          max(int(x0 - dx), 0):min(int(x1 + dx) + 1, rw)] = True
    frame &= valid
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), BLUR)  # noqa: E731
    q, r = lab(blur(warped)), lab(blur(ref_rgb))
    gain = (np.percentile(r[..., 0][frame], WHITE_PCT)
            / max(np.percentile(q[..., 0][frame], WHITE_PCT), 1.0))
    q[..., 0] = np.clip(q[..., 0] * gain, 0, 100)
    delta = np.sqrt(((q - r) ** 2).sum(axis=-1))[hull]
    chroma = np.sqrt(((q[..., 1:] - r[..., 1:]) ** 2).sum(axis=-1))[hull]
    return {"mean": float(delta.mean()), "median": float(np.median(delta)),
            "chroma": float(chroma.mean()), "share25": float((delta > 25).mean()),
            "inliers": int(mask.sum()), "hull": float(hull.mean()), "gain": float(gain)}
