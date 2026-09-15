#!/usr/bin/env python3
"""S1.1 + S1.2 — приведение кадра к виду, сравнимому с эталоном каталога.

Запрос снят у полки: бутылка под углом, рядом чужие бутылки, блик от витрины,
тёплый свет. Эталон — студийное фото одной бутылки на белом. Пока обе стороны
идут в модель «как есть», вектор запроса описывает в том числе фон и соседей.

Конвейер выравнивает обе стороны, шаг за шагом:
  detect  — YOLO находит бутылки, выбирается целевая (крупная и по центру)
  label   — внутри бутылки ищется полоса этикетки по плотности деталей
  dewarp  — разворот цилиндра: края этикетки сжаты проекцией, растягиваем
  glare   — блик снимается инпейнтом по маске пересвета
  balance — серый мир плюс CLAHE, чтобы убрать цвет освещения

Каждый шаг включается отдельно: замер вклада — задача S1.6.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

BOTTLE_CLASS = 39  # COCO: bottle
ALL_STEPS = ("detect", "label", "dewarp", "glare", "balance")
_yolo: dict = {}

# Разрешение и порог детектора. Держатся в переменных окружения, потому что
# менять их можно только одновременно для индекса и для запроса: кроп входит
# в представление, и односторонняя правка сравнивает разные сущности.
CONF = float(os.environ.get("YOLO_CONF", "0.20"))
IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))


def _device() -> str:
    """Устройство инференса. Образ собирается и под CPU, поэтому жёстко
    требовать видеокарту нельзя."""
    if "device" not in _yolo:
        import torch
        _yolo["device"] = 0 if torch.cuda.is_available() else "cpu"
    return _yolo["device"]


def load_detector(weights: str | None = None):
    """Детектор бутылок. Путь к весам можно задать через YOLO_WEIGHTS —
    в образе они лежат рядом с кодом, чтобы не качаться при каждом старте."""
    if "model" not in _yolo:
        from ultralytics import YOLO
        _yolo["model"] = YOLO(weights or os.environ.get("YOLO_WEIGHTS", "yolo11m.pt"))
    return _yolo["model"]


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


def detect_bottles(images: list[Image.Image], conf: float | None = None,
                   batch_size: int = 16) -> list[Box | None]:
    """Целевая бутылка на каждом кадре.

    Пользователь наводит камеру на нужную бутылку, поэтому из нескольких
    найденных выбираем ту, что крупнее и ближе к центру кадра: соседи по полке
    обычно срезаны краем и смещены.
    """
    model = load_detector()
    out: list[Box | None] = []
    for i in range(0, len(images), batch_size):
        chunk = [np.array(im.convert("RGB"))[:, :, ::-1] for im in images[i:i + batch_size]]
        results = model.predict(chunk, classes=[BOTTLE_CLASS],
                                conf=CONF if conf is None else conf,
                                imgsz=IMGSZ, verbose=False, device=_device())
        for im, res in zip(images[i:i + batch_size], results):
            width, height = im.size
            best, best_score = None, 0.0
            for box, score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = (int(v) for v in box)
                candidate = Box(x1, y1, x2, y2)
                rel_area = candidate.area / max(width * height, 1)
                cx = (x1 + x2) / 2 / max(width, 1)
                centrality = 1.0 - min(abs(cx - 0.5) * 2, 1.0)
                rank = score * (rel_area ** 0.5) * (0.35 + 0.65 * centrality)
                if rank > best_score:
                    best, best_score = candidate, rank
            out.append(best)
    return out


def crop_box(im: Image.Image, box: Box, pad: float = 0.04) -> Image.Image:
    width, height = im.size
    dx, dy = int((box.x2 - box.x1) * pad), int((box.y2 - box.y1) * pad)
    return im.crop((max(0, box.x1 - dx), max(0, box.y1 - dy),
                    min(width, box.x2 + dx), min(height, box.y2 + dy)))


def find_label_band(im: Image.Image, min_frac: float = 0.32) -> Image.Image:
    """Полоса этикетки внутри бутылки.

    Стекло и фон гладкие, этикетка — единственная зона с плотными деталями,
    поэтому ищем горизонтальную полосу с максимальной суммой градиента.
    """
    arr = np.array(im.convert("L"))
    height = arr.shape[0]
    if height < 40:
        return im
    energy = np.abs(cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    energy += np.abs(cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=1)
    energy = cv2.GaussianBlur(energy.reshape(-1, 1), (1, 15), 0).ravel()

    window = max(int(height * min_frac), 20)
    cumulative = np.concatenate([[0.0], np.cumsum(energy)])
    sums = cumulative[window:] - cumulative[:-window]
    if sums.size == 0:
        return im
    start = int(np.argmax(sums))
    # Расширяем полосу, пока детали держатся выше трети среднего: у этикетки
    # плотность деталей неровная — гравюра ярче текста, и на строгом пороге
    # полоса схлопывается на одну картинку, теряя название.
    threshold = energy[start:start + window].mean() * 0.35
    top, bottom = start, start + window
    while top > 0 and energy[top - 1] > threshold:
        top -= 1
    while bottom < height and energy[bottom] > threshold:
        bottom += 1
    pad = int((bottom - top) * 0.12)
    return im.crop((0, max(0, top - pad), im.width, min(height, bottom + pad)))


def dewarp_cylinder(im: Image.Image, theta: float = 1.05) -> Image.Image:
    """Разворот этикетки с цилиндра.

    Видимая часть бутылки — дуга: к краям одинаковый кусок этикетки занимает
    меньше пикселей. Обратное отображение x = sin(u·θ)/sin(θ) растягивает края.
    """
    arr = np.array(im.convert("RGB"))
    height, width = arr.shape[:2]
    if width < 16:
        return im
    u = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    src_x = (np.sin(u * theta) / np.sin(theta) + 1.0) * 0.5 * (width - 1)
    # remap требует непрерывные float32-карты: операции выше повышают тип до float64
    map_x = np.ascontiguousarray(np.tile(src_x, (height, 1)), dtype=np.float32)
    map_y = np.ascontiguousarray(
        np.tile(np.arange(height).reshape(-1, 1), (1, width)), dtype=np.float32)
    warped = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(warped)


def suppress_glare(im: Image.Image) -> Image.Image:
    """Инпейнт пересвеченных пятен: блик витрины стирает текст этикетки."""
    arr = np.array(im.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] > 245) & (hsv[:, :, 1] < 40)).astype(np.uint8)
    if mask.mean() < 0.002 or mask.mean() > 0.35:
        return im  # блика нет либо кадр целиком светлый — инпейнт только навредит
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    return Image.fromarray(cv2.inpaint(arr, mask, 4, cv2.INPAINT_TELEA))


def balance(im: Image.Image) -> Image.Image:
    """Серый мир против цвета освещения плюс CLAHE против тусклого света."""
    arr = np.array(im.convert("RGB")).astype(np.float32)
    means = arr.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    if means.min() > 1.0:
        arr = np.clip(arr * (gray / means), 0, 255)
    lab = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def normalize_batch(images: list[Image.Image], steps: tuple[str, ...] = ALL_STEPS,
                    batch_size: int = 16) -> list[Image.Image]:
    """Конвейер целиком. Детекция идёт пачкой — она самая дорогая часть."""
    out = list(images)
    if "detect" in steps:
        boxes = detect_bottles(out, batch_size=batch_size)
        out = [crop_box(im, box) if box is not None else im
               for im, box in zip(out, boxes)]
    result = []
    for im in out:
        if "label" in steps:
            im = find_label_band(im)
        if "dewarp" in steps:
            im = dewarp_cylinder(im)
        if "glare" in steps:
            im = suppress_glare(im)
        if "balance" in steps:
            im = balance(im)
        result.append(im)
    return result
