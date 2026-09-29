"""Декодирование загрузки в сервисе: поворот по EXIF и отказ на мусоре."""
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from service.main import decode_image  # noqa: E402


def jpeg(size, orientation=None) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    img.save(buf, "JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_exif_rotation_applied():
    # Кадр 400×300 с Orientation=6 — на экране телефона он 300×400.
    img = decode_image(jpeg((400, 300), orientation=6))
    assert img.size == (300, 400)
    assert img.mode == "RGB"


def test_without_orientation_unchanged():
    raw = jpeg((400, 300))
    img = decode_image(raw)
    assert img.size == (400, 300)
    assert len(img.info["sha1"]) == 40


def test_garbage_is_none():
    assert decode_image(b"not an image") is None
