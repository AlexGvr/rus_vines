#!/usr/bin/env python3
"""S4.1 — замер времени ответа сервиса по HTTP, как его видит клиент.

Меряем не внутренние тайминги, а полный путь запроса: скрипт кейсодержателя
и браузер видят именно его. Прогревочные запросы не учитываются — первый
вызов поднимает ленивые части модели и портит картину.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
QUERIES = ROOT / "data" / "queries"


def post_image(url: str, path: Path) -> tuple[int, float]:
    boundary = uuid.uuid4().hex
    payload = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    request = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            status = response.status
    except Exception:
        status = 0
    return status, (time.perf_counter() - started) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/search")
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    manifest = list(csv.DictReader(
        (QUERIES / f"manifest_{args.subset}.csv").open(encoding="utf-8")))
    images = [QUERIES / args.subset / row["image"] for row in manifest]
    images = [p for p in images if p.exists()]
    random.Random(7).shuffle(images)
    images = images[:args.count + args.warmup]
    if not images:
        raise SystemExit(f"не нашлось кадров набора {args.subset}")

    for path in images[:args.warmup]:
        post_image(args.endpoint, path)

    timings, failures = [], 0
    for n, path in enumerate(images[args.warmup:], 1):
        status, elapsed = post_image(args.endpoint, path)
        if status != 200:
            failures += 1
            continue
        timings.append(elapsed)
        if n % 10 == 0:
            print(f"  {n}/{len(images) - args.warmup}", flush=True)

    arr = np.array(timings)
    print(f"\nзапросов {len(arr)}, ошибок {failures}, набор {args.subset}")
    print(f"  медиана     {np.median(arr):7.0f} мс")
    print(f"  95-й проц.  {np.percentile(arr, 95):7.0f} мс")
    print(f"  99-й проц.  {np.percentile(arr, 99):7.0f} мс")
    print(f"  максимум    {arr.max():7.0f} мс")
    print(f"  SLA 3000 мс: уложилось {(arr < 3000).mean() * 100:.1f}% запросов")

    out = ROOT / "data" / "index" / f"latency_{args.subset}.json"
    out.write_text(json.dumps({
        "n": len(arr), "failures": failures,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
        "within_sla": float((arr < 3000).mean()),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"результаты: {out}")


if __name__ == "__main__":
    main()
