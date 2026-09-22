#!/usr/bin/env python3
"""Финальная проверка полного пути: файл → HTTP → /v1/eval/predict → slug.

Меряется то, что увидит скрипт кейсодержателя: ответ и время по HTTP,
включая чтение файла и сетевой путь. Кадры берутся из полевого манифеста,
правильный ответ — из него же; эндпоинт оценки порог не применяет, поэтому
здесь нет отказов, только совпадение slug.

Вывод: доля верных среди всех положительных кадров, медиана и p95 времени,
доля запросов дольше SLA. Отчёт пишется в JSON рядом с остальными.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIELD = ROOT / "data" / "field"
SCORABLE = {"front", "collage", "multi"}


def post_image(url: str, path: Path, timeout: float = 30.0) -> tuple[dict | None, float, int]:
    boundary = uuid.uuid4().hex
    payload = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
               f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
               ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            status = response.status
    except Exception:
        body, status = None, 0
    return body, (time.perf_counter() - started) * 1000, status


def locate(name: str) -> Path | None:
    for base in (FIELD, ROOT / "dataset" / "eval" / "queries"):
        if (base / name).exists():
            return base / name
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/eval/predict")
    ap.add_argument("--manifest", default=str(FIELD / "manifest.csv"))
    ap.add_argument("--split", choices=["tune", "test", "all"], default="all")
    ap.add_argument("--sla-ms", type=float, default=3000)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", default=str(ROOT / "data" / "validation" / "api_check.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    pairs = [(r, locate(r["image"])) for r in rows]
    missing = [r["image"] for r, p in pairs if p is None]
    if missing:
        sys.exit(f"нет кадров: {missing[:5]}")
    for r, path in pairs[:args.warmup]:
        post_image(args.endpoint, path)

    results = []
    for r, path in pairs:
        body, ms, status = post_image(args.endpoint, path)
        slug, shown, state, probability = None, None, None, None
        if isinstance(body, dict) and "status" in body:
            # /v1/search: карточка показана только при status != unsure;
            # лидер при отказе лежит первым в alternatives.
            state = body.get("status")
            probability = (body.get("confidence") or {}).get("probability")
            match = body.get("match")
            shown = match is not None
            slug = (match or {}).get("slug") if shown else (
                (body.get("alternatives") or [{}])[0].get("slug"))
        elif isinstance(body, dict):
            slug = body.get("slug")
        elif isinstance(body, list) and body:
            slug = (body[0] or {}).get("slug")
        results.append({"image": r["image"], "gold": r["slug"], "frame": r["frame"],
                        "kind": r["kind"], "split": r["split"], "predicted": slug,
                        "status": status, "latency_ms": round(ms, 1),
                        "correct": bool(r["slug"]) and slug == r["slug"],
                        "search_status": state, "shown": shown, "probability": probability})
        print(f"  {r['image'][:44]:44} {ms:7.0f} мс  {'ВЕРНО' if results[-1]['correct'] else ''}",
              flush=True)

    positives = [x for x in results if x["gold"]]
    scorable = [x for x in positives if x["frame"] in SCORABLE]
    negatives = [x for x in results if not x["gold"]]
    latencies = sorted(x["latency_ms"] for x in results)
    failed = sum(1 for x in results if x["status"] != 200)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * len(latencies))) - 1)] if latencies else 0
    over = sum(1 for v in latencies if v > args.sla_ms)
    summary = {
        "endpoint": args.endpoint, "split": args.split, "n": len(results),
        "http_failures": failed,
        "positives": len(positives),
        "top1_all_positives": sum(x["correct"] for x in positives) / max(len(positives), 1),
        "positives_front_visible": len(scorable),
        "top1_front_visible": sum(x["correct"] for x in scorable) / max(len(scorable), 1),
        "absent_answered": sum(1 for x in negatives if x["predicted"]),
        "absent_n": len(negatives),
        "latency_ms": {"median": statistics.median(latencies) if latencies else None,
                       "p95": p95, "max": latencies[-1] if latencies else None,
                       "over_sla": over, "over_sla_share": over / max(len(latencies), 1),
                       "sla_ms": args.sla_ms},
    }
    if any(x["shown"] is not None for x in results):
        # Поведение отказа видно только через /v1/search.
        shown_pos = [x for x in positives if x["shown"]]
        tp = sum(1 for x in shown_pos if x["correct"])
        fp = (len(shown_pos) - tp) + sum(1 for x in negatives if x["shown"])
        fn = len(positives) - tp
        summary["search"] = {
            "shown_total": sum(1 for x in results if x["shown"]),
            "refused_correct": sum(1 for x in positives if x["correct"] and not x["shown"]),
            "absent_shown": sum(1 for x in negatives if x["shown"]),
            "absent_n": len(negatives),
            "f1_top1_with_rejection": 2 * tp / max(2 * tp + fp + fn, 1),
            "states": {k: sum(1 for x in results if x["search_status"] == k)
                       for k in ("confident", "uncertain", "unsure")},
            "confident_wrong": sum(1 for x in results if x["search_status"] == "confident"
                                   and x["gold"] and not x["correct"]),
        }
    print(f"\nполный путь через API, часть {args.split}: {len(results)} кадров, "
          f"ошибок HTTP {failed}")
    if "search" in summary:
        sm = summary["search"]
        print(f"  /v1/search: показано {sm['shown_total']}, верных отказано {sm['refused_correct']}, "
              f"карточка винам вне каталога {sm['absent_shown']}/{sm['absent_n']}, "
              f"F1 top-1 с отказом {sm['f1_top1_with_rejection'] * 100:.1f}%, состояния {sm['states']}, "
              f"уверенных ошибок {sm['confident_wrong']}")
    print(f"  верных среди всех положительных: {sum(x['correct'] for x in positives)}/{len(positives)} "
          f"({summary['top1_all_positives'] * 100:.1f}%)")
    print(f"  верных среди кадров с видимой лицевой этикеткой: "
          f"{sum(x['correct'] for x in scorable)}/{len(scorable)} "
          f"({summary['top1_front_visible'] * 100:.1f}%)")
    print(f"  время: медиана {summary['latency_ms']['median']:.0f} мс, p95 {p95:.0f} мс, "
          f"максимум {latencies[-1]:.0f} мс, дольше {args.sla_ms:.0f} мс — {over} "
          f"({summary['latency_ms']['over_sla_share'] * 100:.1f}%)")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"отчёт: {out}")


if __name__ == "__main__":
    main()
