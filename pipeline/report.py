#!/usr/bin/env python3
"""Отчёт по метрикам ТЗ: F1 для топ-1 и топ-5, состояния выдачи, время ответа.

ТЗ требует показывать «метрику уверенности (F1) для топ-1 и для топ-5
карточек» и оценивает решение по F1 в поисковой выдаче. Точность top-k
на эту роль не годится: она не знает про отказ, а сервис умеет молчать,
и молчание нужно считать отдельно от ошибки.

Методика, одна на весь проект:

  Запрос считается обслуженным, если сервис показал карточку, то есть
  уверенность лидера не ниже порога показа. Дальше по каждому запросу:

    TP   карточка показана и лидер — правильное вино
    FP   карточка показана, но лидер другой (сюда же попадают запросы
         к винам, которых в каталоге нет: любая карточка для них ошибка)
    FN   правильный ответ существовал, но карточку не показали

  точность = TP / (TP + FP)        — чему можно верить из показанного
  полнота  = TP / (TP + FN)        — сколько существующих ответов дошло
  F1       = 2 · точность · полнота / (точность + полнота)

  F1 для топ-5 считается так же, но TP засчитывается, если правильное
  вино попало в показанную пятёрку.

Считается в двух постановках, потому что они отвечают на разные вопросы:

  по каталогу      только запросы, у которых правильный ответ существует.
                   Это постановка кейсодержателя: в публичном наборе вина
                   из каталога, и его F1 считается именно так.
  со строгим отказом  плюс запросы к винам, вычеркнутым из каталога. Любая
                   карточка для них — ошибка. Числа ниже, зато видно, чего
                   стоит готовность сервиса молчать.

Исходные данные берутся из артефакта калибровки: он получен прогоном того
же конвейера, что работает в сервисе, вместе с перестановкой по тексту.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"


def f1(true_positive: int, shown: int, answerable: int) -> tuple[float, float, float]:
    precision = true_positive / shown if shown else 0.0
    recall = true_positive / answerable if answerable else 0.0
    score = (2 * precision * recall / (precision + recall)
             if precision + recall else 0.0)
    return precision, recall, score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="sharp")
    ap.add_argument("--confident", type=float, default=0.70)
    ap.add_argument("--show", type=float, default=0.06)
    ap.add_argument("--held-out", action="store_true",
                    help="считать только по отложенной половине, на которой "
                         "пороги не подбирались")
    args = ap.parse_args()

    source = INDEX_DIR / f"calibration_{args.subset}.json"
    if not source.exists():
        sys.exit(f"нет {source} — сначала pipeline/calibrate.py --subset {args.subset}")
    data = json.loads(source.read_text(encoding="utf-8"))
    weights, cases = data["weights"], data["cases"]

    probability = np.array([core.probability({k: c[k] for k in core.FEATURE_ORDER},
                                             weights) for c in cases])
    present = np.array([c["present"] for c in cases])
    correct = np.array([c["correct"] for c in cases])
    in_top5 = np.array([c["in_top5"] for c in cases])

    mask = np.ones(len(cases), dtype=bool)
    if args.held_out:
        mask[:len(cases) // 2] = False

    shown = (probability >= args.show) & mask
    confident = (probability >= args.confident) & mask
    answerable = int((present & mask).sum())

    scores = {}
    for label, pool in (("по каталогу", present & mask), ("со строгим отказом", mask)):
        shown_here = shown & pool
        scores[label] = {
            "top1": f1(int((shown_here & correct).sum()),
                       int(shown_here.sum()), answerable),
            "top5": f1(int((shown_here & in_top5).sum()),
                       int(shown_here.sum()), answerable),
        }

    part = "отложенная половина" if args.held_out else "весь набор"
    print(f"\nнабор {args.subset}, {part}: {answerable} запросов с вином "
          f"в каталоге, {int((~present & mask).sum())} с вычеркнутым")
    print(f"пороги: показ {args.show}, без оговорок {args.confident}\n")

    print(f"{'постановка':22} {'метрика':9} {'точность':>10} {'полнота':>9} {'F1':>8}")
    for label, pair in scores.items():
        for name, (precision, recall, score) in pair.items():
            print(f"{label:22} {name:9} {precision * 100:>9.1f}% "
                  f"{recall * 100:>8.1f}% {score * 100:>7.1f}%")

    # Состояния выдачи по запросам, для которых ответ существует.
    live = present & mask
    rows = [
        ("одна карточка без оговорок", confident & live),
        ("карточка с вариантами", shown & ~confident & live),
        ("не удалось определить", ~shown & live),
    ]
    print(f"\n{'состояние выдачи':30} {'запросов':>9} {'из них верных':>14} "
          f"{'ошибок':>8}")
    for name, selector in rows:
        total = int(selector.sum())
        right = int((selector & correct).sum())
        print(f"{name:30} {total:>9} {right:>14} {total - right:>8}")
    missed = int((~shown & live & correct).sum())
    print(f"\nиз отказов правильный лидер был в {missed} случаях — это цена порога")
    print(f"верная карточка дошла до пользователя: "
          f"{int((shown & correct).sum())}/{answerable} "
          f"({(shown & correct).sum() / max(answerable, 1) * 100:.1f}%)")

    absent = ~present & mask
    print(f"\nвина нет в каталоге ({int(absent.sum())} запросов): карточка всё "
          f"равно показана в {int((absent & shown).sum())} случаях "
          f"({(absent & shown).sum() / max(absent.sum(), 1) * 100:.1f}%), "
          f"из них без оговорок — {int((absent & confident).sum())} "
          f"({(absent & confident).sum() / max(absent.sum(), 1) * 100:.1f}%)")

    latency_path = INDEX_DIR / f"latency_{args.subset}.json"
    latency = json.loads(latency_path.read_text()) if latency_path.exists() else {}
    if latency:
        print(f"\nвремя ответа: медиана {latency.get('median_ms', 0):.0f} мс, "
              f"95-й процентиль {latency.get('p95_ms', 0):.0f} мс "
              f"(SLA ТЗ — 3000 мс)")

    out = INDEX_DIR / f"report_{args.subset}.json"
    out.write_text(json.dumps({
        "subset": args.subset, "held_out": args.held_out,
        "show_threshold": args.show, "confident_threshold": args.confident,
        "answerable": answerable,
        "f1": {label: {name: {"precision": p, "recall": r, "f1": s}
                       for name, (p, r, s) in pair.items()}
               for label, pair in scores.items()},
        "states": {name: {"queries": int(selector.sum()),
                          "correct": int((selector & correct).sum())}
                   for name, selector in rows},
        "absent": {"queries": int(absent.sum()),
                   "shown": int((absent & shown).sum()),
                   "confident": int((absent & confident).sum())},
        "latency_ms": latency,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
