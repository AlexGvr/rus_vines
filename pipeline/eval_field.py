#!/usr/bin/env python3
"""Замер на полевом наборе: реальные снимки вместо синтетики.

Полевой набор устроен иначе, чем синтетический, и отвечает на другой вопрос.
Синтетика умеет много положительных примеров — запрос делается из эталона,
поэтому правильный ответ известен всегда. Но такой запрос легче настоящего,
и пороги, обученные на нём, на съёмке отказывают.

Реальные снимки из отзывов покупателей дают обратное распределение: почти
все они — вина, которых в каталоге нет или у которых другое поколение
этикетки. Положительных мало. Зато они впервые позволяют измерить главную
ошибку отказа не на выдуманных, а на настоящих кадрах: как часто сервис
показывает карточку вину, которого у него нет.

Поэтому отчёт делится честно:
  отрицательные — измеряются на полевых данных, их достаточно;
  положительные — перечисляются поимённо, их слишком мало для метрики.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import searchcore as core  # noqa: E402
from embed import embed_images  # noqa: E402
from imageprep import normalize_batch  # noqa: E402
from ocr import read_words  # noqa: E402
from rerank import PrecomputedReranker  # noqa: E402
from text_match import TextChannel  # noqa: E402

FIELD = ROOT / "data" / "field"
PUBLIC = ROOT / "dataset" / "eval" / "queries"
INDEX_DIR = ROOT / "data" / "index"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def locate(name: str) -> Path | None:
    for base in (FIELD, PUBLIC):
        if (base / name).exists():
            return base / name
    return None


EQUIVALENTS = ROOT / "data" / "catalog" / "equivalents.csv"


def load_equivalents(wines: dict) -> set[frozenset]:
    """Пары записей каталога об одном и том же вине — из явного списка.

    Автоматическому признаку тут не место. Похожесть названий склеивала
    «Шато Тамань. Каберне Совиньон» (розовое) с «Шато Тамань. Каберне»
    (красное) — совпадают винодельня, линейка и сорт, расходится цвет,
    и метрика от этого завышалась. Пара попадает в список только после
    ручной сверки шести признаков; если хоть один определить нельзя,
    эквивалентность считается неподтверждённой.

    Файл проверяется при загрузке: пара с разной винодельней или разной
    категорией — ошибка списка, а не повод для тихого исключения.
    """
    if not EQUIVALENTS.exists():
        return set()
    out: set[frozenset] = set()
    with EQUIVALENTS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(line for line in fh if not line.startswith("#")):
            first, second = (row.get("slug_a") or "").strip(), (row.get("slug_b") or "").strip()
            if not first or not second:
                continue
            for slug in (first, second):
                if slug not in wines:
                    sys.exit(f"{EQUIVALENTS.name}: нет такой позиции — {slug}")
            left, right = wines[first], wines[second]
            if left["manufacturer"] != right["manufacturer"]:
                sys.exit(f"{EQUIVALENTS.name}: разные винодельни у {first} и {second}")
            if left["category"].strip().lower() != right["category"].strip().lower():
                sys.exit(f"{EQUIVALENTS.name}: разные категории у {first} и {second}")
            out.add(frozenset((first, second)))
    return out


def resolve_all(rows: list[dict]) -> list[tuple[dict, Path]]:
    """Все кадры манифеста или отказ считать.

    Фотографии в репозиторий не коммитятся, поэтому на чужой машине их
    может не быть. Молча пропускать такие строки нельзя: отчёт получится
    по огрызку набора, а выглядеть будет как полноценный замер. Лучше
    остановиться и сказать, чего не хватает.
    """
    found, missing = [], []
    for row in rows:
        path = locate(row["image"])
        (found.append((row, path)) if path else missing.append(row["image"]))
    if missing:
        print(f"нет {len(missing)} кадров из {len(rows)}, например: "
              f"{', '.join(missing[:5])}")
        sys.exit("полевой набор неполон — соберите его pipeline/collect_field.py "
                 "или укажите другой манифест")
    return found


# Кадры, по которым метрику считать осмысленно: в кадре видна та сторона
# бутылки, которая есть в индексе. Индекс собран из лицевых фотографий
# каталога, поэтому контрэтикетка и горлышко без этикетки не находятся
# ни при каком качестве поиска — это дефект охвата каталога, а не поиска.
# Коллаж и кадр с несколькими бутылками остаются: лицевая этикетка нужного
# вина в них есть.
SCORABLE = {"front", "collage", "multi"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(FIELD / "manifest.csv"))
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--weights", default=str(INDEX_DIR / "confidence.json"),
                    help="веса и пороги; тот же файл читает сервис")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--views", default="raw,detect",
                    help="виды запроса через запятую, шаги внутри вида через '+'")
    ap.add_argument("--split", choices=["tune", "test", "all"], default="all",
                    help="часть выборки: tune — для подбора порогов, "
                         "test — финальная проверка, её для настройки не трогаем")
    args = ap.parse_args()

    query_views = [tuple(x for x in spec.split("+") if x and x != "raw")
                   for spec in args.views.split(",")]
    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    if args.split != "all":
        rows = [r for r in rows if r.get("split") == args.split]
    if not rows:
        sys.exit(f"в манифесте нет кадров части {args.split}")
    artifact = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    weights = artifact["outcome_weights"]
    show_p = artifact["thresholds"]["show_p"]
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}

    vectors = np.load(INDEX_DIR / f"{args.index}.npy")
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        index_slugs = [row["slug"] or None for row in csv.DictReader(fh)]
    # Папка признаков задаётся снаружи: масштаб признаков эталонов должен
    # совпадать с SIFT_MAX_SIDE запроса, иначе сравниваются разные вещи.
    reranker = PrecomputedReranker(
        os.environ.get("SIFT_DIR", str(INDEX_DIR / "sift")))
    channel = TextChannel(wines)
    equivalents = load_equivalents(by_slug)
    # Позиция «находима», если её эталон попал в индекс. Одного поля photo
    # для этого мало: у части позиций фото в дампе нет вовсе, и эталон для
    # них приходит из data/catalog/extra_refs.csv.
    sys.path.insert(0, str(ROOT / "pipeline"))
    from build_index import extra_references
    findable_slugs = {slug for slug, wine in by_slug.items() if wine.get("photo")}
    findable_slugs |= {slug for _, slug in extra_references()}

    # Полевой кадр не должен оказаться среди эталонов: тогда запрос искал бы
    # сам себя. Индекс собирается только из фотографий каталога, но проверка
    # стоит копейки, а молчаливое пересечение обесценило бы весь замер.
    with (INDEX_DIR / f"{args.index}.csv").open(encoding="utf-8") as fh:
        indexed = {Path(row["path"]).name for row in csv.DictReader(fh)}
    leaked = {row["image"] for row in rows} & indexed
    if leaked:
        sys.exit(f"полевые кадры попали в индекс эталонов: {sorted(leaked)[:5]}")

    results = []
    for row, path in resolve_all(rows):
        image = Image.open(path).convert("RGB")
        views = {steps: normalize_batch([image], steps=steps)[0]
                 for steps in query_views}
        query = embed_images(list(views.values()))
        similarity = vectors @ query.T
        scores = similarity.max(axis=1)
        candidates = core.shortlist(scores, index_slugs, args.topk)
        # Тот же выбор кропа, что в сервисе: иначе замер описывает не тот
        # конвейер, который отвечает пользователю.
        shot = list(views.values())[core.pick_view(list(views.values()), similarity)]

        # Позиция правильного ответа после каждой ступени: без этого промах
        # нельзя приписать к этапу, а лечатся этапы по-разному.
        def place(gold: str) -> int:
            order = [c.slug for c in candidates]
            return order.index(gold) if gold in order else -1

        rank_shortlist = place(row["slug"]) if row["slug"] else -1
        core.rerank(shot, candidates, reranker)
        rank_geometry = place(row["slug"]) if row["slug"] else -1
        label = core.LabelText(shot, read_words)
        core.resolve(candidates, shot, channel, label,
                     window=0.80, min_conf=0.60)
        rank_text = place(row["slug"]) if row["slug"] else -1
        core.check_leader(candidates, shot, channel, label,
                          0.60, 0.80, by_slug)
        feats = core.features(candidates)
        results.append({
            "image": row["image"],
            "gold": row["slug"],
            "kind": row.get("kind", "field"),
            "frame": row.get("frame", "front"),
            "rank_shortlist": rank_shortlist,
            "rank_geometry": rank_geometry,
            "rank_text": rank_text,
            "top1": candidates[0].slug if candidates else "",
            "inliers": candidates[0].inliers if candidates else 0,
            "probability": core.outcomes(feats, weights)[0],
            "probability_top5": core.outcomes(feats, weights)[1],
            "plain": core.may_drop_caveat(candidates, by_slug, 0.80) if candidates else False,
            "conflicts": list(candidates[0].conflicts) if candidates else [],
            "findable": bool(row["slug"]) and row["slug"] in findable_slugs
                        and row.get("frame", "front") in SCORABLE,
            "duplicate": bool(row["slug"]) and bool(candidates)
                         and candidates[0].slug != row["slug"]
                         and frozenset((row["slug"], candidates[0].slug)) in equivalents,
            "correct": bool(row["slug"]) and bool(candidates)
                       and candidates[0].slug == row["slug"],
            "in_top5": bool(row["slug"])
                       and row["slug"] in [c.slug for c in candidates[:5]],
        })
        print(f"  {row['image'][:44]:44} p={results[-1]['probability']:.3f} "
              f"инл {results[-1]['inliers']:>4} "
              f"{'ВЕРНО' if results[-1]['correct'] else ''}", flush=True)

    positives = [r for r in results if r["gold"]]
    negatives = [r for r in results if not r["gold"]]
    shot = [r for r in results if r["kind"] == "field"]
    part = args.split if args.split != "all" else "весь набор"
    print(f"\nполевой набор, часть {part}: {len(results)} кадров — "
          f"{len(positives)} с вином из каталога, {len(negatives)} без; "
          f"съёмка {len(shot)}, предметная карточка {len(results) - len(shot)}")
    print(f"{'категория кадров':34} {'кадров':>7} {'лидер верен':>12}")
    groups = [
        ("съёмка, вино с эталоном", lambda r: r["kind"] == "field" and r["findable"]),
        ("предметная, вино с эталоном", lambda r: r["kind"] == "studio" and r["findable"]),
        ("вино в каталоге без эталона", lambda r: bool(r["gold"]) and not r["findable"]),
        ("вина нет в каталоге", lambda r: not r["gold"]),
    ]
    for name, keep in groups:
        chunk = [r for r in results if keep(r)]
        if not chunk:
            continue
        right = sum(1 for r in chunk if r["correct"])
        mark = f"{right}" if any(r["gold"] for r in chunk) else "—"
        print(f"{name:34} {len(chunk):>7} {mark:>12}")

    if negatives:
        probability = np.array([r["probability"] for r in negatives])
        inliers = np.array([r["inliers"] for r in negatives])
        print(f"\nвина в каталоге нет ({len(negatives)} кадров):")
        print(f"  инлаеры лидера: медиана {np.median(inliers):.0f}, "
              f"90-й процентиль {np.percentile(inliers, 90):.0f}, "
              f"максимум {inliers.max()}")
        print(f"  уверенность: медиана {np.median(probability):.3f}, "
              f"90-й процентиль {np.percentile(probability, 90):.3f}, "
              f"максимум {probability.max():.3f}")
        print(f"\n{'порог':>7} {'карточка показана':>19}")
        for threshold in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70):
            share = float((probability >= threshold).mean())
            print(f"{threshold:>7.2f} {share * 100:>18.1f}%")

    if positives:
        # Позиция может быть в каталоге, но без эталонного фото — таких
        # пятнадцать из 2103. Найти их нельзя ни при каких порогах, и мерить
        # по ним качество поиска бессмысленно: это дыра в данных.
        findable = [r for r in positives if r["findable"]]
        blind = [r for r in positives if not r["findable"]]
        right = sum(1 for r in findable if r["correct"])
        twins = sum(1 for r in findable if r["duplicate"])
        print(f"\nвино есть в каталоге: {len(positives)} кадров, из них с эталонным "
              f"фото {len(findable)}")
        print(f"  лидер верен в {right} из {len(findable)} "
              f"({right / max(len(findable), 1) * 100:.0f}%), "
              f"верный в пятёрке — {sum(1 for r in findable if r['in_top5'])}")
        if twins:
            print(f"  ещё в {twins} случаях выдана запись, признанная тем же вином "
                  f"(data/catalog/equivalents.csv) — с ними {right + twins} "
                  f"из {len(findable)} ({(right + twins) / len(findable) * 100:.0f}%)")
        elif not equivalents:
            print("  подтверждённых дублей каталога нет, метрика строгая по slug")
        if blind:
            print(f"  ещё {len(blind)} кадров у позиции без эталонного фото — "
                  f"найти нельзя, в счёт не идут")
        by_wine: dict[str, list] = {}
        for r in findable:
            by_wine.setdefault(r["gold"], []).append(r)
        print(f"\n{'вино':44} {'верно':>7} {'кадров':>7}")
        for gold, group in sorted(by_wine.items(),
                                  key=lambda kv: -sum(x["correct"] for x in kv[1])):
            title = by_slug.get(gold, {}).get("title", gold)[:42]
            print(f"{title:44} {sum(r['correct'] for r in group):>7} {len(group):>7}")

    # Где именно теряется правильный ответ. Этапы не пересекаются: промах
    # приписывается первому, на котором ответ ушёл с первого места.
    losses = [r for r in results if r["gold"] and r["findable"] and not r["correct"]]
    if losses:
        def stage(r):
            if r["rank_shortlist"] < 0:
                return "не попал в шортлист"
            if r["rank_geometry"] != 0:
                return "проиграл геометрии"
            if r["rank_text"] != 0:
                return "потерян перестановкой по тексту"
            return "первый, но карточка не показана"
        counts: dict[str, int] = {}
        for r in losses:
            counts[stage(r)] = counts.get(stage(r), 0) + 1
        hidden = sum(1 for r in results
                     if r["gold"] and r["findable"] and r["correct"]
                     and r["probability"] < show_p)
        print(f"\nгде теряется правильный ответ ({len(losses)} промахов):")
        for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {name:36} {count:>4}")
        if hidden:
            print(f"  {'верен, но ниже порога показа':36} {hidden:>4}")

    # Разбивка по тому, что вообще попало в кадр. Лицевая этикетка, оборот,
    # коллаж и горлышко без этикетки — разные задачи, и смешивать их
    # в одном проценте значит прятать, что оборот не ищется в принципе.
    names = {"front": "лицевая этикетка", "back": "контрэтикетка",
             "collage": "коллаж лицевой и оборота", "multi": "несколько бутылок",
             "neck": "горлышко, этикетки не видно"}
    skipped = [r for r in results if r["frame"] not in SCORABLE]
    if skipped:
        with_wine = [r for r in skipped if r["gold"]]
        right = sum(1 for r in with_wine if r["correct"])
        print(f"\nне учтено в метрике: {len(skipped)} кадров — в кадре нет той "
              f"стороны бутылки, что лежит в индексе")
        for key in sorted({r["frame"] for r in skipped}):
            part = [r for r in skipped if r["frame"] == key]
            print(f"  {names.get(key, key):32} {len(part):>4}")
        print(f"  из них с вином каталога {len(with_wine)}, и поиск угадал "
              f"{right} — засчитывать это как отказ поиска нечестно в обе стороны")

    print(f"\n{'что в кадре':32} {'кадров':>8} {'верных':>8} {'в пятёрке':>11}")
    for key, name in names.items():
        part = [r for r in results if r["frame"] == key and r["gold"] and r["findable"]]
        if not part:
            continue
        print(f"{name:32} {len(part):>8} "
              f"{sum(1 for r in part if r['correct']):>8} "
              f"{sum(1 for r in part if r['in_top5']):>11}")

    for kind in ("field", "studio"):
        part = [r for r in results if r["kind"] == kind]
        if not part:
            continue
        good = [r for r in part if r["gold"]]
        bad = [r for r in part if not r["gold"]]
        name = "съёмка" if kind == "field" else "предметная карточка"
        line = (f"\n{name}: {len(part)} кадров")
        if good:
            line += (f", лидер верен в {sum(1 for r in good if r['correct'])}"
                     f" из {len(good)}")
        if bad:
            shown = sum(1 for r in bad if r["probability"] >= show_p)
            line += (f", карточка для отсутствующего вина в {shown}"
                     f" из {len(bad)}")
        print(line)

    # Три состояния выдачи — ровно те, что видит пользователь. Считаем их
    # здесь же: отдельная таблица по синтетике ничего не говорит о съёмке.
    confident_p = artifact["thresholds"]["confident_p"]
    findable = [r for r in results if r["gold"] and r["findable"]]
    absent = [r for r in results if not r["gold"]
              and r["frame"] in SCORABLE]
    states = {
        "одна карточка без оговорок":
            [r for r in findable if r["probability"] >= confident_p and r["plain"]],
        "карточка с вариантами":
            [r for r in findable if r["probability"] >= show_p
             and not (r["probability"] >= confident_p and r["plain"])],
        "не удалось определить":
            [r for r in findable if r["probability"] < show_p],
    }
    print(f"\n{'состояние выдачи':30} {'кадров':>8} {'доля':>7} {'верных':>8} "
          f"{'ошибок':>8} {'точность':>9}")
    for name, group in states.items():
        right = sum(1 for r in group if r["correct"])
        share = len(group) / max(len(findable), 1) * 100
        precision = right / len(group) * 100 if group else 0.0
        print(f"{name:30} {len(group):>8} {share:>6.0f}% {right:>8} "
              f"{len(group) - right:>8} {precision:>8.0f}%")
    # Почему уверенных ответов может не быть вовсе: порог по вероятности —
    # только половина условия, вторая половина это правило снятия оговорки.
    over = [r for r in findable if r["probability"] >= confident_p]
    if over and not states["одна карточка без оговорок"]:
        right = sum(1 for r in over if r["correct"])
        print(f"  порог {confident_p} прошли {len(over)} кадров "
              f"(верных {right}), но правило снятия оговорки не сработало "
              f"ни на одном: различающего слова на этикетке не прочитано")
    wrong_confident = [r for r in states["одна карточка без оговорок"]
                       if not r["correct"]]
    for r in wrong_confident:
        print(f"  уверенная ошибка: {r['image']} -> {r['top1']} "
              f"(правильно {r['gold']}, p={r['probability']:.3f})")
    shown = states["одна карточка без оговорок"] + states["карточка с вариантами"]
    shown_absent = [r for r in absent if r["probability"] >= show_p]
    right = sum(1 for r in shown if r["correct"])
    right5 = sum(1 for r in shown if r["in_top5"])
    total_shown = len(shown) + len(shown_absent)
    if total_shown and findable:
        for name, hits in (("top-1", right), ("top-5", right5)):
            precision = hits / total_shown
            recall = hits / len(findable)
            score = 2 * precision * recall / max(precision + recall, 1e-9)
            print(f"{name}: точность {precision * 100:.1f}%, "
                  f"полнота {recall * 100:.1f}%, F1 {score * 100:.1f}% "
                  f"(показов {total_shown}, из них для вин вне каталога "
                  f"{len(shown_absent)})")

    out = INDEX_DIR / "field_report.json"
    out.write_text(json.dumps({
        "n": len(results), "positives": len(positives), "negatives": len(negatives),
        "results": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
