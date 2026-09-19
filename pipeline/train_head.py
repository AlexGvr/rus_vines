#!/usr/bin/env python3
"""Линейная голова поверх замороженного SigLIP: развести соседей по линейке.

Зачем именно это. Соседи по линейке — главный источник ошибок на реальных
снимках, и все приёмы поверх готовых векторов уже проверены и отброшены:
взвешивание совпавших точек по зонам этикетки, цвет вокруг совпавших
точек, чтение различающей надписи, дополнительные эталоны. Причина видна
в числах (pipeline/siblings.py): косинус между эталонами соседей в среднем
0.891 против 0.618 у чужих виноделен, а на реальных кадрах разрыв между
правильным эталоном и ближайшим соседом равен -0.005 по медиане — то есть
сосед в среднем ближе. Никакое правило поверх такого представления соседей
не разведёт; надо менять само представление.

Дообучать so400m целиком ради 419 позиций незачем и не на чем. Вместо
этого обучается линейное отображение поверх замороженных векторов:

    v' = нормировать(v + U·v),  U инициализируется нулём

Голова начинается с тождественного отображения, поэтому в худшем случае
она ничего не портит, и обучение можно остановить в любой момент.

Положительные примеры — синтетические запросы к позиции против её эталона.
Отрицательные — эталоны соседей по линейке (data/catalog/siblings.csv)
и случайные позиции. Вес у соседей выше: именно на них модель ошибается.

Честная оговорка. Положительные примеры синтетические, то есть сделаны из
того же файла, что эталон. Голова может выучить устойчивость к аугментации
вместо устойчивости к съёмке. Проверяется это только на полевом наборе,
и только на настроечной его части.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from embed import embed_paths  # noqa: E402
from siblings import series_key  # noqa: E402

INDEX_DIR = ROOT / "data" / "index"
QUERIES = ROOT / "data" / "queries"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"


def load_index(name: str):
    vectors = np.load(INDEX_DIR / f"{name}.npy")
    rows = list(csv.DictReader((INDEX_DIR / f"{name}.csv").open(encoding="utf-8")))
    return vectors, rows


def build_pairs(subsets: list[str], batch_size: int):
    """Векторы синтетических запросов и slug правильной позиции."""
    paths, gold = [], {}
    for subset in subsets:
        manifest = QUERIES / f"manifest_{subset}.csv"
        if not manifest.exists():
            continue
        for row in csv.DictReader(manifest.open(encoding="utf-8")):
            path = QUERIES / subset / row["image"]
            if path.exists():
                paths.append(str(path))
                gold[str(path)] = row["slug"]
    # Тот же вид, что в запросе сервиса: кроп по бутылке.
    vectors, kept = embed_paths(paths, batch_size=batch_size, progress_every=0,
                                steps=("detect",))
    return vectors, [gold[p] for p in kept]


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-9)


def train(queries: np.ndarray, targets: np.ndarray, negatives: np.ndarray,
          steps: int, rate: float, decay: float, temperature: float,
          rank: int) -> np.ndarray:
    """InfoNCE по своему эталону против соседей и случайных позиций.

    U держится низкоранговым: параметров 1152·1152 больше, чем обучающих
    пар, и полная матрица запомнила бы выборку целиком.

    Считается через autograd, а не вручную: нормировка стоит между
    отображением и косинусом, и её якобиан в ручном градиенте пришлось бы
    опустить. Первая версия так и делала — обучение расходилось за сорок
    шагов, доля правильных падала с 81% до 2%.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = queries.shape[1]
    q = torch.tensor(queries, dtype=torch.float32, device=device)
    pos = torch.tensor(targets, dtype=torch.float32, device=device)
    neg = torch.tensor(negatives, dtype=torch.float32, device=device)

    generator = torch.Generator(device="cpu").manual_seed(17)
    left = torch.nn.Parameter(
        torch.randn(dim, rank, generator=generator).to(device) * 0.01)
    right = torch.nn.Parameter(torch.zeros(rank, dim, device=device))
    optimizer = torch.optim.Adam([left, right], lr=rate, weight_decay=decay)

    def head(v):
        return torch.nn.functional.normalize(v + (v @ left) @ right, dim=-1)

    for step in range(steps):
        optimizer.zero_grad()
        hq, hp, hn = head(q), head(pos), head(neg)
        logits_pos = (hq * hp).sum(-1, keepdim=True)
        logits_neg = torch.einsum("nd,nkd->nk", hq, hn)
        logits = torch.cat([logits_pos, logits_neg], dim=1) / temperature
        loss = torch.nn.functional.cross_entropy(
            logits, torch.zeros(len(logits), dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()
        if step % 40 == 0 or step == steps - 1:
            with torch.no_grad():
                correct = (logits_pos > logits_neg).all(dim=1).float().mean().item()
            print(f"  шаг {step:>4}: потеря {loss.item():.4f}, "
                  f"свой впереди всех соперников {correct * 100:.1f}%", flush=True)

    with torch.no_grad():
        return (left @ right).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="clean_mv")
    ap.add_argument("--subsets", default="sharp,hard,neardup")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--rate", type=float, default=1e-3)
    ap.add_argument("--decay", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--randoms", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=str(INDEX_DIR / "head.npy"))
    args = ap.parse_args()

    vectors, rows = load_index(args.index)
    whole = {row["slug"]: i for i, row in enumerate(rows)
             if row.get("view", "raw") == "raw" and row["slug"]}
    wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
    by_slug = {w["slug"]: w for w in wines}

    groups: dict[str, list[str]] = defaultdict(list)
    for wine in wines:
        if wine["slug"] in whole:
            groups[series_key(wine)].append(wine["slug"])

    queries, gold = build_pairs(args.subsets.split(","), args.batch_size)
    print(f"обучающих запросов {len(queries)}, позиций в индексе {len(whole)}")

    rng = np.random.default_rng(17)
    pool = list(whole)
    keep, targets, negatives = [], [], []
    slots = 3 + args.randoms
    for i, slug in enumerate(gold):
        if slug not in whole:
            continue
        siblings = [s for s in groups.get(series_key(by_slug[slug]), []) if s != slug]
        chosen = siblings[:3]
        while len(chosen) < slots:
            other = pool[rng.integers(len(pool))]
            if other != slug and other not in chosen:
                chosen.append(other)
        keep.append(i)
        targets.append(vectors[whole[slug]])
        negatives.append([vectors[whole[s]] for s in chosen[:slots]])

    queries = queries[keep]
    targets = np.array(targets)
    negatives = np.array(negatives)
    with_siblings = sum(
        1 for slug in (gold[i] for i in keep)
        if len(groups.get(series_key(by_slug[slug]), [])) > 1)
    print(f"пар для обучения {len(queries)}, из них с соседом по линейке "
          f"{with_siblings}; отрицательных на пару {negatives.shape[1]}")

    update = train(queries, targets, negatives, args.steps, args.rate,
                   args.decay, args.temperature, args.rank)
    np.save(args.out, update.astype(np.float32))
    print(f"\nголова сохранена: {args.out} (ранг {args.rank}, "
          f"норма обновления {np.linalg.norm(update):.3f})")


if __name__ == "__main__":
    main()
