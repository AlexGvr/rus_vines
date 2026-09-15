#!/usr/bin/env python3
"""Проверка, что образ собирается со всем, что нужно сервису.

Раньше состав образа перечислялся в Dockerfile вручную и отставал от кода:
сервис начал импортировать pipeline/ocr.py, COPY не поправили, и контейнер
падал на старте с ModuleNotFoundError. Ошибка дешёвая и полностью
механическая, поэтому проверяется скриптом, а не вниманием.

Скрипт обходит импорты сервиса по репозиторию, собирает замыкание и
сравнивает с тем, что перечислено в COPY. Запускается без Docker и без
зависимостей — годится и для CI, и перед сборкой.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = ROOT / "service" / "Dockerfile"
ENTRY = ROOT / "service" / "main.py"
PACKAGES = ("pipeline", "core", "service")


def closure(entry: Path) -> set[Path]:
    """Все файлы репозитория, до которых дотягиваются импорты точки входа."""
    seen: set[Path] = set()
    queue = [entry]
    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                head = name.split(".")[0]
                for package in PACKAGES:
                    queue.append(ROOT / package / f"{head}.py")
    return seen


def copied() -> set[Path]:
    """Файлы, которые Dockerfile кладёт в образ."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    text = re.sub(r"\\\s*\n", " ", text)          # склеиваем перенос строки
    out: set[Path] = set()
    for line in text.splitlines():
        if not line.startswith("COPY "):
            continue
        # Последний аргумент — путь назначения, остальные источники.
        sources = line.split()[1:-1]
        for source in sources:
            path = ROOT / source
            if path.is_dir():
                out |= set(path.rglob("*.py"))
            else:
                out.add(path)
    return out


def main() -> int:
    needed = closure(ENTRY)
    have = copied()
    missing = sorted(p.relative_to(ROOT) for p in needed - have)
    extra = sorted(p.relative_to(ROOT) for p in have - needed
                   if p.suffix == ".py" and "__pycache__" not in p.parts)

    print(f"сервису нужно файлов: {len(needed)}, в образ кладётся: {len(have)}")
    if extra:
        print("лишние в образе (не ошибка, но и не нужны):")
        for path in extra:
            print(f"  {path}")
    if missing:
        print("НЕ ПОПАДАЮТ В ОБРАЗ — контейнер упадёт на старте:")
        for path in missing:
            print(f"  {path}")
        return 1
    print("состав образа покрывает импорты сервиса")
    return 0


if __name__ == "__main__":
    sys.exit(main())
