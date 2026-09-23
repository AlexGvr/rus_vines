"""Хранилище векторов индекса: в памяти (numpy) или в PostgreSQL (pgvector).

ТЗ рекомендует для целевой платформы PostgreSQL с pgvector. Индекс здесь
маленький (два вида на позицию, ~4.4 тыс. строк), и полный перебор в
памяти стоит миллисекунды, поэтому по умолчанию остаётся numpy. Вариант
с pgvector нужен для встраивания в платформу: векторы и карточки живут
в той же базе, новая позиция добавляется строкой, без пересборки файла.

Поиск в pgvector точный (без приближённого индекса): на таком объёме он
быстрый, а приближённый мог бы тихо поменять шортлист. Обе реализации
отдают одну и ту же матрицу [строка индекса, вид запроса]; строки,
не попавшие в первые k по какому-либо виду, получают -1 и в шортлист
не проходят. Косинус отобранных строк пересчитывается в numpy по тем же
float32-векторам, и шортлист совпадает с полным перебором на всех 390
кадрах полевого набора (`python pipeline/vector_store.py --check`).

    VECTOR_BACKEND=pgvector DATABASE_URL=postgresql://wine:wine@127.0.0.1:5433/wine
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "index"
CATALOG = ROOT / "data" / "catalog" / "catalog.json"
TOP_ROWS = 400
# Версия формата загрузки входит в отпечаток: смена формата перезаливает базу.
LOAD_FORMAT = 2


def literal(v: np.ndarray) -> str:  # для COPY: текстовый формат, точность float32
    """Вектор в текстовом виде pgvector без потери точности float32."""
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class NumpyStore:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def similarity(self, queries: np.ndarray) -> np.ndarray:
        return self.vectors @ queries.T


class PgStore:
    """Векторы индекса в таблице wine_vectors, карточки — в wines."""

    def __init__(self, dsn: str, variant: str, top_rows: int = TOP_ROWS):
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.variant = variant
        self.top_rows = top_rows
        self.rows = sum(1 for _ in (INDEX_DIR / f"{variant}.csv").open(encoding="utf-8")) - 1
        self.ensure_loaded()

    def __len__(self) -> int:
        return self.rows

    def _fingerprint(self) -> str:
        h = hashlib.sha256(f"format={LOAD_FORMAT}".encode())
        for name in (f"{self.variant}.npy", f"{self.variant}.csv"):
            h.update((INDEX_DIR / name).read_bytes())
        if CATALOG.exists():
            h.update(CATALOG.read_bytes())
        return h.hexdigest()

    def ensure_loaded(self) -> None:
        """Загрузить индекс и каталог, если в базе другая версия."""
        fp = self._fingerprint()
        cur = self.conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE TABLE IF NOT EXISTS index_meta (variant text PRIMARY KEY, "
                    "fingerprint text, rows int)")
        cur.execute("SELECT fingerprint FROM index_meta WHERE variant = %s", (self.variant,))
        row = cur.fetchone()
        if row and row[0] == fp:
            return
        self.load(fp)

    def load(self, fingerprint: str) -> None:
        vectors = np.load(INDEX_DIR / f"{self.variant}.npy").astype(np.float32)
        with (INDEX_DIR / f"{self.variant}.csv").open(encoding="utf-8") as fh:
            meta = list(csv.DictReader(fh))
        dim = vectors.shape[1]
        cur = self.conn.cursor()
        with self.conn.transaction():
            cur.execute("DROP TABLE IF EXISTS wine_vectors")
            cur.execute(f"CREATE TABLE wine_vectors (row_id int PRIMARY KEY, slug text, "
                        f"view text, path text, embedding vector({dim}) NOT NULL)")
            with cur.copy("COPY wine_vectors (row_id, slug, view, path, embedding) FROM STDIN") as cp:
                for i, (m, v) in enumerate(zip(meta, vectors)):
                    cp.write_row((i, m["slug"] or None, m["view"], m["path"], literal(v)))
            if CATALOG.exists():
                cur.execute("DROP TABLE IF EXISTS wines")
                cur.execute("CREATE TABLE wines (slug text PRIMARY KEY, card jsonb NOT NULL)")
                wines = json.loads(CATALOG.read_text(encoding="utf-8"))["wines"]
                with cur.copy("COPY wines (slug, card) FROM STDIN") as cp:
                    for w in wines:
                        cp.write_row((w["slug"], json.dumps(w, ensure_ascii=False)))
            cur.execute("INSERT INTO index_meta VALUES (%s, %s, %s) ON CONFLICT (variant) "
                        "DO UPDATE SET fingerprint = EXCLUDED.fingerprint, rows = EXCLUDED.rows",
                        (self.variant, fingerprint, len(meta)))
        print(f"pgvector: загружено {len(meta)} строк индекса {self.variant}", file=sys.stderr)

    def similarity(self, queries: np.ndarray) -> np.ndarray:
        """База отбирает строки, косинус для них считается здесь.

        Порядок суммирования в pgvector другой, чем в BLAS, и косинусы
        расходятся в седьмом знаке — этого хватало, чтобы почти равные
        кандидаты менялись местами (32 шортлиста из 390). Пересчёт по тем же
        float32-векторам даёт ровно числа numpy-варианта.
        """
        from pgvector.psycopg import register_vector
        register_vector(self.conn)
        cur = self.conn.cursor()
        found: dict[int, np.ndarray] = {}
        for q in queries:
            # <#> — отрицательное скалярное произведение; векторы нормированы.
            cur.execute("SELECT row_id, embedding FROM wine_vectors "
                        "ORDER BY embedding <#> %s LIMIT %s", (q.astype(np.float32), self.top_rows))
            for row_id, emb in cur.fetchall():
                found[row_id] = emb.to_numpy() if hasattr(emb, "to_numpy") else np.asarray(emb)
        out = np.full((self.rows, queries.shape[0]), -1.0, dtype=np.float32)
        ids = np.array(sorted(found))
        out[ids] = np.stack([found[i] for i in ids]).astype(np.float32) @ queries.T
        return out


def open_store(variant: str) -> NumpyStore | PgStore:
    backend = os.environ.get("VECTOR_BACKEND", "numpy")
    if backend == "pgvector":
        return PgStore(os.environ["DATABASE_URL"], variant)
    return NumpyStore(np.load(INDEX_DIR / f"{variant}.npy"))


def check(variant: str = "clean_mv") -> None:
    """Шортлист pgvector против полного перебора на сохранённых запросах."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import searchcore as core
    qfile = ROOT / "data" / "validation" / "dino_20260923" / "queries.npz"
    if not qfile.exists():
        sys.exit(f"нет {qfile}: сначала experiment_dino.py (он кэширует эмбеддинги SigLIP запросов)")
    z = np.load(qfile)
    with (INDEX_DIR / f"{variant}.csv").open(encoding="utf-8") as fh:
        slugs = [r["slug"] or None for r in csv.DictReader(fh)]
    dense = NumpyStore(np.load(INDEX_DIR / f"{variant}.npy"))
    pg = PgStore(os.environ["DATABASE_URL"], variant)
    same = 0
    n = z["siglip_raw"].shape[0]
    for i in range(n):
        q = np.stack([z["siglip_raw"][i], z["siglip_det"][i]])
        a = core.shortlist(dense.similarity(q).max(axis=1), slugs, 20)
        b = core.shortlist(pg.similarity(q).max(axis=1), slugs, 20)
        same += [c.slug for c in a] == [c.slug for c in b]
    print(f"шортлист совпал: {same} из {n}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
