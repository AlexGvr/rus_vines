"""Шортлист: равные косинусы решаются порядком строк индекса, а не случаем."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchcore import shortlist  # noqa: E402


class Ties(unittest.TestCase):
    def test_tie_keeps_index_order_whatever_the_rest(self):
        slugs = [f"s{i}" for i in range(300)]
        base = np.linspace(0.1, 0.5, 300).astype(np.float32)
        base[[7, 3]] = 0.9                      # две позиции с общим эталоном
        order = [c.slug for c in shortlist(base, slugs, 5)]
        self.assertEqual(order[:2], ["s3", "s7"])
        masked = base.copy()
        masked[100:250] = -1.0                  # как у pgvector: строки вне первых k
        self.assertEqual([c.slug for c in shortlist(masked, slugs, 5)][:2], ["s3", "s7"])


if __name__ == "__main__":
    unittest.main()
