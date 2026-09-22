"""Взвешивание инлаеров по уникальности точки (GEOM_WEIGHT)."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402
from searchcore import weighted_scores  # noqa: E402


class FakeReranker:
    """Точки 0..9 общие для всех, 10..14 только у gold, 15..19 только у rival."""

    def stats_points(self, crop, slugs):
        shared = frozenset(range(10))
        table = {"gold": shared | frozenset(range(10, 15)),
                 "rival": shared | frozenset(range(15, 20)) | frozenset({20, 21, 22}),
                 "third": shared}
        out = {}
        for s in slugs:
            pts = table[s]
            out[s] = {"points": pts, "union": pts, "inliers": len(pts), "coverage": 0.5, "entry": 0,
                      "entries": []}
        return out


class WeightedScores(unittest.TestCase):
    def test_shared_points_are_discounted_once_per_sku(self):
        stats = FakeReranker().stats_points(None, ["gold", "rival", "third"])
        scores, mult = weighted_scores(stats, ["gold", "rival", "third"], alpha=1.0)
        # общие 10 точек делятся на три вина: каждая даёт 1/3
        self.assertAlmostEqual(scores["third"], 10 / 3)
        self.assertAlmostEqual(scores["gold"], 10 / 3 + 5)
        self.assertAlmostEqual(scores["rival"], 10 / 3 + 8)
        self.assertEqual(mult[0], 3)
        self.assertEqual(mult[12], 1)

    def test_repeated_references_of_one_sku_do_not_inflate_multiplicity(self):
        stats = {"a": {"points": frozenset({1, 2}), "union": frozenset({1, 2, 3}), "inliers": 2, "coverage": 0.1},
                 "b": {"points": frozenset({1}), "union": frozenset({1}), "inliers": 1, "coverage": 0.1}}
        scores, mult = weighted_scores(stats, ["a", "b"], alpha=1.0)
        self.assertEqual(mult, {1: 2, 2: 1, 3: 1})
        self.assertAlmostEqual(scores["a"], 0.5 + 1.0)
        self.assertAlmostEqual(scores["b"], 0.5)

    def test_flag_changes_order_only_when_enabled(self):
        cands = [core.Candidate("third", 0.9), core.Candidate("gold", 0.85), core.Candidate("rival", 0.8)]
        saved = os.environ.get("GEOM_WEIGHT")
        try:
            os.environ["GEOM_WEIGHT"] = "1"
            core.rerank(None, list(cands), FakeReranker())
            ordered = core.rerank(None, [core.Candidate(c.slug, c.cv) for c in cands], FakeReranker())
            # rival: 3.33+8 против gold 3.33+5 — отрыв меньше 1.5×, порядок по косинусу
            self.assertEqual([c.slug for c in ordered], ["third", "gold", "rival"])
            self.assertGreater(ordered[2].weighted, ordered[1].weighted)
            self.assertEqual(ordered[0].inliers, 10)
        finally:
            if saved is None:
                os.environ.pop("GEOM_WEIGHT", None)
            else:
                os.environ["GEOM_WEIGHT"] = saved


if __name__ == "__main__":
    unittest.main()
