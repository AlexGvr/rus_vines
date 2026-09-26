"""Геометрия за кандидата против отвергнутого лидера (VLM_GEO_FALLBACK)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from searchcore import Candidate, geometry_backs  # noqa: E402


def shortlist(**inliers):
    return [Candidate(slug=slug, cv=0.8, inliers=n) for slug, n in inliers.items()]


class GeometryBacks(unittest.TestCase):
    def test_victor_dravigny_brut(self):
        # Экстра Брют лидирует по косинусу с 16 точками, Брют — 45 при лучших 48.
        c = shortlist(extra=16, krasnoe=10, brut=45, rose=48, polusuhoe=32)
        self.assertTrue(geometry_backs(c[2], c[0], c))

    def test_candidate_outside_best_group(self):
        # Кошерный брют на 12876444_0: 39 точек при лучших 57 — не в группе лучших.
        c = shortlist(leader=10, kosher=39, other=57)
        self.assertFalse(geometry_backs(c[1], c[0], c))

    def test_leader_not_decisively_beaten(self):
        # Кандидат в группе лучших, но лидер по точкам почти не уступает.
        c = shortlist(leader=40, rival=45)
        self.assertFalse(geometry_backs(c[1], c[0], c))

    def test_no_geometry(self):
        c = shortlist(leader=0, rival=0)
        self.assertFalse(geometry_backs(c[1], c[0], c))


if __name__ == "__main__":
    unittest.main()
