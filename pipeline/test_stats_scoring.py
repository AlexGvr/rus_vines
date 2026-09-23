"""Интервалы, парный тест и правила подсчёта оценочного эндпоинта."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_scoring import credited, score  # noqa: E402
from stats import mcnemar_exact, wilson  # noqa: E402


def rec(gold, correct, p):
    return {"gold": gold, "correct": correct, "probability": p}


class Stats(unittest.TestCase):
    def test_wilson_contains_share(self):
        lo, hi = wilson(134, 154)
        self.assertLess(lo, 134 / 154)
        self.assertGreater(hi, 134 / 154)
        self.assertAlmostEqual(hi - lo, 0.108, places=2)

    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(1, 1), 1.0)
        self.assertLess(mcnemar_exact(10, 0), 0.01)
        self.assertAlmostEqual(mcnemar_exact(3, 1), 0.625)


class Scoring(unittest.TestCase):
    rows = [rec("a", True, 0.5), rec("b", True, 0.05), rec("c", False, 0.5),
            rec("", False, 0.01), rec("", False, 0.3)]

    def test_strict_never_credits_absent(self):
        self.assertEqual(score(self.rows, 0.0, "strict"), (2, 5))
        self.assertEqual(score(self.rows, 0.1, "strict"), (1, 5))

    def test_null_ok_credits_refusal_on_absent_only(self):
        self.assertEqual(score(self.rows, 0.0, "null_ok"), (2, 5))
        # отказ: «b» потерян, первый отсутствующий засчитан
        self.assertEqual(score(self.rows, 0.1, "null_ok"), (2, 5))
        self.assertEqual(score(self.rows, 0.4, "null_ok"), (3, 5))

    def test_positives_rule_skips_absent(self):
        self.assertIsNone(credited(self.rows[3], 0.1, "positives"))
        self.assertEqual(score(self.rows, 0.0, "positives"), (2, 3))


if __name__ == "__main__":
    unittest.main()
