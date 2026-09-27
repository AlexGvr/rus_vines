"""Подсчёт null_ok в сквозной проверке через HTTP (api_check.py)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from api_check import null_ok, summarize  # noqa: E402


def row(gold, predicted, status=200):
    return {"gold": gold, "predicted": predicted, "status": status,
            "correct": bool(gold) and predicted == gold}


class NullOk(unittest.TestCase):
    def test_refusal_on_absent_wine_counts(self):
        self.assertEqual(null_ok([row("", None)]), 1)

    def test_http_failure_is_not_a_refusal(self):
        # Упавший запрос оставляет predicted пустым — это не отказ сервиса.
        self.assertEqual(null_ok([row("", None, status=0)]), 0)
        self.assertEqual(null_ok([row("", None, status=500)]), 0)

    def test_answer_on_absent_wine_is_wrong(self):
        self.assertEqual(null_ok([row("", "some-wine")]), 0)

    def test_catalog_wine_needs_right_slug(self):
        self.assertEqual(null_ok([row("a", "a"), row("a", "b"), row("a", None)]), 1)


if __name__ == "__main__":
    unittest.main()


def search_row(gold, predicted, shown, status=200, ms=1000.0, frame="front"):
    r = row(gold, predicted, status)
    r.update(frame=frame, shown=shown, latency_ms=ms,
             search_status=None if status != 200 else ("uncertain" if shown else "unsure"))
    return r


class Summarize(unittest.TestCase):
    def test_failed_absent_frame_is_an_error_everywhere(self):
        ok = [search_row("a", "a", True), search_row("", None, False)]
        failed = [search_row("a", "a", True), search_row("", None, None, status=0, ms=1.0)]
        good, bad = summarize(ok, 3000), summarize(failed, 3000)
        self.assertEqual(good["null_ok"], 2)
        self.assertEqual(bad["null_ok"], 1)
        self.assertEqual(bad["absent_failed"], 1)
        self.assertEqual(bad["search"]["absent_failed"], 1)
        # Сбой на вине вне каталога — ложное срабатывание, а не верный отказ.
        self.assertEqual(good["search"]["f1_top1_with_rejection"], 1.0)
        self.assertLess(bad["search"]["f1_top1_with_rejection"], 1.0)

    def test_failures_are_not_timed(self):
        s = summarize([search_row("a", "a", True, ms=1500.0),
                       search_row("", None, None, status=0, ms=1.0)], 3000)
        self.assertEqual(s["latency_ms"]["median"], 1500.0)
        self.assertEqual(s["latency_ms"]["http_failures"], 1)
