"""Отбор центральных слов по строкам, а не по отдельным словам."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_match  # noqa: E402
from ocr import Word  # noqa: E402
from text_match import central_words, text_lines  # noqa: E402


def word(text, x1, x2, y1=100, y2=140, conf=0.95):
    return Word(text, conf, (x1, y1, x2, y2))


class LineBand(unittest.TestCase):
    def setUp(self):
        text_match.USE_LINE_BAND = True

    def tearDown(self):
        text_match.USE_LINE_BAND = False

    def test_second_half_of_a_line_survives(self):
        # «ПОЛУСЛАДКОЕ РОЗОВОЕ» одной строкой: второе слово правее центра.
        words = [word("НОВЫЙ", 128, 1082, 10, 60), word("ПОЛУСЛАДКОЕ", 252, 1060),
                 word("РОЗОВОЕ", 1069, 1604), word("2023", 653, 1174, 200, 240)]
        kept = {w.text for w in central_words(words, 0.6, band=0.5)}
        self.assertIn("РОЗОВОЕ", kept)

    def test_neighbour_bottle_text_is_still_dropped(self):
        # Слово соседней бутылки: та же высота, но широкий просвет.
        words = [word("МУСКАТ", 300, 700), word("CABERNET", 1400, 1800),
                 word("БЕЛЫЙ", 350, 650, 200, 240)]
        kept = {w.text for w in central_words(words, 0.6, band=0.5)}
        self.assertNotIn("CABERNET", kept)
        # Слово соседа не склеилось ни с одной строкой своей этикетки.
        self.assertTrue(all(len(line) == 1 for line in text_lines(words)))

    def test_lines_are_grouped_by_overlap_and_gap(self):
        words = [word("A", 0, 100), word("B", 110, 200), word("C", 400, 500),
                 word("D", 0, 100, 300, 340)]
        lines = [[w.text for w in line] for line in text_lines(words)]
        self.assertIn(["A", "B"], lines)
        self.assertIn(["C"], lines)
        self.assertIn(["D"], lines)


if __name__ == "__main__":
    unittest.main()
