"""Склейка разрядки и разрыва слова в строках OCR."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_match  # noqa: E402
from text_match import extract_attributes, join_spaced  # noqa: E402


class JoinSpaced(unittest.TestCase):
    def setUp(self):
        text_match.USE_JOIN = True

    def test_spaced_letters_are_joined(self):
        self.assertEqual(join_spaced("Б А Л А K Л А В А"), "БАЛАKЛАВА")
        self.assertEqual(join_spaced("B RUT"), "BRUT")
        self.assertEqual(join_spaced("Б Р ЮТ"), "БРЮТ")
        self.assertEqual(join_spaced("1 8 8 8"), "1888")

    def test_split_category_word_is_joined(self):
        self.assertEqual(join_spaced("КРАСН OE"), "КРАСНOE")
        self.assertEqual(extract_attributes(join_spaced("КРАСН OE")).color, "красное")
        self.assertEqual(extract_attributes(join_spaced("ПОЛУ СЛАДКОЕ")).sweetness, "полусладкое")

    def test_ordinary_phrases_untouched(self):
        self.assertEqual(join_spaced("Шато Тамань"), "Шато Тамань")
        self.assertEqual(join_spaced("ВИНО С ЗАЩИЩЕННЫМ"), "ВИНО С ЗАЩИЩЕННЫМ")
        self.assertEqual(join_spaced("Шампанский Дом"), "Шампанский Дом")
        self.assertEqual(join_spaced("BRUT"), "BRUT")


if __name__ == "__main__":
    unittest.main()
