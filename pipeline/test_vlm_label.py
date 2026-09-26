"""Разбор ответа VLM о стиле вина: только значения из словаря каталога."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_label import parse  # noqa: E402


class ParseLabel(unittest.TestCase):
    def test_fenced_json_mapped_to_catalog_vocabulary(self):
        raw = ('```json\n{"producer": "Mosavali", "name": "Saperavi", "color": "red", '
               '"sweetness": "dry", "sparkling": false, "grapes": ["Саперави"], '
               '"region": null, "country": "Грузия"}\n```')
        self.assertEqual(parse(raw), {
            "producer": "Mosavali", "producer_ru": None, "name": "Saperavi", "color": "красное",
            "sweetness": "сухое", "sparkling": False, "grapes": ["Саперави"],
            "region": None, "country": "Грузия"})

    def test_brut_implies_sparkling(self):
        label = parse('{"color": "white", "sweetness": "Brut", "sparkling": null}')
        self.assertEqual(label["sweetness"], "брют")
        self.assertTrue(label["sparkling"])

    def test_unknown_values_become_none(self):
        label = parse('{"color": "golden", "sweetness": "medium", "sparkling": "yes", '
                      '"grapes": "Мерло", "producer": "null"}')
        self.assertIsNone(label["color"])
        self.assertIsNone(label["sweetness"])
        self.assertIsNone(label["sparkling"])
        self.assertEqual(label["grapes"], [])
        self.assertIsNone(label["producer"])

    def test_words_from_label_beat_classification(self):
        # «Алазанская долина»: модель сказала dry, а слова этикетки — в сортах
        label = parse('{"name": "Алазанская долина", "color": "red", "sweetness": "dry", '
                      '"grapes": ["Красное Полусладкое"], "style_text": null}')
        self.assertEqual(label["sweetness"], "полусладкое")
        self.assertEqual(label["grapes"], [])
        label = parse('{"style_text": "Розовое сухое", "color": "red", '
                      '"grapes": ["Мускат Белый"]}')
        self.assertEqual(label["color"], "розовое")
        self.assertEqual(label["grapes"], ["Мускат Белый"])

    def test_not_json(self):
        self.assertIsNone(parse("I cannot read the label."))
        self.assertIsNone(parse('{"color": "red",'))


if __name__ == "__main__":
    unittest.main()
