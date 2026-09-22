"""Читатель транскрипции и диагностика промахов в eval_transcript.py."""
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_transcript import Typed, diagnose  # noqa: E402
from text_match import TextChannel, central_words  # noqa: E402

WINES = [
    {"slug": "a-beloe-bryut", "title": "Фанагория. Брют белое", "manufacturer": "Фанагория",
     "category": "Белое", "grapes": ["Шардоне"]},
    {"slug": "a-beloe-polusladkoe", "title": "Фанагория белое полусладкое",
     "manufacturer": "Фанагория", "category": "Белое", "grapes": ["Рислинг"]},
]


class TypedReader(unittest.TestCase):
    def test_lines_land_in_crop_coordinates_and_pass_central_filter(self):
        crop = Image.new("RGB", (400, 1000))
        reader = Typed([{"text": "БРЮТ", "x": 0.5, "y": 0.8, "w": 0.2},
                        {"text": "", "x": 0.5, "y": 0.9, "unreadable": True},
                        {"text": "FANAGORIA", "x": 0.5, "y": 0.84, "w": 0.6}])
        words = reader(crop)
        self.assertEqual([w.text for w in words], ["БРЮТ", "FANAGORIA"])
        self.assertEqual(words[0].box, (160, 792, 240, 808))
        self.assertEqual({w.text for w in central_words(words, 0.6, band=0.5)},
                         {"БРЮТ", "FANAGORIA"})

    def test_unreadable_lines_are_not_read(self):
        words = Typed([{"text": "СЛОВО", "unreadable": True}])(Image.new("RGB", (10, 10)))
        self.assertEqual(words, [])


class Diagnosis(unittest.TestCase):
    def setUp(self):
        self.channel = TextChannel(WINES, verified_attributes=None, platform_cards=None)

    def test_categories(self):
        findable = {"a-beloe-bryut", "a-beloe-polusladkoe"}
        before = {"order": ["a-beloe-polusladkoe", "a-beloe-bryut"],
                  "after_geometry": ["a-beloe-polusladkoe", "a-beloe-bryut"]}
        fixed = {"order": ["a-beloe-bryut", "a-beloe-polusladkoe"],
                 "after_geometry": ["a-beloe-polusladkoe", "a-beloe-bryut"]}
        self.assertEqual(diagnose("a-beloe-bryut", before, fixed, self.channel, findable),
                         "исправлено чтением")
        self.assertTrue(diagnose("a-beloe-bryut", before, before, self.channel, findable)
                        .startswith("текст не переставил"))
        self.assertEqual(diagnose("a-beloe-bryut", before, before, self.channel, set()),
                         "эталон отсутствует")
        self.assertEqual(diagnose("a-beloe-bryut", before, {"order": [], "after_geometry": []},
                                  self.channel, findable), "кандидат не в шортлисте")


if __name__ == "__main__":
    unittest.main()
