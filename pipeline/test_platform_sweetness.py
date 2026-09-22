"""Сладость из карточки платформы добирает пропуски slug и не перекрывает его."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_match  # noqa: E402
from text_match import TextChannel, attributes_from_slug  # noqa: E402

WINES = [
    {"slug": "balaklava-muskat", "title": "Балаклава Мускат", "manufacturer": "Золотая Балка",
     "category": "Белое", "grapes": ["Мускат"]},
    {"slug": "x-beloe-polusuhoe-12", "title": "X", "manufacturer": "Y",
     "category": "Белое", "grapes": []},
    {"slug": "agora-vintage-blanc-de-blancs-extra-brut", "title": "Agora", "manufacturer": "Agora",
     "category": "Белое", "grapes": []},
]
CARDS = {"wines": [{"slug": "balaklava-muskat", "sweetness": "Брют"},
                   {"slug": "x-beloe-polusuhoe-12", "sweetness": "Сухое"}]}


class PlatformSweetness(unittest.TestCase):
    def setUp(self):
        text_match.USE_PLATFORM_SWEETNESS = True
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(CARDS, self.tmp, ensure_ascii=False)
        self.tmp.close()

    def tearDown(self):
        text_match.USE_PLATFORM_SWEETNESS = False
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_fills_only_missing(self):
        channel = TextChannel(WINES, verified_attributes=None, platform_cards=Path(self.tmp.name))
        self.assertEqual(channel.attributes_of["balaklava-muskat"].sweetness, "брют")
        # slug говорит «полусухое», карточка «сухое»: slug главнее
        self.assertEqual(channel.attributes_of["x-beloe-polusuhoe-12"].sweetness, "полусухое")

    def test_switch_off_keeps_gaps(self):
        text_match.USE_PLATFORM_SWEETNESS = False
        channel = TextChannel(WINES, verified_attributes=None, platform_cards=Path(self.tmp.name))
        self.assertIsNone(channel.attributes_of["balaklava-muskat"].sweetness)

    def test_extra_brut_in_latin_slug(self):
        self.assertEqual(attributes_from_slug("agora-vintage-blanc-de-blancs-extra-brut", "Белое").sweetness,
                         "экстра брют")


if __name__ == "__main__":
    unittest.main()
