"""Сведение латиницы и кириллицы в текстовом канале.

Проверяется ровно то, ради чего заведён словарь ALIASES: слово линейки или
сорта, написанное в одной карточке латиницей, а в другой кириллицей, не
должно становиться различителем между ними, а прочитанный латинский сорт
должен подтверждать карточку с кириллическим полем сортов.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_match  # noqa: E402
from ocr import Word  # noqa: E402
from text_match import (SURE_MATCH, TextChannel, alphabet_variants,  # noqa: E402
                        best_match, brand_key, brand_slugs, canonical, card_features,
                        discriminating, resolve_close)

WINES = [
    {"slug": "ct-signature-kaberne", "title": "Chateau Tamagne Signature. Каберне",
     "manufacturer": "Кубань-Вино", "category": "Красное", "grapes": ["Каберне Совиньон"]},
    {"slug": "shato-taman-kaberne-sovinon", "title": "Шато Тамань. Каберне Совиньон",
     "manufacturer": "Кубань-Вино", "category": "Красное", "grapes": ["Каберне Совиньон"]},
    {"slug": "golubitskoe-chardonnay", "title": "Golubitskoe Estate Chardonnay",
     "manufacturer": "Поместье Голубицкое", "category": "Белое", "grapes": ["Шардоне"]},
    {"slug": "golubitskoe-risling", "title": "Голубицкое Рислинг",
     "manufacturer": "Поместье Голубицкое", "category": "Белое", "grapes": ["Рислинг"]},
    {"slug": "golubitskoe-shardone", "title": "Голубицкое Шардоне",
     "manufacturer": "Поместье Голубицкое", "category": "Белое", "grapes": ["Шардоне"]},
    {"slug": "ge-rezerv", "title": "Шардоне Резерв",
     "manufacturer": "Golubitskoe Estate", "category": "Белое", "grapes": ["Шардоне"]},
    {"slug": "other-merlo", "title": "Мерло",
     "manufacturer": "Усадьба Перовских", "category": "Красное", "grapes": ["Мерло"]},
    {"slug": "fan-extra-brut-rose", "title": "Fanagoria Extra Brut Rose 2019",
     "manufacturer": "Фанагория", "category": "Розовое", "grapes": ["Пино Нуар"]},
]


def word(text: str, conf: float = 0.95, x: int = 100) -> Word:
    return Word(text, conf, (x, 10, x + 80, 30))


class Aliases(unittest.TestCase):
    def setUp(self):
        self.channel = TextChannel(WINES, verified_attributes=None)

    def test_variants_include_verified_alias(self):
        self.assertIn("тамань", alphabet_variants("tamagne"))
        self.assertIn("шардоне", alphabet_variants("chardonnay"))
        # Незнакомое латинское имя словарь не трогает; при включённой
        # транслитерации оно сводится к кириллице побуквенно, категорийные
        # слова — никогда.
        self.assertEqual(canonical("signature"),
                         "сигнатуре" if text_match.USE_NAME_TRANSLIT else "signature")
        self.assertEqual(canonical("brut"), "brut")

    def test_line_name_is_not_a_discriminator(self):
        marks = discriminating(["ct-signature-kaberne", "shato-taman-kaberne-sovinon"],
                               self.channel)
        for slug, words in marks.items():
            self.assertNotIn("tamagne", words, slug)
            self.assertNotIn("тамань", words, slug)
        # Настоящее различие остаётся: у латинской карточки есть «signature».
        self.assertIn(canonical("signature"), marks["ct-signature-kaberne"])

    def test_latin_grape_matches_cyrillic_card(self):
        features = card_features("golubitskoe-shardone", self.channel)
        self.assertTrue(best_match("chardonnay", features.grapes, SURE_MATCH))
        self.assertFalse(best_match("chardonnay",
                                    card_features("golubitskoe-risling", self.channel).grapes,
                                    SURE_MATCH))

    def test_read_latin_grape_demotes_only_other_grape(self):
        candidates = [("golubitskoe-risling", 40.0), ("golubitskoe-shardone", 38.0),
                      ("golubitskoe-chardonnay", 37.0)]
        ordered, found = resolve_close(candidates, [word("CHARDONNAY")], self.channel,
                                       window=0.8, min_conf=0.6)
        self.assertEqual(found["golubitskoe-shardone"], [])
        self.assertEqual(found["golubitskoe-chardonnay"], [])
        self.assertTrue(found["golubitskoe-risling"])
        self.assertNotEqual(ordered[0][0], "golubitskoe-risling")

    def test_brand_spellings_are_one_winery(self):
        self.assertEqual(brand_key("Поместье Голубицкое"), brand_key("Golubitskoe Estate"))
        self.assertNotEqual(brand_key("Le K2"), brand_key("В2Р"))
        slugs = ["other-merlo", "ge-rezerv", "golubitskoe-shardone", "golubitskoe-chardonnay"]
        own = brand_slugs(slugs, [word("GOLUBITSKOE")], self.channel)
        self.assertEqual(own, {"ge-rezerv", "golubitskoe-shardone", "golubitskoe-chardonnay"})

    def test_latin_producer_in_title_is_not_a_name(self):
        features = card_features("fan-extra-brut-rose", self.channel)
        self.assertNotIn("fanagoria", features.name)
        self.assertNotIn("фанагория", features.name)

    def test_read_line_name_does_not_demote_cyrillic_sibling(self):
        candidates = [("shato-taman-kaberne-sovinon", 30.0), ("ct-signature-kaberne", 29.0)]
        _, found = resolve_close(candidates, [word("TAMAGNE")], self.channel,
                                 window=0.8, min_conf=0.6)
        self.assertEqual(found["shato-taman-kaberne-sovinon"], [])


if __name__ == "__main__":
    unittest.main()


class NameTranslit(unittest.TestCase):
    def test_latin_proper_names_transliterate_when_enabled(self):
        import text_match
        text_match.USE_NAME_TRANSLIT = True
        try:
            self.assertEqual(canonical("caspico"), "каспико")
            self.assertEqual(canonical("golubitskoe"), "голубицкое")
            # словарь важнее побуквенного перевода, категории не трогаются
            self.assertEqual(canonical("chardonnay"), "шардоне")
            self.assertEqual(canonical("brut"), "brut")
        finally:
            text_match.USE_NAME_TRANSLIT = False
        self.assertEqual(canonical("caspico"), "caspico")
