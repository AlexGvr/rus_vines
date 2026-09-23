import unittest

from ocr import Word
from text_match import TextChannel, resolve_close


class GenericWineTerms(unittest.TestCase):
    def setUp(self):
        self.channel = TextChannel([
            {'slug': 'reserve-bryut', 'title': 'Балаклава Брют Резерв Розе',
             'manufacturer': 'Золотая Балка', 'category': 'Розовое', 'grapes': ['Пино Блан']},
            {'slug': 'sparkling-polusladkoe', 'title': 'Игристое розовое полусладкое',
             'manufacturer': 'Золотая Балка', 'category': 'Розовое', 'grapes': ['Пино Блан']},
        ], verified_attributes=None, platform_cards=None)

    def test_generic_type_does_not_demote_card_without_type_in_title(self):
        candidates = [('reserve-bryut', 50.), ('sparkling-polusladkoe', 45.)]
        words = [Word('Вино игристое розовое', .95, (10, 10, 210, 30))]
        ranked, reasons = resolve_close(candidates, words, self.channel)
        self.assertEqual(ranked, candidates)
        self.assertFalse(any(reasons.values()))

    def test_sweetness_still_contradicts_brut(self):
        words = [Word('Игристое розовое полусладкое', .95, (10, 10, 210, 30))]
        self.assertTrue(self.channel.hard_conflicts('reserve-bryut', words))
        self.assertFalse(self.channel.hard_conflicts('sparkling-polusladkoe', words))


if __name__ == '__main__':
    unittest.main()
