import unittest
from types import SimpleNamespace

from searchcore import Candidate, prefer_brand
from text_match import brand_slugs
from unittest.mock import patch


def words(*names):
    return [SimpleNamespace(text=n, conf=.99, box=(0, 0, 100, 10), center_x=50)
            for n in names]


class BrandTests(unittest.TestCase):
    def setUp(self):
        self.channel = SimpleNamespace(by_slug={
            'a': {'manufacturer': 'Мысхако'},
            'b': {'manufacturer': 'Фанагория'},
            'c': {'manufacturer': 'Массандра'},
            'd': {'manufacturer': 'Фанагория'},
        })

    def test_two_brands_preserve_geometry_order(self):
        candidates = [Candidate('a', .95, 100), Candidate('b', .5, 2),
                      Candidate('c', .4, 1)]
        with patch.dict('os.environ', {'BRAND_FIRST': '1'}):
            prefer_brand(candidates, words('Фанагория', 'Массандра'), self.channel, .6)
        self.assertEqual([c.slug for c in candidates], ['a', 'b', 'c'])

    def test_single_brand_keeps_all_its_skus(self):
        self.assertEqual(brand_slugs(list(self.channel.by_slug), words('Фанагория'),
                                    self.channel), {'b', 'd'})

    def test_unrecognized_brand_does_not_promote(self):
        self.assertEqual(brand_slugs(['a', 'b'], words('неизвестный'), self.channel), set())


if __name__ == '__main__':
    unittest.main()
