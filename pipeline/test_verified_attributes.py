import json
import tempfile
import unittest
from pathlib import Path

from ocr import Word
from text_match import TextChannel, attributes_from_words


class VerifiedAttributeTests(unittest.TestCase):
    def words(self, text, confidence=.95):
        return [Word(text, confidence, (0,0,100,20))]

    def test_confident_single_substitution(self):
        self.assertEqual(attributes_from_words(self.words('ПОЛУСЛААКОЕ')).sweetness,
                         'полусладкое')

    def test_uncertain_or_truncated_reading_abstains(self):
        for text, conf in [('ПОЛУСЛААКОЕ', .8), ('ОЛУСЛАДКОЕ', .99), ('ПОЛУСЛААКОА', .99)]:
            self.assertIsNone(attributes_from_words(self.words(text,conf)).sweetness)

    def test_conflicting_corrections_abstain(self):
        self.assertIsNone(attributes_from_words(self.words('ПОЛУСЛААКОЕ ПОЛУСУХОА')).sweetness)

    def test_verified_sweetness_disambiguates_identical_titles(self):
        wines = [dict(slug=s,title='Пино Нуар Мускат',manufacturer='АРАТТИ',category='Розовое')
                 for s in ('dry','sweet')]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'attributes.json'
            p.write_text(json.dumps(dict(version=1,wines={
                'dry':dict(sweetness='сухое'), 'sweet':dict(sweetness='полусладкое')})))
            channel=TextChannel(wines,verified_attributes=p)
            words=self.words('ПОЛУСЛААКОЕ')
            self.assertTrue(channel.hard_conflicts('dry',words))
            self.assertFalse(channel.hard_conflicts('sweet',words))


if __name__ == '__main__':
    unittest.main()
