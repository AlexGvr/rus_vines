"""Владелец уверенно прочитанного имени попадает в группу текстового сравнения."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import text_match  # noqa: E402
from ocr import Word  # noqa: E402
from text_match import TextChannel, name_slugs  # noqa: E402

WINES = [
    {"slug": "portveyn-belyy-krymskiy", "title": "Портвейн белый крымский",
     "manufacturer": "Массандра", "category": "Белое", "grapes": ["Алиготе", "Кокур"]},
    {"slug": "portveyn-krasnyy-livadiya", "title": "Портвейн красный Ливадия",
     "manufacturer": "Массандра", "category": "Красное", "grapes": ["Каберне Совиньон"]},
    {"slug": "madera-krymskaya", "title": "Мадера Крымская",
     "manufacturer": "Массандра", "category": "Белое", "grapes": ["Альбильо"]},
    {"slug": "kagor-gurzuf", "title": "Кагор Гурзуф",
     "manufacturer": "Массандра", "category": "Красное", "grapes": ["Саперави"]},
]


def word(text, conf=0.98, x=100):
    return Word(text, conf, (x, 10, x + 80, 30))


class NameSlugs(unittest.TestCase):
    def setUp(self):
        text_match.USE_NAME_FIRST = True
        self.channel = TextChannel(WINES, verified_attributes=None)
        self.slugs = [w["slug"] for w in WINES]

    def test_exact_owner_wins_over_fuzzy_neighbour(self):
        # «КРЫМСКИЙ» точно у портвейна и с одной правкой у «Мадеры Крымской»:
        # точное совпадение решает, нечёткий сосед группу не блокирует.
        self.assertEqual(name_slugs(self.slugs, [word("КРЫМСКИЙ")], self.channel),
                         {"portveyn-belyy-krymskiy"})

    def test_shared_word_selects_nobody(self):
        # «ПОРТВЕЙН» стоит у двух карточек — ничьё имя.
        self.assertEqual(name_slugs(self.slugs, [word("ПОРТВЕЙН")], self.channel), set())

    def test_low_confidence_is_ignored(self):
        self.assertEqual(name_slugs(self.slugs, [word("КРЫМСКИЙ", conf=0.7)], self.channel), set())

    def test_fuzzy_match_does_not_pull(self):
        # «ЛИВАДИА» с ошибкой в букве: нечёткое совпадение владельца не приводит.
        self.assertEqual(name_slugs(self.slugs, [word("ЛИВАДИА")], self.channel), set())

    def test_weak_geometry_owner_is_not_pulled(self):
        weight = {"portveyn-belyy-krymskiy": 5.0, "portveyn-krasnyy-livadiya": 60.0,
                  "madera-krymskaya": 10.0, "kagor-gurzuf": 8.0}
        self.assertEqual(name_slugs(self.slugs, [word("КРЫМСКИЙ")], self.channel, weight=weight),
                         set())
        weight["portveyn-belyy-krymskiy"] = 36.0
        self.assertEqual(name_slugs(self.slugs, [word("КРЫМСКИЙ")], self.channel, weight=weight),
                         {"portveyn-belyy-krymskiy"})


if __name__ == "__main__":
    unittest.main()
