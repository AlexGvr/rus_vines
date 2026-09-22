"""Четыре правки сопоставления, найденные транскрипцией промахов tune."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402
import text_match  # noqa: E402
from ocr import Word  # noqa: E402
from text_match import TextChannel, resolve_close, strong_name_slugs  # noqa: E402

WINES = [
    {"slug": "alushta-red", "title": "Портвейн красный Алушта", "manufacturer": "Массандра",
     "category": "Красное", "grapes": ["Красные сорта винограда"]},
    {"slug": "surozh", "title": "Портвейн Сурож", "manufacturer": "Массандра",
     "category": "Белое", "grapes": ["Кокур"]},
    {"slug": "krymskiy", "title": "Портвейн белый крымский", "manufacturer": "Массандра",
     "category": "Белое", "grapes": ["Алиготе"]},
    {"slug": "kagor", "title": "Кагор Гурзуф", "manufacturer": "Массандра",
     "category": "Красное", "grapes": ["Саперави"]},
    {"slug": "vd-extra-brut-beloe", "title": "Victor Dravigny. Extra Brut",
     "manufacturer": "Абрау-Дюрсо", "category": "Белое", "grapes": ["Шардоне"]},
    {"slug": "vd-krasnoe-polusladkoe", "title": "Victor Dravigny. Красное полусладкое",
     "manufacturer": "Абрау-Дюрсо", "category": "Красное", "grapes": ["Каберне Совиньон"]},
    {"slug": "vd-polusladkoe-beloe", "title": "Victor Dravigny. Полусладкое",
     "manufacturer": "Абрау-Дюрсо", "category": "Белое", "grapes": ["Совиньон Блан"]},
]


def word(text, conf=0.98, x=100):
    return Word(text, conf, (x, 10, x + 80, 30))


class Env:
    def __init__(self, **values):
        self.values, self.saved = values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.saved[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *_):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class GroupLeader(unittest.TestCase):
    def setUp(self):
        self.channel = TextChannel(WINES, verified_attributes=None, platform_cards=None)

    def test_leader_outside_window_is_compared_when_enabled(self):
        pairs = [("alushta-red", 7.0), ("surozh", 12.0), ("krymskiy", 6.0)]
        words = [word("КРЫМСКИЙ")]
        text_match.USE_GROUP_LEADER = False
        _, reasons = resolve_close(pairs, words, self.channel, extra={"krymskiy"})
        self.assertEqual(reasons.get("alushta-red", []), [])
        text_match.USE_GROUP_LEADER = True
        try:
            ranked, reasons = resolve_close(pairs, words, self.channel, extra={"krymskiy"})
            self.assertTrue(reasons["alushta-red"])
            self.assertEqual(ranked[0][0], "krymskiy")
        finally:
            text_match.USE_GROUP_LEADER = False


class StrongName(unittest.TestCase):
    def setUp(self):
        self.channel = TextChannel(WINES, verified_attributes=None, platform_cards=None)
        self.slugs = ["kagor", "surozh", "krymskiy", "alushta-red"]

    def test_exact_long_word_only(self):
        text_match.USE_NAME_STRONG = True
        try:
            self.assertEqual(strong_name_slugs(self.slugs, [word("КРЫМСКИЙ")], self.channel),
                             {"krymskiy"})
            self.assertEqual(strong_name_slugs(self.slugs, [word("КРЫМСКИИ")], self.channel), set())
            self.assertEqual(strong_name_slugs(self.slugs, [word("СУРОЖ")], self.channel), set())
        finally:
            text_match.USE_NAME_STRONG = False

    def test_everyone_ahead_of_owner_gets_compared(self):
        cands = [core.Candidate("kagor", 0.9, 65, 0.5), core.Candidate("surozh", 0.85, 43, 0.5),
                 core.Candidate("krymskiy", 0.84, 15, 0.4)]
        words = [word("КРЫМСКИЙ", x=120), word("БЕЛЫЙ", x=110)]
        with Env(NAME_STRONG="1", GROUP_LEADER="0", NAME_FIRST="1"):
            text_match.USE_NAME_STRONG = True
            try:
                core.resolve(cands, None, self.channel, lambda *_: words, 0.8, 0.6,
                             {w["slug"]: w for w in WINES})
            finally:
                text_match.USE_NAME_STRONG = False
        self.assertEqual(cands[0].slug, "krymskiy")
        self.assertTrue(cands[1].contradictions or cands[2].contradictions)


class DemoteGeometry(unittest.TestCase):
    def setUp(self):
        self.channel = TextChannel(WINES, verified_attributes=None, platform_cards=None)

    def test_geometry_decides_among_consistent_candidates(self):
        cands = [core.Candidate("vd-extra-brut-beloe", 0.9, 6, 0.3),
                 core.Candidate("vd-krasnoe-polusladkoe", 0.89, 5, 0.3),
                 core.Candidate("vd-polusladkoe-beloe", 0.88, 47, 0.6)]
        words = [word("ПОЛУСЛАДКОЕ")]
        core.demote_contradicted(list(cands), self.channel, words, 0.6)
        plain = list(cands)
        core.demote_contradicted(plain, self.channel, words, 0.6)
        self.assertEqual(plain[0].slug, "vd-krasnoe-polusladkoe")
        with Env(DEMOTE_GEOMETRY="1"):
            fixed = list(cands)
            core.demote_contradicted(fixed, self.channel, words, 0.6)
        self.assertEqual(fixed[0].slug, "vd-polusladkoe-beloe")


class DemoteDepth(unittest.TestCase):
    def test_two_attributes_lift_the_depth_cap(self):
        wines = [{"slug": f"brut-{i}", "title": f"Брют белое {i}", "manufacturer": "X",
                  "category": "Белое", "grapes": []} for i in range(6)]
        wines[0]["slug"] = "beloe-bryut-0"
        wines = [{**w, "slug": w["slug"].replace("brut-", "beloe-bryut-")} for w in wines]
        wines.append({"slug": "rozovoe-polusuhoe-x", "title": "Розовое полусухое",
                      "manufacturer": "X", "category": "Розовое", "grapes": []})
        channel = TextChannel(wines, verified_attributes=None, platform_cards=None)
        cands = [core.Candidate(w["slug"], 0.9 - i * 0.01, 10, 0.3) for i, w in enumerate(wines)]
        words = [word("РОЗОВОЕ", x=100), word("ПОЛУСУХОЕ", x=100)]
        plain = list(cands)
        core.demote_contradicted(plain, channel, words, 0.6)
        self.assertNotEqual(plain[0].slug, "rozovoe-polusuhoe-x")
        with Env(DEMOTE_DEPTH="1"):
            deep = list(cands)
            core.demote_contradicted(deep, channel, words, 0.6)
        self.assertEqual(deep[0].slug, "rozovoe-polusuhoe-x")


if __name__ == "__main__":
    unittest.main()
