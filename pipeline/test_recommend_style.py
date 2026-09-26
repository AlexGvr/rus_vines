"""Аналоги совпадают со стилем: сладость и игристость, а не только цвет и сорт."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "service"))
from recommend import Recommender, sweetness_of  # noqa: E402


def wine(slug, title, maker, grapes=("Совиньон Блан",), category="Белое", rating=5):
    return {"slug": slug, "title": title, "manufacturer": maker, "category": category,
            "region": "Кубань", "grapes": list(grapes), "rating": rating}


WINES = [
    wine("belaya-lvicza", "Белая Львица белое полусладкое", "АРАТТИ"),
    wine("brule-brut", "Brule Sauvignon Blanc Brut", "Фанагория"),
    wine("ice-wine", "ICE Wine", "Шато Пино", rating=5),
    wine("zelenaya-dolina-polusladkoe", "Зелёная Долина Совиньон Блан", "Союз-Вино"),
    wine("mysxako-sovinon-blan-suhoe", "Совиньон Блан", "Мысхако"),
    wine("vibes-pet-nat-2022", "VIBES, Silvaner Pet-Nat 2022", "VIBES"),
    wine("david-2020", "David, 2020", "WINEMAFIA", grapes=("Саперави",), category="Красное"),
    wine("david-2021", "David, 2021", "WINEMAFIA", grapes=("Саперави",), category="Красное"),
    wine("saperavi-suhoe", "Саперави", "Фанагория", grapes=("Саперави",), category="Красное"),
]
# Сладость, которую знает распознавание (TextChannel): у ICE Wine её нет
# ни в slug, ни в названии — пусть будет сладкое из карточки платформы.
KNOWN = {"ice-wine": "сладкое", "zelenaya-dolina-polusladkoe": "полусладкое"}


class Style(unittest.TestCase):
    def setUp(self):
        self.rec = Recommender(WINES, KNOWN)

    def test_sweetness_from_title(self):
        self.assertEqual(sweetness_of("belaya-lvicza", "Белая Львица белое полусладкое"),
                         "полусладкое")
        # «полусладкое» не читается как «сладкое»
        self.assertEqual(sweetness_of("x", "X полусладкое"), "полусладкое")

    def test_analogs_keep_sweetness_and_still(self):
        slugs = [item["wine"]["slug"] for item in self.rec.similar("belaya-lvicza")]
        self.assertIn("zelenaya-dolina-polusladkoe", slugs)
        for other in ("brule-brut", "ice-wine", "mysxako-sovinon-blan-suhoe", "vibes-pet-nat-2022"):
            self.assertNotIn(other, slugs)

    def test_pet_nat_is_sparkling(self):
        self.assertEqual(self.rec.style("vibes-pet-nat-2022"), {"sweetness": None, "sparkling": True})

    def test_label_analogs_skip_vintages_and_unknown_producer(self):
        facets = self.rec.label_facets({"producer": "Мосавали", "color": "красное",
                                        "sweetness": "сухое", "sparkling": False,
                                        "grapes": ["Саперави"], "region": "Кахетия"})
        self.assertEqual(facets.manufacturer, "")
        self.assertEqual(facets.region, "")
        slugs = [item["wine"]["slug"] for item in self.rec.analogs(facets)]
        self.assertEqual(sum(s.startswith("david-") for s in slugs), 1)
        self.assertIn("saperavi-suhoe", slugs)

    def test_label_producer_from_catalog_is_excluded(self):
        facets = self.rec.label_facets({"producer": "Фанагория", "color": "красное",
                                        "grapes": ["саперави"]})
        slugs = [item["wine"]["slug"] for item in self.rec.analogs(facets)]
        self.assertNotIn("saperavi-suhoe", slugs)

    def test_sommelier_after_scan_prefers_same_color_and_skips_found(self):
        wines = [dict(w, dishes=["Сыры"]) for w in WINES + [
            wine("rose-suhoe", "Розе сухое", "Мысхако", category="Розовое"),
            wine("merlo-krasnoe-suhoe", "Мерло", "Кубань-Вино", grapes=("Мерло",),
                 category="Красное", rating=4)]]
        rec = Recommender(wines, KNOWN)
        picks = rec.sommelier({"dish": "Сыры", "taste": "сухое", "like": "saperavi-suhoe"})
        slugs = [p["wine"]["slug"] for p in picks]
        self.assertNotIn("saperavi-suhoe", slugs)
        self.assertEqual(rec.facets[slugs[0]].color, "красное")

    def test_by_label_name_prefers_producer_then_catalog(self):
        # винодельня найдена → её вина; название ближе к Brule
        slug, close = self.rec.by_label_name({"producer": "Фанагория", "name": "BRULE BRUT"})
        self.assertEqual(slug, "brule-brut")
        self.assertGreaterEqual(close, 0.6)
        # винодельни нет в каталоге → поиск по всему каталогу
        slug, _ = self.rec.by_label_name({"producer": "Неизвестная", "name": "Pet-Nat Silvaner"})
        self.assertEqual(slug, "vibes-pet-nat-2022")
        self.assertEqual(self.rec.by_label_name({"producer": "Фанагория"}), (None, 0.0))


if __name__ == "__main__":
    unittest.main()
