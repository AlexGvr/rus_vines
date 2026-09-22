"""Regression checks for persistent field partitions (no model loading)."""
import unittest

from verify_field import assign_splits, split_nodes


def row(image, series, slug="", split="tune"):
    return dict(image=image, series=series, slug=slug, split=split)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("a.jpg", "s1", "wine1"),
                     row("b.jpg", "s2", "wine2", "test")]
        self.lock = {n: r["split"] for r in self.rows for n in split_nodes(r)}

    def test_addition_and_reordering_preserve_existing_splits(self):
        rows = [row("0.jpg", "s0", "wine0"), *reversed(self.rows)]
        splits, find = assign_splits(rows, self.lock)
        self.assertEqual([splits[find("кадр:" + r["image"])] for r in rows],
                         ["tune", "test", "tune"])

    def test_new_photo_inherits_wine_partition(self):
        splits, find = assign_splits([row("c.jpg", "s3", "wine2")], self.lock)
        self.assertEqual(splits[find("кадр:c.jpg")], "test")

    def test_bridge_between_partitions_fails(self):
        with self.assertRaisesRegex(ValueError, "объединяет tune и test"):
            assign_splits([*self.rows, row("c.jpg", "s1", "wine2")], self.lock)

    def test_relabel_cannot_move_existing_image(self):
        with self.assertRaises(ValueError):
            assign_splits([row("a.jpg", "new-series", "wine2")], self.lock)


if __name__ == "__main__":
    unittest.main()
