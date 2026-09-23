"""Переключатель PLATFORM_ADDITIONS: позиции после выгрузки выпадают из индекса."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import searchcore as core  # noqa: E402


class PlatformScope(unittest.TestCase):
    def setUp(self):
        self.saved = (core.USE_PLATFORM_ADDITIONS, core._ADDITIONS)
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump({"wines": [{"slug": "merlo-2"}, {"slug": "sovinon-blan"}]}, self.tmp)
        self.tmp.close()
        core._ADDITIONS = Path(self.tmp.name)

    def tearDown(self):
        core.USE_PLATFORM_ADDITIONS, core._ADDITIONS = self.saved
        Path(self.tmp.name).unlink()

    def test_on_keeps_everything(self):
        core.USE_PLATFORM_ADDITIONS = True
        slugs = ["merlo-litavshhuk", "merlo-2", None]
        self.assertEqual(core.scope_index(slugs), slugs)

    def test_off_masks_additions_only(self):
        core.USE_PLATFORM_ADDITIONS = False
        self.assertEqual(core.scope_index(["merlo-litavshhuk", "merlo-2", "sovinon-blan", None]),
                         ["merlo-litavshhuk", None, None, None])


if __name__ == "__main__":
    unittest.main()
