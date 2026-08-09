from __future__ import annotations

import unittest

from normalizer import normalize_slug


class NormalizeSlugTests(unittest.TestCase):
    def test_normalizes_words(self) -> None:
        self.assertEqual(normalize_slug("Release Candidate"), "release-candidate")

    def test_trims_boundary_whitespace(self) -> None:
        self.assertEqual(normalize_slug("  Already-Ready  "), "already-ready")

    def test_preserves_simple_alphanumeric_values(self) -> None:
        self.assertEqual(normalize_slug("Build42"), "build42")


if __name__ == "__main__":
    unittest.main()
