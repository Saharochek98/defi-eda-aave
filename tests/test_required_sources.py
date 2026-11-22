"""The optional public source must not weaken the mandatory source contract."""

import unittest
from dataclasses import fields
from unittest.mock import patch

from src.config import REQUIRED_SOURCES, Settings, required_sources_verified
from src.onchain_sources import collect_dune


class RequiredSourcesTests(unittest.TestCase):
    def test_all_five_are_required(self):
        self.assertTrue(required_sources_verified(REQUIRED_SOURCES))
        for missing in REQUIRED_SOURCES:
            with self.subTest(missing=missing):
                candidate = set(REQUIRED_SOURCES) - {missing} | {"Dune"}
                self.assertEqual(len(candidate), 5)
                self.assertFalse(required_sources_verified(candidate))

    def test_no_dune_credentials_in_settings(self):
        self.assertFalse(any("dune" in f.name for f in fields(Settings)))

    def test_no_dune_api_request_without_public_csv(self):
        with patch("requests.Session.request", side_effect=AssertionError("Unexpected API call")):
            self.assertIsNone(collect_dune(None, None))


if __name__ == "__main__":
    unittest.main()
