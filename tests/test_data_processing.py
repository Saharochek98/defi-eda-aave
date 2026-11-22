"""Small test fixtures validate alignment rules; never used as research data."""

import unittest

import pandas as pd

from src.data_processing import daily_series


class DailySeriesTests(unittest.TestCase):
    def setUp(self):
        self.calendar = pd.date_range("2026-01-01", periods=3, tz="UTC")
        self.start = int(self.calendar[0].timestamp())

    def series(self, rows):
        return daily_series(rows, "s", "test", self.calendar)

    def test_missing_days_are_not_filled(self):
        series, _, audit = self.series([[self.start, 10], [self.start + 2 * 86400, 12]])
        self.assertTrue(pd.isna(series.iloc[1]))
        self.assertEqual(audit["missing_days"], 1)

    def test_identical_duplicates_are_counted(self):
        series, _, audit = self.series([[self.start, 10], [self.start, 10]])
        self.assertEqual(audit["duplicate_timestamps"], 1)
        self.assertEqual(series.iloc[0], 10)

    def test_conflicting_duplicates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.series([[self.start, 10], [self.start, 11]])

    def test_last_observation_per_day_is_selected(self):
        series, _, audit = self.series([[self.start + 3600, 11], [self.start, 10]])
        self.assertEqual(series.iloc[0], 11)
        self.assertEqual(audit["extra_observations_same_day"], 1)
        self.assertEqual(audit["non_midnight_observations_in_period"], 1)

    def test_invalid_values_are_rejected(self):
        for value in [-1, float("inf"), float("nan")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.series([[self.start, value]])

    def test_missing_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing timestamp"):
            self.series([[None, 10]])

    def test_zero_is_not_replaced(self):
        series, _, audit = self.series([[self.start, 0]])
        self.assertEqual(series.iloc[0], 0)
        self.assertEqual(audit["zero_values"], 1)


if __name__ == "__main__":
    unittest.main()
