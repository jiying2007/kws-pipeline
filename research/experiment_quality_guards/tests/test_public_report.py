"""Report adapters preserve public outcomes and distinguish invented examples."""
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import review_saved


class PublicReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((HERE / "fixtures/public_regressions.json").read_text())
        cls.result = review_saved.report(cls.fixture)

    def test_old6_public_outcome_keeps_both_denominators(self):
        counts = self.result["old6_descriptive_counts"]
        self.assertEqual(counts["original"], dict(wake_recordings_detected=1, wake_recordings=2,
            nonwake_recordings_with_false_events=1, nonwake_recordings=4))
        self.assertEqual(counts["candidate"], dict(wake_recordings_detected=0, wake_recordings=2,
            nonwake_recordings_with_false_events=2, nonwake_recordings=4))
        self.assertEqual(self.result["old6_event_comparison"]["regressions"], ["Z5", "Z6"])

    def test_illustrative_rows_cannot_be_presented_as_saved_recording_results(self):
        self.assertTrue(all(row["fixture_origin"] == "invented" for row in self.fixture["coverage_rows"]))
        self.assertTrue(all(key.startswith("illustrative-timing-") for key in self.fixture["timing_illustration_original"]))
        self.assertTrue(all(row["fixture_origin"] == "invented" for row in self.fixture["supervision_illustration"].values()))
        self.assertIn("Invented", self.result["evidence_basis"]["timing_rows"])
        aggregate = self.result["historical_old98_aggregate"]
        self.assertEqual(aggregate["availability_delays_ms"], [300, 180, 120])
        self.assertEqual(self.result["old98_rss"]["model_memory_regression"], "NOT_ESTABLISHED")
        self.assertFalse(self.result["scientific_qualification"])


if __name__ == "__main__":
    unittest.main()
