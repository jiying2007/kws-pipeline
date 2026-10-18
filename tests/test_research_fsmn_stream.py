"""Hosted opt-in build entry: only synthetic mocked transport, no qualification."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

EXECUTE = False


class ResearchFsmnStreamTest(unittest.TestCase):
    def test_bound_build_and_eight_mock_groups(self):
        if not EXECUTE:
            self.skipTest("explicit test entry --execute is required")
        root = Path(__file__).resolve().parents[1]
        driver = root / "research/fsmn2599-stream-c/build.py"
        spec = importlib.util.spec_from_file_location("fsmn_stream_build_guard", driver)
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)  # Definitions only; does not compile or run.
        with tempfile.TemporaryDirectory(prefix="fsmn-stream-entry-") as temporary:
            output, _ = guard.run_bounded(
                [sys.executable, "-B", str(driver), "--execute"], Path(temporary), 110, 100)
        report = json.loads(output)
        self.assertEqual(report["fixture_groups"], 8)
        self.assertEqual(report["fixture_runs"], 1)
        self.assertEqual(report["compiler_invocations"], 3)
        self.assertEqual(report["compile_budget_seconds"], 60)
        self.assertFalse(report["example_executed"])
        self.assertFalse(report["production_loaded"])
        self.assertFalse(report["qualification_allowed"])
        self.assertEqual(report["fixture_message"], guard.FIXTURE_MESSAGE)
        self.assertEqual(set(report["production_exports"]),
                         guard.ADAPTER | guard.DECODER_EXPORTS | guard.CORE_EXPORTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="build and run the eight mocked glue groups once")
    args = parser.parse_args()
    if args.execute:
        EXECUTE = True
        unittest.main(argv=[sys.argv[0]])
    else:
        parser.print_help()
