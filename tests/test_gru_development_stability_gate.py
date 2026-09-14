#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FINALIZER = ROOT / "tools" / "finalize_gru_frozen_candidate.py"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(round_index: int, strict: bool) -> dict:
    return {
        "round": round_index,
        "calibration_gate": strict,
        "test_gate": strict,
    }


class GruDevelopmentStabilityGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.finalizer = load_module(FINALIZER, "gru_development_stability_gate")

    def test_single_terminal_strict_pass_is_rejected_when_two_are_required(self) -> None:
        development = {
            "records": [record(0, False), record(1, False), record(2, True)],
        }
        policy = {"stable_strict_pass_rounds": 2}
        with self.assertRaisesRegex(ValueError, "stable strict-pass streak"):
            self.finalizer.require_stable_strict_pass(development, policy)

    def test_two_terminal_strict_passes_are_accepted(self) -> None:
        development = {
            "records": [record(0, False), record(1, True), record(2, True)],
        }
        policy = {"stable_strict_pass_rounds": 2}
        self.assertEqual(self.finalizer.require_stable_strict_pass(development, policy), 2)

    def test_non_terminal_historical_streak_does_not_qualify(self) -> None:
        development = {
            "records": [record(0, True), record(1, True), record(2, False)],
        }
        policy = {"stable_strict_pass_rounds": 2}
        with self.assertRaisesRegex(ValueError, "stable strict-pass streak"):
            self.finalizer.require_stable_strict_pass(development, policy)

    def test_non_contiguous_rounds_fail_closed(self) -> None:
        development = {
            "records": [record(0, True), record(2, True)],
        }
        policy = {"stable_strict_pass_rounds": 2}
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.finalizer.require_stable_strict_pass(development, policy)


if __name__ == "__main__":
    unittest.main()
