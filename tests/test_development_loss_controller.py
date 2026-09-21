#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from development_loss_controller import (  # noqa: E402
    initial_controller,
    next_controller,
    validate_controller_config,
)


class DevelopmentLossControllerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gru_policy = json.loads(
            (ROOT / "configs/training/xiaowo.gru-development-stage-a-v1.json").read_text()
        )
        cls.rnn_policy = json.loads(
            (ROOT / "configs/training/xiaowo.rnn-development-stage-a-v1.json").read_text()
        )
        cls.gru_loop_policy = json.loads(
            (ROOT / "configs/training/xiaowo.gru-development-loop.json").read_text()
        )
        cls.rnn_loop_policy = json.loads(
            (ROOT / "configs/training/xiaowo.rnn-development-loop.json").read_text()
        )

    def test_development_policies_use_normalized_rates(self) -> None:
        for policy in (
            self.gru_policy,
            self.rnn_policy,
            self.gru_loop_policy,
            self.rnn_loop_policy,
        ):
            raw = policy["loss_controller"]
            self.assertEqual(validate_controller_config(raw), "normalized-rates-v1")
            self.assertGreater(float(raw["normalization_frr"]), 0.0)
            self.assertGreater(float(raw["normalization_far_per_hour"]), 0.0)
            self.assertGreaterEqual(float(raw["wake_example_weight_initial"]), 1.0)
            self.assertGreater(float(raw["wake_example_weight_step"]), 0.0)

    def test_rate_pressure_corrects_raw_count_inversion(self) -> None:
        policy = self.gru_policy
        current = initial_controller(policy)
        result = next_controller(
            policy,
            current,
            false_rejects=8,
            false_accepts=3,
            frr=0.008,
            far_per_hour=180.0,
        )
        self.assertEqual(result["controller_decision"], "precision")
        self.assertEqual(
            result["positive_example_weight"], current["positive_example_weight"]
        )
        self.assertLess(
            result["wake_example_weight"], current["wake_example_weight"]
        )
        self.assertGreater(
            result["ordered_token_loss_weight"], current["ordered_token_loss_weight"]
        )
        self.assertAlmostEqual(float(result["frr_pressure"]), 0.08)
        self.assertAlmostEqual(float(result["far_pressure"]), 3.0)
        self.assertEqual(result["failure_replay_repeat"], 2)

    def test_recall_pressure_moves_in_opposite_direction(self) -> None:
        policy = self.gru_policy
        current = initial_controller(policy)
        result = next_controller(
            policy,
            current,
            false_rejects=1,
            false_accepts=10,
            frr=0.30,
            far_per_hour=1.0,
        )
        self.assertEqual(result["controller_decision"], "recall")
        self.assertEqual(
            result["positive_example_weight"], current["positive_example_weight"]
        )
        self.assertGreater(
            result["wake_example_weight"], current["wake_example_weight"]
        )
        self.assertLess(
            result["ordered_token_loss_weight"], current["ordered_token_loss_weight"]
        )

    def test_severity_uses_normalized_rate_not_dataset_size(self) -> None:
        policy = self.gru_policy
        current = initial_controller(policy)
        low_counts = next_controller(
            policy, current, 1, 1, frr=0.20, far_per_hour=120.0
        )
        high_counts = next_controller(
            policy, current, 100, 100, frr=0.20, far_per_hour=120.0
        )
        self.assertEqual(
            low_counts["failure_replay_repeat"],
            high_counts["failure_replay_repeat"],
        )
        self.assertEqual(
            low_counts["controller_severity"],
            high_counts["controller_severity"],
        )

    def test_normalized_mode_preserves_requested_replay_latch(self) -> None:
        policy = self.rnn_loop_policy
        current = initial_controller(policy)
        failed = next_controller(
            policy,
            current,
            false_rejects=1,
            false_accepts=0,
            frr=0.20,
            far_per_hour=0.0,
            latch_after_failure=True,
        )
        self.assertEqual(failed["failure_replay_repeat"], 1)
        clean = next_controller(
            policy,
            failed,
            false_rejects=0,
            false_accepts=0,
            frr=0.0,
            far_per_hour=0.0,
            latch_after_failure=True,
        )
        self.assertEqual(clean["failure_replay_repeat"], 1)

    def test_normalized_mode_requires_rate_signals(self) -> None:
        policy = self.gru_policy
        current = initial_controller(policy)
        with self.assertRaisesRegex(ValueError, "requires FRR and FAR/hour"):
            next_controller(policy, current, 1, 1)

    def test_legacy_mode_remains_available_for_old_policies(self) -> None:
        policy = json.loads(json.dumps(self.gru_policy))
        raw = policy["loss_controller"]
        for key in (
            "signal_mode",
            "normalization_frr",
            "normalization_far_per_hour",
            "pressure_deadband",
            "wake_example_weight_initial",
            "wake_example_weight_min",
            "wake_example_weight_max",
            "wake_example_weight_step",
        ):
            raw.pop(key, None)
        current = initial_controller(policy)
        result = next_controller(policy, current, 8, 3)
        self.assertEqual(result["controller_signal_mode"], "legacy-counts-v1")
        self.assertEqual(result["controller_decision"], "recall")
        self.assertGreater(
            result["positive_example_weight"], current["positive_example_weight"]
        )
        self.assertEqual(result["wake_example_weight"], 1.0)


if __name__ == "__main__":
    unittest.main()
