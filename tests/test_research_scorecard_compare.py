#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "compare_kws_v2_research_scorecards.py"


def scorecard(*, margin: float, config_sha: str = "a" * 64, ordered: float = 0.35) -> dict:
    distribution = {
        "positive_true_keyword_confidence": {"p10": 0.45},
        "nonwake_max_keyword_confidence": {"p99": 0.65},
        "tokenized_nonwake_max_keyword_confidence": {"p99": 0.66},
        "empty_target_nonwake_max_keyword_confidence": {"p99": 0.50},
        "separation_gap_p10_wake_minus_p99_nonwake": -0.20,
        "wake_top1_frame_diagnostics": {
            "blank_top1_fraction": 0.90,
            "keyword_root_top1_fraction": 0.04,
        },
    }
    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-research-baseline-scorecard-v1",
        "evidence_scope": "research-only",
        "protected_evidence_used": False,
        "promotion_allowed": False,
        "family": "gru",
        "frontend": "logmel",
        "feature_dim": 32,
        "hidden_dim": 64,
        "config_sha256": config_sha,
        "training_seed": 1337,
        "training_epochs": 8,
        "loss_profile": "fixture",
        "loss_weights": {
            "ordered_token_loss_weight": ordered,
            "keyword_sequence_margin_loss_weight": margin,
            "prefix_completion_loss_weight": 0.0,
            "recurrent_release_loss_weight": 0.0,
        },
        "ordinary_speech_negative_sidecar": {
            "enabled": True,
            "receipt_sha256": "b" * 64,
            "total_recordings": 512,
        },
        "class_balance": {
            "policy": "ctc-target-and-wake-balance-v1",
            "wake_examples": 128,
            "tokenized_nonwake_examples": 384,
            "empty_target_examples": 256,
        },
        "float_ctc_confidence": {
            "splits": {
                "calibration": distribution,
                "test": distribution,
            }
        },
        "soft_operating_points_by_far_budget": {
            "120.0": {
                "threshold": 0.60,
                "calibration": {"frr": 0.90, "far_per_hour": 60.0},
                "test": {"frr": 0.92, "far_per_hour": 80.0},
            }
        },
        "research_negative_exposure": {"far_per_hour": 12.0},
        "classifier_baseline": {
            "test": {
                "positive_recall": 0.50,
                "negative_false_positive_clip_rate": 0.47,
                "score_distribution": {
                    "separation_gap_p10_positive_minus_p99_negative": -0.18
                },
            }
        },
    }


class ResearchScorecardCompareTest(unittest.TestCase):
    def run_compare(self, baseline: dict, candidate: dict) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            left = root / "baseline.json"
            right = root / "candidate.json"
            out = root / "comparison.json"
            left.write_text(json.dumps(baseline), encoding="utf-8")
            right.write_text(json.dumps(candidate), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--baseline",
                    str(left),
                    "--candidate",
                    str(right),
                    "--expect-single-variable",
                    "--output",
                    str(out),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if completed.returncode == 0:
                completed.report = json.loads(out.read_text())  # type: ignore[attr-defined]
            return completed

    def test_single_margin_delta_is_accepted(self) -> None:
        result = self.run_compare(scorecard(margin=0.1), scorecard(margin=0.3))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = result.report  # type: ignore[attr-defined]
        self.assertEqual(
            report["changed_loss_variables"],
            ["keyword_sequence_margin_loss_weight"],
        )

    def test_identity_drift_is_rejected(self) -> None:
        result = self.run_compare(
            scorecard(margin=0.1),
            scorecard(margin=0.3, config_sha="c" * 64),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("identity drift", result.stderr)

    def test_multi_variable_delta_is_rejected(self) -> None:
        result = self.run_compare(
            scorecard(margin=0.1, ordered=0.35),
            scorecard(margin=0.3, ordered=0.50),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly one loss-variable", result.stderr)


if __name__ == "__main__":
    unittest.main()
