#!/usr/bin/env python3
"""Source-only counterexamples for a controlled exposure-budget comparison."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training")]
import research_fit_budget_compare as comparison


def pair():
    rows = [{"source_id": "fixture-" + str(i), "ctc_weight": 0.5, "target_ids": [i + 1]}
            for i in range(8)]
    common = {"candidate": "c1-ctc2601", "prepared_manifest_sha256": "a" * 64,
        "row_binding_sha256": "b" * 64, "vocabulary_sha256": "c" * 64,
        "initial_checkpoint_sha256": "d" * 64, "initial_float_state_sha256": "e" * 64,
        "diagnostic_source_hashes": {"source": "f" * 64}, "trainable_parameter_names": ["weight"],
        "selected_training_rows": 8, "final_checkpoint_sha256": "1" * 64,
        "after": {"blank_argmax_fraction": .95, "exact_token_matches": 1}, "tiny_fit_passed": False}
    result = []
    for steps in (1, 4):
        recipe = {"batch_size": 4, "seed": 20261105, "optimizer_steps": steps,
            "total_exposures": steps * 4, "optimizers": {"fresh": {"lr": .001}},
            "exposure_plan": {"draws": [{"row_index": i % len(rows),
                "source_id": rows[i % len(rows)]["source_id"], "ctc_weight": rows[i % len(rows)]["ctc_weight"]}
                for i in range(steps * 4)]}}
        result.append({"summary": {**copy.deepcopy(common), "optimizer_steps": steps, "sample_exposures": steps * 4},
            "rows": copy.deepcopy(rows), "recipe": recipe, "manifest_sha256": "2" * 64,
            "audit_sha256": "3" * 64, "phases": {"before": [{"numeric_sha256": "4" * 64}]},
            "ledger_lines": [json.dumps({"step": i + 1, "loss": .5}).encode() + b"\n" for i in range(steps)]})
    return result


class ComparisonContracts(unittest.TestCase):
    def test_only_fixed_exposure_changes_with_zero_new_forwards(self):
        short, long = pair()
        result = comparison.compare(short, long)
        self.assertTrue(result["only_exposure_budget_control_verified"])
        self.assertEqual(result["new_model_forwards"], 0)
        self.assertEqual(result["new_optimizer_steps"], 0)
        self.assertFalse(result["independent_effectiveness_claim"])
        self.assertIsNone(result["quality_winner"])

    def test_initialization_vocabulary_labels_and_optimized_parameters_cannot_change(self):
        for key in ("candidate", "vocabulary_sha256", "initial_checkpoint_sha256",
                    "row_binding_sha256", "trainable_parameter_names"):
            short, long = pair()
            long["summary"][key] = "different"
            with self.assertRaisesRegex(ValueError, "FIT_COMPARE_FIXED_CONTROL_CHANGED"):
                comparison.compare(short, long)

    def test_target_and_confidence_mutations_cannot_be_hidden_in_same_row_id(self):
        for key, value in (("target_ids", [99]), ("ctc_weight", .9)):
            short, long = pair()
            long["rows"][0][key] = value
            with self.assertRaisesRegex(ValueError, "FIT_COMPARE_SELECTED_ROWS_CHANGED"):
                comparison.compare(short, long)

    def test_optimizer_change_and_post_prefix_sampling_change_are_rejected(self):
        short, long = pair()
        long["recipe"]["optimizers"]["fresh"]["lr"] *= 2
        with self.assertRaisesRegex(ValueError, "FIT_COMPARE_OPTIMIZATION_POLICY_CHANGED"):
            comparison.compare(short, long)
        short, long = pair()
        long["recipe"]["exposure_plan"]["draws"][-1] = copy.deepcopy(long["recipe"]["exposure_plan"]["draws"][-2])
        with self.assertRaisesRegex(ValueError, "FIT_COMPARE_EXTENDED_SAMPLING_POLICY_CHANGED"):
            comparison.compare(short, long)

    def test_consumption_numerical_drift_is_observed_not_silently_accepted(self):
        short, long = pair()
        long["ledger_lines"][0] = b'{"step": 1, "loss": 0.500001}\n'
        result = comparison.compare(short, long)
        self.assertEqual(result["status"], "numeric-control-mismatch")
        self.assertFalse(result["only_exposure_budget_control_verified"])
        self.assertEqual(result["first_different_common_step"], 1)

    def test_initial_logits_drift_prevents_exposure_only_conclusion(self):
        short, long = pair()
        long["phases"]["before"][0]["numeric_sha256"] = "changed"
        result = comparison.compare(short, long)
        self.assertFalse(result["before_observations_and_numeric_files_identical"])
        self.assertFalse(result["only_exposure_budget_control_verified"])

    def test_high_blank_occupancy_does_not_override_explicit_complete_memorization(self):
        short, long = pair()
        long["summary"]["after"] = {"blank_argmax_fraction": .97, "exact_token_matches": 8}
        long["summary"]["tiny_fit_passed"] = True
        result = comparison.compare(short, long)
        self.assertTrue(result["outcomes"]["long"]["tiny_fit_passed"])
        self.assertTrue(result["only_exposure_budget_control_verified"])
        self.assertIsNone(result["quality_winner"])

    def test_reversed_or_equal_budget_is_rejected(self):
        short, long = pair()
        with self.assertRaisesRegex(ValueError, "FIT_COMPARE_ORDERED_EXPOSURE_BUDGETS_REQUIRED"):
            comparison.compare(long, short)

    def test_audit_receipt_cannot_certify_another_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "MANIFEST.json").write_text("{}\n")
            audit = root / "audit.json"
            audit.write_text(json.dumps({"status": "verified", "new_model_forwards": 0,
                "optimizer_steps": 0, "tiny_fit_passed": False, "manifest_sha256": "a" * 64}))
            with self.assertRaisesRegex(ValueError, "FIT_COMPARE_MANIFEST_CHANGED"):
                comparison.load_audited_output(output, audit, comparison.training.sha_file(audit))
            with self.assertRaisesRegex(ValueError, "FIT_COMPARE_AUDIT_CHANGED"):
                comparison.load_audited_output(output, audit, "b" * 64)

    def test_import_and_help_construct_no_model(self):
        script = "import sys;sys.path.insert(0," + repr(str(ROOT / "tools")) + ");" + \
            "import research_fit_budget_compare;assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-I", "-S", "-B", "-c", script], check=True, capture_output=True)
        subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_fit_budget_compare.py"), "--help"],
                       check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
