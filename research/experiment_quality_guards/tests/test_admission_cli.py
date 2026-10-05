"""Exercise actual CLI exit codes on invented rows and exposed saved labels."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from admit_dataset import assess_admission


def example():
    policy = {"policy_id": "invented-builder-test-v1", "frozen": True,
              "minima": {"K1": 1, "K2": 1}, "nonwake_minima": {"你好小屋": 1}}
    rows = [{"id": str(i), "source_group": "voice-A", "split": "dev",
             "actual_text": text, "review": {"status": "clean", "independent_human": True, "complete": True},
             "pcm_sha256": hashlib.sha256(str(i).encode()).hexdigest(), "exposure": "EXPOSED",
             "voice_identity": "voice-A", "lineage_status": "verified", "use": "development"}
            for i, text in enumerate(("你好小窝", "小窝小窝", "你好小屋"))]
    return {"requested_qualification": "balanced_source_groups", "require_unseen_voice": True,
            "rows": rows, "history": [], "declarations": [{"source_group": "voice-A", "split": "dev", "role": "balanced"}],
            "policy": policy, "frozen_policy_sha256": hashlib.sha256(json.dumps(policy, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()}


class AdmissionCliTests(unittest.TestCase):
    def run_cli(self, payload, expected_code):
        result = subprocess.run([sys.executable, str(HERE / "admit_dataset.py"), "-"],
                                input=json.dumps(payload), text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, expected_code, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def test_success_exposed_balanced_data_does_not_grant_training(self):
        result = self.run_cli(example(), 0)
        self.assertTrue(result["coverage_eligible"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["product_qualified"])
        self.assertFalse(result["fresh_validation_qualified"])

    def test_missing_positive_rejects_even_with_generation_intent(self):
        payload = example()
        payload["rows"][1].update(actual_text="小窝", intended_text="小窝小窝")
        self.assertIn("BALANCED_COVERAGE_INSUFFICIENT", self.run_cli(payload, 1)["reasons"])

    def test_cross_generator_reference_voice_leakage_rejects(self):
        payload = example()
        historical = dict(payload["rows"][0], id="Qwen-original", split="train", use="training")
        payload["history"] = [historical]
        for row in payload["rows"]:
            row.update(voice_identity="Cosy-renamed", reference_sha256=historical["pcm_sha256"])
        self.assertIn("IDENTITY_OR_EXPOSURE_CONFLICT", self.run_cli(payload, 1)["reasons"])

    def test_unknown_lineage_cannot_earn_requested_unseen_voice(self):
        payload = example()
        for row in payload["rows"]:
            row["lineage_status"] = "unknown"
        self.assertIn("UNSEEN_VOICE_NOT_ESTABLISHED", self.run_cli(payload, 1)["reasons"])

    def test_old6_exposed_report_allowed_without_balanced_claim(self):
        fixture = json.loads((HERE / "fixtures/public_regressions.json").read_text())
        payload = example()
        payload.update(requested_qualification="exposed_regression", require_unseen_voice=False,
                       rows=[dict(r, source_group="historical") for r in fixture["old6_rows"]],
                       history=fixture["identity_history"],
                       declarations=[{"source_group": "historical", "split": "regression", "role": "balanced"}],
                       policy=fixture["coverage_policy"], frozen_policy_sha256=fixture["coverage_policy_sha256"])
        result = self.run_cli(payload, 0)
        self.assertFalse(result["coverage_eligible"])
        self.assertIn("report/code regression only", result["qualification_scope"])
        payload["requested_qualification"] = "balanced_source_groups"
        self.run_cli(payload, 1)

    def test_frozen_policy_corruption_has_invalid_input_exit(self):
        payload = example()
        payload["policy"]["minima"]["K1"] = 2
        self.assertEqual(self.run_cli(payload, 2)["status"], "INVALID_INPUT")

    def test_function_and_cli_share_result(self):
        result, code = assess_admission(example())
        self.assertEqual(code, 0)
        self.assertEqual(result, self.run_cli(example(), 0))

    def test_exposed_regression_with_complete_coverage_does_not_grant_balanced_claim(self):
        payload = example()
        payload.update(requested_qualification="exposed_regression", require_unseen_voice=False)
        for row in payload["rows"]:
            row.update(use="regression", split="regression")
        payload["declarations"][0]["split"] = "regression"
        result = self.run_cli(payload, 0)
        self.assertTrue(result["coverage"]["balanced_admission"])
        self.assertFalse(result["coverage_eligible"])
        self.assertFalse(result["training_authorized"])
        self.assertFalse(result["fresh_validation_qualified"])

    def test_unknown_split_cannot_bypass_identity_conflicts(self):
        for location in ("rows", "history", "declarations"):
            with self.subTest(location=location):
                payload = example()
                payload["require_unseen_voice"] = False
                payload["history"] = [dict(payload["rows"][0], id="previous", split="train", use="training")]
                if location == "rows":
                    for row in payload["rows"]:
                        row["split"] = "development"
                    payload["declarations"][0]["split"] = "development"
                else:
                    payload[location][0]["split"] = "training"
                self.assertEqual(self.run_cli(payload, 2)["status"], "INVALID_INPUT")
