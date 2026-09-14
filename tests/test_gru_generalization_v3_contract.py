#!/usr/bin/env python3
from __future__ import annotations

import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "gru_generalization_v3" / "run_candidate.py"
RENDER = ROOT / "experiments" / "gru_generalization_v3" / "render_calibration_repair.py"
WORKFLOW = ROOT / ".github" / "workflows" / "gru-generalization-v3.yml"


def constants(path: pathlib.Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            pass
    return result


class GruGeneralizationV3ContractTest(unittest.TestCase):
    def test_bounded_repair_policy_is_frozen(self) -> None:
        value = constants(RUNNER)
        self.assertEqual(value["POLICY"], "bounded-gru-generalization-v3")
        self.assertEqual(value["POSITIVE_EXAMPLE_WEIGHT"], 2.45)
        self.assertEqual(value["REPAIR_EPOCHS"], 6)
        self.assertEqual(value["REPAIR_LR_SCALE"], 0.25)
        self.assertEqual(value["BASE_REPLAY_EXPOSURE_REPEAT"], 3)
        self.assertEqual(value["CALIBRATION_REPAIR_EXPOSURE_REPEAT"], 1)
        self.assertEqual(value["VALIDATION_SEED_NAMESPACE"], 181_000_019)
        self.assertEqual(value["RESERVED_SHADOW_ARENA"], "gru-independent-shadow-v2")
        self.assertEqual(
            value["SOURCE_MODEL_SHA256"],
            "8dc7d85505147fb6c9b402b2b07f6f0b047335207ca954800ac5e311c46a3c7d",
        )

    def test_calibration_repair_is_resynthesis_only(self) -> None:
        value = constants(RENDER)
        self.assertEqual(value["POLICY"], "gru-v3-calibration-repair-resynthesis-v1")
        self.assertEqual(value["EXAMPLES_PER_FAILURE"], 8)
        self.assertEqual(value["SEED_NAMESPACE"], 181_500_019)
        text = RENDER.read_text(encoding="utf-8")
        self.assertIn('"development_source_wav_bytes_copied": False', text)
        self.assertIn('"source_split": "calibration"', text)
        self.assertIn('"domain-calibration-000095"', text)

    def test_candidate_requires_fresh_untrained_validation(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"fresh_validation_used_for_training": False', text)
        self.assertIn('"used_as_unbiased_validation": False', text)
        self.assertIn('"calibration_wav_bytes_copied": False', text)
        self.assertIn('reserved.get("status") != "reserved-untouched"', text)
        self.assertIn('list(range(951101, 951109))', text)
        self.assertIn('"latest-strict-gate-passing-round"', text)
        self.assertIn('"single-predeclared-bounded-repair-no-shadow-selection"', text)

    def test_workflow_is_one_shot_trigger_only(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("branches:\n      - codex/gru-generalization-v3", text)
        self.assertIn("- '.github/triggers/gru-generalization-v3.txt'", text)
        self.assertNotIn("workflow_dispatch:", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn("SOURCE_CANDIDATE_RUN_ID: '34790831573'", text)
        self.assertIn("CALIBRATION_DIAGNOSTIC_RUN_ID: '34793030728'", text)
        self.assertIn("EXPECTED_SOURCE_CANDIDATE_ARTIFACT_DIGEST: 'sha256:52626953d0a1bd7bbe00c4e17a7519807d1ba13b7fc1ca5f5c802bdcc875f565'", text)
        self.assertIn("EXPECTED_CALIBRATION_DIAGNOSTIC_ARTIFACT_DIGEST: 'sha256:647178320b1f4fd3715a5996642ab5b5407a9c66f787a42aa88dac6f9f3a50f8'", text)


if __name__ == "__main__":
    unittest.main()
