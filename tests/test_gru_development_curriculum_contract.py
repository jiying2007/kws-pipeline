#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ITERATOR = ROOT / "training" / "iterate_gru_development.py"
POLICY = ROOT / "configs" / "training" / "xiaowo.gru-development-loop.json"
FINALIZER = ROOT / "tools" / "finalize_gru_frozen_candidate.py"
VERIFIER = ROOT / "tools" / "verify_gru_frozen_candidate.py"
WORKFLOW = ROOT / ".github" / "workflows" / "gru-development-curriculum.yml"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GruDevelopmentCurriculumContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.iterator = load_module(ITERATOR, "iterate_gru_development_contract")
        cls.finalizer = load_module(FINALIZER, "finalize_gru_frozen_candidate_contract")
        cls.verifier = load_module(VERIFIER, "verify_gru_frozen_candidate_contract")
        cls.policy = json.loads(POLICY.read_text(encoding="utf-8"))

    def test_development_policy_is_explicitly_non_qualification(self) -> None:
        value = self.iterator.validate_policy(POLICY)
        self.assertEqual(value["policy"], "gru-development-curriculum-loop-v1")
        self.assertEqual(value["evidence_scope"], "development-only")
        self.assertFalse(value["qualification_used"])
        self.assertFalse(value["shadow_used"])
        self.assertFalse(value["formal_qualification_used"])
        freeze = value["candidate_freeze"]
        self.assertTrue(freeze["fresh_validation_required"])
        self.assertTrue(freeze["shadow_required"])
        self.assertTrue(freeze["formal_qualification_required"])
        self.assertFalse(freeze["validation_feedback_allowed"])
        self.assertFalse(freeze["threshold_feedback_allowed"])
        self.assertFalse(freeze["training_rule_feedback_allowed"])

    def test_policy_fails_closed_if_candidate_feedback_is_enabled(self) -> None:
        value = copy.deepcopy(self.policy)
        value["candidate_freeze"]["validation_feedback_allowed"] = True
        with tempfile.TemporaryDirectory() as temp:
            path = pathlib.Path(temp) / "policy.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_feedback_allowed=false"):
                self.iterator.validate_policy(path)

    def test_loss_controller_is_bounded_and_directional(self) -> None:
        current = self.iterator.controller_initial(self.policy)
        recall = self.iterator.controller_next(self.policy, current, 3, 0)
        self.assertGreater(recall["positive_example_weight"], current["positive_example_weight"])
        self.assertLess(recall["ordered_token_loss_weight"], current["ordered_token_loss_weight"])
        precision = self.iterator.controller_next(self.policy, current, 0, 5)
        self.assertLess(precision["positive_example_weight"], current["positive_example_weight"])
        self.assertGreater(precision["ordered_token_loss_weight"], current["ordered_token_loss_weight"])
        self.assertEqual(precision["failure_replay_repeat"], 2)
        heavy = self.iterator.controller_next(self.policy, current, 30, 30)
        self.assertLessEqual(
            heavy["failure_replay_repeat"], self.policy["failure_replay_repeat_max"]
        )

    def test_iterator_does_not_evaluate_candidate_qualification(self) -> None:
        text = ITERATOR.read_text(encoding="utf-8")
        self.assertNotIn('references=dataset / "qualification.references.jsonl"', text)
        self.assertNotIn("qualification_used_for_selection\": True", text)
        self.assertIn('"selection_evidence": ["development-calibration", "development-test"]', text)
        self.assertIn('"validation_feedback_allowed": False', text)
        self.assertIn("render_development_failure_replay", text)
        self.assertIn("render_hard_negative_replay", text)
        self.assertIn("update_curriculum", text)

    def _write_valid_candidate(self, root: pathlib.Path) -> dict:
        members = {
            "model_sha256": "model.kwm",
            "checkpoint_sha256": "model.pt",
            "pack_sha256": "keywords.kwk",
            "keywords_sha256": "keywords.tsv",
            "provenance_sha256": "model-provenance.json",
        }
        digests = {}
        for field, name in members.items():
            path = root / name
            path.write_bytes((name + "\n").encode("utf-8"))
            digests[field] = self.verifier.sha256_file(path)

        config = {
            "domain_gates": {
                "max_frr": 0.0,
                "max_far_per_hour": 0.0,
                "max_p95_latency_ms": 800.0,
                "max_far_frr": 0.0,
            }
        }
        source_policy = {
            "policy": "gru-development-curriculum-loop-v1",
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
        }
        selection = {
            "schema_version": 1,
            "evidence_class": "gru-frozen-development-selection",
            "source_policy": "gru-development-curriculum-loop-v1",
            "selected_round": 3,
            "selected_frontend": "logmel",
            "selected_score": 0.0,
            "model_sha256": digests["model_sha256"],
            "calibration_gate": True,
            "test_gate": True,
            "calibration": {
                "frr": 0.0,
                "far_per_hour": 0.0,
                "p95_post_end_latency_ms": 0.0,
            },
            "calibration_domains": {"domains": {"distance:far": {"frr": 0.0}}},
            "test": {
                "frr": 0.0,
                "far_per_hour": 0.0,
                "p95_post_end_latency_ms": 0.0,
            },
            "test_domains": {"domains": {"distance:far": {"frr": 0.0}}},
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
        }
        corpus = {
            "schema_version": 1,
            "evidence_class": "gru-frozen-development-wav-identities",
            "development_only": True,
            "qualification_used": False,
            "shadow_used": False,
            "formal_qualification_used": False,
            "wav_sha256_count": 1,
            "wav_sha256": ["0" * 64],
        }
        files = {
            "source-config.json": config,
            "source-development-policy.json": source_policy,
            "selection-evidence.json": selection,
            "development-wav-sha256.json": corpus,
        }
        for name, value in files.items():
            (root / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

        manifest = {
            "schema_version": 1,
            "policy": "gru-frozen-candidate-v1",
            "source_policy": "gru-development-curriculum-loop-v1",
            "evidence_scope": "development-only",
            "selected_round": 3,
            "selected_score": 0.0,
            "selection_evidence": ["development-calibration", "development-test"],
            "qualification_used_for_selection": False,
            "shadow_used_for_selection": False,
            "formal_qualification_used_for_selection": False,
            "candidate_stage": {
                "fresh_validation_required": True,
                "shadow_required": True,
                "formal_qualification_required": True,
                "bounded_repair_only": True,
                "validation_feedback_allowed": False,
                "threshold_feedback_allowed": False,
                "training_rule_feedback_allowed": False,
            },
            "selected_model_matches_selection_evidence": True,
            "config_sha256": self.verifier.sha256_file(root / "source-config.json"),
            "development_policy_sha256": self.verifier.sha256_file(
                root / "source-development-policy.json"
            ),
            "selection_evidence_sha256": self.verifier.sha256_file(
                root / "selection-evidence.json"
            ),
            "development_wav_identities_sha256": self.verifier.sha256_file(
                root / "development-wav-sha256.json"
            ),
            "source_config_snapshot_sha256": self.verifier.sha256_file(
                root / "source-config.json"
            ),
            "source_development_policy_snapshot_sha256": self.verifier.sha256_file(
                root / "source-development-policy.json"
            ),
            **digests,
        }
        (root / "freeze-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest

    def test_frozen_candidate_verifier_recomputes_strict_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            self._write_valid_candidate(root)
            self.verifier.verify(root)
            evidence_path = root / "selection-evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["test"]["frr"] = 0.1
            evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
            manifest_path = root / "freeze-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["selection_evidence_sha256"] = self.verifier.sha256_file(evidence_path)
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "strict development pass"):
                self.verifier.verify(root)

    def test_frozen_candidate_verifier_rejects_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            manifest = self._write_valid_candidate(root)
            self.verifier.verify(root)
            manifest["candidate_stage"]["validation_feedback_allowed"] = True
            (root / "freeze-manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "validation_feedback_allowed=false"):
                self.verifier.verify(root)

    def test_workflow_finalizes_before_verifying_handoff(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("tools/finalize_gru_frozen_candidate.py", text)
        self.assertIn("tools/verify_gru_frozen_candidate.py", text)
        self.assertLess(
            text.index("Finalize self-contained frozen candidate evidence"),
            text.index("Verify frozen candidate handoff"),
        )
        finalizer = FINALIZER.read_text(encoding="utf-8")
        self.assertIn("development-wav-sha256.json", finalizer)
        self.assertIn("selection-evidence.json", finalizer)
        self.assertIn('{"train", "calibration", "test"}', finalizer)


if __name__ == "__main__":
    unittest.main()
