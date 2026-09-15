#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "gru-frozen-candidate-qualification.yml"
POLICY = ROOT / "configs" / "training" / "xiaowo.gru-development-loop.json"
FRESH_REGISTRY = ROOT / "experiments" / "model_family" / "fresh_validation_registry.json"
SHADOW_REGISTRY = ROOT / "experiments" / "model_family" / "shadow_arena_registry.json"
FRESH = ROOT / "training" / "validate_frozen_gru_candidate.py"
FORMAL = ROOT / "training" / "qualify_frozen_gru_formal.py"
MATERIALIZE = ROOT / "tools" / "materialize_gru_candidate_workspace.py"
SHADOW_CONFIG = ROOT / "tools" / "prepare_gru_shadow_config.py"
INDEPENDENCE = ROOT / "tools" / "verify_gru_evaluation_independence.py"


class GruFrozenQualificationContractTest(unittest.TestCase):
    def test_candidate_namespaces_are_frozen_before_qualification(self) -> None:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        fresh_registry = json.loads(FRESH_REGISTRY.read_text(encoding="utf-8"))
        shadow_registry = json.loads(SHADOW_REGISTRY.read_text(encoding="utf-8"))
        freeze = policy["candidate_freeze"]

        self.assertEqual(freeze["fresh_validation_seed_namespace"], 195000019)
        fresh_by_namespace = {
            int(row["namespace"]): row for row in fresh_registry["namespaces"]
        }
        current_fresh = fresh_by_namespace[freeze["fresh_validation_seed_namespace"]]
        self.assertEqual(current_fresh["name"], "gru-fresh-validation-v4")
        self.assertEqual(current_fresh["model_family"], "gru")
        self.assertEqual(current_fresh["status"], "reserved-untouched")
        consumed_fresh = fresh_by_namespace[194000019]
        self.assertEqual(consumed_fresh["name"], "gru-fresh-validation-v3")
        self.assertEqual(consumed_fresh["status"], "opened")
        self.assertEqual(consumed_fresh["source_run_id"], 34987002760)
        self.assertEqual(
            consumed_fresh["source_head"],
            "ca3a78c476a6adaf99f210943951b790c952126c",
        )
        self.assertEqual(
            consumed_fresh["source_artifact_digest"],
            "sha256:4e18ebfd74ea74fbc6fcbb7c7ecd2ed3c92452cbbd4648c2f4f4efae3697fcf4",
        )
        self.assertEqual(consumed_fresh["result"], "failed")
        self.assertFalse(consumed_fresh["shadow_consumed"])
        self.assertEqual(
            consumed_fresh["candidate_model_sha256"],
            "02d9011dd564c1544619e5535b8065e4dc192fecd02cb66fb4c8d33f9708946d",
        )
        self.assertFalse(consumed_fresh["formal_qualification_seed_consumed"])

        self.assertEqual(freeze["shadow_arena"], "gru-independent-shadow-v4")
        arenas = {row["name"]: row for row in shadow_registry["arenas"]}
        current = arenas[freeze["shadow_arena"]]
        self.assertEqual(current["model_family"], "gru")
        self.assertEqual(current["status"], "reserved-untouched")
        self.assertEqual(current["seeds"], list(range(981101, 981109)))
        consumed = arenas["gru-independent-shadow-v3"]
        self.assertEqual(consumed["status"], "opened")
        self.assertEqual(consumed["source_run_id"], 34935659562)
        self.assertEqual(
            consumed["candidate_model_sha256"],
            "3b1e0b43126fa5d761a005f9cff3b3bdc2e9c14402d77a27024c2aa3578154de",
        )
        self.assertEqual(consumed["result"], "failed-3-of-8")
        self.assertEqual(consumed["runtime_result"], "failed-3-of-8")
        self.assertFalse(consumed["formal_qualification_seed_consumed"])
        self.assertTrue(freeze["formal_qualification_required"])
        self.assertFalse(freeze["validation_feedback_allowed"])
        self.assertFalse(freeze["threshold_feedback_allowed"])
        self.assertFalse(freeze["training_rule_feedback_allowed"])

    def test_qualification_workflow_is_manual_for_data_consuming_stages(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_dispatch:", text)
        self.assertIn("if: github.event_name == 'workflow_dispatch'", text)
        self.assertIn("expected_artifact_digest", text)
        self.assertIn("expected_source_head", text)
        self.assertIn("cancel-in-progress: false", text)
        self.assertIn("experiments/model_family/fresh_validation_registry.json", text)
        self.assertIn("experiments/model_family/shadow_arena_registry.json", text)
        fresh = text.index("Run feedback-free fresh validation")
        shadow = text.index("Run reserved shadow qualification")
        formal = text.index("Run isolated formal qualification")
        self.assertLess(fresh, shadow)
        self.assertLess(shadow, formal)
        self.assertIn("steps.fresh.outputs.exit_code == '0'", text)
        self.assertIn("steps.shadow.outputs.exit_code == '0'", text)

    def test_candidate_stages_cannot_train_or_recalibrate(self) -> None:
        fresh = FRESH.read_text(encoding="utf-8")
        formal = FORMAL.read_text(encoding="utf-8")
        joined = fresh + "\n" + formal
        self.assertNotIn("train_gru_ctc.py", joined)
        self.assertNotIn("--warm-start", joined)
        self.assertNotIn("calibrate(", fresh)
        self.assertIn('"thresholds_recalibrated": False', fresh)
        self.assertIn('"training_performed": False', fresh)
        self.assertIn('"validation_feedback_allowed": False', fresh)
        self.assertIn('"threshold_feedback_allowed": False', formal)
        self.assertIn('"training_rule_feedback_allowed": False', formal)

    def test_fresh_validation_checks_one_shot_registry_and_development_disjointness(self) -> None:
        text = FRESH.read_text(encoding="utf-8")
        self.assertIn("fresh_validation_registry.json", text)
        self.assertIn("fresh validation namespace is no longer reserved-untouched", text)
        self.assertIn('consumed_model = row.get("candidate_model_sha256")', text)
        self.assertIn(
            "frozen candidate model already consumed by an earlier fresh validation namespace",
            text,
        )
        self.assertIn("development-wav-sha256.json", text)
        self.assertIn("development_hashes & fresh_hashes", text)
        self.assertIn('{"calibration", "test", "qualification"}', text)
        self.assertIn("fresh validation seed overlaps protected qualification namespace", text)
        self.assertIn('"fresh_validation_wav_sha256": sorted(fresh_hashes)', text)
        self.assertIn('"fresh_validation_registry_entry":', text)

    def test_fresh_and_shadow_internal_sha_independence_precede_protected_stages(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        verifier = INDEPENDENCE.read_text(encoding="utf-8")
        fresh = workflow.index("Run feedback-free fresh validation")
        fresh_independence = workflow.index("Verify fresh validation split SHA independence")
        shadow = workflow.index("Run reserved shadow qualification")
        shadow_independence = workflow.index("Verify reserved shadow seed SHA independence")
        formal = workflow.index("Run isolated formal qualification")
        self.assertLess(fresh, fresh_independence)
        self.assertLess(fresh_independence, shadow)
        self.assertLess(shadow, shadow_independence)
        self.assertLess(shadow_independence, formal)
        self.assertIn("fresh-split-independence.json", workflow)
        self.assertIn("shadow-seed-independence.json", workflow)
        self.assertIn("tools/verify_gru_evaluation_independence.py", workflow)
        self.assertIn("contains duplicate WAV SHA256 evidence", verifier)
        self.assertIn("with an earlier fresh split", verifier)
        self.assertIn("with an earlier shadow seed", verifier)
        self.assertIn('"all_fresh_splits_sha_disjoint": True', verifier)
        self.assertIn('"all_shadow_seeds_sha_disjoint": True', verifier)

    def test_shadow_is_bound_to_reserved_registry_arena(self) -> None:
        text = SHADOW_CONFIG.read_text(encoding="utf-8")
        self.assertIn("shadow_arena_registry.json", text)
        self.assertIn('arena.get("status") != "reserved-untouched"', text)
        self.assertIn('raw["seeds"] = seeds', text)
        self.assertIn("formal qualification namespace", text)
        self.assertIn('consumed_model = row.get("candidate_model_sha256")', text)
        self.assertIn(
            "frozen candidate model already consumed by an earlier shadow arena",
            text,
        )

    def test_formal_gate_requires_fresh_and_shadow_and_all_sha_isolation(self) -> None:
        text = FORMAL.read_text(encoding="utf-8")
        self.assertIn("fresh frozen-candidate validation is not qualified", text)
        self.assertIn("reserved shadow qualification is not qualified", text)
        self.assertIn("development_hashes & fresh_hashes", text)
        self.assertIn("development_hashes & seen_shadow_hashes", text)
        self.assertIn("fresh_hashes & seen_shadow_hashes", text)
        self.assertIn("active_hashes & development_hashes", text)
        self.assertIn("active_hashes & fresh_hashes", text)
        self.assertIn("active_hashes & seen_shadow_hashes", text)
        self.assertIn('(\"frozen development/training corpus\", development_hashes)', text)
        self.assertIn('(\"fresh validation cohort\", fresh_hashes)', text)
        self.assertIn('(\"reserved shadow cohort\", seen_shadow_hashes)', text)
        self.assertIn('(\"another retired formal cohort\", seen_retired_hashes)', text)
        self.assertIn('"all_pairwise_sha_disjoint": True', text)
        self.assertIn("retired formal seed", text)
        self.assertIn('"formal_seed_consumed": True', text)
        self.assertIn('"validation_feedback_allowed": False', text)

    def test_formal_cohorts_recheck_internal_sha_uniqueness(self) -> None:
        text = FORMAL.read_text(encoding="utf-8")
        self.assertIn("from verify_gru_evaluation_independence import split_hashes", text)
        self.assertIn('values = split_hashes(index, "qualification")', text)
        self.assertIn("with an earlier shadow seed", text)
        self.assertIn('active_hashes = split_hashes(active_index, "qualification")', text)
        self.assertIn('hashes = split_hashes(retired_index, "qualification")', text)
        self.assertIn("raw WAV identities disagree with renderer evidence", text)
        self.assertIn('"wav_sha256_count": len(hashes)', text)
        self.assertIn('"all_formal_cohorts_internal_sha_unique": True', text)

    def test_materialized_workspace_preserves_strict_selection_contract(self) -> None:
        text = MATERIALIZE.read_text(encoding="utf-8")
        self.assertIn('"policy": "latest-strict-gate-passing-round"', text)
        self.assertIn('"calibration_gate": True', text)
        self.assertIn('"test_gate": True', text)
        self.assertIn('"qualification_used_for_selection": False', text)
        self.assertIn("verify(candidate)", text)


if __name__ == "__main__":
    unittest.main()
