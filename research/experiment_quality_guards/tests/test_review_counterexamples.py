"""Independent saved-record counterexamples; standard library and no execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import quality_gates as gates

FIXTURE = json.loads((HERE / "fixtures/public_regressions.json").read_text())
ASR_FIXTURE = json.loads((HERE / "fixtures/observed_asr_decisions.json").read_text())


def row(identity, text, **changes):
    return dict(id=identity, actual_text=text, source_group="voice-A", split="dev",
                review=dict(status="clean", independent_human=True, complete=True),
                pcm_sha256=hashlib.sha256(identity.encode()).hexdigest(), **changes)


class ReviewCounterexamples(unittest.TestCase):
    def coverage(self, rows, declarations):
        return gates.coverage_admission(rows, declarations, FIXTURE["coverage_policy"],
                                        FIXTURE["coverage_policy_sha256"])

    def test_invalid_negative_cohort_blocks_overall_admission(self):
        rows = [row(str(i), text) for i, text in enumerate(
            ("你好小窝", "小窝小窝", "你好小屋", "小屋小屋", "你好你好", "小窝", "小挖"))]
        negative = row("negative", "你好小窝")
        negative.update(source_group="background", split="background")
        result = self.coverage(rows + [negative], [
            dict(source_group="voice-A", split="dev", role="balanced"),
            dict(source_group="background", split="background", role="negative_only")])
        self.assertEqual(result["groups"][0]["status"], "ADMITTED_BALANCED")
        self.assertEqual(result["groups"][1]["status"], "INVALID_NEGATIVE_ONLY")
        self.assertFalse(result["balanced_admission"])

    def test_missing_event_arm_stays_unknown_even_when_other_arm_is_empty(self):
        for present in ([], [{"keyword": 1, "available_samples": 100}]):
            for original, candidate in (({"a": present}, {}), ({}, {"a": present})):
                with self.subTest(original=original, candidate=candidate):
                    result = gates.availability_report(original, candidate, {"a": 1000})
                    self.assertEqual(result["status"], "AMBIGUOUS_OR_UNKNOWN")
                    self.assertTrue(result["unknown"])
                    self.assertFalse(result["unmatched"])
        known_zero = gates.availability_report(
            {"a": [{"keyword": 1, "available_samples": 100}]}, {"a": []}, {"a": 1000})
        self.assertEqual(known_zero["unmatched"][0]["change"], "removed")

    def test_unrecognized_supervision_status_or_schema_cannot_pass(self):
        for changes in ({"status": "RUNNING"}, {"status": "FAILED"}, {"status": ""},
                        {"status": "UNRECOGNIZED"}, {"schema": "unrecognized"}):
            with self.subTest(changes=changes):
                resource = copy.deepcopy(FIXTURE["supervision_illustration"]["original_A20"])
                resource.update(changes)
                result = gates.supervision_interpretation(resource)
                self.assertEqual(result["integrity_status"], "FAIL")
                self.assertEqual(result["original_execution_status"], resource["status"])

    def test_prego_children_and_missing_live_children_status_cannot_pass(self):
        for changes in (
                {"children_count": 1, "children": "123", "children_observation": "AVAILABLE_AT_SAMPLE"},
                {"children_count": 0, "children_observation": "NOT_AVAILABLE"},
                {"children_observation": "FAILED_UNAVAILABLE"}):
            with self.subTest(changes=changes):
                resource = copy.deepcopy(FIXTURE["supervision_illustration"]["original_A20"])
                resource["pre_go_observation"].update(changes)
                self.assertEqual(gates.supervision_interpretation(resource)["integrity_status"], "FAIL")
        resource = copy.deepcopy(FIXTURE["supervision_illustration"]["original_A20"])
        resource["proc_samples_compact"][0][9] = "UNKNOWN"
        self.assertEqual(gates.supervision_interpretation(resource)["integrity_status"], "FAIL")

    def test_inherited_recording_restriction_matches_explicit_restriction(self):
        old = FIXTURE["old6_rows"][0]
        for use in ("calibration", "evaluation", None):
            with self.subTest(use=use):
                alias = dict(old, id="renamed", use=use)
                alias.pop("restriction")
                self.assertEqual(gates.identity_audit([alias], [old])["status"], "FAIL")
        sibling = dict(old, id="different-recording", use="calibration", wav_sha256="new-wave",
                       pcm_sha256="new-pcm")
        sibling.pop("restriction")
        self.assertEqual(gates.identity_audit([sibling], [old])["status"], "CHECKED_DECLARED_LINEAGE")

    def test_overlapping_keyword_occurrences_require_instance_annotation(self):
        original = {"a": [{"keyword": 2}]}
        result = gates.event_comparison([row("a", "小窝小窝小窝")], original, original)
        self.assertEqual(result["event_status"], "UNKNOWN")
        self.assertEqual(result["unscored"], ["a"])

    def test_whitespace_and_punctuation_do_not_change_actual_label_truth(self):
        for text, expected in (("你好，小窝", {"K1"}), (" 小窝 小窝。", {"K2"}),
                               ("， 。", None), ("小挖。", set())):
            with self.subTest(text=text):
                self.assertEqual(gates.human_truth(row("a", text)), expected)
        result = self.coverage([row("a", "你好，小屋。")], [
            dict(source_group="voice-A", split="dev", role="balanced")])
        self.assertEqual(result["groups"][0]["counts"]["nonwake:你好小屋"], 1)
        self.assertEqual(result["groups"][0]["counts"]["nonwake:out_of_vocabulary"], 0)


class ObservedAsrDecisionRegressions(unittest.TestCase):
    def setUp(self):
        self.rows = {r["id"]: copy.deepcopy(r) for r in ASR_FIXTURE["rows"]}

    def check(self, alias):
        return gates.label_preparation(self.rows[alias])

    def test_six_observed_wrong_consensuses_do_not_support_original_plan(self):
        for alias in ("E", "X", "L", "I", "P", "D"):
            with self.subTest(alias=alias):
                source = self.rows[alias]
                before = copy.deepcopy(source)
                result = self.check(alias)
                self.assertEqual(result["machine_state"], "machinesAgreeWeak")
                self.assertEqual(result["planned_lexical_support"], "REJECTED")
                self.assertFalse(result["original_complete_label_eligible"])
                self.assertEqual(result["actual_text"], source["actual_text"])
                self.assertEqual(source, before)
                self.assertFalse(result["automatic_relabel"])

    def test_observed_D_I_are_xiaowu_errors_not_plan_repetition_completion(self):
        for alias, actual, plan in (("D", "小窝", "小窝小窝"), ("I", "小屋", "小屋小屋")):
            with self.subTest(alias=alias):
                result = self.check(alias)
                self.assertEqual(result["machine_normalized_texts"], ["小五", "小五"])
                self.assertEqual((result["actual_text"], result["intended_text"]), (actual, plan))
                self.assertFalse(result["plan_matches_human_actual"])
                self.assertEqual(result["actual_expected_keywords"], [])

    def test_missing_plan_never_uses_matching_actual_as_substitute(self):
        source = self.rows["Z1"]
        source["intended_text"] = None
        result = gates.label_preparation(source)
        self.assertEqual(result["machine_normalized_texts"], ["你好", "你好"])
        self.assertEqual(result["planned_lexical_support"], "UNKNOWN")
        self.assertIsNone(result["plan_matches_human_actual"])
        self.assertFalse(result["original_complete_label_eligible"])
        self.assertEqual(result["actual_text"], "你好")

    def test_verified_Z1_original_plan_is_separate_from_actual_relabel(self):
        result = self.check("Z1")
        self.assertEqual(result["intended_text"], "你好你好")
        self.assertEqual(result["actual_text"], "你好")
        self.assertEqual(result["machine_normalized_texts"], ["你好", "你好"])
        self.assertEqual(result["planned_lexical_support"], "REJECTED")
        self.assertFalse(result["plan_matches_human_actual"])
        self.assertFalse(result["original_complete_label_eligible"])
        self.assertEqual(result["actual_expected_keywords"], [])

    def test_matching_W_text_cannot_certify_endpoint_completeness(self):
        result = self.check("W")
        self.assertEqual(result["planned_lexical_support"], "SUPPORTED")
        self.assertEqual(result["actual_text"], "小窝小窝")
        self.assertEqual(result["acoustic_completeness"], "UNKNOWN")
        self.assertFalse(result["original_complete_label_eligible"])
        self.assertIsNone(result["actual_expected_keywords"])

    def test_inaudible_Q_T_never_become_blank_or_machine_truth(self):
        for alias in ("Q", "T"):
            with self.subTest(alias=alias):
                result = self.check(alias)
                self.assertEqual(result["machine_state"], "unresolved")
                self.assertEqual(result["planned_lexical_support"], "UNKNOWN")
                self.assertIsNone(result["actual_text"])
                self.assertIsNone(result["actual_expected_keywords"])
                self.assertIsNone(result["ctc_target"])
                self.assertFalse(result["silence_or_blank_target_inferred"])
                self.assertFalse(result["original_complete_label_eligible"])

    def test_Z3_oov_actual_is_retained_by_existing_actual_label_policy(self):
        sys.path.insert(0, str(HERE / "tests/vendor"))
        import actual_label_policy
        result = self.check("Z3")
        policy = actual_label_policy.assess_actual_label(self.rows["Z3"])
        self.assertEqual(result["actual_text"], "小挖")
        self.assertEqual(result["intended_text"], "小窝")
        self.assertEqual(result["planned_lexical_support"], "REJECTED")
        self.assertEqual(result["actual_expected_keywords"], [])
        self.assertEqual(policy["oov_characters"], ["挖"])
        self.assertFalse(policy["ctc_label_eligible"])
        self.assertIsNone(policy["ctc_target"])
        self.assertEqual(result["machine_normalized_texts"], ["小ia啊", "小窝"])

    def test_genuine_lexical_support_survives_comparison_normalization(self):
        for alias, expected in (("A", ["K1"]), ("B", ["K2"]), ("Z2", []), ("Z4", ["K2"])):
            with self.subTest(alias=alias):
                result = self.check(alias)
                self.assertEqual(result["planned_lexical_support"], "SUPPORTED")
                self.assertTrue(result["original_complete_label_eligible"])
                self.assertEqual(result["actual_expected_keywords"], expected)
                self.assertTrue(result["human_actual_review_complete"])
                self.assertEqual(result["acoustic_completeness"], "UNKNOWN")
                self.assertIsNone(result["ctc_target"])
                self.assertFalse(result["training_admitted"])
                self.assertFalse(result["independent_accuracy_qualified"])

    def test_observed_disagreement_preserves_correct_human_actual_labels(self):
        for alias, expected in (("Z5", ["K1"]), ("Z6", [])):
            with self.subTest(alias=alias):
                result = self.check(alias)
                self.assertEqual(result["machine_state"], "disagree")
                self.assertEqual(result["planned_lexical_support"], "REJECTED")
                self.assertEqual(result["actual_expected_keywords"], expected)
                self.assertTrue(result["plan_matches_human_actual"])

    def test_hypothetical_plan_matching_consensus_cannot_override_human(self):
        source = self.rows["D"]
        # Counterfactual: unlike the observed 小五 output, both complete the plan.
        source["asr_results"] = [dict(status="complete", raw_text="小窝小窝")] * 2
        result = gates.label_preparation(source)
        self.assertEqual(result["planned_lexical_support"], "SUPPORTED")
        self.assertFalse(result["original_complete_label_eligible"])
        self.assertIn("HUMAN_ACTUAL_DIFFERS_FROM_PLAN", result["reasons"])
        self.assertEqual(result["actual_text"], "小窝")

    def test_machine_match_without_complete_human_review_is_not_admission(self):
        source = self.rows["A"]
        for review in ({}, dict(status="clean", independent_human=False, complete=True),
                       dict(status="clean", independent_human=True, complete=False)):
            with self.subTest(review=review):
                source["review"] = review
                result = gates.label_preparation(source)
                self.assertEqual(result["planned_lexical_support"], "SUPPORTED")
                self.assertFalse(result["original_complete_label_eligible"])

    def test_failed_empty_missing_or_one_sided_machine_evidence_stays_unresolved(self):
        source = self.rows["A"]
        for bad in (None, {}, dict(status="failed", raw_text="你好小窝"),
                    dict(status="complete", raw_text=""), dict(status="complete", raw_text=None)):
            with self.subTest(bad=bad):
                source["asr_results"][1] = bad
                result = gates.label_preparation(source)
                self.assertEqual(result["machine_state"], "unresolved")
                self.assertEqual(result["planned_lexical_support"], "UNKNOWN")
                self.assertFalse(result["original_complete_label_eligible"])

    def test_one_matching_machine_does_not_supply_dual_support(self):
        source = self.rows["A"]
        source["asr_results"][1]["raw_text"] = "你好小屋"
        result = gates.label_preparation(source)
        self.assertEqual(result["machine_state"], "disagree")
        self.assertEqual(result["planned_lexical_support"], "REJECTED")
        self.assertFalse(result["original_complete_label_eligible"])

    def test_quality_flags_block_complete_label_but_preserve_lexical_fact(self):
        source = self.rows["A"]
        source["asr_results"][0]["quality_flags"] = ["saved_incomplete_observation"]
        result = gates.label_preparation(source)
        self.assertEqual(result["planned_lexical_support"], "SUPPORTED")
        self.assertFalse(result["original_complete_label_eligible"])
        self.assertIn("ASR_QUALITY_FLAGS_RETAINED", result["reasons"])

    def test_fixture_only_contains_sanitized_scientific_regression_fields(self):
        keys = {"id", "actual_text", "intended_text", "review", "asr_results", "exposure",
                "use", "restriction", "allowed_purposes"}
        for source in self.rows.values():
            self.assertEqual(set(source), keys)
            self.assertEqual(source["exposure"], "EXPOSED")
            self.assertEqual(source["use"], "regression")
            self.assertEqual(source["restriction"], "regression_only")
            self.assertEqual(source["allowed_purposes"], ["asr_review_tool_validation"])
            for observation in source["asr_results"]:
                self.assertEqual(set(observation), {"raw_text", "status", "quality_flags"})


if __name__ == "__main__":
    unittest.main()
