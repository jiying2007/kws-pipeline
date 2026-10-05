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


if __name__ == "__main__":
    unittest.main()
