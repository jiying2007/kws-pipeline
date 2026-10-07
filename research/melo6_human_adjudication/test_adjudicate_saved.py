"""Regress the frozen public supplement with saved JSON only."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import adjudicate_saved as review


class SavedHumanReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = [ARGS.comparison.read_bytes()] + [
            (ARGS.review_root / name).read_bytes() for name in
            ("human-labels.json", "audio-bindings.json", "vocabulary-binding.json")]
        cls.output = review.adjudicate_saved(*cls.raw)
        cls.labels = json.loads(cls.raw[1])

    def test_reproduces_exact_saved_result(self):
        expected = (ARGS.review_root / "adjudication-result.json").read_bytes()
        self.assertEqual(review.serialized(self.output), expected)

    def test_machine_disagreement_and_human_prompt_match_can_coexist(self):
        for index in (0, 1, 3):
            row = self.output["clips"][index]
            self.assertFalse(row["original_machine_text_agreement_only"])
            self.assertEqual(row["original_machine_assessment"], "quarantine")
            self.assertEqual(row["human_prompt_match"], "MATCH")
        self.assertEqual(self.output["original_both_asr_exact_intent_matches"], 2)
        gate = self.output["original_machine_gate"]
        self.assertFalse(gate["all_six_weak_lexical_gate_passed"])
        self.assertEqual(gate["overall_result"], "failed_or_inconclusive_quarantine")
        self.assertFalse(gate["automatic_source_advancement"])

    def test_m6_partial_cannot_be_completed_by_old_label_or_two_asrs(self):
        row = self.output["clips"][5]
        self.assertEqual(set(row["saved_asr_texts"].values()), {"明天天气很好"})
        self.assertEqual(row["superseded_human_revisions"][0]["full_text"], "明天天气很好")
        self.assertTrue(row["superseded_human_revisions"][0]["superseded"])
        self.assertEqual(row["current_human_revision"], 2)
        self.assertEqual(row["human_partial_text"], "[首字听不清]天天气很好")
        self.assertIsNone(row["human_full_text"])
        self.assertEqual(row["first_character"], "UNKNOWN")
        self.assertEqual(row["human_prompt_match"], "UNKNOWN")
        self.assertIsNone(row["full_transcript_representable"])

    def test_revision_priority_is_numeric_not_order_or_completeness(self):
        revisions = copy.deepcopy(self.labels["clips"][5]["revisions"])
        self.assertEqual(review.latest_revision(list(reversed(revisions))), revisions[1])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            review.latest_revision(revisions + [revisions[1]])
        revisions[1]["full_text"] = "明天天气很好"
        with self.assertRaisesRegex(ValueError, "partial"):
            review.latest_revision(revisions)

    def test_wrong_hash_or_alias_binding_is_rejected(self):
        for index in range(4):
            with self.subTest(input=index):
                changed = list(self.raw)
                changed[index] += b" "
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    review.adjudicate_saved(*changed)
        for index, rows_key, hash_key in ((1, "clips", "heard_native_wav_sha256"),
                                          (2, "bindings", "native_wav_sha256")):
            for key, value in (("alias", "M2"), (hash_key, "0" * 64)):
                with self.subTest(input=index, field=key):
                    changed = list(self.raw)
                    data = json.loads(changed[index])
                    data[rows_key][0][key] = value
                    changed[index] = review.serialized(data)
                    with self.assertRaises(ValueError):
                        review.adjudicate_saved(*changed)

    def test_old_full_snapshot_cannot_roll_back_current_partial_revision(self):
        changed = list(self.raw)
        labels = copy.deepcopy(self.labels)
        labels["clips"][5]["revisions"].pop()
        changed[1] = review.serialized(labels)
        with self.assertRaisesRegex(ValueError, "Human labels hash mismatch"):
            review.adjudicate_saved(*changed)

    def test_oov_and_partial_speech_are_never_blank_ctc(self):
        alphabet = json.loads(self.raw[3])["symbols"][1:]
        for full, partial in (("明天天气很好", None), (None, "[首字听不清]天天气很好"),
                              (None, None), ("你好小窝", None)):
            coverage = review.transcript_coverage(full, partial, alphabet)
            self.assertIsNone(coverage["ctc_target"])
            self.assertFalse(coverage["training_admission"])
        self.assertFalse(review.transcript_coverage("明天天气很好", None, alphabet)
                         ["full_transcript_representable"])
        for row in self.output["clips"]:
            self.assertIsNone(row["ctc_target"])
            self.assertFalse(row["training_admission"])

    def test_clip_coverage_does_not_fill_panel_or_representation_gaps(self):
        for row in self.output["clips"][:5]:
            self.assertTrue(row["full_transcript_available"])
            self.assertTrue(row["full_transcript_representable"])
        scope = self.output["human_review_scope"]
        self.assertFalse(scope["fully_blinded"])
        self.assertFalse(scope["derived_audio_independently_heard"])
        self.assertEqual(scope["heard_representation"], "native_44100_float32")
        self.assertEqual(scope["asr_representation"], "derived_16000_pcm16")
        panel = self.output["panel_readiness"]
        self.assertEqual((panel["full_human_transcripts"], panel["partial_human_transcripts"]), (5, 1))
        self.assertEqual((panel["human_prompt_matches"], panel["human_prompt_mismatches"],
                          panel["human_prompt_unknown"]), (5, 0, 1))
        self.assertEqual(panel["word_level_positive_coverage"], {
            "K1": {"text": "你好小窝", "human_transcript_aliases": ["M1"]},
            "K2": {"text": "小窝小窝", "human_transcript_aliases": ["M2"]},
        })
        self.assertEqual(panel["split_allocation"], "UNASSIGNED")
        self.assertFalse(panel["split_integrity_qualified"])
        self.assertFalse(panel["qualified_positive_development_set"])
        self.assertFalse(panel["identity_held_out_evaluation_established"])
        self.assertEqual(panel["voice_rights_cleared"], "UNKNOWN")
        for row in self.output["clips"]:
            self.assertEqual(row["confidence"], "UNKNOWN")
            self.assertEqual(row["acoustic_tail_completeness"], "UNKNOWN")

    def test_input_bytes_unchanged_and_output_overwrite_rejected(self):
        before = list(self.raw)
        review.adjudicate_saved(*self.raw)
        self.assertEqual(self.raw, before)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            review.write_new(path, self.output)
            initial = path.read_bytes()
            with self.assertRaises(FileExistsError):
                review.write_new(path, {"changed": True})
            self.assertEqual(path.read_bytes(), initial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--review-root", required=True, type=Path)
    ARGS, unittest_args = parser.parse_known_args()
    unittest.main(argv=[__file__, *unittest_args])
