"""Invented text/metadata mocks only. No ASR, audio, private labels or downloads."""
import sys, pathlib, unittest, copy
ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'research/experiment_quality_guards')]
import speech_asr_consensus as consensus
import asr_native_contracts as native
import quality_gates as research

class ExistingBoundaries(unittest.TestCase):

    def test_punctuation_compare_does_not_rewrite_raw(self):
        raw = {'text': '你好，小窝。'}
        before = copy.deepcopy(raw)
        self.assertEqual(consensus.normalized_transcript(raw['text']), '你好小窝')
        self.assertEqual(raw, before)

    def test_confusables_remain_distinct(self):
        values = ['小窝', '小屋', '小哦', '小欧']
        self.assertEqual(len({consensus.normalized_transcript(x) for x in values}), 4)

    def test_two_engines_cannot_be_three_family_policy(self):
        with self.assertRaisesRegex(ValueError, 'exactly three'):
            consensus.validate_recognizer_policy({'schema_version': 1, 'evidence_class': consensus.POLICY_CLASS, 'normalization': consensus.NORMALIZATION, 'frozen_before_recognition': True, 'target_conditioning_allowed': False, 'recognizers': [{}, {}]})

    def row(self, a, b):
        return {'intended_text': '你好小窝', 'actual_text': None, 'review': {}, 'asr_results': [{'status': 'complete', 'raw_text': a}, {'status': 'complete', 'raw_text': b}]}

    def test_agreement_remains_weak_no_admission(self):
        r = research.label_preparation(self.row('你好，小窝。', '你好小窝'))
        self.assertEqual(r['machine_state'], 'machinesAgreeWeak')
        self.assertFalse(r['training_admitted'])
        self.assertFalse(r['original_complete_label_eligible'])
        self.assertEqual(r['acoustic_completeness'], 'UNKNOWN')

    def test_disagreement_is_preserved(self):
        r = research.label_preparation(self.row('你好小窝', '你好小屋'))
        self.assertEqual(r['machine_state'], 'disagree')
        self.assertFalse(r['automatic_relabel'])
        self.assertIsNone(r['ctc_target'])

    def test_agree_other_string_does_not_support_plan(self):
        r = research.label_preparation(self.row('你好小屋', '你好，小屋。'))
        self.assertEqual(r['planned_lexical_support'], 'REJECTED')

    def test_missing_result_unknown_not_blank(self):
        r = research.label_preparation(self.row('', None))
        self.assertEqual(r['machine_state'], 'unresolved')
        self.assertFalse(r['silence_or_blank_target_inferred'])

    def test_contract_versions_cannot_be_relabelled(self):
        self.assertEqual(native.QWEN_REQUEST['max_new_tokens'], 256)
        self.assertNotEqual(native.QWEN_REQUEST['max_new_tokens'], 512)
        self.assertEqual(native.EXECUTION['dtype'], 'float32')

    def test_product_keywords_remain_wo_not_wu(self):
        self.assertEqual(research.KEYWORDS, {'K1': '你好小窝', 'K2': '小窝小窝'})


import hashlib
import json
import subprocess
import tempfile
from unittest import mock
sys.path.insert(0, str(ROOT))
from eval import score_keyword_presence as keyword_presence


class KeywordPresenceScoringTests(unittest.TestCase):
    def row(self, name="a", k1="present", k2="unknown"):
        return dict(recording=name, duration_s=2.0,
            pcm_sha256=hashlib.sha256((name + "-pcm").encode()).hexdigest(),
            label_evidence_sha256=hashlib.sha256((name + "-labels").encode()).hexdigest(),
            split="development", keyword_states={"1": k1, "2": k2})

    def event(self, name="a", kid=1, time_s=1.0):
        return dict(recording=name, keyword_id=kid, time_s=time_s, confidence=0.8)

    def score(self, rows=None, events=None):
        return keyword_presence.compare_keyword_presence(
            [self.row()] if rows is None else rows,
            {"c1": [] if events is None else events})

    def test_known_positive_remains_scorable_with_other_unknown(self):
        r = self.score(events=[self.event()])
        self.assertEqual(r["arms"]["c1"]["1"]["hit_positive_clips"], 1)
        self.assertEqual(r["arms"]["c1"]["2"]["status"], "UNSCORED")
        self.assertEqual((r["known_keyword_labels"], r["unknown_keyword_labels"]), (1, 1))

    def test_unknown_event_is_not_a_negative(self):
        r = self.score(events=[self.event(kid=2)])["arms"]["c1"]
        self.assertEqual(r["2"]["unknown_events"], 1)
        self.assertEqual(r["2"]["negative_events"], 0)
        self.assertEqual(r["2"]["known_negative_clips"], 0)
        self.assertEqual(r["1"]["missed_positive_clips"], 1)

    def test_all_73_unknown_is_unscored_not_zero_error_pass(self):
        rows = [self.row("n" + str(n), "unknown", "unknown") for n in range(73)]
        r = self.score(rows, [self.event("n0", 1)])
        self.assertEqual(r["status"], "UNSCORED")
        self.assertEqual(r["unknown_keyword_labels"], 146)
        self.assertEqual(r["accepted_negative_seconds"], 0)
        self.assertIsNone(r["winner"])
        self.assertFalse(r["natural_far_available"])

    def test_known_absence_does_not_create_natural_negative_hours(self):
        r = self.score([self.row(k1="absent", k2="absent")], [self.event()])
        self.assertEqual(r["arms"]["c1"]["1"]["negative_events"], 1)
        self.assertEqual(r["accepted_negative_seconds"], 0)
        self.assertFalse(r["natural_far_available"])

    def test_duplicate_events_count_but_not_independent_hits(self):
        r = self.score(events=[self.event(), self.event()])["arms"]["c1"]["1"]
        self.assertEqual((r["hit_positive_clips"], r["total_events"], r["repeat_events"]), (1, 2, 1))

    def test_wrong_keyword_cannot_rescue_known_positive(self):
        r = self.score([self.row(k2="absent")], [self.event(kid=2)])["arms"]["c1"]
        self.assertEqual(r["1"]["missed_positive_clips"], 1)
        self.assertEqual(r["2"]["negative_clips_with_events"], 1)

    def test_two_known_positive_words_are_not_forced_into_one_class(self):
        r = self.score([self.row(k2="present")], [self.event(), self.event(kid=2)])["arms"]["c1"]
        self.assertEqual([r[str(k)]["hit_positive_clips"] for k in (1, 2)], [1, 1])

    def test_multiple_models_share_one_reference_without_ranking(self):
        rows = [self.row()]
        r = keyword_presence.compare_keyword_presence(rows,
            {"fsmn": [], "direct": [self.event()], "c1": [self.event()]})
        self.assertEqual(list(r["arms"]), ["c1", "direct", "fsmn"])
        self.assertIsNone(r["winner"])
        self.assertFalse(r["context_comparability_verified"])

    def test_counts_cover_every_recording_and_keyword(self):
        rows = [self.row("p", "present", "absent"), self.row("n", "absent", "present"),
                self.row("u", "unknown", "unknown")]
        r = self.score(rows, [self.event("p"), self.event("n", 1), self.event("u", 2)])
        for value in r["arms"]["c1"].values():
            self.assertEqual(value["known_positive_clips"] + value["known_negative_clips"] + value["unknown_clips"], 3)
            self.assertEqual(value["hit_positive_clips"] + value["missed_positive_clips"], value["known_positive_clips"])
        self.assertEqual(r["arms"]["c1"]["1"]["known_label_coverage"], 2 / 3)

    def test_no_event_metrics_or_admission_are_emitted(self):
        r = self.score(events=[self.event()])
        for key in ("far_per_hour", "frr", "p95_post_end_latency_ms", "event_times"):
            self.assertNotIn(key, r)
        for key in ("training_admitted", "independent_evaluation", "qualification_allowed",
                    "upstream_evidence_authenticated", "acoustic_labels_verified", "event_annotations_available"):
            self.assertIs(r[key], False)

    def test_input_objects_are_not_relabelled_or_modified(self):
        rows, events = [self.row()], {"c1": [self.event()]}
        before = copy.deepcopy((rows, events))
        keyword_presence.compare_keyword_presence(rows, events)
        self.assertEqual((rows, events), before)

    def test_missing_extra_or_event_reference_fields_rejected(self):
        for edit in (lambda r: r.pop("keyword_states"), lambda r: r.update(expected=[]),
                     lambda r: r.update(annotation_status="complete"), lambda r: r.update(ctc_target=[])):
            row = self.row(); edit(row)
            with self.assertRaises(ValueError): self.score([row])

    def test_each_keyword_state_must_be_explicit(self):
        for states in ({"1": "present"}, {"1": "present", "2": None},
                       {"1": True, "2": "absent"}, {"1": "present", "2": "not_detected"},
                       {"1": "present", "2": "absent", "3": "absent"}):
            row = self.row(); row["keyword_states"] = states
            with self.assertRaises(ValueError): self.score([row])

    def test_unexposed_roles_are_not_relabelled_development(self):
        for role in ("train", "test", "calibration", None):
            row = self.row(); row["split"] = role
            with self.assertRaises(ValueError): self.score([row])

    def test_duplicate_name_or_pcm_denominator_rejected(self):
        for rows in ([self.row(), self.row()], [self.row(), dict(self.row(), recording="renamed")]):
            with self.assertRaises(ValueError): self.score(rows)

    def test_invalid_hashes_are_rejected_not_authenticated(self):
        for key in ("pcm_sha256", "label_evidence_sha256"):
            for value in (None, "f" * 63, "A" * 64, "0" * 64):
                row = self.row(); row[key] = value
                with self.assertRaises(ValueError): self.score([row])

    def test_invalid_duration_or_numeric_types_rejected(self):
        for value in (True, "2", None, -1, 0, float("nan"), float("inf"), 10**1000):
            row = self.row(); row["duration_s"] = value
            with self.assertRaises(ValueError): self.score([row])

    def test_event_identity_and_types_rejected(self):
        for key, value in (("recording", "missing"), ("recording", []), ("keyword_id", True),
                           ("keyword_id", 1.0), ("keyword_id", 3), ("time_s", -0.1),
                           ("time_s", 2.001), ("time_s", True), ("time_s", float("nan")),
                           ("confidence", 1.1), ("confidence", "0.9"), ("confidence", float("inf"))):
            event = self.event(); event[key] = value
            with self.assertRaises(ValueError): self.score(events=[event])

    def test_event_at_exact_declared_endpoint_allowed(self):
        r = self.score(events=[self.event(time_s=2.0)])
        self.assertEqual(r["arms"]["c1"]["1"]["hit_positive_clips"], 1)

    def test_missing_or_extra_event_fields_rejected(self):
        for edit in (lambda e: e.pop("confidence"), lambda e: e.update(expected=True)):
            event = self.event(); edit(event)
            with self.assertRaises(ValueError): self.score(events=[event])

    def test_empty_bad_name_and_excessive_arms_rejected(self):
        for arms in ({}, {"": []}, {"a b": []}, {"a" + str(i): [] for i in range(9)}):
            with self.assertRaises(ValueError): keyword_presence.compare_keyword_presence([self.row()], arms)
        with self.assertRaises(ValueError): self.score([])

    def test_jsonl_duplicate_keys_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "bad.jsonl"
            for raw in ('{"a":1,"a":2}\n', '{"a": NaN}\n', '{"a": Infinity}\n'):
                path.write_text(raw)
                with self.assertRaises(ValueError): keyword_presence.load_jsonl(path, 10)
            path.write_text('{}\n{}\n')
            with self.assertRaises(ValueError): keyword_presence.load_jsonl(path, 1)
            with mock.patch.object(keyword_presence, "MAX_BYTES", 3):
                with self.assertRaises(ValueError): keyword_presence.load_jsonl(path, 10)

    def test_cli_writes_input_bindings_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            refs, events, output = [root / n for n in ("refs.jsonl", "events.jsonl", "out.json")]
            refs.write_text(json.dumps(self.row()) + "\n")
            events.write_text(json.dumps(self.event()) + "\n")
            argv = [sys.executable, "-I", "-S", "-B", keyword_presence.__file__,
                    "--references", str(refs), "--events", "c1=" + str(events), "--output", str(output)]
            p = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            original = output.read_bytes()
            r = json.loads(original)
            self.assertEqual(r["input_file_sha256"]["references"], hashlib.sha256(refs.read_bytes()).hexdigest())
            p = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertEqual(output.read_bytes(), original)

    def test_cli_rejects_invalid_before_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            refs, events, output = [root / n for n in ("refs.jsonl", "events.jsonl", "out.json")]
            refs.write_text(json.dumps(self.row()) + "\n")
            events.write_text(json.dumps(self.event(kid=3)) + "\n")
            argv = [sys.executable, "-I", "-S", "-B", keyword_presence.__file__,
                    "--references", str(refs), "--events", "c1=" + str(events), "--output", str(output)]
            p = subprocess.run(argv, capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(output.exists())
            events.write_text("")
            p = subprocess.run(argv + ["--events", "c1=" + str(events)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertFalse(output.exists())


# Run complete-keyword regressions through this existing normal/-O/-OO CI entry.
from research.cnative_landing_v1.test_metrics import MetricsTests


if __name__ == '__main__':
    unittest.main()
