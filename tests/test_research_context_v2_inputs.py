#!/usr/bin/env python3
"""Context full2601 projections; no models, fixture PCM only when data is supplied."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import research_context_v2_inputs as inputs
import research_d1_vocab as vocab

TOKENS = ROOT / "tests/fixtures/d1-fulltext/fsmn-official-tokens.txt"
DATA_ROOT = os.environ.get("KWS_CONTEXT_DATA_ROOT")
VOCABULARY_SHA = vocab.digest(vocab.make_vocabulary(TOKENS.read_bytes()))


class Reader:
    @staticmethod
    def read_bytes(path, *args):
        return Path(path).read_bytes()

    @staticmethod
    def loads(raw):
        return json.loads(raw)


def observed_row(text="你好小窝请把灯打开"):
    return {"consumer": "context-v2-verified-full-transcript-v1", "context_stage": "context-supplement-v2",
            "source_adapter_schema": "kws-active-context-v2-native-source-v1",
            "source_id": "bulk-" + "1" * 24, "source_plan_sha256": "a" * 64,
            "global_source_key": ["a" * 64, "bulk-" + "1" * 24],
            "generation_category": "full-positive", "intended_kind": "positive-intent", "intended_keyword_id": 1,
            "generation_outcome": "completed", "label_status": "machine_candidate",
            "purpose": "technical-research-only", "human_gold": False, "independent_test": False,
            "split": "train", "source_role": "train-candidate", "full_text": text, "selected_raw_text": text,
            "samples": 16000, "ctc_mask": True, "ctc_weight": .5, "target_ids": [1, 2, 3, 4],
            "keyword_targets": [int(word in text) for word in inputs.KEYWORDS], "keyword_mask": [True, True],
            "keyword_weight": [.5, .5], "asr_transcripts": [{"model": m, "text": text} for m in ("qwen06", "sensevoice")],
            "requested_vocabulary_sha256": VOCABULARY_SHA,
            "context_observation": {"fixture": True}, "retained_row_sha256": "c" * 64}


class ContextV2ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = vocab.make_vocabulary(TOKENS.read_bytes())

    def project(self, row):
        return inputs.project_row(row, self.vocabulary)

    def test_full_context_and_appended_character_are_projected_without_legacy_truncation(self):
        original = observed_row()
        before = copy.deepcopy(original)
        result = self.project(original)
        self.assertEqual(original, before)
        self.assertEqual(result["consumer"], "kws-context-v2-fulltext-projection-v1")
        self.assertEqual(result["full_text"], original["full_text"])
        self.assertEqual(result["selected_raw_text"], original["selected_raw_text"])
        self.assertEqual(result["target_ids"], [self.vocabulary["token_to_id"][c] for c in original["full_text"]])
        self.assertEqual(len(result["target_ids"]), len(original["full_text"]))
        self.assertIn(2600, result["target_ids"])
        self.assertTrue(result["ctc_mask"])
        self.assertEqual(result["observed_label"]["keyword_weight"], [.5, .5])
        self.assertEqual(result["asr_transcripts"], original["asr_transcripts"])

    def test_oov_keeps_complete_observation_but_has_no_ctc_or_keyword_training_weight(self):
        original = observed_row("你好小窝请把灯打开🧭")
        result = self.project(original)
        self.assertFalse(result["ctc_mask"])
        self.assertEqual(result["ctc_weight"], 0)
        self.assertIsNone(result["target_ids"])
        self.assertEqual(result["keyword_targets"], [None, None])
        self.assertEqual(result["keyword_mask"], [False, False])
        self.assertEqual(result["keyword_weight"], [0, 0])
        self.assertEqual(result["ctc_reason"], "full_transcript_out_of_vocabulary")
        self.assertEqual(result["oov_characters"], ["🧭"])
        self.assertEqual(result["full_text"], original["full_text"])
        self.assertTrue(result["observed_label"]["ctc_mask"])
        self.assertEqual(result["observed_label"]["keyword_targets"], [1, 0])

    def test_unknown_and_failed_native_never_turn_into_training_negatives(self):
        for outcome in ("completed", "failed-native"):
            row = observed_row()
            row.update(generation_outcome=outcome, label_status="uncertain", full_text=None,
                       ctc_mask=False, ctc_weight=0, keyword_targets=[None, None],
                       keyword_mask=[False, False], keyword_weight=[0, 0], target_ids=None)
            result = self.project(row)
            with self.subTest(outcome=outcome):
                self.assertIsNone(result["target_ids"])
                self.assertEqual(result["keyword_weight"], [0, 0])
                self.assertEqual(result["asr_transcripts"], row["asr_transcripts"])
        forged = observed_row(); forged["generation_outcome"] = "failed-native"
        with self.assertRaisesRegex(ValueError, "FAILED_NATIVE_ZERO_LOSS"):
            self.project(forged)

    def test_unconfirmed_complete_text_is_retained_without_objective_weight(self):
        row = observed_row("今天晚上吃什么")
        row.update(label_status="uncertain", ctc_mask=False, ctc_weight=0,
                   keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0])
        result = self.project(row)
        self.assertEqual(result["full_text"], "今天晚上吃什么")
        self.assertEqual(result["ctc_weight"], 0)
        self.assertIsNone(result["target_ids"])

    def test_geometry_uses_full_sequence_including_adjacent_repeat_cost(self):
        row = observed_row("你好小窝窝窝")
        row["samples"] = 400 + 320 * 6
        result = self.project(row)
        self.assertEqual(result["expected_c1_frames"], 7)
        self.assertEqual(result["ctc_minimum_frames"], 8)
        self.assertFalse(result["ctc_mask"])
        self.assertEqual(result["keyword_weight"], [0, 0])

    def test_d1_d2_or_wrong_split_and_owner_scope_cannot_masquerade_as_context(self):
        for key, value in (("consumer", "d1-verified-full-transcript-v2"),
                           ("consumer", "d2-verified-full-transcript-v1"),
                           ("consumer", "context-verified-full-transcript-v1"),
                           ("context_stage", "context-supplement"),
                           ("source_adapter_schema", "kws-active-context-native-source-v1"),
                           ("context_stage", "existing-eight"),
                           ("source_adapter_schema", "kws-active-d2-native-source-v1"),
                           ("source_role", "development-only"),
                           ("global_source_key", ["f" * 64, "different"])):
            row = observed_row(); row[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ROW_SCOPE"):
                self.project(row)

    def test_consumer_cannot_use_a_different_map_than_the_planned_fixed_vocabulary(self):
        changed = copy.deepcopy(self.vocabulary)
        changed["token_to_id"]["灯"] = 2599
        with self.assertRaisesRegex(ValueError, "REQUESTED_AND_CONSUMER_VOCABULARY_DIFFER"):
            inputs.project_row(observed_row(), changed)

    def test_keyword_label_must_match_the_entire_observed_transcript_and_intended_target(self):
        row = observed_row(); row["keyword_targets"] = [0, 0]
        with self.assertRaises(ValueError):
            self.project(row)
        row = observed_row("我听见小窝小窝请回应")
        with self.assertRaisesRegex(ValueError, "INTENDED_WAKE_NOT_OBSERVED"):
            self.project(row)

    def test_vocabulary_file_and_original_token_bytes_are_both_pinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, tokens = root / "vocabulary.json", root / "tokens.txt"
            path.write_bytes(vocab.canonical(self.vocabulary) + b"\n")
            tokens.write_bytes(TOKENS.read_bytes())
            expected = inputs.sha(path.read_bytes())
            value, identity = inputs.load_vocabulary(path, expected, tokens, Reader)
            self.assertEqual(value, self.vocabulary)
            self.assertEqual(identity["file_sha256"], expected)
            self.assertEqual(identity["output_count"], 2601)
            self.assertEqual(identity["blank_id"], 0)
            with self.assertRaisesRegex(ValueError, "FILE_PIN_CHANGED"):
                inputs.load_vocabulary(path, "0" * 64, tokens, Reader)
            tokens.write_bytes(tokens.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "OFFICIAL_TOKEN_BYTES_CHANGED"):
                inputs.load_vocabulary(path, expected, tokens, Reader)


@unittest.skipUnless(DATA_ROOT, "set KWS_CONTEXT_DATA_ROOT for cross-repository no-model evidence replay")
class ContextV2PackageIntegrationTests(unittest.TestCase):
    def test_real_context_fixture_package_replays_before_full2601_projection(self):
        root = Path(DATA_ROOT).absolute()
        loader, source = inputs.load_data_module(root / "automation/active_context_v2_weak_view.py")
        sys.path.insert(0, str(root / "automation"))
        fixtures = importlib.import_module("test_active_context_v2_weak_view")
        fixture = fixtures.ContextV2WeakViewTests("test_context_native_asr_package_replays_offline_without_changing_source_roles")
        fixture.setUp(); self.addCleanup(fixture.doCleanups)
        package, _ = fixture.create()
        declaration = {"path": str(package), "manifest_sha256": inputs.sha((package / "MANIFEST.json").read_bytes())}
        vocabulary = vocab.make_vocabulary(TOKENS.read_bytes())
        vocabulary_path = fixture.root / "full2601.json"
        vocabulary_path.write_bytes(vocab.canonical(vocabulary) + b"\n")
        options = dict(expected_plan_sha256=fixture.plan_sha, vocabulary_path=vocabulary_path,
                       vocabulary_sha256=inputs.sha(vocabulary_path.read_bytes()), original_tokens_path=TOKENS)
        rows, report = inputs.verified_projection([declaration], loader, source["files"], **options)
        self.assertEqual(report["observed_rows"], 8)
        self.assertEqual(report["planned_rows"], 16)
        self.assertFalse(report["complete_observed_coverage"])
        self.assertEqual(len(report["unobserved_cell_ids"]), 8)
        self.assertTrue(report["all_rows_replayed_before_projection"])
        self.assertTrue(all(row["consumer"] == inputs.ROW_SCHEMA for row in rows))
        failed = next(row for row in rows if row["generation_outcome"] == "failed-native")
        self.assertIsNone(failed["target_ids"])
        self.assertEqual(failed["keyword_weight"], [0, 0])
        eligible = [row for row in rows if row["ctc_mask"]]
        self.assertTrue(eligible)
        self.assertTrue(any(len(row["target_ids"]) > 8 for row in eligible))
        self.assertTrue(all(row["target_ids"] == [vocabulary["token_to_id"][c] for c in row["full_text"]]
                            for row in eligible))
        self.assertTrue(all(row["keyword_weight"] == [0, 0] for row in rows if not row["ctc_mask"]))
        with self.assertRaisesRegex(ValueError, "PACKAGE_IDENTITY_OR_DUPLICATE"):
            inputs.verified_projection([declaration, declaration], loader, source["files"], **options)
        with self.assertRaisesRegex(ValueError, "PLAN_AND_SOURCE_SCOPE"):
            inputs.verified_projection([declaration], loader, source["files"],
                                      **{**options, "expected_plan_sha256": "0" * 64})
        (package / "report.json").write_bytes(b"{}\n")
        with self.assertRaises(ValueError):
            inputs.verified_projection([declaration], loader, source["files"], **options)


if __name__ == "__main__":
    unittest.main()
