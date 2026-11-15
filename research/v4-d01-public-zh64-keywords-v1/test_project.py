#!/usr/bin/env python3
"""Label-boundary fixtures and optional saved-byte integration; no model/audio IO."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import project as p
from speech_asr_consensus import require_independent_reference_labels, require_training_label_authority


def keyword(texts, *, keyword_id=1, quarantine=False, split="train", complete=None):
    return p.project_keyword(dict(zip(p.PROVIDERS, texts)),
                             complete or dict.fromkeys(p.PROVIDERS, True),
                             keyword_id=keyword_id, quarantine=quarantine, split=split)


def row(number, texts, *, preset="Dylan", quarantine=False):
    split = p.SPLITS[preset]
    return {"audio_id": f"fixture-{number}", "source_id": f"fixture-source-{number}",
            "global_source_key": ["fixture", str(number)], "audio": {"fixture": True},
            "research_split": split,
            "source_identity": {"preset": preset, "nearphone_quarantined": quarantine},
            "provider_evidence": {family: {"actual_text": text, "status": "complete_native_observation"}
                                  for family, text in zip(p.PROVIDERS, texts)},
            "keywords": [keyword(texts, keyword_id=k, quarantine=quarantine, split=split) for k in (1, 2)],
            "historical_authority": dict.fromkeys(p.LEGACY_FALSE, False),
            "training_admitted": False, "formal_admission": False, "human_gold": False,
            "independent_test": False, "ctc_mask": False, "target_ids": None, "event_times": None,
            "label_provenance": {"status": "PENDING_EXTERNAL_VERIFICATION", "scope": "fixture-only"}}


def snapshot(*rows):
    return {"schema": p.SCHEMA, "protocol": copy.deepcopy(p.PROTOCOL), "rows": list(rows)}


class KeywordEvidenceTests(unittest.TestCase):
    def test_extra_suffix_retained_without_word_end_or_ctc(self):
        k = keyword(("你好，小窝。", "你好小窝", "你好小窝儿"))
        self.assertEqual(k["state"], "present")
        self.assertEqual(k["weak_target"], 1)
        self.assertEqual(k["text_matches"]["whisper"][0]["retained_normalized_suffix"], "儿")
        self.assertEqual(k["text_matches"]["qwen06"][0]["raw_character_indices"], [0, 1, 3, 4])
        self.assertIsNone(k["event_times"])
        self.assertFalse(k["ctc_allowed"])

    def test_no_detection_is_unknown_not_absent(self):
        k = keyword(("今天天气很好",) * 3)
        self.assertEqual(k["state"], "unknown")
        self.assertIsNone(k["weak_target"])
        self.assertEqual(k["weak_weight"], 0.0)

    def test_homophone_agreement_is_not_keyword(self):
        for text in ("笑我笑我", "小我小我", "小屋小屋"):
            self.assertEqual(keyword((text,) * 3, keyword_id=2)["state"], "unknown")

    def test_two_votes_and_partial_words_stay_unknown(self):
        for texts in (("你好小窝", "你好小窝", "你好小屋"),
                      ("你好小", "你好小窝", "小窝")):
            self.assertEqual(keyword(texts)["state"], "unknown")

    def test_complete_nonempty_native_outputs_required(self):
        self.assertEqual(keyword(("你好小窝", "你好小窝", ""))["state"], "unknown")
        k = keyword(("你好小窝",) * 3, complete={"qwen06": True, "sensevoice": True, "whisper": False})
        self.assertEqual(k["state"], "unknown")

    def test_quarantine_overrides_even_three_positive_votes(self):
        k = keyword(("你好小窝",) * 3, quarantine=True)
        self.assertEqual(k["state"], "unknown")
        self.assertEqual(k["reason"], "inherited_nearphone_quarantine")
        self.assertFalse(k["research_optimizer_eligible"])

    def test_native_normalization_is_not_nfkc_or_general_whitespace(self):
        # NBSP would fold under NFKC, but it is outside the original fixed list.
        k = keyword(("你\u00a0好小窝", "你好小窝", "你好小窝"))
        self.assertEqual(k["state"], "unknown")
        self.assertFalse(k["literal_votes"]["qwen06"])

    def test_keyword_must_not_cross_sentence_or_line_boundaries(self):
        for separator in p.HARD_BOUNDARIES:
            for text, ident in (("你好" + separator + "小窝", 1),
                                ("小窝" + separator + "小窝", 2)):
                with self.subTest(separator=repr(separator), keyword_id=ident):
                    k = keyword((text,) * 3, keyword_id=ident)
                    self.assertEqual(k["state"], "unknown")
                    self.assertFalse(any(k["literal_votes"].values()))

    def test_sentence_rejection_retains_raw_span_and_allows_internal_comma(self):
        k = keyword(("你好。小窝", "你好，小窝", "你好小窝！"))
        self.assertEqual(k["state"], "unknown")
        blocked = k["text_matches"]["qwen06"][0]
        self.assertFalse(blocked["eligible_literal_span"])
        self.assertEqual(blocked["raw_hard_boundary_character_indices"], [2])
        self.assertTrue(k["literal_votes"]["sensevoice"])
        self.assertTrue(k["literal_votes"]["whisper"])
        self.assertEqual(keyword(("你好，小窝。", "你好小窝!", "你好小窝？"))["state"], "present")

    def test_each_keyword_has_its_own_unknown_mask(self):
        source = row(1, ("你好小窝",) * 3)
        consumed = p.research_inputs(snapshot(source), "train")["rows"][0]
        self.assertEqual(consumed["keyword_targets"], [1, None])
        self.assertEqual(consumed["keyword_mask"], [True, False])
        self.assertEqual(consumed["keyword_weight"], [0.25, 0.0])
        self.assertTrue(consumed["research_optimizer_allowed"])
        self.assertIsNone(consumed["full_text"])

    def test_development_does_not_enter_optimizer(self):
        sources = snapshot(row(1, ("小窝小窝",) * 3),
                           row(2, ("小窝小窝",) * 3, preset="Serena"),
                           row(3, ("小窝小窝",) * 3, quarantine=True))
        train = p.research_inputs(sources, "train")
        dev = p.research_inputs(sources, "development")
        self.assertEqual([r["audio_id"] for r in train["rows"]], ["fixture-1"])
        self.assertEqual([r["audio_id"] for r in dev["rows"]], ["fixture-2"])
        self.assertFalse(dev["rows"][0]["research_optimizer_allowed"])

    def test_changed_masks_weights_types_roles_and_times_rejected(self):
        mutations = (
            lambda r: r["keywords"][0].update(weak_target=True),
            lambda r: r["keywords"][0].update(weak_weight=1.0),
            lambda r: r["keywords"][1].update(state="absent", weak_target=0, weak_mask=True),
            lambda r: r.update(event_times={"end_frame": 16000}),
            lambda r: r.update(ctc_mask=True, target_ids=[1, 2, 3, 4]),
            lambda r: r.update(training_admitted=True),
            lambda r: r.update(research_split="dev"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                source = row(1, ("你好小窝",) * 3)
                mutate(source)
                with self.assertRaises(ValueError):
                    p.research_inputs(snapshot(source), "train")

    def test_old_formal_training_and_independent_reference_gates_still_reject(self):
        consumed = p.research_inputs(snapshot(row(1, ("你好小窝",) * 3)), "train")["rows"][0]
        for allow_machine in (False, True):
            with self.assertRaises(ValueError):
                require_training_label_authority(consumed, allow_machine=allow_machine, split="train")
        with self.assertRaises(ValueError):
            require_independent_reference_labels([consumed])

    def test_duplicate_sources_and_other_uses_rejected(self):
        source = row(1, ("你好小窝",) * 3)
        with self.assertRaises(ValueError):
            p.research_inputs(snapshot(source, copy.deepcopy(source)), "train")
        for phase in ("evaluation", "locked-test", "ctc", "production"):
            with self.assertRaises(ValueError):
                p.research_inputs(snapshot(source), phase)


@unittest.skipUnless(os.environ.get("D01_SAVED_DELIVERY"), "optional saved-output integration")
class SavedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.delivery = Path(os.environ["D01_SAVED_DELIVERY"])
        cls.temp = tempfile.TemporaryDirectory(prefix="d01-saved-projection-")
        cls.output = Path(cls.temp.name) / "package"
        cls.created = p.create(cls.delivery, cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_actual_snapshot_and_consumer_counts(self):
        manifest = self.created["manifest_sha256"]
        saved = p.verify(self.delivery, self.output, manifest)
        self.assertEqual(saved["summary"]["states"], {"present": 6, "absent": 0, "unknown": 122})
        self.assertEqual(saved["summary"]["new_present_from_fulltext_disagreement"], ["clip-000029"])
        self.assertEqual(saved["summary"]["nearphone_quarantined_keyword_slots"], 96)
        train = p.consume(self.delivery, self.output, manifest, "train")
        dev = p.consume(self.delivery, self.output, manifest, "development")
        self.assertEqual([r["audio_id"] for r in train["rows"]],
                         ["clip-000002", "clip-000018", "clip-000029", "clip-000030"])
        self.assertEqual([r["audio_id"] for r in dev["rows"]], ["clip-000006", "clip-000026"])

    def test_self_rehashed_label_and_manifest_cannot_promote_snapshot(self):
        path = self.output / "snapshot.json"
        before = path.read_bytes()
        manifest_path = self.output / "MANIFEST.json"
        manifest_before = manifest_path.read_bytes()
        try:
            snapshot = p.loads(before)
            snapshot["rows"][0]["keywords"][0].update(state="present", weak_target=1, weak_mask=True)
            changed = p.canonical(snapshot) + b"\n"
            path.write_bytes(changed)
            manifest = p.loads(manifest_before)
            manifest["files"]["snapshot.json"] = p.record(changed)
            updated_manifest = p.canonical(manifest) + b"\n"
            manifest_path.write_bytes(updated_manifest)
            with self.assertRaisesRegex(ValueError, "PACKAGE_NOT_SAVED_PROJECTION"):
                p.verify(self.delivery, self.output, p.sha(updated_manifest))
        finally:
            path.write_bytes(before)
            manifest_path.write_bytes(manifest_before)


if __name__ == "__main__":
    unittest.main()
