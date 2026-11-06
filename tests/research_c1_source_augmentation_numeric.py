#!/usr/bin/env python3
"""Tiny PCM/frontend and invented-logit checks; no model/forward/optimizer."""
import copy
from contextlib import ExitStack
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tests")]
import research_c1_source_augmentation as stage
from test_research_c1_source_augmentation import fixture


class TinyNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

    def audio_fixture(self, root, *, uncertain=False, text="你好小窝"):
        vocabulary, raw = fixture(text=text)
        if uncertain:
            raw.update(label_status="uncertain", full_text=None, ctc_mask=False, ctc_weight=0.0,
                keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0.0, 0.0])
        pcm = struct.pack("<%dh" % raw["samples"], *[(i % 97 - 48) * 127 for i in range(raw["samples"])])
        path = root / raw["audio"]
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(pcm)
        raw.update(pcm_sha256=hashlib.sha256(pcm).hexdigest(), file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return vocabulary, stage.project_row(raw, vocabulary), pcm

    def no_model_calls(self):
        guards = ExitStack()
        for obj, name in ((stage.training, "build_fresh"), (self.torch.nn.Module, "__init__"),
                (self.torch.nn.Module, "__call__"), (self.torch.optim.Optimizer, "__init__")):
            guards.enter_context(patch.object(obj, name, side_effect=AssertionError("NO_MODEL_OR_OPTIMIZER")))
        return guards

    def test_known_pcm_features_and_original_evidence_are_byte_identical(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory, self.no_model_calls():
            root = Path(directory)
            _, row, pcm = self.audio_fixture(root)
            old_bank, old_evidence, old_pcm = stage.prior.feature_bank(root, [row], torch)
            bank, evidence, pcms = stage.feature_bank(root, [row], torch)
            self.assertEqual(old_evidence, evidence)
            self.assertEqual(old_pcm, pcms)
            self.assertEqual(pcms, [pcm])
            for ms in stage.OFFSETS:
                self.assertTrue(bank[0, ms].equal(old_bank[0, ms]))
                self.assertEqual(stage.training.tensor_sha(bank[0, ms]), stage.training.tensor_sha(old_bank[0, ms]))

    def test_target_free_pcm_retains_all_offsets_and_unmodified_label_identity(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory, self.no_model_calls():
            root = Path(directory)
            vocabulary, row, pcm = self.audio_fixture(root, uncertain=True)
            original = copy.deepcopy(row)
            # Exercise the real former failure before testing the local adapter.
            with self.assertRaises(TypeError):
                stage.prior.feature_bank(root, [row], torch)
            bank, evidence, pcms = stage.feature_bank(root, [row], torch)
            self.assertEqual(row, original)
            self.assertEqual(pcms, [pcm])
            self.assertEqual(len(evidence), 3)
            self.assertEqual({e["label_binding_sha256"] for e in evidence}, {stage.fit.row_identity(original)})
            self.assertNotEqual(stage.fit.row_identity(stage.feature_geometry_row(row)), stage.fit.row_identity(original))
            model_pin = {"model_id": "invented-no-model", "checkpoint": {"sha256": "0" * 64}}
            baseline_logits = None
            for ms in stage.OFFSETS:
                feature = bank[0, ms]
                self.assertEqual(feature.dtype, torch.float32)
                self.assertEqual(tuple(feature.shape), (stage.onset.geometry(row["samples"], ms)["frames"], 32))
                logits = torch.zeros((len(feature), 1, 2601), dtype=torch.float32)
                if ms == 0:
                    baseline_logits = logits
                target = root / ("numeric/ms-%03d.pt" % ms)
                artifact = {"path": target.relative_to(root).as_posix(),
                    **stage.evaluation.save_numeric(target, row["source_id"], feature, logits)}
                observed = stage.observed_score(row, model_pin, pcm, ms, feature, logits,
                    baseline_logits, bank[0, 0], 14, artifact, vocabulary)
                self.assertEqual(observed["evaluation_status"], "not-evaluable")
                self.assertIsNone(observed["full_text"])
                self.assertEqual(observed["keyword_targets"], [None, None])
                self.assertEqual(observed["keyword_mask"], [False, False])
                self.assertEqual(observed["ctc_weight"], 0)
                self.assertNotIn("ctc_loss_target_length_normalized", observed)
                self.assertTrue(all(not k["known"] for k in observed["keywords"]))
                self.assertEqual(observed["label_binding_sha256"], stage.fit.row_identity(original))
                self.assertEqual(observed["numeric_artifact"]["sha256"], stage.training.sha_file(target))

    def test_oov_pcm_features_keep_original_fulltext_and_known_keyword_mask(self):
        torch = self.torch
        with tempfile.TemporaryDirectory() as directory, self.no_model_calls():
            root = Path(directory)
            _, row, _ = self.audio_fixture(root, text="你好小窝龘")
            original = copy.deepcopy(row)
            self.assertIsNone(row["ctc_minimum_frames"])
            bank, evidence, _ = stage.feature_bank(root, [row], torch)
            self.assertEqual(len(bank), 3)
            self.assertEqual(row, original)
            self.assertEqual(row["full_text"], "你好小窝龘")
            self.assertEqual(row["keyword_mask"], [True, True])
            self.assertEqual(row["observed_label"]["ctc_weight"], .5)
            self.assertEqual({e["label_binding_sha256"] for e in evidence}, {stage.fit.row_identity(original)})

    def test_complete_observed_score_keeps_weight_and_full_target(self):
        torch = self.torch
        vocabulary, original = fixture()
        row = stage.project_row(original, vocabulary)
        feature = torch.zeros((row["expected_c1_frames"], 32), dtype=torch.float32)
        logits = torch.zeros((len(feature), 1, 2601), dtype=torch.float32)
        logits[:, 0, 0] = 4
        for i, token in enumerate(row["target_ids"]):
            logits[5 + 3 * i, 0, token] = 9
        model_pin = {"model_id": "invented", "checkpoint": {"sha256": "0" * 64}}
        with patch.object(stage.training, "build_fresh", side_effect=AssertionError("NO_MODEL")):
            item = stage.observed_score(row, model_pin, b"\0\0" * row["samples"], 0, feature, logits,
                logits, feature, 14, {}, vocabulary)
        self.assertEqual(item["evaluation_status"], "evaluated")
        self.assertEqual(item["target_text"], original["full_text"])
        self.assertEqual(item["target_ids"], row["target_ids"])
        self.assertAlmostEqual(item["ctc_loss_confidence_weighted"], item["ctc_loss_target_length_normalized"] * .5)
        self.assertTrue(item["keywords"][0]["complete_sequence_present"])

    def test_oov_numeric_result_keeps_known_keyword_but_no_fulltext_loss(self):
        torch = self.torch
        vocabulary, original = fixture(text="你好小窝龘")
        row = stage.project_row(original, vocabulary)
        feature = torch.zeros((row["expected_c1_frames"], 32), dtype=torch.float32)
        logits = torch.zeros((len(feature), 1, 2601), dtype=torch.float32)
        logits[:, 0, 0] = 4
        model_pin = {"model_id": "invented", "checkpoint": {"sha256": "0" * 64}}
        item = stage.observed_score(row, model_pin, b"\0\0" * row["samples"], 0, feature, logits,
            logits, feature, 14, {}, vocabulary)
        self.assertEqual(item["evaluation_status"], "not-evaluable")
        self.assertEqual(item["full_text"], original["full_text"])
        self.assertEqual(item["observed_label"]["ctc_weight"], .5)
        self.assertNotIn("ctc_loss_target_length_normalized", item)
        self.assertTrue(item["keywords"][0]["known"])
        self.assertEqual(item["keywords"][0]["reference"], 1)


if __name__ == "__main__":
    unittest.main()
