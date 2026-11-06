#!/usr/bin/env python3
"""Adapter regression fixtures; numeric mode uses invented tensors only."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools")]
import research_c1_posterior_audit as stage


def unknown_fixture(text=None):
    row = {"source_id": "invented-unknown", "source_family_id": "invented-family", "split": "train",
        "ctc_mask": False, "ctc_weight": 0.0, "full_text": text, "target_ids": None,
        "keyword_targets": [None, None], "keyword_mask": [False, False], "keyword_weight": [0.0, 0.0],
        "human_gold": False, "independent_test": False, "intended_keyword_id": 1}
    observation = {k: row[k] for k in ("source_id", "source_family_id", "split", "ctc_weight", "full_text",
        "keyword_targets", "keyword_mask", "keyword_weight")}
    observation["label_binding_sha256"] = stage.fit.row_identity(row)
    return row, observation


def known_fixture():
    contract = stage.plan()
    target = [*contract["keywords"]["1"], *contract["keywords"]["2"], *contract["keywords"]["1"]]
    row = {"source_id": "invented-known", "source_family_id": "invented-family", "speaker_id": "invented-speaker",
        "source_role": "development-only", "split": "development", "ctc_mask": True, "ctc_weight": .5,
        "full_text": "你好小窝小窝小窝你好小窝", "target_ids": target, "keyword_targets": [1, 1],
        "keyword_mask": [True, True], "keyword_weight": [.5, .5], "samples": 32000, "pcm_sha256": "a" * 64,
        "human_gold": False, "independent_test": False}
    observation = {k: row[k] for k in ("source_id", "source_family_id", "speaker_id", "source_role", "split", "ctc_mask",
        "ctc_weight", "target_ids", "keyword_targets", "keyword_mask", "keyword_weight")}
    shape = stage.kernel.onset.geometry(row["samples"], 0)
    observation.update(target_text=row["full_text"], label_binding_sha256=stage.fit.row_identity(row),
        original_pcm_sha256=row["pcm_sha256"], checkpoint_sha256=contract["checkpoint_sha256"]["baseline-old"],
        offset_ms=0, frames=shape["frames"], known_zero_prefix_frames=0, greedy_token_ids=target,
        keywords=[{"complete_sequence_present": True}, {"complete_sequence_present": True}],
        numeric_artifact={"path": "numeric/invented.pt", "bytes": 123, "sha256": "b" * 64})
    previous = {"origin": "context52", "row": row, "offset_ms": 0, "saved_greedy": target,
        "saved_keyword_hits": [True, True], "numeric_artifact": {**observation["numeric_artifact"], "path": "/invented/source.pt"},
        "hypotheses": stage.kernel.variants(target, target, [1, 1], [True, True], contract["keywords"])}
    previous["condition_binding_sha256"] = stage.fit.digest(previous)
    result = {"origin": "context52", "source_id": row["source_id"], "split": row["split"], "offset_ms": 0,
        "label_binding_sha256": stage.fit.row_identity(row), "condition_binding_sha256": previous["condition_binding_sha256"],
        "checkpoint_sha256": observation["checkpoint_sha256"], "original_weak_weight": .5,
        "original_target_positions": [{"target_position": i, "token_id": token} for i, token in enumerate(target)]}
    return row, observation, previous, result


def aggregate_fixture(side, source="private-source", negative=False):
    contract = stage.plan()
    targets = [0, 0] if negative else [1, 1]
    tokens = [123] if negative else [*contract["keywords"]["1"], *contract["keywords"]["2"], *contract["keywords"]["1"]]
    hypotheses = stage.kernel.variants(tokens, [], targets, [True, True], contract["keywords"])
    amount = 2. if side == "augmentation" else 1.
    raw = {"origin": "fixture", "source_id": source, "split": "development", "offset_ms": 0,
        "reference_log_probability": -4. + amount,
        "logical_hypotheses": [{**h, "original_minus_candidate_log_probability": amount} for h in hypotheses["logical_hypotheses"]],
        "original_target_positions": [{"token_id": token, **{name: amount / 10 for name in stage.METRICS}} for token in tokens]}
    return {"comparison_side": side, "cohort": "new-development", "source_id": source,
        "split": "development", "offset_ms": 0, "label_binding_sha256": source,
        "keyword_mask": [True, True], "keyword_targets": targets,
        "saved_keyword_hits": [side == "augmentation", False if negative else side == "augmentation"], "result": raw}


class MetadataTests(unittest.TestCase):
    def test_metadata_import_has_no_torch_or_model(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_original_kernel_closure_is_frozen(self):
        contract = stage.plan()
        actual = stage.sources(contract)
        self.assertEqual({k: actual[k] for k in stage.kernel.CODE}, contract["kernel_source_hashes"])
        broken = copy.deepcopy(contract)
        broken["kernel_source_hashes"][stage.kernel.SOURCE] = "0" * 64
        with self.assertRaisesRegex(ValueError, "KERNEL_BYTES_CHANGED"):
            stage.sources(broken)

    def test_recipe_typed_boundaries_and_source_are_not_relaxed(self):
        original = {"schema": stage.SCHEMA, "contract": stage.plan(), "inputs": {}, "source_hashes": {"x": "a" * 64}}
        with patch.object(stage, "assemble", return_value=original):
            stage.validate_recipe(copy.deepcopy(original))
            for field, value in (("worker_seconds", 601), ("audit_seconds", 601), ("automatic_retry", True),
                    ("new_model_forwards", True), ("blank_id", False), ("threshold_selection", True)):
                changed = copy.deepcopy(original)
                changed["contract"][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    stage.validate_recipe(changed)
            changed = copy.deepcopy(original)
            changed["source_hashes"]["x"] = "b" * 64
            with self.assertRaises(ValueError):
                stage.validate_recipe(changed)

    def test_unknown_nonempty_observation_is_preserved_without_targets(self):
        for text in (None, "你好小窝"):
            with self.subTest(text=text):
                row, item = unknown_fixture(text)
                before = copy.deepcopy((row, item))
                stage.validate_unknown(row, item)
                self.assertEqual((row, item), before)
                self.assertNotIn("frames", item)
                self.assertNotIn("ctc_mask", item)
                self.assertNotIn("target_ids", item)
                self.assertEqual(item["full_text"], text)

    def test_unknown_cannot_be_relabelled_or_given_fake_targets(self):
        row, item = unknown_fixture("保留的未准入观察")
        for field, value in (("full_text", None), ("ctc_weight", .5), ("target_ids", [1027, 1401, 1462, 2093]),
                ("ctc_mask", True), ("keyword_targets", [0, 0]), ("keyword_mask", [True, True])):
            changed = copy.deepcopy(item)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                stage.validate_unknown(row, changed)

    def test_reuse_requires_exact_model_label_numeric_and_original_positions(self):
        row, item, previous, result = known_fixture()
        stage.validate_reuse(row, item, previous, result, stage.plan())
        for field, value in (("origin", "known-silence-corrected"), ("label_binding_sha256", "c" * 64),
                ("checkpoint_sha256", "d" * 64), ("original_weak_weight", 1.0)):
            changed = copy.deepcopy(result)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                stage.validate_reuse(row, item, previous, changed, stage.plan())
        changed = copy.deepcopy(result)
        changed["original_target_positions"][1]["token_id"] = 1
        with self.assertRaises(ValueError):
            stage.validate_reuse(row, item, previous, changed, stage.plan())
        changed = copy.deepcopy(previous)
        changed["saved_greedy"] = []
        with self.assertRaises(ValueError):
            stage.validate_reuse(row, item, changed, result, stage.plan())

    def test_manifest_binding_uses_metadata_and_worker_still_verifies_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "derived").mkdir()
            file = root / "derived/test.pt"
            file.write_bytes(b"opaque invented fixture")
            valid = stage.io.file_record(file)
            artifact = {"path": "derived/test.pt", **valid}
            with patch.object(stage.io, "file_record", side_effect=AssertionError("declaration must not read tensor bytes")):
                actual = stage.artifact_from_manifest(root, {"derived/test.pt": valid}, artifact)
            self.assertEqual(actual, {"path": str(file), **valid})
            file.write_bytes(b"changed invented bytes!")
            with self.assertRaises(ValueError):
                stage.kernel.check_pin(actual)
            with self.assertRaises(ValueError):
                stage.artifact_from_manifest(root, {}, {**artifact, "path": "../escape"})

    def test_aggregate_retains_positions_repeats_negatives_unknown_and_no_private_fields(self):
        records = [aggregate_fixture(side) for side in ("baseline", "augmentation")]
        records += [aggregate_fixture(side, "private-negative", True) for side in ("baseline", "augmentation")]
        coverage = [{"comparison_side": side, "cohort": "new-development", "offset_ms": 0, "analysis_mode": mode}
            for side in ("baseline", "augmentation") for mode in ("computed", "computed", "unscored-target-free")]
        public = stage.aggregate(records, coverage)
        self.assertEqual(len(public["conditions"]), 2)
        for group in public["conditions"]:
            positions = group["keyword_positions"]
            self.assertEqual([(p["keyword_id"], p["keyword_position"]) for p in positions], [(k, p) for k in (1, 2) for p in range(1, 5)])
            # K2 also overlaps the suffix of the first K1 in this invented text.
            self.assertTrue(all(p["reference_mentions"] == 2 for p in positions))
        control = next(r for r in public["keyword_controls"] if r["comparison_side"] == "augmentation" and r["keyword_id"] == 1)
        self.assertEqual((control["known_negative_rows"], control["negative_greedy_hits"]), (1, 1))
        self.assertEqual(sum(r["conditions"] for r in public["coverage_counts"] if r["analysis_mode"] == "unscored-target-free"), 2)
        self.assertEqual(len(public["paired_changes"][0]["keyword_positions"]), 8)
        encoded = json.dumps(public, ensure_ascii=False)
        for forbidden in ("private-source", "private-negative", "source_id", "full_text", "/workspace/", "target_ids"):
            self.assertNotIn(forbidden, encoded)

    def test_output_is_create_once(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.json"
            stage.io.write_json(path, {"first": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                stage.io.write_json(path, {"second": True})
            self.assertEqual(path.read_bytes(), before)

    def test_failed_terminal_is_rejected_before_numeric_runtime(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage.io.write_json(root / "summary.json", {"schema": stage.SCHEMA, "status": "failed"})
            with patch.object(stage.kernel.onset, "verify_manifest"), patch.object(stage, "pin", return_value={}), \
                    patch.object(stage, "terminal_metadata"), self.assertRaisesRegex(ValueError, "SUCCESS_REQUIRED"):
                # A failed status also cannot masquerade as a successful terminal.
                stage.completed_output(root, "a" * 64)


class NumericalTests(unittest.TestCase):
    def test_guarded_kernel_roundtrip_uses_only_invented_tensor(self):
        import torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        with stage.kernel.no_model_or_optimizer(torch), torch.inference_mode():
            logits = torch.full((6, 1, 2601), -20., dtype=torch.float32)
            logits[:, 0, 0] = 2.
            logits[1:3, 0, 1] = 3.
            logits[4:6, 0, 2] = 3.
            target = [1, 2]
            greedy = stage.kernel.evaluation.collapse_ctc(logits[:, 0].argmax(-1).tolist())
            row = {"source_id": "invented-numeric", "source_family_id": "invented-family", "split": "development",
                "ctc_weight": .5, "target_ids": target}
            saved = {"schema": "kws-d1-development-numeric-evidence-v2", "source_id": row["source_id"],
                "feature": torch.zeros((6, 32), dtype=torch.float32), "logits": logits}
            for name in ("feature", "logits"):
                saved[name + "_sha256"] = stage.training.tensor_sha(saved[name])
            record = {"row": row, "frames": 6, "origin": "fixture", "ordinal": 0, "offset_ms": 0,
                "checkpoint_sha256": "c" * 64, "label_binding_sha256": "d" * 64,
                "condition_binding_sha256": "e" * 64, "saved_greedy": greedy, "class_ids": [0, 1, 2],
                "added_samples": 0, "fully_digital_silence_frames": 0,
                "feature_sha256": saved["feature_sha256"], "logits_sha256": saved["logits_sha256"],
                "numeric_artifact": {"sha256": "f" * 64},
                "saved_original_ctc_nll": -stage.kernel.sequence_score(logits[:, 0].double().log_softmax(-1), target, torch),
                "hypotheses": stage.kernel.variants(target, greedy, [0, 0], [True, True], stage.plan()["keywords"])}
            stage.kernel.verify_numeric(saved, record, torch)
            arrays, result = stage.kernel.numerical_condition(saved, record, stage.plan(), torch)
            self.assertEqual(len(result["original_target_positions"]), 2)
            self.assertEqual(set(arrays), set(stage.kernel.ARRAY_KEYS))
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "fixture.pt"
                stage.kernel.save_arrays(path, arrays, record, torch)
                retained = torch.load(path, weights_only=True, map_location="cpu")
                self.assertEqual(retained["array_hashes"], {k: stage.training.tensor_sha(v) for k, v in arrays.items()})
                self.assertEqual(retained["condition_binding_sha256"], record["condition_binding_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NumericalTests if args.numeric else MetadataTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
