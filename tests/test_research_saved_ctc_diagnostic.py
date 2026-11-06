#!/usr/bin/env python3
"""Metadata guards and tiny enumerated CTC paths; never read campaign tensors."""
from __future__ import annotations

import argparse
import copy
import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools")]
import research_saved_ctc_diagnostic as stage


def observation_fixture():
    row = {"source_id": "invented-row", "source_family_id": "invented-family", "speaker_id": "invented-speaker",
        "split": "development", "source_role": "development-only", "ctc_mask": True, "ctc_weight": .5,
        "full_text": "你好小窝", "target_ids": [1027, 1401, 1462, 2093], "keyword_targets": [1, 0],
        "keyword_mask": [True, True], "keyword_weight": [.5, .5], "human_gold": False, "independent_test": False,
        "pcm_sha256": "a" * 64, "samples": 32000}
    item = {**{k: row[k] for k in ("source_id", "source_family_id", "speaker_id", "split", "source_role", "ctc_mask",
        "ctc_weight", "keyword_targets", "keyword_mask", "keyword_weight", "target_ids")}, "target_text": row["full_text"],
        "label_binding_sha256": stage.fit.row_identity(row), "original_pcm_sha256": row["pcm_sha256"],
        "checkpoint_sha256": stage.plan()["checkpoint_sha256"]["context52"], "offset_ms": 0,
        "frames": stage.onset.geometry(row["samples"], 0)["frames"], "known_zero_prefix_frames": 0}
    return row, item


class MetadataTests(unittest.TestCase):
    def test_metadata_import_has_no_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_fixed_typed_contract_and_source_are_not_relaxed(self):
        original = {"schema": stage.SCHEMA, "contract": stage.plan(), "inputs": {}, "source_hashes": {"source": "a" * 64}}
        with patch.object(stage, "assemble", return_value=original):
            stage.validate_recipe(copy.deepcopy(original))
            for key, value in (("saved_conditions", 263), ("single_token_deletion_contrasts", 875), ("unique_sequence_scores", 1383),
                    ("worker_seconds", 541), ("new_model_forwards", 1), ("optimizer_steps", 1),
                    ("automatic_retry", True), ("blank_id", False), ("threshold_selection", True),
                    ("offsets_ms", [False, 200, 500]), ("independent_test", True)):
                changed = copy.deepcopy(original)
                changed["contract"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    stage.validate_recipe(changed)
            changed = copy.deepcopy(original)
            changed["source_hashes"]["source"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "RECIPE_OR_SOURCE"):
                stage.validate_recipe(changed)

    def test_all_positions_both_keywords_and_repeated_mentions(self):
        keywords = stage.plan()["keywords"]
        k1, k2 = keywords["1"], keywords["2"]
        target = [18, *k1, 19, *k2, 20, *k2, 21]
        result = stage.variants(target, target, [1, 1], [True, True], keywords)
        self.assertEqual(len(result["keyword_mentions"]), 3)
        self.assertEqual(len(result["logical_hypotheses"]), 14)
        self.assertEqual(len(result["unique_targets"]), 13)
        self.assertEqual(result["unique_targets"][0], target)
        logical = result["logical_hypotheses"][2:]
        self.assertEqual({x["keyword_id"] for x in logical}, {1, 2})
        self.assertEqual([x["keyword_position"] for x in logical], [1, 2, 3, 4] * 3)
        for item in logical:
            index = item["target_position"]
            self.assertEqual(result["unique_targets"][item["hypothesis_index"]], target[:index] + target[index + 1:])
        self.assertEqual([m["reference_start"] for m in result["keyword_mentions"]], [1, 6, 11])

    def test_analytic_deletion_does_not_relabel_or_drop_other_mentions(self):
        k = stage.plan()["keywords"]
        target = [*k["1"], 18, *k["1"]]
        greedy = target[:1] + target[2:]
        saved = copy.deepcopy(target)
        result = stage.variants(target, greedy, [1, 0], [True, True], k)
        self.assertEqual(target, saved)
        deleted = [h for h in result["logical_hypotheses"] if h.get("target_position") == 1][0]
        self.assertEqual(result["logical_hypotheses"][1]["hypothesis_index"], deleted["hypothesis_index"])
        self.assertEqual(result["unique_targets"][deleted["hypothesis_index"]][-4:], k["1"])
        self.assertEqual(len(result["logical_hypotheses"]), 10)

    def test_unknown_is_not_promoted_to_negative_or_positive(self):
        k = stage.plan()["keywords"]
        ordinary = stage.variants([18, 19], [], [None, None], [False, False], k)
        self.assertEqual(ordinary["keyword_mentions"], [])
        for targets, masks in (([0, 0], [False, False]), ([None, 0], [False, True])):
            with self.assertRaises(ValueError):
                stage.variants(k["1"], [], targets, masks, k)

    def test_source_label_weight_pcm_checkpoint_and_geometry_guards(self):
        row, item = observation_fixture()
        stage.validate_observation(row, item, "context52", stage.plan())
        for key, value in (("target_text", "edited"), ("ctc_weight", 1.0), ("target_ids", [1]),
                ("source_family_id", "swapped"), ("original_pcm_sha256", "b" * 64),
                ("checkpoint_sha256", "c" * 64), ("frames", 100), ("offset_ms", False)):
            changed = copy.deepcopy(item)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stage.validate_observation(row, changed, "context52", stage.plan())
        bad = copy.deepcopy(row)
        bad["ctc_weight"] = True
        with self.assertRaises(ValueError):
            stage.validate_observation(bad, item, "context52", stage.plan())

    def test_exact_workload_and_output_budget(self):
        keys = ("saved_conditions", "saved_serialized_numeric_bytes", "saved_frames", "original_target_positions",
            "keyword_reference_mentions", "single_token_deletion_contrasts", "logical_sequence_scores", "unique_sequence_scores", "empty_greedy_sequences")
        value = {**{k: stage.plan()[k] for k in keys}, "derived_array_payload_bytes": 4096}
        stage.validate_workload(value, stage.plan())
        for mutate in (lambda v: v.pop("logical_sequence_scores"), lambda v: v.update(saved_conditions=265),
                lambda v: v.update(derived_array_payload_bytes=True), lambda v: v.update(derived_array_payload_bytes=65 * 1024 * 1024)):
            changed = copy.deepcopy(value)
            mutate(changed)
            with self.assertRaises(ValueError):
                stage.validate_workload(changed, stage.plan())

    def test_failed_queue_cannot_be_audited_as_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "queue/jobs/job/outputs"
            output.mkdir(parents=True)
            summary = {"status": "completed"}
            result = {"status": "completed", "exit_code": 0, "job_id": stage.JOB_ID,
                "recipe_sha256": "a" * 64, "queue_sha256": "a" * 64, "claim_sha256": "a" * 64,
                "outputs": {"MANIFEST.json": {"bytes": 1, "sha256": "a" * 64}}}
            def read(path):
                return summary if Path(path).name == "status.json" else result
            with patch.object(stage.training, "read_json", side_effect=read), patch.object(stage.training, "sha_file", return_value="a" * 64), \
                    patch.object(stage.io, "file_record", return_value={"bytes": 1, "sha256": "a" * 64}):
                stage.completed_queue(output, summary)
                result["status"] = "failed"
                result["exit_code"] = 1
                with self.assertRaises(ValueError):
                    stage.completed_queue(output, summary)
                result.update(status="completed", exit_code=0)
                (output / "failure.json").write_text("{}")
                with self.assertRaises(ValueError):
                    stage.completed_queue(output, summary)

    def test_terminal_counts_and_boundaries_are_verified(self):
        job = {"contract": stage.plan(), "source_hashes": {"one": "a" * 64}, "workload": {"saved_conditions": 264}}
        summary = {"schema": stage.SCHEMA, "recipe_sha256": "b" * 64, "wall_seconds": 1.0, "peak_rss_kib": 1024,
            "actual_counts": {**stage.computation_limits(job["contract"]), "conditions_started": 264, "conditions_completed": 264},
            "source_hashes": job["source_hashes"], "workload": job["workload"],
            "new_model_constructions": 0, "new_model_forwards": 0, "optimizer_steps": 0, "backward_calls": 0,
            "asr_or_tts_calls": 0, "automatic_retry": False, "human_gold": False, "independent_test": False,
            "previously_exposed_development": True, "quality_winner": None, "generalization_passed": None,
            "pairwise_preferences_only": True, "conditional_alignment_is_not_acoustic_truth": True,
            "original_logits_duplicated": False, "derived_fullprecision_arrays_retained": True}
        stage.terminal_metadata(job, summary, "b" * 64)
        for key, value in (("actual_counts", {}), ("new_model_forwards", False), ("optimizer_steps", 1),
                ("quality_winner", "K1"), ("wall_seconds", 541), ("peak_rss_kib", 3 * 1024 ** 2), ("source_hashes", {})):
            changed = copy.deepcopy(summary)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                stage.terminal_metadata(job, changed, "b" * 64)

    def test_create_only_records_keep_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claim.json"
            stage.io.write_json(path, {"first": True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                stage.io.write_json(path, {"second": True})
            self.assertEqual(path.read_bytes(), before)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)

    def analyze(self, values, target):
        t = self.torch
        logits = t.tensor(values, dtype=t.float32).unsqueeze(1)
        greedy = stage.evaluation.collapse_ctc(logits[:, 0].argmax(-1).tolist())
        with stage.no_model_or_optimizer(t), t.inference_mode():
            computed = stage.compute_arrays(logits, list(target), greedy, stage.plan(), t)
        return logits, greedy, computed

    def enumerate_paths(self, log_probs, target):
        # Independent label-path enumeration, not a second dynamic program.
        total, maximum = 0.0, -math.inf
        masses = [[0.0] * len(log_probs) for _ in target]
        for path in itertools.product(range(len(log_probs[0])), repeat=len(log_probs)):
            if stage.evaluation.collapse_ctc(path) != list(target):
                continue
            logp = sum(log_probs[i][label] for i, label in enumerate(path))
            p = math.exp(logp)
            total += p
            maximum = max(maximum, logp)
            position, previous = -1, None
            for frame, token in enumerate(path):
                if token != 0:
                    if token != previous:
                        position += 1
                    masses[position][frame] += p
                previous = token
        q = [[p / sum(row) for p in row] for row in masses]
        occupancy = [sum(row) / total for row in masses]
        return math.log(total), q, occupancy, maximum

    def test_exhaustive_distinct_and_adjacent_repeated_targets(self):
        values = [[1.1, .4, -.8], [-.2, 1.4, .1], [1.8, -.3, .2], [.2, .6, 1.3]]
        for target in ((1, 2), (1, 1)):
            with self.subTest(target=target):
                logits, _, computed = self.analyze(values, target)
                arrays, columns, score, positions, viterbi = computed
                original = logits[:, 0].double().log_softmax(-1).tolist()
                expected, q, occupancy, best_path = self.enumerate_paths(original, target)
                self.assertAlmostEqual(score, expected, places=11)
                self.assertTrue(self.torch.allclose(arrays["temporal_weights"], self.torch.tensor(q, dtype=self.torch.float64), atol=1e-11, rtol=1e-11))
                for actual, want in zip(positions, occupancy):
                    self.assertAlmostEqual(actual["state_occupancy_frames"], want, places=11)
                path = [0] * len(values)
                for token, segment in zip(target, viterbi):
                    path[segment["first_frame"]:segment["last_frame"] + 1] = [token] * segment["frames"]
                self.assertEqual(stage.evaluation.collapse_ctc(path), list(target))
                self.assertAlmostEqual(sum(original[i][x] for i, x in enumerate(path)), best_path, places=11)
                if target[0] == target[1]:
                    self.assertGreater(viterbi[1]["first_frame"] - viterbi[0]["last_frame"], 1)

    def test_repeated_sequence_retains_distinct_original_positions(self):
        values = [[.2, .3, -.4], [.6, -.1, .8], [.9, .2, -.5], [-.1, 1., .2], [.1, -.3, .8], [.7, .1, -.2]]
        logits, _, (arrays, _, score, positions, _) = self.analyze(values, (1, 2, 1, 2))
        expected, q, _, _ = self.enumerate_paths(logits[:, 0].double().log_softmax(-1).tolist(), (1, 2, 1, 2))
        self.assertAlmostEqual(score, expected, places=10)
        self.assertEqual([p["occurrence"] for p in positions], [1, 2, 3, 4])
        self.assertEqual(tuple(arrays["temporal_weights"].shape), (4, 6))
        self.assertTrue(self.torch.allclose(arrays["temporal_weights"], self.torch.tensor(q, dtype=self.torch.float64), atol=1e-10, rtol=1e-10))

    def test_full_vocabulary_normalization_survives_gather_and_not_renormalization(self):
        values = [[2., 0., -.2, 1., .5], [2., .4, 0., 1., .5], [2., 0., .4, 1., .5], [2., -.2, 0., 1., .5]]
        logits, greedy, (arrays, columns, score, _, _) = self.analyze(values, (1, 2))
        self.assertEqual(greedy, [])
        self.assertEqual(columns, [0, 1, 2])
        full = logits[:, 0].double().log_softmax(-1)
        self.assertAlmostEqual(score, stage.sequence_score(full, (1, 2), self.torch), places=11)
        altered = stage.sequence_score(arrays["class_log_probs"].log_softmax(-1), (1, 2), self.torch)
        self.assertGreater(abs(score - altered), .1)
        self.assertTrue(bool((arrays["class_log_probs"].exp().sum(-1) < 1).all()))

    def test_empty_greedy_is_exact_all_blank_path(self):
        logits, greedy, (arrays, _, _, _, _) = self.analyze([[5., -1., -2.]] * 4, (1, 2))
        self.assertEqual(greedy, [])
        actual = stage.sequence_score(arrays["class_log_probs"], [], self.torch)
        expected = float(logits[:, 0].double().log_softmax(-1)[:, 0].sum())
        self.assertEqual(actual, expected)

    def test_conditional_occupancy_is_not_acoustic_support_and_rank_ties_are_not_argmax(self):
        _, _, (arrays, _, score, _, _) = self.analyze([[10., -10., -10.]] * 4, (1, 2))
        self.assertLess(score, -30.)
        self.assertTrue(self.torch.allclose(arrays["temporal_weights"].sum(-1), self.torch.ones(2, dtype=self.torch.float64)))
        self.assertLess(float(arrays["class_log_probs"][:, 1:].exp().max()), 1e-8)
        _, _, (tied, _, _, _, _) = self.analyze([[0., 0., 0.]] * 3, (1, 2))
        self.assertTrue(bool((tied["rank_all"] == 1).all()))
        self.assertTrue(bool((tied["argmax"] == 0).all()))

    def test_model_backward_optimizer_guard_allows_only_tensor_math(self):
        t = self.torch
        with stage.no_model_or_optimizer(t):
            with self.assertRaisesRegex(RuntimeError, "FORBIDDEN"):
                t.nn.Module()
            with self.assertRaisesRegex(RuntimeError, "FORBIDDEN"):
                t.tensor(1., requires_grad=True).backward()
            with self.assertRaisesRegex(RuntimeError, "FORBIDDEN"):
                t.optim.Optimizer([], {})
            self.assertEqual(float(t.tensor([1., 2.]).sum()), 3.)

    def test_original_numeric_schema_dtype_shape_and_hash(self):
        t = self.torch
        saved = {"schema": "kws-d1-development-numeric-evidence-v2", "source_id": "invented",
            "feature": t.zeros((3, 32), dtype=t.float32), "logits": t.zeros((3, 1, 2601), dtype=t.float32)}
        for key in ("feature", "logits"):
            saved[key + "_sha256"] = stage.training.tensor_sha(saved[key])
        record = {"frames": 3, "row": {"source_id": "invented"}, "feature_sha256": saved["feature_sha256"], "logits_sha256": saved["logits_sha256"]}
        stage.verify_numeric(saved, record, t)
        with self.assertRaises(ValueError):
            stage.verify_numeric({**saved, "logits": saved["logits"].double()}, record, t)
        with self.assertRaises(ValueError):
            stage.verify_numeric(saved, {**record, "logits_sha256": "0" * 64}, t)

    def test_nonfinite_and_infeasible_alignment_are_rejected(self):
        with self.assertRaises(ValueError):
            self.analyze([[float("nan"), 0., 1.]] * 3, (1, 2))
        with self.assertRaises(ValueError):
            self.analyze([[0., 1., 0.]] * 2, (1, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true", help="Run only tiny synthetic numeric fixtures, no campaign artifacts")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NumericalTests if args.numeric else MetadataTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
