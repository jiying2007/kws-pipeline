#!/usr/bin/env python3
"""Invented metadata and queue-failure fixtures; no Torch, models or forwards."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
import research_c1_context_development as diagnostic


def fixture():
    vocabulary = {"schema": "kws-d1-fulltext-vocabulary-v2", "output_count": 2601,
        "token_to_id": {char: i + 1 for i, char in enumerate("你好小窝今天")}}
    rows = []
    for i in range(18):
        text = ("你好小窝", "小窝小窝", "今天")[i % 3]
        target = diagnostic.fit.fulltext_targets(text, vocabulary)["target_ids"]
        rows.append({"source_id": "fixture-%02d" % i, "source_family_id": "dev-family-%d" % (i // 3),
            "speaker_id": "dev-speaker-%d" % (i // 3), "split": "development", "source_role": "development-only",
            "ctc_mask": True, "ctc_weight": .5, "quarantined": False, "generation_outcome": "completed",
            "human_gold": False, "independent_test": False, "full_text": text, "target_ids": target,
            "vocabulary_sha256": diagnostic.fit.digest(vocabulary), "samples": 4800, "expected_c1_frames": 14,
            "ctc_minimum_frames": len(target), "keyword_mask": [True, True], "keyword_weight": [.5, .5],
            "keyword_targets": [int(word in text) for word in diagnostic.fit.KEYWORDS],
            "pcm_sha256": hashlib.sha256(("invented-pcm-%d" % i).encode()).hexdigest()})
    return vocabulary, rows


def invented_observations(rows, vocabulary):
    output = []
    for model in diagnostic.MODELS:
        for row in rows:
            for ms in diagnostic.onset.OFFSETS:
                target = row["target_ids"]
                output.append({"model_id": model, "source_id": row["source_id"], "offset_ms": ms,
                    "greedy_token_ids": target, "exact_token_match": True,
                    "substitutions": 0, "deletions": 0, "insertions": 0, "edit_numerator": 0,
                    "target_token_denominator": len(target), "ctc_loss_target_length_normalized": .5,
                    "ctc_loss_confidence_weighted": .25, "ctc_weight": .5, "ctc_nll": .5 * len(target),
                    "frames": 14, "blank_argmax_frames": 7, "blank_probability_mean": .5,
                    "known_zero_prefix_frames": {0: 0, 200: 9, 500: 24}[ms], "known_zero_nonblank_frames": 0,
                    "emissions_entirely_in_added_silence": 0,
                    "keywords": diagnostic.keyword_observations(target, row, vocabulary)})
    return output


class DevelopmentContractTests(unittest.TestCase):
    def test_import_and_declaration_helpers_do_not_import_torch(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("model", sys.modules)

    def test_no_budget_expansion_training_threshold_or_reclassification(self):
        original = {"schema": diagnostic.SCHEMA, **diagnostic.fixed_contract()}
        diagnostic.validate_contract(original)
        for key, value in (("forward_limit", 109), ("worker_seconds", 601), ("queue_minutes", 11),
                ("optimizer_steps", 1), ("training_forwards", 1), ("threshold_selection", True),
                ("independent_test", True), ("previously_exposed_development", False),
                ("full_float32_logits_retained", False), ("automatic_retry", True),
                ("model_order", list(reversed(diagnostic.MODELS))), ("offsets_ms", [False, 200, 500]),
                ("reset_each_condition", False), ("development_rows", 17)):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                diagnostic.validate_contract(changed)

    def test_all_eighteen_full_transcripts_keep_original_development_role(self):
        vocabulary, rows = fixture()
        diagnostic.validate_rows(rows, vocabulary, [])
        for mutate in (lambda rs: rs.pop(), lambda rs: rs.__setitem__(1, copy.deepcopy(rs[0])),
                lambda rs: rs[0].update(split="train"), lambda rs: rs[0].update(source_role="train-candidate"),
                lambda rs: rs[0].update(target_ids=rs[0]["target_ids"][:2]),
                lambda rs: rs[2].update(full_text="", target_ids=[]),
                lambda rs: rs[0].update(ctc_mask=False), lambda rs: rs[0].update(ctc_weight=float("nan")),
                lambda rs: rs[0].update(expected_c1_frames=100), lambda rs: rs[0].update(keyword_targets=[0, 0]),
                lambda rs: rs[0].update(keyword_mask=[0, True])):
            altered = copy.deepcopy(rows)
            mutate(altered)
            with self.assertRaises(ValueError):
                diagnostic.validate_rows(altered, vocabulary, [])

    def test_overlap_in_family_speaker_or_original_pcm_is_rejected(self):
        vocabulary, rows = fixture()
        train = {"source_family_id": "different-family", "speaker_id": "different-speaker"}
        diagnostic.validate_rows(rows, vocabulary, [train])
        for key, value in (("source_family_id", rows[0]["source_family_id"]),
                           ("speaker_id", rows[0]["speaker_id"]), ("original_pcm_sha256", rows[0]["pcm_sha256"])):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "TRAIN_SOURCE_OVERLAP"):
                diagnostic.validate_rows(rows, vocabulary, [{**train, key: value}])

    def test_full_keyword_requires_contiguous_complete_sequence(self):
        vocabulary, rows = fixture()
        k1 = diagnostic.fit.fulltext_targets("你好小窝", vocabulary)["target_ids"]
        k2 = diagnostic.fit.fulltext_targets("小窝小窝", vocabulary)["target_ids"]
        for hypothesis, expected in ((k1[:3], [False, False]), (k1, [True, False]),
                                    (k2[:2], [False, False]), (k2, [False, True]),
                                    (k1[:2] + [6] + k1[2:], [False, False])):
            result = diagnostic.keyword_observations(hypothesis, rows[0], vocabulary)
            self.assertEqual([r["complete_sequence_present"] for r in result], expected)

    def test_unknown_keyword_is_preserved_and_never_enters_negative_denominator(self):
        vocabulary, rows = fixture()
        rows[0].update(keyword_mask=[False, True], keyword_targets=[None, 0], keyword_weight=[0, .5])
        diagnostic.validate_rows(rows, vocabulary, [])
        observations = invented_observations(rows, vocabulary)
        report = diagnostic.summarize(observations, {"development": rows})
        for model in diagnostic.MODELS:
            for ms in ("0", "200", "500"):
                k1 = report[model][ms]["keywords"][0]
                self.assertEqual(k1, {"keyword_id": 1, "known_positive_rows": 5, "matched_positive_rows": 5,
                    "known_negative_rows": 12, "false_positive_rows": 0, "unknown_rows": 1})
        rows[0]["keyword_targets"][0] = 0
        with self.assertRaisesRegex(ValueError, "UNKNOWN_IS_NOT_NEGATIVE"):
            diagnostic.validate_rows(rows, vocabulary, [])

    def test_complete_paired_roster_and_known_zero_frame_denominators(self):
        vocabulary, rows = fixture()
        observations = invented_observations(rows, vocabulary)
        result = diagnostic.summarize(observations, {"development": rows})
        self.assertEqual(len(observations), 108)
        for model in diagnostic.MODELS:
            self.assertEqual([result[model][str(ms)]["known_zero_frames"] for ms in (0, 200, 500)], [0, 162, 432])
            self.assertEqual([result[model][str(ms)]["keywords"][1]["matched_positive_rows"] for ms in (0, 200, 500)], [6, 6, 6])
        for altered in (observations[:-1], observations + observations[:1], list(reversed(observations))):
            with self.assertRaisesRegex(ValueError, "ALL_CONDITIONS_REQUIRED"):
                diagnostic.summarize(altered, {"development": rows})

    def test_recipe_changes_to_checkpoint_or_complete_row_are_rejected(self):
        original = {"schema": diagnostic.SCHEMA, **diagnostic.fixed_contract(), "inputs": {},
                    "models": [{"checkpoint": {"sha256": "a" * 64}}], "development": fixture()[1]}
        with patch.object(diagnostic, "assemble", return_value=(original, {})):
            diagnostic.validate_recipe(copy.deepcopy(original))
            for mutate in (lambda job: job["models"][0]["checkpoint"].update(sha256="b" * 64),
                           lambda job: job["development"][0].update(full_text="改写"),
                           lambda job: job["development"].reverse()):
                changed = copy.deepcopy(original)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "RECIPE_OR_INPUT_CHANGED"):
                    diagnostic.validate_recipe(changed)

    def test_frozen_verifier_is_pinned_and_numeric_source_cannot_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in diagnostic.context.CODE:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            hashes = {name: diagnostic.training.sha_file(source / name) for name in diagnostic.context.CODE}
            diagnostic.frozen_sources("context", source, hashes)
            helper = "tools/research_d2_checkpoints.py"
            (source / helper).write_bytes((source / helper).read_bytes() + b"\n# invented old metadata helper fixture\n")
            with self.assertRaisesRegex(ValueError, "FROZEN_VERIFIER_CHANGED"):
                diagnostic.frozen_sources("context", source, hashes)
            hashes[helper] = diagnostic.training.sha_file(source / helper)
            diagnostic.frozen_sources("context", source, hashes)
            numeric = "training/research_d1_evaluate.py"
            (source / numeric).write_bytes((source / numeric).read_bytes() + b"\n# invented changed numeric source\n")
            hashes[numeric] = diagnostic.training.sha_file(source / numeric)
            with self.assertRaisesRegex(ValueError, "NUMERIC_OR_READER_SOURCE_CHANGED"):
                diagnostic.frozen_sources("context", source, hashes)

    def test_runtime_metadata_only_drift_does_not_relax_package_or_interpreter_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for name in diagnostic.silence.CODE:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            name = "configs/runtime/research-runtime.json"
            original = diagnostic.training.read_json(source / name)
            changed = {**original, "validation": {"invented_metadata_only": True}}
            (source / name).write_bytes(diagnostic.io.canonical(changed))
            hashes = {path: diagnostic.training.sha_file(source / path) for path in diagnostic.silence.CODE}
            diagnostic.frozen_sources("correction", source, hashes)
            changed["invented_numeric_contract_change"] = True
            (source / name).write_bytes(diagnostic.io.canonical(changed))
            hashes[name] = diagnostic.training.sha_file(source / name)
            with self.assertRaisesRegex(ValueError, "NUMERIC_RUNTIME_CHANGED"):
                diagnostic.frozen_sources("correction", source, hashes)

    def test_direct_unqueued_run_has_no_output_or_model_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe, lock, output = root / "recipe.json", root / "fixture.lock", root / "outputs"
            diagnostic.io.write_json(recipe, {"schema": diagnostic.SCHEMA, **diagnostic.fixed_contract(), "heavy_lock": str(lock)})
            with self.assertRaisesRegex(ValueError, "D1_HASH_BOUND_DURABLE_QUEUE_REQUIRED"):
                diagnostic.run(recipe, output, lock)
            self.assertFalse(output.exists())
            self.assertNotIn("torch", sys.modules)

    def test_real_queue_retains_failed_claim_and_reentry_never_retries(self):
        # Deliberately incomplete invented input: the actual worker must fail
        # before runtime loading. It never reaches any model call.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe, request, lock, queue = root / "recipe.json", root / "queue-request.json", root / "fixture.lock", root / "queue"
            diagnostic.io.write_json(recipe, {"schema": diagnostic.SCHEMA, **diagnostic.fixed_contract(),
                "heavy_lock": str(lock), "inputs": {}})
            definition = diagnostic.queue_definition(recipe, str(Path(sys.executable).resolve()))
            self.assertEqual(definition["global_budget_minutes"], 10)
            self.assertEqual(definition["jobs"][0]["timeout_minutes"], 10)
            diagnostic.io.write_json(request, definition)
            diagnostic.io.create(queue, request, ROOT / "configs/research/active-plan-20261105.json", lock)
            _, code = diagnostic.io.run(queue, max_jobs=1)
            self.assertEqual(code, 1)
            folder = queue / "jobs" / diagnostic.JOB_ID
            claim = (folder / "claim.json").read_bytes()
            result = (folder / "result.json").read_bytes()
            status = diagnostic.training.read_json(folder / "outputs/status.json")
            self.assertEqual(status["status"], "failed")
            self.assertIn("CONTEXT_DIAGNOSTIC_INPUT_REQUEST", status["error"])
            self.assertEqual(status["model_constructions"], 0)
            self.assertEqual(status["started_forward_calls"], 0)
            self.assertFalse(status["automatic_retry"])
            _, code = diagnostic.io.run(queue, max_jobs=1)
            self.assertEqual(code, 1)
            self.assertEqual((folder / "claim.json").read_bytes(), claim)
            self.assertEqual((folder / "result.json").read_bytes(), result)
            self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
