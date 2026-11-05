#!/usr/bin/env python3
"""Development metric/receipt contracts; optional bounded synthetic numerics."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tests")]
import research_d1_evaluate as evaluation
import research_d1_recipe as recipes
import test_research_d1 as fixture


def job():
    return {"schema": evaluation.SCHEMA, "prepared_from_dependency": "prepare",
            "training_from_dependency": "train", "candidate": "rnn64-ctc2601", "near_percent": 0,
            "held_out_source_family_ids": ["synthetic-family-6", "synthetic-family-7"],
            "evaluator_source_sha256": evaluation.training.sha_file(ROOT / evaluation.SOURCE_PATH),
            "timeout_seconds": 10}


class DevelopmentContracts(unittest.TestCase):
    def test_explicit_groups_source_timeout_and_candidate(self):
        self.assertEqual(evaluation.validate_job(job()), job())
        for field, value in (("held_out_source_family_ids", ["one"]),
                             ("held_out_source_family_ids", ["same", "same"]),
                             ("held_out_source_family_ids", ["z", "a"]),
                             ("near_percent", True), ("candidate", "fsmn2599"),
                             ("timeout_seconds", 60000), ("timeout_seconds", float("nan")),
                             ("training_from_dependency", "prepare"), ("evaluator_source_sha256", "wrong")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                evaluation.validate_job({**job(), field: value})

    def test_ctc_collapse_keeps_blank_separated_repetitions(self):
        self.assertEqual(evaluation.collapse_ctc([0, 1, 1, 0, 1, 2, 2, 0, 2600]), [1, 1, 2, 2600])
        self.assertEqual(evaluation.collapse_ctc([0, 0]), [])
        with self.assertRaisesRegex(ValueError, "GREEDY_TOKEN_ID"):
            evaluation.collapse_ctc([2601])

    def test_edit_numerator_denominator_and_empty_hypothesis(self):
        exact = evaluation.token_errors([1, 1, 2], [1, 1, 2])
        self.assertTrue(exact["exact_token_match"])
        self.assertEqual((exact["edit_numerator"], exact["target_token_denominator"]), (0, 3))
        for reference, hypothesis, expected in (([1, 2], [1, 3], (1, 0, 0)),
                                                 ([1, 2], [1], (0, 1, 0)),
                                                 ([1], [1, 2], (0, 0, 1)),
                                                 ([1, 1], [], (0, 2, 0)),
                                                 ([1], [2, 2, 2], (1, 0, 2))):
            value = evaluation.token_errors(reference, hypothesis)
            self.assertEqual(tuple(value[k] for k in ("substitutions", "deletions", "insertions")), expected)
            self.assertEqual(value["edit_numerator"], sum(expected))
        with self.assertRaises(ValueError):
            evaluation.token_errors([], [])

    def test_aggregate_uses_unweighted_micro_tokens_and_explicit_exclusions(self):
        def scored(length, loss, weight):
            return {"evaluation_status": "evaluated", "edit_numerator": 1,
                    "target_token_denominator": length, "substitutions": 1, "deletions": 0, "insertions": 0,
                    "exact_token_match": False, "ctc_loss_target_length_normalized": loss,
                    "ctc_loss_confidence_weighted": loss * weight, "ctc_weight": weight}
        rows = [scored(9, 2, .5), scored(1, 3, 1)]
        rows += [{"evaluation_status": "not_evaluable", "reason": reason} for reason in
                 ("failed_native_generation_audit_only", "full_transcript_out_of_vocabulary", "no_full_transcript")]
        result = evaluation.summarize(rows)
        self.assertEqual((result["observed_clips"], result["evaluable_clips"]), (5, 2))
        self.assertEqual((result["edit_numerator"], result["target_token_denominator"]), (2, 10))
        self.assertEqual(result["token_edit_rate"], .2)
        self.assertEqual(result["mean_ctc_loss_target_length_normalized"], 2.5)
        self.assertEqual(result["mean_ctc_loss_confidence_weighted"], 2)
        self.assertEqual(result["exact_token_match_denominator"], 2)
        self.assertEqual(len(result["not_evaluable_reasons"]), 3)
        empty = evaluation.summarize(rows[2:])
        self.assertIsNone(empty["token_edit_rate"])
        self.assertIsNone(empty["exact_token_match_rate"])

    def selected(self, values=None, families=None):
        rows = recipes.project_rows(values or fixture.raw_rows(), fixture.vocabulary())
        record = {"observed_rows": len(rows), "projected_rows_by_split": dict(Counter(row["split"] for row in rows))}
        return evaluation.development_rows(rows, recipes.cohort_rows(rows), record, fixture.vocabulary(),
                                            families or job()["held_out_source_family_ids"])

    def test_same_two_development_groups_keep_oov_uncertain_and_failed_rows(self):
        values = fixture.raw_rows()
        values[193].update(full_text="请把𠮷灯打开", keyword_targets=[0, 0])
        values[194].update(label_status="uncertain", full_text=None, target_ids=None, ctc_mask=False,
                           ctc_weight=0, keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0])
        values[195].update(label_status="uncertain", generation_outcome="failed-native", full_text=None,
                           target_ids=None, ctc_mask=False, ctc_weight=0, keyword_targets=[None, None],
                           keyword_mask=[False, False], keyword_weight=[0, 0])
        selected = self.selected(values)
        self.assertEqual((len(selected), sum(row["ctc_mask"] for row in selected)), (64, 61))
        self.assertEqual({row["source_family_id"] for row in selected}, set(job()["held_out_source_family_ids"]))
        self.assertTrue(all(row["split"] == "development" for row in selected))
        self.assertEqual(len({row["ctc_reason"] for row in selected if not row["ctc_mask"]}), 3)

    def test_development_names_cannot_relabel_train_speakers_or_ancestry(self):
        rows = fixture.raw_rows()
        rows[192]["speaker_id"] = rows[0]["speaker_id"]
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_LEAK"):
            self.selected(rows)
        rows = fixture.raw_rows()
        rows[192]["reference_wav_sha256"] = "a" * 64
        rows[0]["parent_source_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "SOURCE_ROLE_LEAK"):
            self.selected(rows)
        with self.assertRaisesRegex(ValueError, "HELD_OUT_ROSTER_CHANGED"):
            self.selected(families=["synthetic-family-5", "synthetic-family-6"])

    def test_real_queue_completion_binds_training_outputs_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            producer = root / "producer.py"
            producer.write_text("""import hashlib,json,os
from pathlib import Path
out=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])
# Hypothetical training metadata only: this probe performs no optimizer/model call.
(out/'final.pt').write_bytes(b'fictional checkpoint bytes, never deserialized')
(out/'consumption.jsonl').write_bytes(b'fictional consumption evidence\\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
value={'schema':'kws-d1-training-result-v2','status':'completed','actual_training_executed':True,
'candidate':'rnn64-ctc2601','near_percent':0,'job_recipe_sha256':os.environ['KWS_RESEARCH_RECIPE_SHA256'],
'final_checkpoint_sha256':sha(out/'final.pt'),'consumption_sha256':sha(out/'consumption.jsonl'),'fixture':True}
(out/'summary.json').write_text(json.dumps(value))
""")
            probe = root / "probe.py"
            probe.write_text("""import json,os,sys
from pathlib import Path
sys.path.insert(0, TRAINING)
import research_d1_evaluate as e
job=e.training.read_json(os.environ['KWS_RESEARCH_RECIPE'])
out,summary,pin=e.completed_training(job)
if any(name in sys.modules for name in ('torch','numpy','onnxruntime')): raise RuntimeError('ML import')
Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'],'verified.json').write_text(json.dumps(pin))
""".replace("TRAINING", repr(str(ROOT / "training"))))
            recipes_by_id = {"train": {"fixture": "no model or optimizer"}, "inspect": job()}
            definitions = []
            for name, value in recipes_by_id.items():
                path = root / (name + ".json")
                evaluation.training.io_contract.write_json(path, value)
                definitions.append({"job_id": name, "recipe": str(path), "recipe_sha256": evaluation.training.sha_file(path),
                    "argv": [sys.executable, "-I", "-S", "-B", str(producer if name == "train" else probe)],
                    "cwd": str(root), "depends_on": [] if name == "train" else ["train"], "timeout_minutes": 1,
                    "outputs": ["final.pt", "summary.json", "consumption.jsonl"] if name == "train" else ["verified.json"]})
            request = root / "request.json"
            evaluation.training.io_contract.write_json(request, {"schema": "kws-local-research-queue-v1",
                "queue_id": "d1-dev-dependency-fixture", "global_budget_minutes": 1, "jobs": definitions})
            queue = root / "queue"
            argv = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_queue.py")]
            environment = {key: value for key, value in os.environ.items() if not key.startswith("KWS_RESEARCH_")}
            for command in (argv + ["init", "--request", str(request), "--root", str(queue), "--heavy-lock", str(root / "dummy-heavy.lock")],
                            argv + ["run", "--root", str(queue)]):
                result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            dependencies = queue / "jobs/inspect/dependencies.json"
            output, summary, pin = evaluation.completed_training(job(), dependencies_path=dependencies)
            self.assertEqual(pin["job_id"], "train")
            with self.assertRaisesRegex(ValueError, "COMPLETED_TRAINING_SUMMARY"):
                evaluation.completed_training({**job(), "candidate": "c1-ctc2601"}, dependencies_path=dependencies)
            (output / "final.pt").write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "TRAINING_OUTPUT_CHANGED"):
                evaluation.completed_training(job(), dependencies_path=dependencies)

    def test_isolated_help_imports_no_machine_learning(self):
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / evaluation.SOURCE_PATH), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class NumericContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, cls.reference = evaluation.training.torch_runtime()

    def test_known_ctc_path_and_confidence_do_not_change_edit_denominator(self):
        torch = self.torch
        logits = torch.full((5, 1, 2601), -12.0)
        for frame, token in enumerate((1, 1, 0, 1, 2)):
            logits[frame, 0, token] = 12
        half = evaluation.score_logits(logits, {"target_ids": [1, 1, 2], "ctc_weight": .5})
        full = evaluation.score_logits(logits, {"target_ids": [1, 1, 2], "ctc_weight": 1})
        self.assertEqual((half["greedy_token_ids"], half["edit_numerator"], half["target_token_denominator"]), ([1, 1, 2], 0, 3))
        self.assertAlmostEqual(half["ctc_loss_confidence_weighted"], full["ctc_loss_confidence_weighted"] / 2)
        self.assertEqual(half["ctc_loss_target_length_normalized"], full["ctc_loss_target_length_normalized"])

    def test_full_evaluation_reads_sealed_preparation_and_leaves_checkpoint_unchanged(self):
        from research_job_lock import heavy_model_slot, PATH_ENV, SUPERVISED_ENV
        from training_state import atomic_save_checkpoint, load_checkpoint_snapshot
        torch = self.torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared, result = fixture.synthetic_preparation(root, torch)
            record, vocabulary, _ = evaluation.training.verify_prepared(prepared, result["manifest_sha256"])
            request = job()
            path = root / "evaluation.json"
            evaluation.training.io_contract.write_json(path, request)
            trained = root / "hypothetical-trained-checkpoint"
            trained.mkdir()
            initial, initial_sha = load_checkpoint_snapshot(prepared / "initial/rnn64-ctc2601.pt")
            recipe = evaluation.training.read_json(prepared / "recipes/0.json")
            # Hypothetical completed checkpoint metadata for software boundary
            # coverage. Its weights are untouched initialization; optimizer=0
            # in the test execution, with no real speech or training claim.
            payload = {"schema": "kws-d1-trained-checkpoint-v2", "candidate": request["candidate"], "near_percent": 0,
                "state_dict": initial["state_dict"], "float_state_identity": initial["float_state_identity"],
                "prepared_manifest_sha256": result["manifest_sha256"], "vocabulary_sha256": record["vocabulary_sha256"],
                "source_hashes": record["source_hashes"], "job_recipe_sha256": "a" * 64,
                "initial_checkpoint_sha256": initial_sha, "exposure_plan_sha256": recipe["exposure_plan_sha256"],
                "optimizer_steps": 150, "sample_exposures": 600, "seed": evaluation.training.SEED,
                "human_gold": False, "quality_winner": None, "fixture": "zero actual optimizer steps"}
            atomic_save_checkpoint(payload, trained / "final.pt")
            summary = {**{key: payload[key] for key in ("prepared_manifest_sha256", "vocabulary_sha256",
                "job_recipe_sha256", "initial_checkpoint_sha256", "exposure_plan_sha256", "optimizer_steps", "sample_exposures")},
                "final_checkpoint_sha256": evaluation.training.sha_file(trained / "final.pt"),
                "final_float_state_sha256": initial["float_state_identity"]["sha256"],
                "exposure_recipe_sha256": evaluation.training.sha_file(prepared / "recipes/0.json"),
                "description": initial["description"]}
            before = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                      for source in (prepared, trained) for p in source.rglob("*") if p.is_file()}
            lock = Path(os.environ.get(PATH_ENV, str(root / "fixture-heavy.lock")))
            with heavy_model_slot(lock) as lease:
                environment = lease.child_environment()
                environment.update({SUPERVISED_ENV: "1", "KWS_RESEARCH_RECIPE_SHA256": evaluation.training.sha_file(path)})
                with patch.dict(os.environ, environment), \
                        patch.object(evaluation.training.research_runtime, "check_interpreter"), \
                        patch.object(evaluation.training.research_runtime, "check_packages"), \
                        patch.object(evaluation.training, "resolve_prepared", return_value=(prepared, result["manifest_sha256"], {"fixture": True})), \
                        patch.object(evaluation, "completed_training", return_value=(trained, summary, {"fixture": True})):
                    value = evaluation.evaluate(path, root / "evaluation", lock)
            self.assertEqual(value["status"], "completed")
            self.assertEqual((value["optimizer_steps"], value["model_forward_calls"]), (0, 2))
            self.assertEqual(value["aggregate"]["evaluable_clips"], 2)
            self.assertEqual(len(value["by_source_family"]), 2)
            self.assertTrue(value["checkpoint_state_unchanged"])
            self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in before})
            observed = [json.loads(line) for line in (root / "evaluation/observations.jsonl").read_text().splitlines()]
            for row in observed:
                retained = torch.load(root / "evaluation" / row["numeric_artifact"]["path"], weights_only=True)
                self.assertEqual(evaluation.training.tensor_sha(retained["logits"]), row["logits_sha256"])
                self.assertEqual(evaluation.score_logits(retained["logits"], row)["edit_numerator"], row["edit_numerator"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DevelopmentContracts)
    if args.numeric:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NumericContracts))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
