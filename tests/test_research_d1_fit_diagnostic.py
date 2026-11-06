#!/usr/bin/env python3
"""Tiny-fit selection/receipt regressions; optional saved-logit tensor checks.

No real speech, downloaded weights or candidate model execution is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools"), str(ROOT / "tests")]
import research_d1_fit_diagnostic as fit
from test_research_d1 import raw_rows, vocabulary
import research_d1_recipe as recipes


def cohort():
    return recipes.cohort_rows(recipes.project_rows(raw_rows(), vocabulary()))


def job_for(rows):
    return {"schema": fit.SCHEMA, "prepared": "/fixture/not-read", "prepared_manifest_sha256": "a" * 64,
            "candidate": "rnn64-ctc2601", "source_ids": [row["source_id"] for row in rows],
            "row_binding_sha256": fit.digest([recipes.row_identity(row) for row in rows]),
            "optimizer_steps": 4, "timeout_seconds": 60,
            "diagnostic_source_hashes": dict.fromkeys(fit.SOURCE_PATHS, "b" * 64)}


def base_recipe():
    return {"fixed_ctc_normalizer": 1.0, "gradient_clip_max_norm": 5.0,
            "optimizers": {"fresh-rnn64-or-c1": {"name": "AdamW", "lr": .001, "weight_decay": .01},
                           "extended-fsmn-head": {"name": "AdamW", "lr": .0001, "weight_decay": 0.0}}}


def ledger_for(recipe, rows):
    ledger = []
    for step in range(recipe["optimizer_steps"]):
        batch = recipe["exposure_plan"]["draws"][step * 4:(step + 1) * 4]
        ledger.append({"optimizer_step": step + 1, "draw_offset": step * 4, "loss": 1.0,
            "gradient_norm_before_clip": 0.5 if step % 2 else 0.0,
            "draws": [{"source_id": rows[item["row_index"]]["source_id"],
                       "label_binding_sha256": recipes.row_identity(rows[item["row_index"]]),
                       "target_ids": rows[item["row_index"]]["target_ids"], "ctc_mask": True,
                       "ctc_weight": rows[item["row_index"]]["ctc_weight"], "keyword_loss_used": False}
                      for item in batch]})
    return ledger


class SelectionContracts(unittest.TestCase):
    def test_four_and_eight_are_deterministic_train_only_observed_strata(self):
        rows, vocab = cohort(), vocabulary()
        for count in (4, 8):
            selected = fit.choose_rows(rows, vocab, count)
            self.assertEqual(selected, fit.choose_rows(list(reversed(rows)), vocab, count))
            self.assertEqual(len({row["source_id"] for row in selected}), count)
            self.assertEqual([fit.selection_bucket(row, vocab) for row in selected], list(fit.BUCKETS) * (count // 4))
            self.assertEqual({row["split"] for row in selected}, {"train"})
            self.assertGreater(len({row["source_family_id"] for row in selected}), 1)
            self.assertEqual(fit.verify_selection(job_for(selected), rows, vocab), selected)

    def test_uncertain_and_development_cannot_enter_diagnostic(self):
        rows = raw_rows()
        rows[0].update(label_status="uncertain", ctc_mask=False, ctc_weight=0,
                       keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0])
        projected = recipes.project_rows(rows, vocabulary())
        for row in (projected[0], projected[192]):
            with self.assertRaisesRegex(ValueError, "D1_FIT_KNOWN_TRAINING_ROWS_ONLY"):
                fit.selection_bucket(row, vocabulary())
        selected = fit.choose_rows(recipes.cohort_rows(projected), vocabulary(), 4)
        self.assertNotIn(projected[0]["source_id"], [row["source_id"] for row in selected])

    def test_wrong_target_blank_or_weakened_keyword_mask_cannot_be_hidden(self):
        row = copy.deepcopy(fit.choose_rows(cohort(), vocabulary(), 4)[0])
        row["target_ids"][0] = 0
        with self.assertRaisesRegex(ValueError, "D1_FIT_FULLTEXT_TARGET_CHANGED"):
            fit.selection_bucket(row, vocabulary())
        row = copy.deepcopy(fit.choose_rows(cohort(), vocabulary(), 4)[0])
        row["keyword_mask"] = [True, False]
        with self.assertRaisesRegex(ValueError, "D1_FIT_KEYWORD_MASK_CHANGED"):
            fit.selection_bucket(row, vocabulary())

    def test_missing_partial_pool_never_silently_fills_with_other_samples(self):
        rows = [row for row in cohort() if row["generation_category"] != "partial-negative"]
        with self.assertRaisesRegex(ValueError, "D1_FIT_MISSING_KNOWN_POOL:partial-negative"):
            fit.choose_rows(rows, vocabulary(), 4)

    def test_declared_label_bindings_survive_paths_but_reject_label_changes(self):
        rows = cohort()
        selected = fit.choose_rows(rows, vocabulary(), 4)
        job = job_for(selected)
        selected[0]["audio_path"] = "/relocated/same-source.wav"
        self.assertEqual(len(fit.verify_selection(job, rows, vocabulary())), 4)
        selected[0]["ctc_weight"] *= .75
        with self.assertRaisesRegex(ValueError, "D1_FIT_SELECTED_LABELS_CHANGED"):
            fit.verify_selection(job, rows, vocabulary())

    def test_declared_ids_cannot_name_dev_unknown_or_duplicate_rows(self):
        rows = cohort()
        selected = fit.choose_rows(rows, vocabulary(), 4)
        job = job_for(selected)
        job["source_ids"][0] = "missing-development-or-unknown"
        with self.assertRaisesRegex(ValueError, "D1_FIT_SELECTION_OUTSIDE_TRAIN_COHORT"):
            fit.verify_selection(job, rows, vocabulary())
        job["source_ids"][0] = job["source_ids"][1]
        with self.assertRaisesRegex(ValueError, "D1_FIT_FOUR_OR_EIGHT_DISTINCT_SOURCES"):
            fit.validate_job(job)

    def test_schema_rejects_unbounded_steps_time_and_unfixed_subset(self):
        job = job_for(fit.choose_rows(cohort(), vocabulary(), 4))
        self.assertEqual(fit.validate_job(job), job)
        for key, values in (("optimizer_steps", (0, True, 2001, 1.5)),
                            ("timeout_seconds", (0, float("inf"), float("nan"), 601, True)),
                            ("source_ids", ([], job["source_ids"][:3], job["source_ids"] + ["extra"]))):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    fit.validate_job({**job, key: value})

    def test_same_consumer_and_settings_receive_exact_fixed_exposure_schedule(self):
        rows = fit.choose_rows(cohort(), vocabulary(), 8)
        base = base_recipe()
        recipe = fit.tiny_recipe(rows, base, 4)
        self.assertEqual(recipe["total_exposures"], 16)
        self.assertEqual(Counter(row["source_id"] for row in recipe["exposure_plan"]["draws"]),
                         dict.fromkeys([row["source_id"] for row in rows], 2))
        self.assertEqual(recipe["optimizers"], base["optimizers"])
        self.assertEqual(recipe["fixed_ctc_normalizer"], base["fixed_ctc_normalizer"])
        for draw in recipe["exposure_plan"]["draws"]:
            self.assertEqual(draw["ctc_weight"], rows[draw["row_index"]]["ctc_weight"])

    def test_zero_gradient_steps_are_reported_and_receipt_mismatches_rejected(self):
        rows = fit.choose_rows(cohort(), vocabulary(), 4)
        recipe = fit.tiny_recipe(rows, base_recipe(), 4)
        ledger = ledger_for(recipe, rows)
        summary = fit.gradient_observation(ledger, recipe, rows)
        self.assertEqual(summary["nonzero_gradient_steps"], 2)
        self.assertEqual(summary["zero_gradient_steps"], 2)
        with self.assertRaisesRegex(ValueError, "D1_FIT_INCOMPLETE_STEP_LEDGER"):
            fit.gradient_observation(ledger[:-1], recipe, rows)
        ledger[0]["draws"][0]["ctc_mask"] = False
        with self.assertRaisesRegex(ValueError, "D1_FIT_CONSUMED_LABEL_CHANGED"):
            fit.gradient_observation(ledger, recipe, rows)
        ledger = ledger_for(recipe, rows)
        ledger[1]["gradient_norm_before_clip"] = float("nan")
        with self.assertRaisesRegex(ValueError, "D1_FIT_NONFINITE_GRADIENT_OR_LOSS"):
            fit.gradient_observation(ledger, recipe, rows)


class EntryContracts(unittest.TestCase):
    def test_import_and_help_do_not_import_torch(self):
        code = ("import sys; sys.path.insert(0, " + repr(str(ROOT / "training")) + "); "
                "import research_d1_fit_diagnostic; assert 'torch' not in sys.modules; "
                "print('metadata-only')")
        imported = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(imported.stdout.strip(), "metadata-only")
        help_result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / fit.SOURCE_PATHS[0]), "--help"],
                                     capture_output=True, text=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("declare", help_result.stdout)


class NumericContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch, _ = fit.training.torch_runtime()

    def logits(self, *, frames=16):
        return self.torch.full((frames, 1, fit.OUTPUTS), -8.0)

    def test_blank_collapse_and_exact_target_are_distinct(self):
        row = fit.choose_rows(cohort(), vocabulary(), 4)[0]
        logits = self.logits()
        logits[:, 0, 0] = 5.0
        blank = fit.score(logits, row)
        self.assertEqual(blank["greedy_token_ids"], [])
        self.assertEqual(blank["blank_argmax_fraction"], 1.0)
        self.assertEqual({item["best_competition_rank"] for item in blank["target_tokens"]}, {2})
        for index, token in enumerate(row["target_ids"]):
            logits[index * 2 + 1, 0, token] = 10.0
        exact = fit.score(logits, row)
        self.assertTrue(exact["exact_token_match"])
        self.assertLess(exact["ctc_nll"], blank["ctc_nll"])
        self.assertEqual({item["best_competition_rank"] for item in exact["target_tokens"]}, {1})
        self.assertAlmostEqual(exact["blank_probability_mean"] + exact["target_probability_mass_mean"] +
                               exact["other_nonblank_probability_mass_mean"], 1, places=5)

    def test_constant_one_token_is_visible_even_when_edit_count_decreases(self):
        rows = fit.choose_rows(cohort(), vocabulary(), 4)
        token = rows[0]["target_ids"][0]
        observations = []
        for row in rows:
            logits = self.logits()
            logits[:, 0, token] = 5.0
            observations.append({**fit.score(logits, row), "ctc_weight": row["ctc_weight"]})
        summary = fit.summarize(observations)
        self.assertTrue(summary["constant_single_token_prediction"])
        self.assertEqual(summary["single_token_outputs"], 4)
        self.assertEqual(summary["exact_token_matches"], 0)
        self.assertEqual(summary["empty_outputs"], 0)

    def test_unknown_cannot_be_scored_as_an_empty_or_negative_label(self):
        row = copy.deepcopy(fit.choose_rows(cohort(), vocabulary(), 4)[0])
        row.update(ctc_mask=False, ctc_weight=0, target_ids=None)
        with self.assertRaisesRegex(ValueError, "D1_FIT_CANNOT_SCORE_UNKNOWN_LABEL"):
            fit.score(self.logits(), row)

    def test_parameter_observation_detects_no_update_nonfinite_and_frozen_changes(self):
        torch = self.torch
        initial = {"head": torch.zeros(2), "frozen": torch.ones(2)}
        changed, delta = fit.parameter_changes(initial, copy.deepcopy(initial), ["head"])
        self.assertEqual((changed, delta), ([], 0.0))
        final = copy.deepcopy(initial)
        final["head"][0] = 3.0
        self.assertEqual(fit.parameter_changes(initial, final, ["head"]), (["head"], 3.0))
        final["frozen"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "D1_FIT_FROZEN_PARAMETER_CHANGED"):
            fit.parameter_changes(initial, final, ["head"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric", action="store_true")
    args = parser.parse_args()
    cases = [SelectionContracts, EntryContracts]
    if args.numeric:
        cases.append(NumericContracts)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in cases)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
