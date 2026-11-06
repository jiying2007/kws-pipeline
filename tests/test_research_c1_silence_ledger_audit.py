#!/usr/bin/env python3
"""Invented receipts only; no checkpoint load, tensor import or model call."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_c1_silence_ledger_audit as supplement


def fixture():
    draws, events = [], []
    for step in range(supplement.STEPS):
        items = []
        for draw_index in range(step * 4, (step + 1) * 4):
            offset = (0, 200, 500)[(draw_index // 8) % 3]
            draw = {"source_id": "invented-%d" % (draw_index % 8), "ctc_weight": 0.5,
                "offset_ms": offset, "row_visit_index": draw_index // 8,
                "known_zero_prefix_frames": {0: 0, 200: 9, 500: 24}[offset]}
            draws.append(draw)
            auxiliary = 1.0 if offset else 0.0
            items.append({**draw, "ctc_mask": True, "auxiliary_scale": 1.0,
                "raw_length_normalized_ctc": 2.0, "aux_blank_nll_mean": auxiliary,
                "weighted_batch_contribution": 0.25, "aux_batch_contribution": auxiliary / 4,
                "total_batch_contribution": 0.25 + auxiliary / 4})
        events.append({"optimizer_step": step + 1, "draw_offset": step * 4, "draws": items,
            "fulltext_ctc_batch_loss": 1.0, "aux_batch_loss": auxiliary,
            "loss": 1.0 + auxiliary})
    recipe = {"schema": supplement.EXECUTION_SCHEMA, "optimizer_steps": 2000, "batch_size": 4,
        "fixed_ctc_normalizer": 1.0, "auxiliary_scale": 1.0, "exposure_plan": {"draws": draws},
        "source_hashes": {supplement.EXECUTION_SOURCE: supplement.EXECUTION_SHA}}
    return recipe, events


class ContributionTests(unittest.TestCase):
    def setUp(self):
        self.recipe, self.events = fixture()

    def test_all_three_offsets_and_full_denominator_are_checked(self):
        result = supplement.check_ledger(self.events, self.recipe)
        self.assertEqual(result["optimizer_steps_checked"], 2000)
        self.assertEqual(result["draw_contributions_checked"], 8000)
        self.assertTrue(result["per_draw_total_equals_primary_plus_auxiliary"])

    def test_corrupt_row_totals_cannot_cancel_within_a_correct_batch(self):
        items = self.events[-1]["draws"]
        before = sum(row["total_batch_contribution"] for row in items)
        items[0]["total_batch_contribution"] += 0.125
        items[1]["total_batch_contribution"] -= 0.125
        self.assertEqual(sum(row["total_batch_contribution"] for row in items), before)
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_DRAW_TOTAL_CHANGED"):
            supplement.check_ledger(self.events, self.recipe)

    def test_primary_and_auxiliary_cannot_cancel_within_a_correct_row(self):
        item = self.events[2]["draws"][0]
        item["weighted_batch_contribution"] += 0.125
        item["aux_batch_contribution"] -= 0.125
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_PRIMARY_CONTRIBUTION_CHANGED"):
            supplement.check_ledger(self.events, self.recipe)

    def test_missing_draws_and_wrong_order_are_rejected(self):
        for change in (lambda: self.events.pop(),
                lambda: self.events[0]["draws"].pop(),
                lambda: self.events[0].update(optimizer_step=2),
                lambda: self.events[0]["draws"][0].update(source_id="different")):
            self.recipe, self.events = fixture()
            change()
            with self.assertRaises(ValueError):
                supplement.check_ledger(self.events, self.recipe)

    def test_nonfinite_and_boolean_values_are_not_losses(self):
        for invalid in (float("nan"), float("inf"), True):
            with self.subTest(value=invalid):
                self.events[0]["draws"][0]["total_batch_contribution"] = invalid
                with self.assertRaisesRegex(ValueError, "SUPPLEMENT_FINITE_NUMBER_REQUIRED"):
                    supplement.check_ledger(self.events, self.recipe)

    def test_changed_batch_total_and_loss_scale_are_rejected(self):
        self.events[0]["loss"] += 0.125
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_BATCH_TOTAL_CHANGED"):
            supplement.check_ledger(self.events, self.recipe)
        self.events[0]["loss"] -= 0.125
        self.recipe["auxiliary_scale"] = 0.5
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_FIXED_LOSS_CONTRACT_CHANGED"):
            supplement.check_ledger(self.events, self.recipe)


class SealedBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.original = self.root / "original-audit.json"
        recipe, events = fixture()
        source = self.output / "execution-source" / supplement.EXECUTION_SOURCE
        source.parent.mkdir(parents=True)
        source.write_bytes((ROOT / supplement.EXECUTION_SOURCE).read_bytes())
        self.assertEqual(supplement.io.file_record(source)["sha256"], supplement.EXECUTION_SHA)
        self.write_member("request.json", recipe)
        summary = {"schema": supplement.EXECUTION_SCHEMA, "status": "completed",
            "recipe_sha256": supplement.io.file_record(self.output / "request.json")["sha256"]}
        self.write_member("summary.json", summary)
        (self.output / "consumption.jsonl").write_bytes(b"".join(supplement.io.canonical(row) for row in events))
        self.seal()

    def write_member(self, name, value):
        (self.output / name).write_bytes(supplement.io.canonical(value))

    def seal(self):
        manifest = {p.relative_to(self.output).as_posix(): supplement.io.file_record(p)
            for p in sorted(self.output.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
        self.write_member("MANIFEST.json", manifest)
        self.manifest_sha = supplement.io.file_record(self.output / "MANIFEST.json")["sha256"]
        self.prior = {"schema": supplement.EXECUTION_SCHEMA, "status": "verified", "manifest_sha256": self.manifest_sha,
            "saved_optimizer_steps": 2000, "saved_training_forwards": 8000, "saved_final_conditions": 24,
            "new_model_constructions": 0, "new_model_forwards": 0, "new_optimizer_steps": 0}
        self.save_prior()

    def save_prior(self):
        self.original.write_bytes(supplement.io.canonical(self.prior))
        self.audit_sha = supplement.io.file_record(self.original)["sha256"]

    def audit(self):
        return supplement.audit(self.output, self.manifest_sha, self.original, self.audit_sha)

    def test_valid_receipt_separates_frozen_and_supplement_source_identities(self):
        before = supplement.io.file_record(self.output / "MANIFEST.json")
        result = self.audit()
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["frozen_execution_source"]["recorded_commit"], supplement.FROZEN_SOURCE_COMMIT)
        self.assertEqual(result["supplement_source_files"][supplement.SOURCE], supplement.io.file_record(ROOT / supplement.SOURCE))
        self.assertEqual(result["new_model_forwards"], 0)
        self.assertEqual(before, supplement.io.file_record(self.output / "MANIFEST.json"))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("research_c1_known_silence", sys.modules)

    def test_manifest_pin_and_bound_member_tampering_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_VERIFIED_ORIGINAL_AUDIT_REQUIRED"):
            supplement.audit(self.output, "0" * 64, self.original, self.audit_sha)
        with (self.output / "consumption.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_FILE_CHANGED:consumption.jsonl"):
            self.audit()

    def test_changed_or_unverified_original_audit_cannot_be_reused(self):
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_ORIGINAL_AUDIT_CHANGED"):
            supplement.audit(self.output, self.manifest_sha, self.original, "0" * 64)
        for field, value in (("status", "failed"), ("manifest_sha256", "0" * 64),
                ("saved_training_forwards", 7999), ("new_model_forwards", False)):
            prior = copy.deepcopy(self.prior)
            self.prior[field] = value
            self.save_prior()
            with self.assertRaisesRegex(ValueError, "SUPPLEMENT_VERIFIED_ORIGINAL_AUDIT_REQUIRED"):
                self.audit()
            self.prior = prior

    def test_resealed_arithmetic_counterexample_still_fails(self):
        path = self.output / "consumption.jsonl"
        rows = [supplement.io.loads(line) for line in path.read_bytes().splitlines()]
        rows[0]["draws"][0]["total_batch_contribution"] += 0.125
        rows[0]["draws"][1]["total_batch_contribution"] -= 0.125
        path.write_bytes(b"".join(supplement.io.canonical(row) for row in rows))
        self.seal()
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_DRAW_TOTAL_CHANGED"):
            self.audit()

    def test_changed_execution_source_is_not_the_frozen_experiment(self):
        path = self.output / "execution-source" / supplement.EXECUTION_SOURCE
        path.write_bytes(b"invented different execution source\n")
        recipe = supplement.io.loads((self.output / "request.json").read_bytes())
        recipe["source_hashes"][supplement.EXECUTION_SOURCE] = supplement.io.file_record(path)["sha256"]
        self.write_member("request.json", recipe)
        summary = supplement.io.loads((self.output / "summary.json").read_bytes())
        summary["recipe_sha256"] = supplement.io.file_record(self.output / "request.json")["sha256"]
        self.write_member("summary.json", summary)
        self.seal()
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_FROZEN_EXECUTION_SOURCE_CHANGED"):
            self.audit()

    def test_receipt_cannot_rewrite_the_sealed_bundle(self):
        receipt = self.output / "supplement.json"
        with self.assertRaisesRegex(ValueError, "SUPPLEMENT_RECEIPT_MUST_BE_OUTSIDE_SEALED_OUTPUT"):
            supplement.main(["--output", str(self.output), "--manifest-sha256", self.manifest_sha,
                "--original-audit", str(self.original), "--original-audit-sha256", self.audit_sha,
                "--receipt", str(receipt)])
        self.assertFalse(receipt.exists())


if __name__ == "__main__":
    unittest.main()
