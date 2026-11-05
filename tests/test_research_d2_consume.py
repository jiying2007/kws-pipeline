#!/usr/bin/env python3
"""D2 bridge contracts; fictional rows and verifier doubles, no model execution."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools"), str(ROOT / "tests")]
import research_d2_inputs as bridge
import research_d2_consume as probe
from test_research_d1 import raw_rows, vocabulary
from test_research_d1_fit_diagnostic import base_recipe


def row(index=0):
    value = copy.deepcopy(raw_rows()[index])
    value.update(consumer="d2-verified-full-transcript-v1", generation_outcome="completed")
    return value


def unknown(value, *, failed=False):
    value.update(label_status="uncertain", ctc_mask=False, ctc_weight=0.0,
                 keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0.0, 0.0])
    if failed:
        value.update(generation_outcome="failed-native", full_text=None)
    return value


class ProjectionTests(unittest.TestCase):
    def test_ordinary_negative_keeps_every_character(self):
        source = row(1)
        actual = bridge.project_row(source, vocabulary())
        self.assertEqual(actual["target_ids"], [vocabulary()["token_to_id"][c] for c in source["full_text"]])
        self.assertGreater(len(actual["target_ids"]), 4)
        self.assertTrue(actual["ctc_mask"])
        self.assertEqual(actual["observed_label"]["ctc_weight"], source["ctc_weight"])

    def test_fixed_extended_character_uses_2601_mapping(self):
        actual = bridge.project_row(row(5), vocabulary())
        self.assertIn(2600, actual["target_ids"])
        self.assertEqual(actual["full_text"], "请把灯打开")

    def test_oov_keeps_text_but_all_objectives_have_zero_weight(self):
        source = row(1)
        source["full_text"] += "𠮷"
        actual = bridge.project_row(source, vocabulary())
        self.assertEqual(actual["full_text"], source["full_text"])
        self.assertEqual(actual["oov_characters"], ["𠮷"])
        self.assertIsNone(actual["target_ids"])
        self.assertEqual((actual["ctc_mask"], actual["ctc_weight"], actual["keyword_weight"]), (False, 0, [0, 0]))
        self.assertGreater(actual["observed_label"]["ctc_weight"], 0)

    def test_uncertain_text_and_native_failure_stay_audit_only(self):
        for source in (unknown(row()), unknown(row(1), failed=True)):
            actual = bridge.project_row(source, vocabulary())
            self.assertIsNone(actual["target_ids"])
            self.assertFalse(actual["ctc_mask"])
            self.assertEqual(actual["keyword_targets"], [None, None])
            self.assertEqual(actual["full_text"], source["full_text"])

    def test_failed_native_cannot_be_promoted(self):
        source = row()
        source["generation_outcome"] = "failed-native"
        with self.assertRaisesRegex(ValueError, "FAILED_NATIVE_ZERO_LOSS"):
            bridge.project_row(source, vocabulary())

    def test_infeasible_ctc_is_not_padded_or_relabelled(self):
        source = row(1)
        source["samples"] = 400
        actual = bridge.project_row(source, vocabulary())
        self.assertEqual(actual["ctc_reason"], "unpadded_c1_ctc_geometry_infeasible")
        self.assertFalse(actual["ctc_mask"])
        self.assertEqual(actual["keyword_weight"], [0, 0])

    def test_d1_row_and_changed_role_are_rejected(self):
        for change in ({"consumer": "d1-verified-full-transcript-v2"}, {"source_role": "development-only"}):
            source = row(); source.update(change)
            with self.assertRaisesRegex(ValueError, "D2_ROW_SCOPE"):
                bridge.project_row(source, vocabulary())

    def test_probe_excludes_development_uncertain_and_oov(self):
        sources = [row(0), row(1), row(192), unknown(row(3))]
        sources[-1]["full_text"] = "保留未知观察"
        actual = [bridge.project_row(x, vocabulary()) for x in sources]
        selected = probe.rows_for_probe(actual)
        self.assertEqual([x["source_id"] for x in selected], [sources[0]["source_id"], sources[1]["source_id"]])
        recipe = probe.probe_recipe(selected, base_recipe())
        self.assertEqual((recipe["optimizer_steps"], len(recipe["exposure_plan"]["draws"])), (25, 100))
        self.assertEqual(recipe["optimizers"]["fresh-rnn64-or-c1"]["lr"], .001)


class Roles:
    def __init__(self):
        self.roles = {}

    def add(self, role, **identities):
        for kind, identity in identities.items():
            if identity is None:
                continue
            key = kind, identity
            if key in self.roles and self.roles[key] != role:
                raise ValueError("SOURCE_IDENTITY_SPLIT_LEAK:" + kind)
            self.roles[key] = role


class AggregateTests(unittest.TestCase):
    """Only orchestration is mocked; production always invokes verify_view."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root, self.replayed = Path(self.temp.name), []
        self.source_files = {"fixture-verifier": "a" * 64}

        def verify(path, **kwargs):
            self.replayed.append(path)
            self.assertEqual(kwargs["expected_manifest_sha256"], bridge.sha((path / "MANIFEST.json").read_bytes()))
            value = json.loads((path / "report.json").read_bytes())
            if getattr(self, "reject", False):
                raise ValueError("NATIVE_EVIDENCE_REJECTED")
            return value

        def add_role(roles, source, clip, measured):
            roles.add(source["role"], family=source["source_family_id"], speaker=source["speaker_id"],
                      parent_source_sha256=source["parent_source_sha256"])

        weak = SimpleNamespace(SourceRoles=Roles, add_source_role=add_role, relative=Path,
                               training_row=lambda raw, consumer: dict(raw["training"]))
        self.loader = SimpleNamespace(weak=weak, read_bytes=lambda path, *args: Path(path).read_bytes(),
                                      loads=json.loads, verify_view=verify)

    def package(self, name, index, *, source_id=None, family=None, producer=None, pcm=None):
        training = row(index)
        training["item_id"] = "clip-000"
        if source_id is not None:
            training["source_id"] = source_id
        source = {**training, "role": training["source_role"],
                  "parent_source_sha256": "parent-" + training["source_family_id"],
                  "reference_wav_sha256": "reference-wav", "reference_pcm_sha256": "reference-pcm"}
        if family is not None:
            source["source_family_id"] = family
        clip = {"id": "clip-000", "original_pcm": {"sha256": pcm or "pcm-" + str(index)}}
        raw = {"item_id": "clip-000", "source": source, "audio_derivation": {"pcm_sha256": "derived-" + str(index),
               "file_sha256": "file-" + str(index)}, "audio": "audio/a.wav", "training": training,
               "decision": {"status": training["label_status"], "reason": "fixture"}, "asr": []}
        report = {"producer": {"source_files_sha256": producer or self.source_files}, "items": [raw],
                  "source_adapter": {"stage": "existing-eight", "plan_sha256": "b" * 64}}
        root = self.root / name
        (root / "evidence").mkdir(parents=True)
        (root / "report.json").write_text(json.dumps(report))
        (root / "evidence/inputs.json").write_text(json.dumps({"clips": [clip]}))
        (root / "MANIFEST.json").write_text(json.dumps({"test-package": name}))
        return {"path": str(root), "manifest_sha256": bridge.sha((root / "MANIFEST.json").read_bytes())}

    def test_replays_every_package_and_local_ids_can_repeat(self):
        packages = [self.package("a", 0), self.package("b", 1)]
        rows, report = bridge.aggregate(packages, self.loader, self.source_files)
        self.assertEqual(len(self.replayed), 2)
        self.assertEqual([r["item_id"] for r in rows], ["clip-000", "clip-000"])
        self.assertEqual(report["observed_rows"], 2)

    def test_verifier_rejection_is_never_replaced_by_export(self):
        self.reject = True
        with self.assertRaisesRegex(ValueError, "NATIVE_EVIDENCE_REJECTED"):
            bridge.aggregate([self.package("a", 0)], self.loader, self.source_files)

    def test_duplicate_source_rejected_even_with_a_different_package(self):
        a = self.package("a", 0)
        b = self.package("b", 1, source_id=row()["source_id"])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_GLOBAL_SOURCE_ID"):
            bridge.aggregate([a, b], self.loader, self.source_files)

    def test_family_leakage_across_packages_is_rejected(self):
        a = self.package("a", 0)
        b = self.package("b", 192, family=row()["source_family_id"])
        with self.assertRaisesRegex(ValueError, "SPLIT_LEAK"):
            bridge.aggregate([a, b], self.loader, self.source_files)

    def test_prior_d1_native_pcm_and_source_ids_are_not_reconsumed(self):
        package = self.package("a", 0)
        for prior, reason in (({"source_id": row()["source_id"]}, "DUPLICATE_GLOBAL_SOURCE_ID"),
                              ({"original_pcm_sha256": "pcm-0"}, "DUPLICATE_NATIVE_PCM")):
            with self.assertRaisesRegex(ValueError, reason):
                bridge.aggregate([package], self.loader, self.source_files, prior_sources=[{"role": "train-candidate", **prior}])

    def test_changed_verifier_source_identity_rejected(self):
        package = self.package("a", 0, producer={"fixture-verifier": "b" * 64})
        with self.assertRaisesRegex(ValueError, "VERIFIER_DIFFERS_FROM_PRODUCER"):
            bridge.aggregate([package], self.loader, self.source_files)


class ReceiptTests(unittest.TestCase):
    def test_original_d1_verifier_failure_is_not_ignored(self):
        with patch.object(probe.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout=b"", stderr=b"D1_PREPARED_EXECUTION_SOURCE_CHANGED")):
            with self.assertRaisesRegex(ValueError, "D2_ORIGINAL_D1_VALIDATION_FAILED"):
                probe.anchor("/fixture", "a" * 64)

    def test_plain_help_imports_no_torch_and_constructs_no_model(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import research_d2_consume;assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code, str(ROOT / "training")], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_fulltext_and_weight_drift_in_actual_ledger_rejected(self):
        rows = [bridge.project_row(row(1), vocabulary())]
        recipe = probe.probe_recipe(rows, base_recipe())
        events = []
        for step in range(25):
            draws = []
            for _ in range(4):
                r = rows[0]
                draws.append({**{k: r[k] for k in ("source_id", "global_source_key", "source_family_id", "pcm_sha256",
                    "full_text", "target_ids", "ctc_mask", "ctc_weight", "keyword_targets", "keyword_mask", "keyword_weight",
                    "generation_category")}, "label_binding_sha256": probe.row_identity(r), "keyword_loss_used": False,
                    "raw_length_normalized_ctc": 2.0, "weighted_batch_contribution": 2.0 * r["ctc_weight"] / 4})
            events.append({"optimizer_step": step + 1, "draw_offset": step * 4, "loss": 2.0 * rows[0]["ctc_weight"],
                           "gradient_norm_before_clip": 1.0, "draws": draws})
        self.assertEqual(probe.ledger_readback(events, recipe, rows)["nonzero_gradient_steps"], 25)
        broken = copy.deepcopy(events); broken[0]["draws"][0]["full_text"] = "小窝"
        with self.assertRaisesRegex(ValueError, "D2_CONSUMED_FULL_LABEL_CHANGED"):
            probe.ledger_readback(broken, recipe, rows)
        broken = copy.deepcopy(events); broken[0]["draws"][0]["weighted_batch_contribution"] = 0
        with self.assertRaisesRegex(ValueError, "D2_WEIGHTED_LOSS_CHANGED"):
            probe.ledger_readback(broken, recipe, rows)


if __name__ == "__main__":
    unittest.main()
