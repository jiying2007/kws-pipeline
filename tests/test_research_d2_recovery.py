#!/usr/bin/env python3
"""Offline counterexamples for lost-history accounting and persistent barriers."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_d2_recovery as recovery
import research_d2_checkpoints as checkpoints

q = recovery.q


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def archive(self, name, originals):
        root = self.root / name
        root.mkdir()
        files = {}
        for index, (original, raw) in enumerate(originals.items()):
            relative = f"files/{index:03d}.json"
            path = root / relative
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(raw)
            files[relative] = {**q.file_record(path), "source_path": original, "category": "explicit-fixture"}
        q.write_json(root / "CHECKPOINT_SCOPE.json", {"fixture": True})
        files["CHECKPOINT_SCOPE.json"] = {**q.file_record(root / "CHECKPOINT_SCOPE.json"), "source_path": None}
        q.write_json(root / "MANIFEST.json", {"schema": "explicit-fixture", "hash_algorithm": "SHA-256",
            "files": files, "self_excluded": ["MANIFEST.json", "SHA256SUMS"]})
        (root / "SHA256SUMS").write_text("explicit fixture: the pinned manifest is the trust anchor\n")
        return {"root": str(root), "manifest_sha256": q.file_record(root / "MANIFEST.json")["sha256"]}

    def test_missing_claims_in_archived_subset_are_unknown_not_pending(self):
        plan = {"cells": [{"cell_id": f"cell-{i}"} for i in range(3)],
                "d1_parent": {"plan_sha256": "a" * 64}, "stage": "existing-eight", "batch_id": "original"}
        plan_raw = q.canonical(plan)
        stage = {"prepared": "/old/stage", "plan": {"bytes": len(plan_raw), "sha256": q.sha(plan_raw)},
                 "chunks": [[f"cell-{i}"] for i in range(3)], "queue_root": "/old/stage/queue", "automatic_retries": 0}
        stage_raw = q.canonical(stage)
        campaign = {"root": "/old/campaign", "schema": "kws-d2-campaign-v1", "automatic_retries": 0,
                    "model_concurrency": 1, "stage": {"path": "/old/stage/stage.json", "bytes": len(stage_raw), "sha256": q.sha(stage_raw)},
                    "generation_queue": {"path": "/old/stage/queue", "queue_sha256": "b" * 64},
                    "job_ids": [f"generate-{i + 1:03d}" for i in range(3)], "max_chunks": 3}
        anchor = self.archive("snapshot", {"/old/stage/plan.json": plan_raw,
            "/old/stage/stage.json": stage_raw, "/old/campaign/campaign.json": q.canonical(campaign)})
        result = recovery.reconcile_campaign(recovery.Checkpoints([anchor]), "/old/campaign/campaign.json")
        self.assertEqual(result["post_checkpoint_outcome_unknown_cell_ids"], ["cell-0", "cell-1", "cell-2"])
        self.assertEqual(result["archived_completed_cells"], 0)
        self.assertIsNone(result["post_checkpoint_pending_count"])
        self.assertIsNone(result["actual_post_checkpoint_runner_nanoseconds"])
        self.assertFalse(result["resubmission_allowed"])

    def test_two_individually_valid_archives_cannot_disagree_about_original_bytes(self):
        a = self.archive("a", {"/old/receipt.json": b"original\n"})
        b = self.archive("b", {"/old/receipt.json": b"replacement\n"})
        with self.assertRaisesRegex(ValueError, "CONFLICTING_ORIGINAL_BYTES"):
            recovery.Checkpoints([a, b])

    def test_pinned_manifest_rejects_changed_or_extra_files(self):
        anchor = self.archive("a", {"/old/receipt.json": b"original\n"})
        path = Path(anchor["root"]) / "files/000.json"
        path.write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "FILE_CHANGED"):
            recovery.Checkpoints([anchor])
        path.write_bytes(b"original\n")
        (Path(anchor["root"]) / "untracked.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "FILE_SET_CHANGED"):
            recovery.Checkpoints([anchor])

    def test_guard_rejects_original_identity_and_overlapping_cells(self):
        plan = {"cells": [{"cell_id": "original-cell"}], "d1_parent": {"plan_sha256": "a" * 64},
                "stage": "existing-eight", "batch_id": "original"}
        loss = self.root / "loss.json"
        q.write_json(loss, {"schema": recovery.SCHEMA, "resubmission_allowed": False,
            "campaigns": [{"declared_cell_ids": ["original-cell"], "campaign_identity": recovery.campaign_identity(plan)}]})
        pin = recovery.pin(loss)
        for change in ({}, {"batch_id": "renamed"}, {"cells": [{"cell_id": "renamed-cell"}]}):
            with self.assertRaisesRegex(ValueError, "RESUBMISSION_FORBIDDEN"):
                recovery.guard_plan({**plan, **change}, [pin])
        fresh = {**plan, "batch_id": "explicit-new-experiment", "cells": [{"cell_id": "new-cell"}]}
        recovery.guard_plan(fresh, [pin])
        loss.write_text("{}")
        with self.assertRaisesRegex(ValueError, "LOSS_PIN_CHANGED"):
            recovery.guard_plan(fresh, [pin])


class CapsuleTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "recipe.json"
        q.write_json(self.source, {"fixture": True, "allowed_calls": 1})

    def save(self, folder, descriptor):
        receipt = folder / "test-save.json"
        q.write_json(receipt, {"schema": checkpoints.SAVE, "status": "saved", "provider": "explicit-fixture",
            "object_id": "fixture-object-1", "provider_response": {"saved_id": "fixture-object-1"},
            "artifact": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"]})
        checkpoints.acknowledge(folder, receipt)

    def test_generic_intent_checks_actual_live_sources_after_external_ack(self):
        def save_then_change(folder, descriptor):
            self.save(folder, descriptor)
            self.source.write_text('{"fixture":"changed"}\n')
        gate = checkpoints.Gate(max_wait_seconds=0, on_wait=save_then_change)
        with self.assertRaisesRegex(ValueError, "LIVE_SOURCE_CHANGED"):
            gate.checkpoint(self.root / "intent", {"job_recipe": checkpoints.pin(self.source)}, [self.source], intent=True)

    def test_copied_intent_is_not_a_new_invocation_permission(self):
        folder = self.root / "intent"
        scope = {"job_recipe": checkpoints.pin(self.source)}
        checkpoints.Gate(max_wait_seconds=0, on_wait=self.save).checkpoint(folder, scope, [self.source], intent=True)
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
            checkpoints.Gate(max_wait_seconds=0, on_wait=self.save).checkpoint(folder, scope, [self.source], intent=True)
        self.assertFalse(checkpoints.verify_ack(folder)["remote_bytes_independently_retrieved"])

    def test_interrupted_ack_resumes_identical_receipt_without_granting_new_intent(self):
        folder = self.root / "intent"
        scope = {"job_recipe": checkpoints.pin(self.source)}
        original_write = q.write_json
        def stop_before_ack(path, value):
            if Path(path).name == "ack.json":
                raise RuntimeError("fixture crash after provider receipt")
            return original_write(path, value)
        gate = checkpoints.Gate(max_wait_seconds=0, on_wait=self.save)
        with mock.patch.object(q, "write_json", side_effect=stop_before_ack):
            with self.assertRaisesRegex(RuntimeError, "fixture crash"):
                gate.checkpoint(folder, scope, [self.source], intent=True)
        self.assertTrue((folder / "provider-save.json").exists())
        self.assertFalse((folder / "ack.json").exists())
        original = (folder / "provider-save.json").read_bytes()
        receipt = folder / "test-save.json"
        ack = checkpoints.acknowledge(folder, receipt)
        self.assertEqual(checkpoints.acknowledge(folder, receipt), ack)
        self.assertEqual((folder / "provider-save.json").read_bytes(), original)
        changed = checkpoints.read(receipt)
        changed["object_id"] = "different-fixture-object"
        changed["provider_response"] = {"saved_id": changed["object_id"]}
        receipt.write_bytes(q.canonical(changed))
        with self.assertRaisesRegex(ValueError, "PROVIDER_SAVE_CONFLICT"):
            checkpoints.acknowledge(folder, receipt)
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
            checkpoints.Gate(max_wait_seconds=0).checkpoint(folder, scope, [self.source], intent=True)

    def test_wrong_provider_response_and_changed_capsule_are_rejected(self):
        folder = self.root / "complete"
        descriptor = checkpoints.create_capsule(folder, {"fixture": True}, [self.source])
        receipt = folder / "bad-save.json"
        q.write_json(receipt, {"schema": checkpoints.SAVE, "status": "saved", "provider": "explicit-fixture",
            "object_id": "fictional-id", "provider_response": {"error": "upload failed"},
            "artifact": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"]})
        with self.assertRaisesRegex(ValueError, "PROVIDER_SAVE_RECEIPT_REQUIRED"):
            checkpoints.acknowledge(folder, receipt)
        failed = checkpoints.read(receipt)
        failed["provider_response"] = {"file_id": "fictional-id", "status": "failed"}
        receipt.write_bytes(q.canonical(failed))
        with self.assertRaisesRegex(ValueError, "PROVIDER_SAVE_RECEIPT_REQUIRED"):
            checkpoints.acknowledge(folder, receipt)
        self.save(folder, descriptor)
        with Path(descriptor["capsule"]["path"]).open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "CAPSULE_CHANGED"):
            checkpoints.verify_ack(folder)

    def test_actual_zip_restore_preserves_intent_and_does_not_grant_a_second_launch(self):
        work = self.root / "lost-workspace"
        work.mkdir()
        recipe = work / "recipe.json"
        q.write_json(recipe, {"fixture": True, "allowed_calls": 1})
        scope = {"job_recipe": checkpoints.pin(recipe)}
        folder = work / "intent"
        checkpoints.Gate(max_wait_seconds=0, on_wait=self.save).checkpoint(folder, scope, [recipe], intent=True)
        descriptor = checkpoints.read(folder / "capsule.json")
        downloaded = self.root / "downloaded.zip"
        shutil.copyfile(descriptor["capsule"]["path"], downloaded)
        shutil.rmtree(work)
        result = checkpoints.restore_capsule(downloaded, descriptor["capsule"]["sha256"], [work])
        self.assertEqual(result["new_model_calls"], 0)
        self.assertFalse((folder / "ack.json").exists())
        self.assertEqual(checkpoints.read(folder / "capsule.json"), descriptor)
        with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
            checkpoints.Gate(max_wait_seconds=0, on_wait=self.save).checkpoint(folder, scope, [recipe], intent=True)

    def test_restore_rejects_out_of_scope_paths_before_creating_files(self):
        folder = self.root / "complete"
        descriptor = checkpoints.create_capsule(folder, {"fixture": True}, [self.source])
        destination = self.root / "unrelated-root"
        with self.assertRaisesRegex(ValueError, "OUTSIDE_ROOT"):
            checkpoints.restore_capsule(Path(descriptor["capsule"]["path"]), descriptor["capsule"]["sha256"], [destination])
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
