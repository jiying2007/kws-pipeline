#!/usr/bin/env python3
"""Exact mixed closure and retained-reader admission; no model execution."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools"), str(ROOT / "tests")]
import research_context_v3_consume as probe
from test_research_context_v3_consume import fixture_plan, fixture_rows
import test_research_context_v3_consume as original_tests

q, cp, mixed = probe.training.io_contract, probe.checkpoints, probe.handoff_inputs
DATA = os.environ.get("KWS_CONTEXT_METADATA_DATA_ROOT")


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(q.canonical(value) + b"\n")
    return probe.pin(path)


def fixture_reader(root, revision):
    for name in mixed.READERS[revision]["files"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# explicit reader identity fixture: " + name + "\n")
    return {"revision": revision, "root": str(root),
        "source_files_sha256": {name: probe.training.sha_file(root / name) for name in mixed.READERS[revision]["files"]}}


def relation():
    return {"schema": mixed.CONTAINER_POLICY, "relation": "peak-timestamp-plus-one-second",
        "returned_wav": {"bytes": 96080, "sha256": "a" * 64},
        "bulk_generation_wav": {"bytes": 96080, "sha256": "b" * 64},
        "native_array": {"bytes": 96128, "sha256": "c" * 64},
        "native_float32_pcm_sha256": "d" * 64, "native_frames": 24000, "native_rate": 24000,
        "bulk_peak_timestamp": 1001, "returned_peak_timestamp": 1000, "differing_byte_offsets": [60],
        "all_other_container_bytes_equal": True, "original_native_array_to_pcm_join_verified": True,
        "signal_samples_changed": False, "original_files_rewritten": False,
        "technical_gate_changed": False, "generation_outcome_changed": False}


def closure_fixture(root):
    """A real fresh status subprocess plus genuinely authenticated fixture ZIPs."""
    jobs = [f"generate-{i:03d}" for i in range(1, 10)]
    old, new = root / "original", root / "continuation"
    old.mkdir(); new.mkdir()
    script = root / "owner/tools/research_context_v3_metadata_handoff.py"
    script.parent.mkdir(parents=True)
    script.write_text("""# Explicit test-only status validator. No model or queue imports.
import json,sys
from pathlib import Path
if sys.argv[1:3] != ['status','--handoff']: raise SystemExit(2)
print((Path(sys.argv[3]).parent/'fixture-status.json').read_text())
""")
    original_validator = root / "old-owner/tools/research_context_v3_campaign.py"
    original_validator.parent.mkdir(parents=True)
    original_validator.write_text("# explicit original validator pin fixture\n")
    stage = put(root / "stage.json", {"chunks": [[f"cell-{i}-{n}" for n in range(8)] for i in range(9)]})
    driver = {"executable": sys.executable, "python": "fixture-status-only"}
    campaign = put(old / "campaign.json", {"root": str(old), "stage": stage,
        "driver_runtime": driver, "asr_runtime": str(root / "asr")})
    readers = {revision: fixture_reader(root / revision, revision) for revision in mixed.READERS}
    value = {"schema": probe.HANDOFF, "root": str(new), "parent_campaign": campaign,
        "original_validator": probe.pin(original_validator), "original_stage": stage,
        "pipeline_files": {"tools/research_context_v3_metadata_handoff.py": q.file_record(script)},
        "driver_runtime": driver, "heavy_lock": str(root / "asr/heavy-model.lock"),
        "original_completed_chunks": jobs[:4], "continuation_chunks": jobs[4:],
        "read_only_generation_jobs": jobs[4:5], "new_generation_jobs": jobs[5:],
        "total_actual_runner_minutes": 180, "new_runner_allocation_minutes": 90,
        "automatic_retries": 0, "model_concurrency": 1,
        "original_reader": readers["context-v3-r1"], "metadata_reader": readers["context-v3-metadata-r2"]}
    handoff = put(new / "handoff.json", value)
    request = {"schema": probe.HANDOFF_PREP, "campaign": campaign,
        "campaign_validator": probe.pin(original_validator), "handoff": handoff,
        "handoff_validator": probe.pin(script)}
    gate = cp.Gate(max_wait_seconds=0, on_wait=original_tests.QueueOwnerTests.save_fixture)
    packages = []
    for i, job in enumerate(jobs):
        base = old if i < 4 else new
        chunk = base / "chunks" / job
        complete = put(chunk / "complete.json", {"explicit_closure_fixture": True, "job_id": job})
        asr_request = put(chunk / "asr-request.json", {"explicit_asr_fixture": True})
        asr_queue = put(chunk / "asr-queue/queue.json", {"explicit_queue_fixture": True})
        put(chunk / "asr-queue/jobs/asr-selective/outputs/asr/claims/qwen06/cell.json",
            {"started_at": "2026-11-06T01:05:00+00:00"})
        view = chunk / "weak-view"
        package = put(view / "MANIFEST.json", {"explicit_weak_fixture": True, "job_id": job})
        reader = readers["context-v3-r1" if i < 4 else "context-v3-metadata-r2"]
        packages.append({"path": str(view), "manifest_sha256": package["sha256"], "reader": reader})
        if i < 4:
            scope = {"schema": "kws-context-v3-campaign-phase-v1", "phase": "chunk-complete", "campaign": campaign,
                "stage": stage, "job_id": job, "cell_ids": [f"cell-{i}-{n}" for n in range(8)],
                "max_tts_calls": 8, "max_asr_calls": 24, "automatic_retries": 0,
                "resume_pending_intent_after_restart": False, "asr_request": asr_request,
                "asr_queue": asr_queue, "completion": complete}
        else:
            scope = {"schema": probe.HANDOFF, "phase": "chunk-complete", "handoff": handoff,
                "parent_campaign": campaign, "stage": stage, "job_id": job, "reader": reader,
                "max_asr_calls": 24, "automatic_retries": 0, "pending_intent_restart_allowed": False,
                "asr_request": asr_request, "asr_queue": asr_queue, "completion": complete}
        gate.checkpoint(base / "checkpoints" / job / "chunk-complete", scope, [complete["path"]], intent=False)
    budget = {"unresolved_jobs": [], "failed_jobs": [], "actual_total_is_complete": True,
        "overrun_nanoseconds": 0, "new_overrun_nanoseconds": 0, "tts_claims_consumed": 72,
        "asr_claims_consumed": 156, "asr_results_saved": 156, "new_tts_claims": 32, "new_asr_claims": 85}
    final = {"schema": probe.HANDOFF, "status": "completed", "handoff": handoff,
        "completed_chunks": jobs, "budget": budget, "packages": packages, "human_gold": False,
        "automatic_retries": 0, "original_prefix_modified": False}
    final_pin = put(new / "complete.json", final)
    folder = new / "checkpoints/handoff-complete"
    gate.checkpoint(folder, {"schema": probe.HANDOFF, "phase": "handoff-complete", "handoff": handoff,
        "completion": final_pin}, [new / "complete.json"], intent=False)
    status = {"schema": probe.HANDOFF, "handoff": handoff, "completed_chunks": jobs,
        "driver_active": False, "budget": budget, "packages": packages, "new_model_calls": 0,
        "completion": final_pin, "completion_checkpoint": str(folder)}
    put(new / "fixture-status.json", status)
    return request, status, readers


class HandoffAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_new_request_is_explicit_and_cannot_mix_a_single_loader_with_two_readers(self):
        source = put(self.root / "pin.json", {"fixture": True})
        request = {"schema": probe.HANDOFF_PREP, "anchor": str(self.root), "anchor_manifest_sha256": "a" * 64,
            "selection_checkpoint": str(self.root), **{key: source for key in
            ("plan", "selection", "campaign", "campaign_validator", "handoff", "handoff_validator")}}
        self.assertEqual(probe.validate_request(request), request)
        for change in ({"view_loader": str(self.root / "reader.py")}, {"schema": probe.PREP}):
            with self.assertRaisesRegex(ValueError, "PREPARATION_REQUEST"):
                probe.validate_request({**request, **change})
        missing = dict(request); missing.pop("handoff_validator")
        with self.assertRaisesRegex(ValueError, "PREPARATION_REQUEST"):
            probe.validate_request(missing)

    def test_real_status_subprocess_authenticates_all_nine_and_final_scopes(self):
        request, status, _readers = closure_fixture(self.root)
        with mock.patch.object(probe.subprocess, "run", wraps=subprocess.run) as run:
            result = probe.handoff_closure(request)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-3:], ["status", "--handoff", request["handoff"]["path"]])
        self.assertEqual(result["status"], status)
        self.assertEqual(len(result["closures"]), 9)
        self.assertEqual([p["reader"]["revision"] for p in result["packages"]],
            ["context-v3-r1"] * 4 + ["context-v3-metadata-r2"] * 5)
        self.assertEqual(result["new_model_calls"], 0)

    def test_active_unknown_failed_overrun_or_partial_status_is_not_admitted(self):
        request, original, _ = closure_fixture(self.root)
        changes = [lambda s: s.update(driver_active=True), lambda s: s["completed_chunks"].pop(),
            lambda s: s["budget"].update(unresolved_jobs=["fixture-unknown"]),
            lambda s: s["budget"].update(failed_jobs=["fixture-failed"]),
            lambda s: s["budget"].update(overrun_nanoseconds=1),
            lambda s: s["budget"].update(new_overrun_nanoseconds=1),
            lambda s: s["budget"].update(tts_claims_consumed=71),
            lambda s: s["budget"].update(asr_results_saved=155)]
        for change in changes:
            status = copy.deepcopy(original); change(status)
            put(self.root / "continuation/fixture-status.json", status)
            with self.assertRaisesRegex(ValueError, "NOT_TERMINAL_AND_CLOSED"):
                probe.handoff_closure(request)

    def test_a_real_ack_for_another_completion_cannot_authorize_this_scope(self):
        request, _status, _ = closure_fixture(self.root)
        folder = self.root / "continuation/checkpoints/generate-005/chunk-complete"
        context = q.loads((folder / "capsule.json").read_bytes())["context"]
        expected = {k: v for k, v in context.items() if k not in ("owner_invocation", "launch_intent")}
        expected["job_id"] = "generate-006"
        with self.assertRaisesRegex(ValueError, "CLOSED_ACK_SCOPE_CHANGED"):
            probe.verify_closed_scope(folder, expected)
        request["handoff_validator"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "PIN_CHANGED"):
            probe.handoff_closure(request)

    def test_both_reader_snapshots_are_exact_and_old_common_source_cannot_drift(self):
        readers = {revision: fixture_reader(self.root / "live" / revision, revision) for revision in mixed.READERS}
        records = {revision: {"source_root": reader["root"], "source_files_sha256": reader["source_files_sha256"],
            "retained_relative_root": "data-sources/" + revision} for revision, reader in readers.items()}
        data = {"schema": probe.HANDOFF_DATA, "readers": records}
        prepared = self.root / "prepared"; prepared.mkdir()
        probe.copy_data_sources(prepared, data)
        probe.verify_retained_data_sources(prepared, data)
        self.assertEqual(sum(len(v["source_files_sha256"]) for v in records.values()), 29)
        changed = prepared / "data-sources/context-v3-r1" / mixed.READERS["context-v3-r1"]["files"][0]
        changed.write_text("# explicit tampered source fixture\n")
        with self.assertRaisesRegex(ValueError, "RETAINED_READER_SOURCE_CHANGED"):
            probe.verify_retained_data_sources(prepared, data)

    def test_replay_allows_only_exact_retained_loader_location_and_keeps_provenance(self):
        declaration = {"path": "/fixture/package", "manifest_sha256": "a" * 64,
            "reader": {"revision": "context-v3-metadata-r2", "root": "/fixture/original-reader",
                       "source_files_sha256": {"reader.py": "b" * 64}},
            "observed_rows": 8, "source_adapter_schema": "explicit-fixture-r2"}
        original = {"schema": mixed.SCHEMA, "parents": [declaration], "metadata_reconciled_cell_ids": ["cell1"]}
        before = copy.deepcopy(original)
        declarations = [{k: declaration[k] for k in ("path", "manifest_sha256", "reader")}]
        replayed = copy.deepcopy(original)
        replayed["parents"][0]["reader"]["root"] = str(self.root / "data-sources/context-v3-metadata-r2")
        probe.verify_replayed_aggregate(original, replayed, self.root, declarations)
        self.assertEqual(original, before)
        mutations = [lambda r: r["parents"][0]["reader"].update(root="/fixture/other"),
            lambda r: r["parents"][0]["reader"].update(revision="context-v3-r1"),
            lambda r: r["parents"][0]["reader"]["source_files_sha256"].update({"reader.py": "c" * 64}),
            lambda r: r["parents"][0].update(source_adapter_schema="cast-to-old"),
            lambda r: r["metadata_reconciled_cell_ids"].clear()]
        for mutate in mutations:
            wrong = copy.deepcopy(replayed); mutate(wrong)
            with self.assertRaisesRegex(ValueError, "REPLAYED_.*CHANGED"):
                probe.verify_replayed_aggregate(original, wrong, self.root, declarations)
        forged = copy.deepcopy(original)
        forged["parents"][0]["reader"]["root"] = "/fixture/relabelled-origin"
        with self.assertRaisesRegex(ValueError, "DECLARED_AGGREGATE_PROVENANCE_CHANGED"):
            probe.verify_replayed_aggregate(forged, replayed, self.root, declarations)

    def test_mixed_rows_keep_fixed_preselected_train_dev_and_complete_targets(self):
        plan = fixture_plan(); selection = probe.selection_layout(plan); rows = fixture_rows(plan)
        for index, row in enumerate(rows):
            if index % 2:
                row.update(consumer=mixed.METADATA_ROW_SCHEMA,
                    source_adapter_schema=mixed.READERS["context-v3-metadata-r2"]["adapter"],
                    native_container_relation=relation(), original_pcm_sha256="d" * 64, original_wav_sha256="a" * 64)
                if index % 5 == 0:
                    row.update(generation_outcome="failed-native", label_status="uncertain", full_text=None,
                        native_container_relation=None, ctc_mask=False, ctc_weight=0, target_ids=None, quarantined=True,
                        keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0],
                        ctc_reason="no_complete_observed_transcript")
        train, dev, ledger = probe.enrollment(rows, selection, plan, mixed_readers=True)
        self.assertEqual(len(ledger), 72)
        self.assertTrue(all(r["generation_outcome"] == "completed" for r in train + dev))
        self.assertTrue(all(r["split"] == "train" for r in train))
        self.assertTrue(all(r["split"] == "development" for r in dev))
        self.assertTrue(any(r["consumer"] == mixed.METADATA_ROW_SCHEMA for r in train))
        self.assertTrue(all(not r["optimizer_allowed"] for r in ledger if r["planned_use"] == "evaluation"))
        ordinary = [r for r in train if r["generation_category"] == "ordinary-negative"]
        self.assertTrue(ordinary and all(len(r["target_ids"]) == len(r["full_text"]) > 4 for r in ordinary))
        with self.assertRaisesRegex(ValueError, "OBSERVED_ROLE_CHANGED"):
            probe.enrollment(rows, selection, plan)
        wrong = copy.deepcopy(rows); wrong[1]["source_adapter_schema"] = mixed.READERS["context-v3-r1"]["adapter"]
        with self.assertRaisesRegex(ValueError, "ROW_READER_CHANGED"):
            probe.enrollment(wrong, selection, plan, mixed_readers=True)

    @unittest.skipUnless(DATA, "set KWS_CONTEXT_METADATA_DATA_ROOT for exact old/new retained-reader replay")
    def test_real_packages_replay_retained_reader_copies_after_live_fixture_roots_are_removed(self):
        script = ROOT / "tests/test_research_context_v3_handoff_inputs.py"
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(script), "--build-fixtures",
            str(Path(DATA).absolute()), str(self.root)], capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        declaration = q.loads((self.root / "declaration.json").read_bytes())
        packages = declaration["packages"]
        live = self.root / "live-fixture-readers"
        for package in packages:
            reader = package["reader"]
            origin = Path(reader["root"])
            root = live / reader["revision"]
            for name in reader["source_files_sha256"]:
                target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((origin / name).read_bytes())
            reader["root"] = str(root)
        anchor = self.root / "anchor"; (anchor / "official").mkdir(parents=True)
        (anchor / "vocabulary.json").write_bytes(Path(declaration["vocabulary_path"]).read_bytes())
        (anchor / "official/tokens.txt").write_bytes(Path(declaration["original_tokens_path"]).read_bytes())
        plan = q.loads((Path(packages[0]["path"]) / "evidence/source/context-v3/plan.json").read_bytes())
        request = {"schema": probe.HANDOFF_PREP, "anchor": str(anchor)}
        closure = {"packages": packages, "readers": {p["reader"]["revision"]: p["reader"] for p in packages}}
        rows, report, sources = probe.project(request, {"prior_sources": []}, None, closure, plan)
        prepared = self.root / "prepared"; prepared.mkdir()
        probe.copy_data_sources(prepared, sources)
        probe.verify_retained_data_sources(prepared, sources)
        shutil.rmtree(live)
        replayed, replay_report, replay_sources = probe.project(request, {"prior_sources": []},
            prepared / "data-sources", closure, plan)
        self.assertEqual(replayed, rows)
        self.assertEqual(replay_sources, sources)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(report["metadata_reconciled_cell_ids"]), 1)
        probe.verify_replayed_aggregate(report, replay_report, prepared, packages)
        self.assertEqual(report["row_consumer_schemas"], {probe.inputs.ROW_SCHEMA: 4, mixed.METADATA_ROW_SCHEMA: 4})
        altered = prepared / "data-sources/context-v3-metadata-r2/automation/active_context_v3_metadata_weak_view.py"
        altered.write_bytes(altered.read_bytes() + b"\n# explicit test-only source drift\n")
        with self.assertRaisesRegex(ValueError, "READER_SOURCE_CHANGED"):
            probe.project(request, {"prior_sources": []}, prepared / "data-sources", closure, plan)


if __name__ == "__main__":
    unittest.main()
