#!/usr/bin/env python3
"""Synthetic protocol tests only: no native engine, optimizer or ML import."""
from collections import Counter
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import run_d1_zipformer_extension as target
import run_d1_zipformer_continuation as continuation
import test_research_d1 as fixture
import test_d1_zipformer_system as old_fixture
from research_d1_recipe import project_rows


def full():
    return project_rows(fixture.raw_rows(), fixture.vocabulary())


def recipe():
    rows = full()
    value = old_fixture.job()
    return {**value, "schema": target.SCHEMA, "source_ids": sorted(row["source_id"] for row in rows),
        "source_files": target.source_records(), "previous_source_root": "/private/frozen-original-source",
        "previous_source_files": target.previous.source_records(),
        "final_prior_evidence": {"path": "/private/real-closed-final-prior.json", "sha256": "c" * 64},
        "previous_native": {"queue_root": "/private/completed-original-native-queue", "job_id": "native64",
                            "receipt_sha256": "a" * 64, "manifest_sha256": "b" * 64}}


def saved_native(folder, rows, identity):
    """Fictional empty-event receipts; these never represent actual inference."""
    for name in ("claims", "native"):
        (folder / name).mkdir(parents=True)
    result = []
    for row in rows:
        claim = folder / "claims" / (row["source_id"] + ".json")
        native = folder / "native" / claim.name
        item = target.previous.input_binding(row)
        target.previous.native.write_new_json(claim, {"schema": 1, "identity_sha256": identity,
            "input": item, "attempt": 1, "fixture_not_model_call": True})
        value = {"schema": 1, "status": "completed", "identity_sha256": identity, "input": item,
            "claim_sha256": target.previous.training.sha_file(claim), "events": [], "audio_samples": row["samples"],
            "input_finished_calls": 1, "added_padding_samples": 0, "inference_seconds": 0.0,
            "fixture_not_model_result": True}
        value["result_sha256"] = target.previous.native.sha256(target.previous.native.canonical(value))
        target.previous.native.write_new_json(native, value)
        result.append(target.previous.observed_row(row, value, target.previous.training.sha_file(native)))
    target.io.write_new(folder / "observations.jsonl", b"".join(target.io.canonical(row) for row in result))
    return result


class NativeExtensionContracts(unittest.TestCase):
    def test_full_roster_fixed_sources_and_real_hash_slots_required(self):
        value = recipe()
        self.assertEqual(target.validate_job(value), value)
        for key, changed in (("source_ids", value["source_ids"][:-1]), ("source_ids", value["source_ids"][::-1]),
                ("source_files", {}), ("timeout_seconds", 121), ("timeout_seconds", float("nan")),
                ("previous_source_root", "relative")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                target.validate_job({**value, key: changed})
        value["completed_preparation"]["receipt_sha256"] = "unknown future hash"
        with self.assertRaisesRegex(ValueError, "COMPLETED_HASHES"):
            target.validate_job(value)

    def test_exact_64_old_192_new_and_only_location_may_change(self):
        old, rows = old_fixture.projected(), full()
        for row in rows:
            row["audio"] = row["audio_path"] = "new-preparation/" + row["source_id"] + ".wav"
        new = target.join_unchanged(old, rows)
        self.assertEqual(len(new), 192)
        self.assertEqual(Counter(row["split"] for row in new), {"train": 144, "development": 48})
        self.assertEqual(Counter(row["split"] for row in old), {"train": 48, "development": 16})

    def test_any_previous_pcm_label_mask_weight_or_source_drift_blocks_reuse(self):
        old, rows = old_fixture.projected(), full()
        index = next(i for i, row in enumerate(rows) if row["source_id"] == old[0]["source_id"])
        changes = {"pcm_sha256": "f" * 64, "file_sha256": "e" * 64, "samples": rows[index]["samples"] + 1,
            "source_family_id": "changed-family", "speaker_id": "changed-speaker", "split": "development",
            "keyword_mask": [False, False], "keyword_targets": [None, None], "keyword_weight": [.25, .25]}
        for key, value in changes.items():
            changed = copy.deepcopy(rows)
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "PREVIOUS_ROW_DRIFT"):
                target.join_unchanged(old, changed)
        for key, value in (("label_reason", "later label reinterpretation"), ("label_ctc_weight", .25),
                           ("full_text", "different synthetic fixture text")):
            changed = copy.deepcopy(rows)
            changed[index][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "LABEL_IDENTITY_CHANGED"):
                target.join_unchanged(old, changed)

    def test_missing_or_duplicate_global_rows_cannot_be_replaced(self):
        old, rows = old_fixture.projected(), full()
        with self.assertRaisesRegex(ValueError, "PREVIOUS_GLOBAL_ROSTER"):
            target.join_unchanged(old[:-1], rows)
        with self.assertRaisesRegex(ValueError, "PREVIOUS_GLOBAL_ROSTER"):
            target.join_unchanged(old, rows[:-1] + [rows[0]])

    def test_merged_evidence_keeps_old_identity_objects_and_new_call_flags(self):
        old_rows, rows = old_fixture.projected(), full()
        new_rows = target.join_unchanged(old_rows, rows)
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            old_values = saved_native(output / "reused/original-output", old_rows, "original-native-identity")
            new_values = saved_native(output / "new", new_rows, "new-native-identity")
            old = {"observations": old_values, "identity_sha256": "original-native-identity",
                   "terminal": {"receipt_sha256": "a" * 64}}
            merged = target.merged_observations(rows, old, new_values, output, "new-native-identity")
            self.assertEqual(len(merged), 256)
            reused = [row for row in merged if not row["new_native_call"]]
            self.assertEqual(len(reused), 64)
            self.assertEqual({row["native_identity_sha256"] for row in reused}, {"original-native-identity"})
            self.assertEqual({row["origin"] for row in reused}, {"reused-previous64"})
            original_by_id = {row["source_id"]: row for row in old_values}
            self.assertTrue(all(row["observation"] == original_by_id[row["source_id"]] for row in reused))
            self.assertTrue(all(row["previous_terminal_sha256"] == "a" * 64 for row in reused))
            counts = target.grouped_counts(merged)
            self.assertEqual(counts["all_observable"]["development"]["raw_observed_clips"], 64)
            self.assertEqual(counts["previous64"]["development"]["raw_observed_clips"], 16)
            self.assertEqual(counts["new_observable"]["development"]["raw_observed_clips"], 48)
            self.assertEqual(counts["all_observable"]["development"]["formal_qualification_denominator"], 0)
            first = output / "reused/original-output/native" / (old_rows[0]["source_id"] + ".json")
            first.unlink()
            with self.assertRaisesRegex(ValueError, "ORIGINAL_OBSERVATION_CHANGED"):
                target.merged_observations(rows, old, new_values, output, "new-native-identity")

    def test_retained_original_output_and_observation_are_byte_exact(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            old_root, original, output = root / "old-queue", root / "old-output", root / "extension"
            original.mkdir()
            old_root.mkdir()
            output.mkdir()
            raw = b'{"synthetic":true,"never_a_model_result":true}\n'
            (original / "observations.jsonl").write_bytes(raw)
            target.io.write_json(original / "MANIFEST.json", {"observations.jsonl": target.io.file_record(original / "observations.jsonl")})
            for relative in ("queue.json", "request.json", "active-plan.json", "recipes/original.json",
                "jobs/original/result.json", "jobs/original/claim.json", "jobs/original/stdout.log", "jobs/original/stderr.log"):
                path = old_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic preserved bytes\n")
            old = {"root": old_root, "output": original, "terminal": {"job_id": "original"}}
            target.retain_previous(output, old)
            self.assertEqual((output / "reused/original-output/observations.jsonl").read_bytes(), raw)
            self.assertEqual((output / "reused/original-output/MANIFEST.json").read_bytes(), (original / "MANIFEST.json").read_bytes())
            with self.assertRaises(FileExistsError):
                target.retain_previous(output, old)

    def test_no_override_of_old_source_contract_and_guard_before_ml(self):
        records = target.source_records()
        self.assertEqual({key: value for key, value in records.items() if key not in target.ADDED_SOURCE_PATHS},
                         target.previous.source_records())
        helper_source = (ROOT / "tools/research_d1_previous_native.py").read_text()
        self.assertIn("original.audit_finished(recipe_path, output)", helper_source)
        self.assertIn("original.source_records() == files", helper_source)
        source = inspect.getsource(target.old_source_readback)
        self.assertIn('"-I", "-S", "-B"', source)
        self.assertIn("start_new_session=False", source)
        self.assertIn("deadline - time.monotonic()", source)
        run_source = inspect.getsource(target.run)
        self.assertLess(run_source.index("research_no_network.install()"), run_source.index("new_native_rows("))
        self.assertNotIn("source_records =", helper_source + source)
        self.assertFalse(any(name in sys.modules for name in target.previous.ML_MODULES))

    def test_zero_new_observable_rows_never_load_engine_or_machine_learning(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            for child in ("claims", "native"):
                (output / child).mkdir()
            with patch.object(target.previous.native, "make_engine", side_effect=AssertionError("engine must not load")):
                self.assertEqual(target.new_native_rows([], output, output, "identity", {}, output,
                    output / "keywords.txt", time.monotonic() + 2), ([], 0.0, 0.0))
            self.assertEqual(target.previous.readback_native(output, [], "identity"), ([], 0.0))
            self.assertEqual((output / "observations.jsonl").read_bytes(), b"")
            self.assertFalse(any(name in sys.modules for name in target.previous.ML_MODULES))

    def test_isolated_help_needs_no_machine_learning(self):
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / target.SOURCE_PATH), "--help"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)


class NativeContinuationContracts(unittest.TestCase):
    """Fictional receipts exercise accounting; no fixture is a native model run."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = full()[:-1]  # One separate generation absence, 255 legal inputs.
        self.old_rows = old_fixture.projected()
        self.new_rows = target.join_unchanged(self.old_rows, self.rows)
        self.original = self.root / "original-output"
        self.original.mkdir()
        self.values = saved_native(self.original / "new", self.new_rows[:128], "partial-identity")
        unknown = self.new_rows[128]
        target.previous.native.write_new_json(self.original / "new/claims" / (unknown["source_id"] + ".json"),
            {"schema": 1, "attempt": 1, "identity_sha256": "partial-identity",
             "input": target.previous.input_binding(unknown), "fixture_not_model_call": True})
        self.pending = [row["source_id"] for row in self.new_rows[129:]]

    def job(self):
        evidence = self.root / "fictional-evidence.json"
        if not evidence.exists():
            evidence.write_bytes(b'{"fixture":true,"no_model_calls":true}\n')
        return {"schema": continuation.SCHEMA,
            "failed_native": {"queue_root": str(self.root / "original-queue"), "job_id": "original",
                              "receipt_sha256": "a" * 64},
            "partial_inventory": {"path": str(evidence), "sha256": target.previous.training.sha_file(evidence)},
            "partial_accounting": {"path": str(evidence), "sha256": target.previous.training.sha_file(evidence)},
            "pending_source_ids": self.pending, "source_files": continuation.source_records(),
            "timeout_seconds": 180, "maximum_attempts": 1}

    def partition(self, pending=None):
        return continuation.partition_partial(self.original, self.new_rows, "partial-identity",
            self.pending if pending is None else pending)

    def test_only_exact_never_claimed_prefix_complement_is_eligible(self):
        proof = self.partition()
        self.assertEqual((len(proof["returned_rows"]), len(proof["unknown_rows"]), len(proof["pending_rows"])), (128, 1, 62))
        self.assertEqual(proof["observations"], self.values)
        for replacement in (self.new_rows[0]["source_id"], self.new_rows[128]["source_id"]):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(ValueError, "EXACT_UNCLAIMED_COMPLEMENT"):
                self.partition([replacement, *self.pending[1:]])
        with self.assertRaisesRegex(ValueError, "EXACT_UNCLAIMED_COMPLEMENT"):
            self.partition(self.pending[::-1])

    def test_unknown_claim_cannot_be_deleted_to_reclassify_pending(self):
        claim = self.original / "new/claims" / (self.new_rows[128]["source_id"] + ".json")
        claim.unlink()
        with self.assertRaisesRegex(ValueError, "129_CLAIMS_128_RETURNS_PREFIX"):
            self.partition()

    def test_partial_native_or_observation_tampering_is_not_repairable(self):
        path = self.original / "new/native" / (self.new_rows[0]["source_id"] + ".json")
        original = path.read_bytes()
        value = json.loads(original); value["audio_samples"] += 1
        path.write_bytes(target.io.canonical(value))
        with self.assertRaises(ValueError):
            self.partition()
        path.write_bytes(original)
        observations = self.original / "new/observations.jsonl"
        observations.write_bytes(b"".join(target.io.canonical(value) for value in reversed(self.values)))
        with self.assertRaisesRegex(ValueError, "ORIGINAL_OBSERVATIONS_CHANGED"):
            self.partition()

    def test_closed_partial_inventory_rejects_extra_missing_and_changed_bytes(self):
        output = self.root / "small-partial"; output.mkdir()
        (output / "original.json").write_bytes(b"original\n")
        terminal = self.root / "terminal.json"; terminal.write_bytes(b"fictional timed out\n")
        inventory = {"schema": "d1-native-partial-output-inventory-v1", "root": str(output),
            "terminal": continuation.pin(terminal), "files": {"original.json": target.io.file_record(output / "original.json")},
            "file_count": 1, "total_bytes": 9, "ml_imports": 0, "new_model_calls": 0,
            "verification_scope": "byte_inventory_and_roster_only"}
        path = self.root / "inventory.json"; path.write_bytes(target.io.canonical(inventory))
        continuation.verify_inventory(path, output, terminal)
        (output / "extra.json").write_bytes(b"unclaimed")
        with self.assertRaisesRegex(ValueError, "PARTIAL_EXACT_ROSTER"):
            continuation.verify_inventory(path, output, terminal)
        (output / "extra.json").unlink(); (output / "original.json").write_bytes(b"changed!\n")
        with self.assertRaisesRegex(ValueError, "PARTIAL_BYTES_CHANGED"):
            continuation.verify_inventory(path, output, terminal)
        (output / "original.json").unlink()
        with self.assertRaisesRegex(ValueError, "PARTIAL_EXACT_ROSTER"):
            continuation.verify_inventory(path, output, terminal)

    def test_unknown_preserves_labels_without_events_and_never_enters_counts(self):
        proof = self.partition()
        unknown = proof["unknown_rows"][0]
        # The observed absence is inference-only, never an unknown/negative label.
        unknown.update(label_status="machine_candidate", quarantined=False, split="development",
            keyword_targets=[0, 1], keyword_mask=[True, True], keyword_weight=[.5, .5])
        old_values = saved_native(self.original / "reused/original-output", self.old_rows, "old-identity")
        proof.update(output=self.original, rows=self.rows, terminal={"receipt_sha256": "a" * 64},
            identity_sha256="partial-identity", old={"observations": old_values, "identity_sha256": "old-identity"})
        output = self.root / "continuation-output"
        new_values = saved_native(output / "new", proof["pending_rows"], "continuation-identity")
        unavailable = continuation.unknown_predictions(proof)
        self.assertEqual(len(unavailable), 1)
        self.assertNotIn("events", unavailable[0])
        self.assertEqual(unavailable[0]["original_projected_row"]["keyword_targets"], [0, 1])
        self.assertEqual(unavailable[0]["label_binding_sha256"], target.previous.row_identity(unknown))
        merged = continuation.merged_predictions(proof, new_values, output, "continuation-identity")
        self.assertEqual(len(merged), 254)
        self.assertNotIn(unknown["source_id"], {row["source_id"] for row in merged})
        counts = continuation.counts(proof, merged)
        for keyword in ("1", "2"):
            self.assertEqual(counts["label_eligible_keyword_clips"]["development"][keyword],
                counts["prediction_available"]["development"]["keywords"][keyword]["known_machine_weak_clips"] + 1)
        reused = {row["source_id"]: row["observation"] for row in merged if row["origin"] == "original64"}
        self.assertEqual(reused, {row["source_id"]: row for row in old_values})

    def test_declaration_fixed_source_180_seconds_and_one_attempt(self):
        job = self.job()
        self.assertEqual(continuation.validate_job(job), job)
        for key, changed in (("timeout_seconds", 181), ("timeout_seconds", 180.0),
                             ("maximum_attempts", 2), ("source_files", {}),
                             ("pending_source_ids", self.pending[:-1] + [self.pending[0]])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                continuation.validate_job({**job, key: changed})
        self.assertEqual({k: v for k, v in continuation.source_records().items() if k != continuation.SOURCE_PATH},
                         target.source_records())

    def test_real_queue_producer_binding_and_cross_queue_single_registration(self):
        job = self.job()
        recipe_path = self.root / "continuation-recipe.json"
        recipe_path.write_bytes(target.io.canonical(job))
        probe = self.root / "stdlib-probe.py"
        probe.write_text("\n".join((
            "from pathlib import Path; import os,sys",
            "sys.path.insert(0, " + repr(str(ROOT / "tools")) + ")",
            "import run_d1_zipformer_continuation as c",
            "recipe=Path(os.environ['KWS_RESEARCH_RECIPE']); output=Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])",
            "job=c.validate_job(c.read(recipe))",
            "with c.previous.heavy_model_slot(Path(sys.argv[1])) as lease:",
            "    queue=c.current_queue_binding(recipe,output,job,lease)",
            "    c.register(job,queue,recipe,output)",
            "    for name in c.QUEUE_OUTPUTS: c.io.write_json(output/name, {'fictional_probe_only':True,'model_calls':0})",
        )) + "\n")
        lock = self.root / "heavy.lock"

        def execute(name):
            definition = {"job_id": "probe", "recipe": str(recipe_path), "recipe_sha256": target.previous.training.sha_file(recipe_path),
                "argv": [sys.executable, "-I", "-S", "-B", str(probe), str(lock)], "cwd": str(self.root),
                "depends_on": [], "timeout_minutes": 3, "outputs": list(continuation.QUEUE_OUTPUTS)}
            request = self.root / (name + ".json")
            request.write_bytes(target.io.canonical({"schema": target.io.SCHEMA, "queue_id": name,
                "global_budget_minutes": 3, "jobs": [definition]}))
            root = self.root / name
            target.io.create(root, request, ROOT / "configs/research/active-plan-20261105.json", lock)
            result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_queue.py"),
                "run", "--root", str(root)], capture_output=True, text=True, timeout=10)
            return result, target.io.inspect(root)

        first, status = execute("first")
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        self.assertEqual(status["jobs"]["probe"]["status"], "completed")
        registration = continuation.registration_path(job)
        retained = (registration.read_bytes(), registration.stat().st_ino, registration.stat().st_mtime_ns)
        second, status = execute("second")
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(status["jobs"]["probe"]["status"], "failed")
        self.assertIn("ALREADY_REGISTERED_NO_SECOND_ATTEMPT", (self.root / "second/jobs/probe/stderr.log").read_text())
        self.assertEqual((registration.read_bytes(), registration.stat().st_ino, registration.stat().st_mtime_ns), retained)
        self.assertFalse(any((self.root / "second/jobs/probe/outputs").iterdir()))

    def test_readonly_import_guard_and_isolated_help(self):
        guard = continuation.NoML()
        sys.meta_path.insert(0, guard)
        try:
            for name in ("numpy", "torch", "sherpa_onnx"):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "READ_ONLY_FORBIDS_ML"):
                    __import__(name)
        finally:
            sys.meta_path.remove(guard)
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / continuation.SOURCE_PATH), "--help"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(name in sys.modules for name in target.previous.ML_MODULES))


if __name__ == "__main__":
    unittest.main()
