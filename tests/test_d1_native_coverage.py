#!/usr/bin/env python3
"""Synthetic closed-plan evidence; no real generation or native inference."""
import copy
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tests")]
import research_d1_native_coverage as coverage
import test_d1_zipformer_extension as fixture

io = coverage.io


def plan_fixture():
    rows, old = fixture.full(), fixture.old_fixture.projected()
    previous_ids = [row["source_id"] for row in old]
    ordered = old + [row for row in rows if row["source_id"] not in set(previous_ids)]
    cells = [{"cell_id": row["source_id"], "source_family_id": row["source_family_id"],
              "role": row["source_role"], "reference_id": "synthetic-reference"} for row in ordered]
    return {"cells": cells, "references": [{"reference_id": "synthetic-reference"}],
            "fixture_not_a_generation_plan": True}, rows, previous_ids


def record_fixture(cell_id, identity, output, stage_sha, recipe_sha):
    owner = {"stage_sha256": stage_sha, "job_id": "chunk-001", "job_recipe_sha256": recipe_sha}
    claim = {"schema": "active-tts-plan-call-claim-v1", "cell_id": cell_id, "identity_sha256": identity,
             "attempt": 1, "owner": owner, "fixture_not_a_model_call": True}
    record = {"schema": "active-tts-plan-cell-v1", "cell_id": cell_id, "identity_sha256": identity,
        "claim_sha256": io.sha(io.canonical(claim)), "status": "FAILED_NO_RETRY", "origin": "extension-call",
        "new_generation_calls": 1, "training_admitted": False, "output_root": str(output), "files": {},
        "native_wav": None, "technical_failure_reason": "synthetic invalid returned dimension", "fixture": True}
    return {"cell_id": cell_id, "record": record, "claim": claim,
            "ledger_record_sha256": io.sha(io.canonical(record))}


class ClosedTerminalContracts(unittest.TestCase):
    def native_record(self, folder, sample=.1, frames=9600, channels=1, *, malformed=False):
        fmt = struct.pack("<HHIIHH", 3, channels, 24000, 96000 * channels, 4 * channels, 32)
        samples = struct.pack("<f", sample) * (frames * channels)
        body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
        wav = b"RIFF" + struct.pack("<I", len(body)) + body
        path = folder / "native.wav"
        path.write_bytes(wav[:-1] if malformed else wav)
        return {"native_wav": path.name, "output_root": str(folder), "files": {path.name: io.file_record(path)},
                "status": "FAILED_NO_RETRY", "technical_failure_reason": "synthetic quality gate failure"}

    def test_finite_quiet_and_clipped_quality_failures_cannot_be_excluded(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for sample in (0.0, .00001, 1.0, 2.0, 1_000_000):
                with self.subTest(sample=sample):
                    record = self.native_record(root, sample=sample)
                    self.assertTrue(coverage.native_eligibility(record)["eligible"])
                    with self.assertRaisesRegex(ValueError, "EXCLUSION_REQUIRES_UNDECODABLE_NATIVE"):
                        coverage.excluded_native({"record": record, "reason": "quiet_or_clipped"})

    def test_invalid_native_geometry_values_and_missing_wav_keep_real_reason(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            cases = [({"sample": float("nan")}, "ASR_NONFINITE_SAMPLE"),
                     ({"sample": 1_000_001}, "ASR_UNSAFE_NUMERIC_RANGE"),
                     ({"frames": 31 * 24000}, "ASR_FRAME_BOUND"),
                     ({"channels": 2}, "ASR_MONO_FLOAT_24000_REQUIRED"),
                     ({"malformed": True}, "ASR_RIFF_GEOMETRY")]
            for options, reason in cases:
                with self.subTest(options=options):
                    record = self.native_record(root, **options)
                    self.assertEqual(coverage.excluded_native({"record": record, "reason": reason}),
                                     {"eligible": False, "reason": reason})
                    with self.assertRaisesRegex(ValueError, "EXCLUSION_REQUIRES_UNDECODABLE_NATIVE"):
                        coverage.excluded_native({"record": record, "reason": "invented"})
            self.assertEqual(coverage.excluded_native({"record": {"native_wav": None},
                "reason": "NO_DECODABLE_NATIVE_WAV"}), {"eligible": False, "reason": "NO_DECODABLE_NATIVE_WAV"})
            record = self.native_record(root, sample=float("nan"))
            (root / "native.wav").write_bytes(b"corrupt after original inventory")
            with self.assertRaisesRegex(ValueError, "NATIVE_BYTES_CHANGED"):
                coverage.excluded_native({"record": record, "reason": "ASR_NONFINITE_SAMPLE"})

    def test_256_terminal_union_is_exact_and_previous64_cannot_be_excluded(self):
        plan, rows, old = plan_fixture()
        removed = plan["cells"][-1]["cell_id"]
        eligible = [row for row in rows if row["source_id"] != removed]
        omitted = [{"exported_row": {"cell_id": removed}, "original_outcome_row": {"fixture": True}}]
        closed = coverage.close_roster(plan, eligible, omitted, old)
        self.assertEqual((closed["planned_cells"], closed["observable_cells"], closed["new_native_cells"],
                          closed["excluded_cells"]), (256, 255, 191, 1))
        self.assertEqual(closed["unobservable"][0]["keyword_known"], [False, False])
        self.assertIsNone(closed["unobservable"][0]["observed_text"])
        for actual, exclusions in ((eligible, []), (rows, omitted), (eligible, omitted * 2)):
            with self.subTest(length=len(actual)), self.assertRaisesRegex(ValueError, "TERMINAL_PARTITION"):
                coverage.close_roster(plan, actual, exclusions, old)
        old_removed = old[0]
        with self.assertRaisesRegex(ValueError, "PREVIOUS64_CANNOT_BE_EXCLUDED"):
            coverage.close_roster(plan, [row for row in rows if row["source_id"] != old_removed],
                [{"exported_row": {"cell_id": old_removed}}], old)

    def test_all192_new_unobservable_still_requires192_explicit_terminal_evidences(self):
        plan, rows, old = plan_fixture()
        old_set = set(old)
        omitted = [{"exported_row": {"cell_id": row["source_id"]}} for row in rows if row["source_id"] not in old_set]
        retained = [row for row in rows if row["source_id"] in old_set]
        result = coverage.close_roster(plan, retained, omitted, old)
        self.assertEqual((result["observable_cells"], result["excluded_cells"], result["new_native_cells"]), (64, 192, 0))
        with self.assertRaisesRegex(ValueError, "TERMINAL_PARTITION"):
            coverage.close_roster(plan, retained, omitted[:-1], old)

    def test_hash_bound_exclusion_joins_real_completed_queue_and_not_mutable_ledger(self):
        plan, rows, old = plan_fixture()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            plan_path, stage_path, recipe = root / "plan.json", root / "stage.json", root / "recipe.json"
            io.write_json(plan_path, plan)
            queue_root, selected = root / "queue", [cell["cell_id"] for cell in plan["cells"][232:]]
            job = {"job_id": "chunk-001", "chunk_index": 0, "cells": selected, "fixture": True}
            io.write_json(recipe, job)
            recipe_sha = io.file_record(recipe)["sha256"]
            stage = {"schema": "active-tts-extension-stage-v1", "source_revision": coverage.SOURCE_REVISION,
                "initial_reused_cells": [cell["cell_id"] for cell in plan["cells"][:232]], "target_count": 256,
                "target_cells": [cell["cell_id"] for cell in plan["cells"]], "chunks": [selected],
                "maximum_new_calls": 24, "maximum_job_seconds": 600, "stage_budget_minutes": 20,
                "max_retries": 0, "paired_cases_included": False, "training_admitted": False,
                "queue_root": str(queue_root), "binding": {"plan_canonical_sha256": coverage.plain_digest(plan),
                    "plan_file_sha256": io.file_record(plan_path)["sha256"]}}
            io.write_json(stage_path, stage)
            stage_sha = io.file_record(stage_path)["sha256"]
            output = queue_root / "jobs/chunk-001/outputs"
            outcome_rows = [record_fixture(cell["cell_id"], coverage.plain_digest({"binding": stage["binding"],
                "cell": cell, "reference": plan["references"][0]}), output, stage_sha, recipe_sha)
                for cell in plan["cells"][232:]]
            outcome = {"schema": "active-tts-stage-outcomes-v1", "job_id": "chunk-001", "stage_sha256": stage_sha,
                "job_recipe_sha256": recipe_sha, "plan_canonical_sha256": coverage.plain_digest(plan),
                "binding": stage["binding"], "selected_cells": selected, "new_call_ids": selected,
                "reused_cell_ids": [], "rows": outcome_rows}
            fixture_output = root / "synthetic-outcomes.json"
            io.write_json(fixture_output, outcome)
            producer = root / "producer.py"
            producer.write_text("import os,shutil\nfrom pathlib import Path\n"
                "shutil.copyfile(" + repr(str(fixture_output)) + ", Path(os.environ['KWS_RESEARCH_OUTPUT_DIR'])/'outcomes.json')\n")
            request = root / "request.json"
            io.write_json(request, {"schema": io.SCHEMA, "queue_id": "synthetic-terminal-coverage",
                "global_budget_minutes": 1, "jobs": [{"job_id": "chunk-001", "recipe": str(recipe),
                    "recipe_sha256": recipe_sha, "argv": [sys.executable, "-I", "-S", "-B", str(producer)],
                    "cwd": str(root), "depends_on": [], "timeout_minutes": 1, "outputs": ["outcomes.json"]}]})
            cli = [sys.executable, "-I", "-S", "-B", str(ROOT / "tools/research_queue.py")]
            env = {key: value for key, value in os.environ.items() if not key.startswith("KWS_RESEARCH_")}
            for command in (cli + ["init", "--root", str(queue_root), "--request", str(request),
                                  "--heavy-lock", str(root / "fixture-heavy.lock")], cli + ["run", "--root", str(queue_root)]):
                completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            binding = {"job_id": "chunk-001", "outcomes": coverage.pin(output / "outcomes.json"),
                       "job_recipe": coverage.pin(recipe)}
            item = outcome_rows[-1]
            excluded = {key: item[key] for key in ("cell_id", "record", "ledger_record_sha256")}
            excluded.update(job_id="chunk-001", reason="NO_DECODABLE_NATIVE_WAV",
                outcomes=binding["outcomes"], asr_calls=0, observed_text=None, training_admitted=False)
            index = {"schema": "active-tts-asr-partitions-v1", "stage": str(stage_path), "stage_sha256": stage_sha,
                "plan_canonical_sha256": coverage.plain_digest(plan), "plan_file_sha256": io.file_record(plan_path)["sha256"],
                "aggregate_cell_count": 256, "paired_cases_included": False, "training_admitted": False,
                "asr_calls_executed": 0, "reused_cells_without_new_asr_request": stage["initial_reused_cells"],
                "job_outcomes": [binding], "partitions": [{**binding, "cell_ids": selected[:-1], "recognizer_context_text": None}],
                "excluded_cells": [excluded], "new_generation_cell_count": 24, "asr_eligible_new_cell_count": 23,
                "asr_excluded_new_cell_count": 1}
            index_path = root / "index.json"
            io.write_json(index_path, index)
            def reader(path):
                folder, _, request, queue_sha = io.load_queue(path)
                return {"root": str(folder), "request": request,
                    "jobs": {job["job_id"]: io.result_for(folder, job, queue_sha, check_active=False) for job in request["jobs"]}}
            def verify(pin=None):
                return coverage.excluded_export(pin or coverage.pin(index_path), plan, coverage.plain_digest(plan),
                    io.file_record(plan_path)["sha256"], reader)
            verified, pins = verify()
            self.assertEqual(verified[0]["original_outcome_row"], item)
            self.assertEqual(len(pins), 5)
            closed = coverage.close_roster(plan, [row for row in rows if row["source_id"] != item["cell_id"]], verified, old)
            self.assertEqual((closed["observable_cells"], closed["excluded_cells"]), (255, 1))
            original_pin = coverage.pin(index_path)
            changed = copy.deepcopy(index)
            changed["excluded_cells"][0]["record"]["status"] = "GENERATED_PENDING_ASR"
            index_path.write_bytes(io.canonical(changed))
            with self.assertRaisesRegex(ValueError, "PIN_CHANGED"):
                verify(original_pin)
            with self.assertRaisesRegex(ValueError, "EXCLUDED_TERMINAL_HASH"):
                verify()


class AccountedCoverageContracts(unittest.TestCase):
    """Adapter and independent roster contracts, never generation evidence."""
    def accounted_fixture(self):
        plan, rows, old = plan_fixture()
        missing = next(row for row in rows if row["source_id"] not in old and row["split"] == "train")
        unknown = {key: missing[key] for key in ("source_id", "source_family_id", "source_role")}
        unknown.update(status="unobservable-consumed", reason="fixture consumed without return",
            asr_calls=0, observed_text=None, training_admitted=False,
            original_proof={"fixture_not_real_generation": True, "status": "timed_out"})
        retained = [row for row in rows if row["source_id"] != missing["source_id"]]
        return plan, rows, retained, old, unknown

    def test_255_returned_plus_one_unknown_is_accounted_not_terminal_complete(self):
        plan, _, retained, old, unknown = self.accounted_fixture()
        closed = coverage.close_accounted_roster(plan, retained, [], [unknown], old)
        self.assertEqual((closed["planned_cells"], closed["accounted_cells"], closed["returned_generation_cells"],
            closed["observable_cells"], closed["new_native_cells"], closed["consumed_unknown_cells"]),
            (256, 256, 255, 255, 191, 1))
        self.assertTrue(closed["accounted_complete"])
        self.assertFalse(closed["complete_terminal_coverage"])
        self.assertEqual(closed["unobservable_by_split"], {"train": 1})
        item = closed["unobservable"][0]
        self.assertFalse(item["new_native_call"])
        self.assertEqual(item["keyword_known"], [False, False])
        self.assertEqual(item["keyword_weight"], [0, 0])
        self.assertEqual(item["original_proof"], unknown["original_proof"])
        self.assertFalse(any(key in item for key in ("pcm", "audio", "keyword_targets", "full_text")))
        self.assertEqual(len(fixture.target.join_unchanged(fixture.old_fixture.projected(), retained)), 191)

    def test_returned_invalid_and_consumed_unknown_have_distinct_counts(self):
        plan, _, retained, old, unknown = self.accounted_fixture()
        removed = next(row for row in retained if row["source_id"] not in old)
        excluded = {key: removed[key] for key in ("source_id", "source_family_id", "source_role")}
        excluded.update(status="returned-unrecognizable", reason="fixture invalid returned dimensions",
            asr_calls=0, observed_text=None, training_admitted=False,
            original_proof={"fixture_not_real_generation": True, "returned": True})
        observed = [row for row in retained if row["source_id"] != removed["source_id"]]
        closed = coverage.close_accounted_roster(plan, observed, [excluded], [unknown], old)
        self.assertEqual((closed["observable_cells"], closed["returned_generation_cells"],
            closed["excluded_returned_cells"], closed["consumed_unknown_cells"], closed["unobservable_cells"]),
            (254, 255, 1, 1, 2))
        self.assertEqual(fixture.target.coverage_counts(closed)["returned_generation_cells"], 255)
        self.assertFalse(closed["complete_terminal_coverage"])

    def test_missing_overlapping_duplicate_and_old_unknown_are_rejected(self):
        plan, rows, retained, old, unknown = self.accounted_fixture()
        for observed, excluded, consumed in ((retained, [], []), (rows, [], [unknown]),
                (retained, [], [unknown, unknown]), (retained, [unknown], [unknown])):
            with self.subTest(count=len(observed)), self.assertRaisesRegex(ValueError, "EXACT_DISJOINT_PARTITION"):
                coverage.close_accounted_roster(plan, observed, excluded, consumed, old)
        old_row = next(row for row in rows if row["source_id"] == old[0])
        old_unknown = {**unknown, **{key: old_row[key] for key in ("source_id", "source_family_id", "source_role")}}
        with self.assertRaisesRegex(ValueError, "PREVIOUS64_MUST_REMAIN_OBSERVABLE"):
            coverage.close_accounted_roster(plan, [row for row in rows if row["source_id"] != old[0]], [], [old_unknown], old)
        for key, value in (("observed_text", "invented negative"), ("training_admitted", True),
                ("asr_calls", 1), ("status", "returned-unrecognizable"), ("source_role", "development-candidate"),
                ("original_proof", {})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "UNOBSERVABLE_RECORD"):
                coverage.close_accounted_roster(plan, retained, [], [{**unknown, key: value}], old)

    def test_accounted_adapter_calls_retained_verifier_and_checks_original_evidence(self):
        plan, _, retained, old, unknown = self.accounted_fixture()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            prior, proof = root / "accounted-prior.json", root / "synthetic-proof.json"
            io.write_json(prior, {"schema": coverage.ACCOUNTED_PRIOR, "fixture_not_evidence": True})
            io.write_json(proof, {"fixture_not_evidence": True})
            views = [{"path": "/synthetic/view", "manifest_sha256": "a" * 64}]
            verified = {"schema": coverage.ACCOUNTED_VERIFICATION, "plan": plan,
                "plan_sha256": coverage.plain_digest(plan), "accounted_prefix_count": 256,
                "observed_ids": [cell["cell_id"] for cell in plan["cells"] if cell["cell_id"] != unknown["source_id"]],
                "excluded_returned": [], "unobservable_consumed": [unknown], "pending_ids": [], "views": views,
                "evidence_files": [coverage.pin(proof)], "accounted_complete": True,
                "complete_terminal_coverage": False, "ml_imports": 0, "new_model_calls": 0}
            job = {"plan_sha256": coverage.plain_digest(plan), "source_ids": sorted(cell["cell_id"] for cell in plan["cells"]),
                   "final_prior_evidence": {"path": str(prior), "sha256": coverage.pin(prior)["sha256"]}}
            record = {"aggregate": {"parents": views}}
            calls = []
            def replay(path, expected_sha, *, expected_count):
                calls.append((path, expected_sha, expected_count))
                return verified
            loader = SimpleNamespace(verify_accounted_prior=replay)
            def bind(selected_loader=loader):
                return coverage.bind(job, record, retained, old,
                    lambda _: self.fail("old excluded-terminal parser must not handle accounted prior"), loader=selected_loader)
            closed = bind()
            self.assertEqual(calls, [(str(prior), coverage.pin(prior)["sha256"], 256)])
            self.assertEqual(closed["schema"], coverage.ACCOUNTED_SCHEMA)
            self.assertEqual(closed["evidence_files"], [coverage.pin(prior), coverage.pin(proof)])
            with self.assertRaisesRegex(ValueError, "RETAINED_VERIFIER_REQUIRED"):
                bind(None)
            original = copy.deepcopy(verified)
            mutations = (("pending_ids", [unknown["source_id"]]), ("accounted_complete", False),
                ("complete_terminal_coverage", True), ("observed_ids", verified["observed_ids"][::-1]),
                ("ml_imports", 1), ("new_model_calls", 1), ("views", []))
            for key, value in mutations:
                verified.clear()
                verified.update(copy.deepcopy(original))
                verified[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    bind()
            verified.clear()
            verified.update(original)
            proof.write_bytes(b"changed after verified receipt")
            with self.assertRaisesRegex(ValueError, "ORIGINAL_EVIDENCE_CHANGED"):
                bind()
            def rejected(*args, **kwargs):
                raise ValueError("fixture original partial proof rejected")
            with self.assertRaisesRegex(ValueError, "original partial proof rejected"):
                bind(SimpleNamespace(verify_accounted_prior=rejected))


if __name__ == "__main__":
    unittest.main()
