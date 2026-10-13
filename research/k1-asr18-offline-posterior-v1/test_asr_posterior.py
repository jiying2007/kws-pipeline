"""Offline adversarial acceptance matrix; every API/log/ASR fixture is synthetic.

No network, model execution, external state mutation, publication, source-tree
write, training admission or acoustic-accuracy claim is exercised or authorized.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture_cases as f
import asr_posterior as a


def replace_json_member(name, edit):
    def mutate(files):
        value = f.t.decode(files[name])
        edit(value)
        files[name] = f.t.canonical(value)
    return mutate


REJECTION = "completed supervised cleanup satisfies original pause and deadline conditions"
FAILURES = (("host_deadline_expired_at_start", True), ("host_deadline_overrun", True),
            ("pause_acknowledged", False), ("cleanup_overrun_seconds", 1), ("cleanup_seconds", 16))


def cleanup_mutation(f, field, value):
    def mutate(files):
        receipt = f.t.decode(files["cleanup-receipt.json"])
        receipt["containers"][0][field] = value
        files["cleanup-receipt.json"] = f.t.canonical(receipt)
    return mutate


def valid_remove_fallback(f):
    def mutate(files):
        from resource_host import safe_error
        error = safe_error(subprocess.CalledProcessError(1, ["docker", "rm", "--force"]))
        f.t._host()._safe_failure(error)
        receipt = f.t.decode(files["cleanup-receipt.json"])
        row = receipt["containers"][0]
        row["errors"] = [{"operation": "force_remove", **error}]
        row["removal_proof"] = "successful_empty_exact_id_query"
        row["operations"] = [
            {"operation": "force_remove", "status": "failed", "wall_seconds": 0.0, **error},
            {"operation": "verify_absence", "status": "succeeded", "wall_seconds": 0.0}]
        files["cleanup-receipt.json"] = f.t.canonical(receipt)
    return mutate


class PosteriorMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.complete = f.normal_case()
        cls.partial = f.normal_case("first_native_failure18")
        cls.notrun = f.normal_case("all_notrun")
        cls.recovered = f.recovery_case()

    def accept(self, case):
        return a.accept(**case.arguments())

    def reject(self, case):
        with self.assertRaises(ValueError):
            self.accept(case)

    def test_A_public_fixture_is_generated_and_deterministic(self):
        case = copy.deepcopy(self.complete)
        self.assertEqual(case, f.normal_case())
        self.assertIn("Synthetic", case.synthetic_notice)
        files = f.members(case.payload)
        self.assertEqual(sum(name.endswith('/started.json') for name in files), 18)
        self.assertEqual(sum(name.endswith('/completed.json') for name in files), 18)
        self.assertTrue(f.RESOURCE_MEMBERS.issubset(files))
        self.assertTrue(all(not name.endswith('.wav') for name in files))
        before = copy.deepcopy(case)
        result = self.accept(case)
        self.assertEqual(result["status"], "complete_native_verified")
        self.assertEqual(case, before, "read-only acceptance must not mutate caller observations")

    def test_A_fixture_cache_isolated_and_rebuild_deterministic(self):
        names = ('success18', 'first_native_failure18', 'all_notrun')
        pristine = {name: f.normal_case(name) for name in names}
        for name in names:
            changed = f.normal_case(name)
            changed.expected['archive']['sha256'] = 'f' * 64
            changed.records['producer_run']['conclusion'] = 'cancelled'
            changed.chunks.clear()
            changed.payload = b'changed caller copy'
            self.assertEqual(f.normal_case(name), pristine[name])
        f._normal_case.cache_clear()
        for name in names:
            rebuilt = f.normal_case(name)
            self.assertEqual(rebuilt.payload, pristine[name].payload)
            self.assertEqual(rebuilt.outer_raw, pristine[name].outer_raw)
            self.assertEqual(rebuilt.expected, pristine[name].expected)
            self.assertEqual(rebuilt.records, pristine[name].records)

    def test_A_complete_summary_is_exactly_eighteen_slots_and_six_machine_only_candidates(self):
        result = self.accept(self.complete)
        self.assertEqual(result["verified_complete_slots"], 18)
        self.assertEqual(result["ledger_reported_attempts"], 18)
        self.assertIsNone(result["consumed_attempts"])
        self.assertEqual(len(result["rows"]), 18)
        self.assertEqual(len({row["slot_id"] for row in result["rows"]}), 18)
        self.assertEqual(len(result["machine_candidates"]), 6)
        self.assertEqual(result["normalization"], a.consensus.NORMALIZATION)
        for row in result["rows"]:
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["normalized_transcript"], a.consensus.normalized_transcript(row["raw_transcript"]))
        for row in result["machine_candidates"]:
            self.assertIs(row["three_complete"], True)
            self.assertIs(row["unanimous_machine_candidate"], True)
            self.assertIs(row["training_admitted"], False)
            self.assertIs(row["human_gold"], False)
        self.assertIs(result["training_admitted"], False)
        self.assertIs(result["human_gold"], False)
        self.assertIs(result["new_attempts_authorized"], False)
        self.assertEqual(result["acoustic_verbatim_completeness"], "unknown")

    def test_A_uses_canonical_punctuation_only_normalizer_without_homophones(self):
        with mock.patch.object(a.consensus, "normalized_transcript", wraps=a.consensus.normalized_transcript) as normalize:
            result = self.accept(self.complete)
        self.assertEqual(normalize.call_count, 18)
        normalize = a.consensus.normalized_transcript
        self.assertEqual(normalize("你，好。"), "你好")
        for first, second in (("在", "再"), ("窝", "沃"), ("窝", "屋"), ("窝", "我"), ("繁體", "繁体"), ("A", "a"), ("１", "1")):
            self.assertNotEqual(normalize(first), normalize(second))
        self.assertIs(result["training_admitted"], False)

    def test_A_calls_frozen_strict_validators_not_a_fixture_selector(self):
        with mock.patch.object(f.t, "validate_archive", wraps=f.t.validate_archive) as archive, \
             mock.patch.object(f.t, "validate_source_binding", wraps=f.t.validate_source_binding) as source_binding, \
             mock.patch.object(f.t._host(), "validate_public_evidence", wraps=f.t._host().validate_public_evidence) as native:
            self.assertEqual(self.accept(self.complete)["status"], "complete_native_verified")
        self.assertGreater(archive.call_count, 0)
        self.assertGreater(source_binding.call_count, 0)
        self.assertGreater(native.call_count, 0)
        self.assertTrue(all(call.args[1] == self.complete.expected["producer"]["identity"] for call in archive.call_args_list))

    def test_B_original_failed_publisher_and_separate_successful_recovery(self):
        case = copy.deepcopy(self.recovered)
        self.assertEqual(case.payload, self.complete.payload)
        self.assertEqual(case.expected["producer"]["terminal"]["publisher_step_conclusion"], "failure")
        self.assertNotEqual(case.expected["producer"]["identity"], case.expected["recovery"]["identity"])
        self.assertEqual(self.accept(case)["status"], "complete_native_verified")
        self.assertEqual(case.records["producer_jobs"]["jobs"][0]["steps"][1]["conclusion"], "failure")
        result = self.accept(case)
        self.assertEqual(result["producer_terminal"], case.expected["producer"]["terminal"])
        self.assertIs(result["original_publisher_reclassified"], False)
        self.assertIs(result["new_attempts_authorized"], False)

    def test_B_recovery_receipt_must_be_unique_in_raw_stdout(self):
        for mode in ("missing", "only_record", "duplicated", "changed_record", "changed_stdout"):
            case = copy.deepcopy(self.recovered)
            if mode in ("missing", "only_record"):
                case.logs.pop("recovery")
                if mode == "missing":
                    case.records.pop("recovery_receipt")
            elif mode == "duplicated":
                case.logs["recovery"] *= 2
            elif mode == "changed_record":
                case.records["recovery_receipt"]["archive_bytes"] += 1
            else:
                value = copy.deepcopy(case.records["recovery_receipt"])
                value["archive_bytes"] += 1
                case.logs["recovery"] = f.t.canonical(value) + b"\n"
            with self.subTest(mode=mode):
                self.reject(case)

    def test_B_recovery_receipt_never_authorizes_native_retry_or_publisher_reclassification(self):
        for key in ("native_success_established", "new_native_calls_authorized", "retry_authorized", "original_publisher_reclassified"):
            case = copy.deepcopy(self.recovered)
            case.records["recovery_receipt"][key] = True
            case.logs["recovery"] = f.t.canonical(case.records["recovery_receipt"]) + b"\n"
            with self.subTest(key=key): self.reject(case)

    def test_B_recovery_checkpoint_and_original_parent_are_fixed(self):
        for target in ("checkpoint_ref", "result_ref", "result_commit_parent"):
            case = copy.deepcopy(self.recovered)
            if target == "result_commit_parent":
                case.records["result_commit"]["parents"][0]["sha"] = case.expected["recovery"]["identity"]["executed_head_sha"]
            else:
                case.records[target]["object"]["sha"] = "f" * 40
            with self.subTest(target=target):
                self.reject(case)

    def test_C_partial_and_all_notrun_do_not_become_complete(self):
        for case in (self.partial, self.notrun):
            with self.subTest(bytes=len(case.payload)):
                result = self.accept(case)
                self.assertEqual(result["status"], "partial_native_verified")
                self.assertEqual(result["verified_complete_slots"], 0)
                self.assertIsNone(result["consumed_attempts"])
                self.assertIs(result["native_success_established"], False)
                self.assertTrue(all(row["normalized_transcript"] is None for row in result["rows"]))
                self.assertTrue(all(not row["unanimous_machine_candidate"] for row in result["machine_candidates"]))
                self.assertIs(result["input_verified_by_original_host"], case is self.partial)

    def test_C_started_without_native_completion_or_failure_remains_partial_unknown(self):
        case = f.started_unknown_case()
        files = f.members(case.payload)
        self.assertIn("output/qwen06/clip-000001/started.json", files)
        self.assertNotIn("output/qwen06/clip-000001/completed.json", files)
        self.assertNotIn("output/qwen06/worker-failure.json", files)
        result = self.accept(case)
        self.assertEqual(result["status"], "partial_native_verified")
        self.assertEqual(result["verified_complete_slots"], 0)
        self.assertEqual(result["ledger_reported_attempts"], 1)
        self.assertIsNone(result["consumed_attempts"])
        self.assertIs(result["native_success_established"], False)
        self.assertIs(result["new_attempts_authorized"], False)
        self.assertTrue(all(row["normalized_transcript"] is None for row in result["rows"]))

    def test_C_successful_host_cannot_admit_partial_native_receipt(self):
        for original in (self.partial, self.notrun):
            case = copy.deepcopy(original)
            case.expected["producer"]["terminal"]["host_step_conclusion"] = "success"
            case.records["producer_jobs"]["jobs"][0]["steps"][0]["conclusion"] = "success"
            with self.subTest(bytes=len(case.payload)):
                self.reject(case)

    def test_D_actual_archive_body_required_even_with_all_hashes(self):
        for payload in (None, b"", self.complete.expected["archive"]["sha256"], {"sha256": self.complete.expected["archive"]["sha256"]}):
            case = copy.deepcopy(self.complete)
            case.payload = payload
            with self.subTest(payload_type=type(payload).__name__):
                self.reject(case)

    def test_D_complete_ordered_chunk_bodies_required(self):
        for mutation in ("missing", "reverse", "hash_only", "wrong_path", "corrupt", "duplicate", "list_pair"):
            case = copy.deepcopy(self.complete)
            if mutation == "missing": case.chunks.pop()
            elif mutation == "reverse": case.chunks.reverse()
            elif mutation == "hash_only": case.chunks[0] = (case.chunks[0][0], f.t.sha(case.chunks[0][1]))
            elif mutation == "wrong_path": case.chunks[0] = ("unexpected.bin", case.chunks[0][1])
            elif mutation == "corrupt": case.chunks[0] = (case.chunks[0][0], b"!" + case.chunks[0][1][1:])
            elif mutation == "duplicate": case.chunks[1] = case.chunks[0]
            else: case.chunks[0] = list(case.chunks[0])
            with self.subTest(mutation=mutation): self.reject(case)

    def test_D_outer_manifest_raw_bytes_and_exact_canonical_body_required(self):
        for raw in (None, b"", f.t.decode(self.complete.outer_raw), self.complete.outer_raw + b"\n"):
            case = copy.deepcopy(self.complete); case.outer_raw = raw
            with self.subTest(type=type(raw).__name__): self.reject(case)

    def test_E_producer_run_identity_and_workflow_drift(self):
        changes = {"id": 124, "run_attempt": 2, "head_sha": "f" * 40,
                   "head_branch": "foreign-branch", "event": "workflow_dispatch", "path": "foreign.yml"}
        for key, value in changes.items():
            case = copy.deepcopy(self.complete); case.records["producer_run"][key] = value
            with self.subTest(key=key): self.reject(case)
        case = copy.deepcopy(self.complete)
        case.records["producer_run"]["repository"]["private"] = True
        self.reject(case)

    def test_E_complete_unique_source_history_and_numeric_job_binding(self):
        for mode in ("history_count", "history_repeat", "history_identity", "numeric_job", "job_attempt", "job_repeat", "missing_steps", "foreign_step"):
            case = copy.deepcopy(self.complete)
            if mode == "history_count": case.records["producer_history"]["total_count"] = 2
            elif mode == "history_repeat": case.records["producer_history"]["workflow_runs"].append(copy.deepcopy(case.records["producer_run"]))
            elif mode == "history_identity": case.records["producer_history"]["workflow_runs"][0]["head_sha"] = "f" * 40
            elif mode == "numeric_job": case.records["producer_jobs"]["jobs"][0]["id"] += 1
            elif mode == "job_attempt": case.records["producer_jobs"]["jobs"][0]["run_attempt"] = 2
            elif mode == "job_repeat":
                case.records["producer_jobs"]["jobs"].append(copy.deepcopy(case.records["producer_jobs"]["jobs"][0]))
                case.records["producer_jobs"]["total_count"] = 2
            elif mode == "missing_steps": case.records["producer_jobs"]["jobs"][0]["steps"].pop()
            else: case.records["producer_jobs"]["jobs"][0]["steps"][1]["name"] = "Another publisher"
            with self.subTest(mode=mode): self.reject(case)

    def test_E_source_commit_tree_freeze_and_input_pins(self):
        for mode in ("source_sha", "source_tree", "timestamp", "truncated", "blob", "source_freeze", "input_authority"):
            case = copy.deepcopy(self.complete)
            if mode == "source_sha": case.records["producer_source_commit"]["sha"] = "f" * 40
            elif mode == "source_tree": case.records["producer_source_commit"]["tree"]["sha"] = "f" * 40
            elif mode == "timestamp": case.records["producer_source_commit"]["committer"]["date"] = "2026-10-12T00:00:00Z"
            elif mode == "truncated": case.records["producer_source_tree"]["truncated"] = True
            elif mode == "blob": case.records["producer_source_tree"]["tree"][0]["sha"] = "f" * 40
            elif mode == "source_freeze": case.expected["producer"]["source_closure_sha256"] = "f" * 64
            else: case.expected["input_authorities"]["tts_authority_sha256"] = "f" * 64
            with self.subTest(mode=mode): self.reject(case)

    def test_E_each_independent_input_authority_must_match(self):
        for key in self.complete.expected["input_authorities"]:
            case = copy.deepcopy(self.complete); case.expected["input_authorities"][key] = "f" * 64
            with self.subTest(key=key): self.reject(case)

    def test_E_result_commit_metadata_tree_and_foreign_additions(self):
        for mode in ("sha", "message", "author", "parent", "tree_sha", "truncated", "file_blob", "ref"):
            case = copy.deepcopy(self.complete)
            if mode == "sha": case.records["result_commit"]["sha"] = "f" * 40
            elif mode == "message": case.records["result_commit"]["message"] += "changed\n"
            elif mode == "author": case.records["result_commit"]["author"]["name"] = "Unbound author"
            elif mode == "parent": case.records["result_commit"]["parents"][0]["sha"] = "f" * 40
            elif mode == "tree_sha": case.records["result_commit"]["tree"]["sha"] = "f" * 40
            elif mode == "truncated": case.records["result_tree"]["truncated"] = True
            elif mode == "file_blob": case.records["result_tree"]["tree"][0]["sha"] = "f" * 40
            else: case.records["result_ref"]["ref"] = "refs/heads/foreign-evidence"
            with self.subTest(mode=mode): self.reject(case)

    def test_E_coherent_result_tree_may_only_add_exact_original_evidence(self):
        for mode in ("extra_file", "changed_source", "executable_evidence"):
            case = copy.deepcopy(self.complete)
            rows = [row for row in case.records["result_tree"]["tree"] if row["type"] != "tree"]
            if mode == "extra_file":
                rows.append({"path": "unapproved-extra.txt", "mode": "100644", "type": "blob", "sha": f.t.git_sha(b"synthetic extra")})
            elif mode == "changed_source":
                next(row for row in rows if row["path"] == f.CAMPAIGN + "native_worker.py")["sha"] = f.t.git_sha(b"synthetic changed source")
            else:
                next(row for row in rows if row["path"] == f.t.chunk_path(0))["mode"] = "100755"
            tree = f.source_tree_response(rows)
            body, commit = f.p.commit_body(tree["sha"], {"source_sha": case.expected["producer"]["identity"]["executed_head_sha"],
                "commit_timestamp": case.expected["producer"]["source_commit_timestamp"]})
            case.expected["result"] = {"commit": commit, "tree": tree["sha"]}
            case.records["result_tree"] = tree
            case.records["result_ref"]["object"]["sha"] = commit
            case.records["result_commit"].update(sha=commit, tree={"sha": tree["sha"]})
            with self.subTest(mode=mode): self.reject(case)

    def test_E_recovery_identity_workflow_source_and_freeze_drift(self):
        for mode in ("run", "job", "branch", "workflow", "attempt", "source", "freeze", "control", "workflow_body", "step"):
            case = copy.deepcopy(self.recovered)
            if mode == "run": case.records["recovery_run"]["id"] = case.expected["producer"]["identity"]["run_id"]
            elif mode == "job": case.records["recovery_jobs"]["jobs"][0]["id"] += 1
            elif mode == "branch": case.records["recovery_run"]["head_branch"] = "foreign"
            elif mode == "workflow": case.records["recovery_run"]["path"] = f.t.POLICY["workflow"]
            elif mode == "attempt": case.records["recovery_run"]["run_attempt"] = 2
            elif mode == "source": case.records["recovery_source_commit"]["tree"]["sha"] = "f" * 40
            elif mode == "freeze": case.records["recovery_source_freeze"] += b"\n"
            elif mode == "control": case.records["recovery_control"] += b"\n"
            elif mode == "workflow_body": case.records["recovery_workflow"] += b"# foreign\n"
            else: case.records["recovery_jobs"]["jobs"][0]["steps"][0]["name"] = "Unbound recovery step"
            with self.subTest(mode=mode): self.reject(case)

    def test_F_missing_original_source_native_resource_and_claim_bodies(self):
        names = ["authority/source-claim.json", "cpu-receipt.json", "cleanup-receipt.json",
                 "supervision-receipt.json", "policy/budget-proof.json",
                 "output/qwen06/clip-000001/raw-native-output.json",
                 "source/" + f.CAMPAIGN + "native_worker.py"]
        for name in names:
            case = f.mutate_archive(self.complete, lambda files, name=name: files.pop(name))
            with self.subTest(name=name): self.reject(case)

    def test_F_resource_accounting_claims_cannot_be_weakened(self):
        mutations = [("cpu-receipt.json", lambda v: v.update(all_final_usage_verified=False)),
                     ("host-receipt.json", lambda v: v.update(accounting_valid=False)),
                     ("host-receipt.json", lambda v: v["boundary"].update(training_admitted=True)),
                     ("authority/source-claim.json", lambda v: v.update(old_calls_allowed=1))]
        for name, mutation in mutations:
            with self.subTest(name=name):
                self.reject(f.mutate_archive(self.complete, replace_json_member(name, mutation)))

    def test_F_complete_cpu_totals_and_original_thresholds_are_replayed(self):
        def exceeds_threshold(value):
            name = next(name for name, row in value["containers"].items() if row["stage"] == "setup")
            value["containers"][name]["usage_usec"] = 3600 * 1000000
            value["total_usage_usec"] = sum(row["usage_usec"] for row in value["containers"].values())
        mutations = [("total", lambda value: value.update(total_usage_usec=value["total_usage_usec"] + 1)),
                     ("per_stage_threshold", lambda value: value["thresholds_seconds"].update(qwen06=601)),
                     ("total_threshold", lambda value: value.update(total_threshold_seconds=6601)),
                     ("coherent_over_threshold_usage", exceeds_threshold)]
        for label, mutation in mutations:
            with self.subTest(label=label):
                self.reject(f.mutate_archive(self.complete, replace_json_member("cpu-receipt.json", mutation)))

    def test_F_coherent_resource_names_must_match_the_actual_writer(self):
        def wrong_names(files):
            names = f.t.decode(files["cpu-receipt.json"])["containers"]
            f.remap_resource_names(files, {name: name + "-foreign" for name in names})
        self.reject(f.mutate_archive(self.complete, wrong_names))

    def test_F_cpu_and_cleanup_stage_must_match_each_writer(self):
        for member in ("cpu-receipt.json", "cleanup-receipt.json"):
            for stage in ("setup", "sensevoice"):
                def mutate(value, member=member, stage=stage):
                    rows = value["containers"].values() if member == "cpu-receipt.json" else value["containers"]
                    for row in rows:
                        if row["stage"] == "qwen06":
                            row["stage"] = stage
                            return
                    raise AssertionError("original qwen decode resource row required")
                with self.subTest(member=member, stage=stage):
                    self.reject(f.mutate_archive(self.complete, replace_json_member(member, mutate)))

    def test_F_recovery_resource_names_are_bound_to_producer_not_recovery_run(self):
        result = self.accept(self.recovered)
        self.assertEqual(result["status"], "complete_native_verified")
        producer_run = self.recovered.expected["producer"]["identity"]["run_id"]
        recovery_run = self.recovered.expected["recovery"]["identity"]["run_id"]
        self.assertNotEqual(producer_run, recovery_run)
        names = f.t.decode(f.members(self.recovered.payload)["cpu-receipt.json"])["containers"]
        self.assertTrue(all(name.startswith(f.t.c.EXPERIMENT + "-" + str(producer_run) + "-") for name in names))
        foreign = f.mutate_archive(self.complete, lambda files: f.writer_resource_names(files, recovery_run))
        recovered_foreign = f.recovery_case(producer_case=foreign)
        self.reject(recovered_foreign)

    def test_F_completed_cleanup_postconditions_and_partial_observation_boundary(self):
        for field, value in FAILURES:
            with self.subTest(scope="complete_rejection", field=field, value=value):
                case = f.mutate_archive(self.complete, cleanup_mutation(f, field, value))
                with self.assertRaisesRegex(ValueError, REJECTION):
                    self.accept(case)
            with self.subTest(scope="partial_observation", field=field, value=value):
                case = f.mutate_archive(self.partial, cleanup_mutation(f, field, value))
                result = self.accept(case)
                self.assertEqual(result["status"], "partial_native_verified")
                self.assertIs(result["native_success_established"], False)
                self.assertIsNone(result["consumed_attempts"])
                self.assertIs(result["new_attempts_authorized"], False)
        with self.subTest(scope="legitimate_remove_error_then_exact_absence"):
            case = f.mutate_archive(self.complete, valid_remove_fallback(f))
            row = f.t.decode(f.members(case.payload)["cleanup-receipt.json"])["containers"][0]
            self.assertEqual(row["removal_proof"], "successful_empty_exact_id_query")
            self.assertTrue(row["errors"])
            f.t._host()._safe_failure({key: value for key, value in row["errors"][0].items() if key != "operation"})
            result = self.accept(case)
            self.assertEqual(result["status"], "complete_native_verified")
            self.assertIs(result["new_attempts_authorized"], False)

    def test_F_native_text_cannot_be_rewritten_without_bound_native_output(self):
        mutation = replace_json_member("output/qwen06/clip-000001/raw-transcript.json", lambda v: v.update(actual_text="invented alternate text"))
        self.reject(f.mutate_archive(self.complete, mutation))

    def test_F_archive_audio_or_unknown_member_is_not_admitted(self):
        for name, raw in (("private/audio.wav", b"RIFF synthetic no actual audio"), ("unknown.json", b"{}")):
            with self.subTest(name=name):
                self.reject(f.mutate_archive(self.complete, lambda files, name=name, raw=raw: files.__setitem__(name, raw)))

    def test_G_unknown_or_nonterminal_producer_never_becomes_success(self):
        for mode in ("running", "unknown", "job_running", "step_unknown", "expected_unknown"):
            case = copy.deepcopy(self.complete)
            if mode == "running": case.records["producer_run"].update(status="in_progress", conclusion=None)
            elif mode == "unknown": case.records["producer_run"]["conclusion"] = None
            elif mode == "job_running": case.records["producer_jobs"]["jobs"][0].update(status="in_progress", conclusion=None)
            elif mode == "step_unknown": case.records["producer_jobs"]["jobs"][0]["steps"][1]["conclusion"] = None
            else: case.expected["producer"]["terminal"]["publisher_step_conclusion"] = None
            with self.subTest(mode=mode): self.reject(case)

    def test_H_null_ci_diagnostic_is_isolated_and_cannot_upgrade_partial(self):
        for original, status in ((self.complete, "complete_native_verified"), (self.partial, "partial_native_verified")):
            case = copy.deepcopy(original)
            case.logs["producer"] = f.diagnostic_checkpoint(None, stage="complete")
            with self.subTest(status=status):
                result = self.accept(case)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["diagnostics"]["unattributed_ci_or_precontext"], 1)
                self.assertEqual(result["diagnostics"]["attributed"], [])

    def test_H_nonnull_foreign_ci_diagnostic_rejected(self):
        for original in (self.complete, self.recovered):
            case = copy.deepcopy(original)
            identity = copy.deepcopy(case.expected["producer"]["identity"])
            identity["run_id"] += 1
            case.logs["producer"] = f.diagnostic_checkpoint(identity)
            self.reject(case)

    def test_H_checkpoint_envelope_and_fallback_remain_observations(self):
        identity = copy.deepcopy(self.complete.expected["producer"]["identity"])
        logs = (f.diagnostic_checkpoint(identity, stage="readback_verified", role="publisher") +
                f.diagnostic_envelope(identity, stage="complete") + f.diagnostic_fallback(stage="complete"))
        case = copy.deepcopy(self.partial); case.logs["producer"] = logs
        result = self.accept(case)
        self.assertEqual(result["status"], "partial_native_verified")
        self.assertEqual(len(result["diagnostics"]["attributed"]), 2)
        self.assertEqual(result["diagnostics"]["unattributed_ci_or_precontext"], 1)
        self.assertIs(result["native_success_established"], False)
        self.assertIsNone(result["consumed_attempts"])
        self.assertIs(result["original_writer_receipt_observed"], False)
        self.assertEqual(result["original_publisher_http_telemetry"], "unavailable")

    def test_H_diagnostic_envelope_digest_and_role_binding(self):
        for mode in ("digest", "bytes", "foreign_identity", "cross_role"):
            case = copy.deepcopy(self.complete)
            identity = copy.deepcopy(case.expected["producer"]["identity"])
            if mode == "foreign_identity": identity["run_id"] += 1
            raw = f.diagnostic_envelope(identity)
            if mode in ("digest", "bytes"):
                value = f.t.decode(raw)
                if mode == "digest": value["record_sha256"] = "f" * 64
                else: value["record_bytes"] += 1
                raw = f.t.canonical(value) + b"\n"
            case.logs["recovery" if mode == "cross_role" else "producer"] = raw
            with self.subTest(mode=mode): self.reject(case)

    def test_I_failed_or_unknown_recovery_is_observation_only(self):
        for state in ("failure", None):
            case = copy.deepcopy(self.recovered)
            status = "completed" if state else "in_progress"
            case.records["recovery_run"].update(status=status, conclusion=state)
            case.records["recovery_history"]["workflow_runs"][0].update(status=status, conclusion=state)
            job = case.records["recovery_jobs"]["jobs"][0]
            job.update(status=status, conclusion=state)
            job["steps"][0].update(status=status, conclusion=state)
            case.records.pop("recovery_receipt")
            case.logs["recovery"] = f.diagnostic_checkpoint(None, stage="checkpoint_verified", role="publisher")
            with self.subTest(state=state):
                result = self.accept(case)
                self.assertEqual(result["status"], "recovery_not_verified")
                self.assertIs(result["native_success_established"], False)
                self.assertIs(result["new_attempts_authorized"], False)
                self.assertIsNone(result["consumed_attempts"])
                self.assertEqual(result["rows"], [])
                self.assertEqual(result["machine_candidates"], [])
                self.assertIs(result["training_admitted"], False)
                self.assertIs(result["original_publisher_reclassified"], False)

    def test_I_raw_success_receipt_cannot_override_failed_or_unknown_recovery_api(self):
        for conclusion in ("failure", None):
            case = copy.deepcopy(self.recovered)
            for run in (case.records["recovery_run"], case.records["recovery_history"]["workflow_runs"][0]):
                run["conclusion"] = conclusion
            case.records["recovery_jobs"]["jobs"][0]["conclusion"] = conclusion
            case.records["recovery_jobs"]["jobs"][0]["steps"][0]["conclusion"] = conclusion
            result = self.accept(case)
            with self.subTest(conclusion=conclusion):
                self.assertEqual(result["status"], "recovery_not_verified")
                self.assertIs(result["native_success_established"], False)
                self.assertEqual(result["rows"], [])
                self.assertIs(result["new_attempts_authorized"], False)

    def test_Z_no_network_subprocess_model_or_writes_during_accept(self):
        # Install the effect guard before importing or generating fixtures.
        # Fixture temporary writes are allowed; acceptance itself stays read-only.
        script = r'''
import sys,json
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
forbidden = {'torch','numpy','transformers','whisper','funasr','qwen_asr'}
phase = 'fixture'
observed = {'external_attempts':0,'model_import_attempts':0,'accept_write_attempts':0}
def guard(event, args):
    if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system', 'os.fork',
                                              'http.client.connect', 'http.client.send', 'urllib.Request'):
        observed['external_attempts'] += 1
        raise AssertionError('forbidden external effect')
    if event == 'import' and args[0].split('.')[0] in forbidden:
        observed['model_import_attempts'] += 1
        raise AssertionError('forbidden model import')
    if phase == 'accept' and event == 'open':
        mode, flags = args[1], args[2]
        if (isinstance(mode, str) and any(k in mode for k in 'wax+')) or (
            isinstance(flags, int) and flags & (1 | 2 | 64 | 512 | 1024)):
            observed['accept_write_attempts'] += 1
            raise AssertionError('forbidden acceptance write')
sys.addaudithook(guard)
import fixture_cases as f
case = f.normal_case()
phase = 'accept'
import asr_posterior as a
result = a.accept(**case.arguments())
if result['status'] != 'complete_native_verified' or forbidden.intersection(sys.modules):
    raise AssertionError('guarded result or imports')
if any(observed.values()):
    raise AssertionError('caught forbidden operation is still a failed guard')
print(json.dumps({'status':'synthetic guarded acceptance passed', **observed,
                  'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,
                  'no_bytecode':sys.dont_write_bytecode,'optimization':sys.flags.optimize}))
'''
        flags = ['-I', '-S', '-B'] + (['-' + 'O' * sys.flags.optimize] if sys.flags.optimize else [])
        run = subprocess.run([sys.executable, *flags, "-c", script, str(f.HERE)], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        value = json.loads(run.stdout)
        self.assertEqual(value, {'status':'synthetic guarded acceptance passed',
            'external_attempts':0,'model_import_attempts':0,'accept_write_attempts':0,
            'isolated':1,'no_site':1,'no_bytecode':True,'optimization':sys.flags.optimize})

    def test_Z_public_ci_is_fixed_offline_and_separate_from_existing_contract_job(self):
        workflow = (f.SOURCE / '.github/workflows/ci.yml').read_text()
        job = workflow.split('  asr-offline-posterior:\n', 1)[1].split('  research-sources:\n', 1)[0]
        self.assertIn('    runs-on: ubuntu-24.04\n', job)
        self.assertIn('    timeout-minutes: 15\n', job)
        self.assertIn('persist-credentials: false', job)
        self.assertIn('actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1', job)
        self.assertIn('("test_asr_posterior.py", "test_offline_custody.py")', job)
        for flags in ('', '-O ', '-OO '):
            self.assertIn('python3 -I -S -B ' + flags + '-c "$posterior_tests"', job)
        self.assertIn('if: always()', job)
        self.assertIn('git diff --exit-code HEAD --', job)
        self.assertIn('  contents: read\n', workflow.split('jobs:', 1)[0])
        for forbidden in ('contents: write', 'pip install', 'GITHUB_TOKEN', 'upload-artifact',
                          'qualification.py', 'hosted_run.py', 'transfer_recovery.py'):
            self.assertNotIn(forbidden, job)
        shared = workflow.split('  python-contracts:\n', 1)[1].split('  asr-offline-posterior:\n', 1)[0]
        self.assertNotIn('test_asr_posterior.py', shared)
        self.assertNotIn('test_offline_custody.py', shared)

    def test_Z_frozen_validator_sources_remain_identical(self):
        manifest = json.loads((f.HERE / "validator-dependencies.json").read_bytes())
        for row in manifest["files"]:
            raw = (f.SOURCE / row["path"]).read_bytes()
            with self.subTest(path=row["path"]):
                self.assertEqual(len(raw), row["bytes"])
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
                self.assertEqual(f.t.git_sha(raw), row["git_blob_sha1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
