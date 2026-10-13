"""Fresh public zero-model ASR fixtures; never actual run or acoustic evidence.

The existing six-WAV host/native/receipt/collector test creates archives in a
TemporaryDirectory. This module only binds its current public source closure
and explicit synthetic API/Git observations. It reads no historical ZIP, Git
checkout, model, private corpus or external service. Production controls and
validators remain unchanged. All generated success claims are synthetic.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
CAMPAIGN = "research/k1-punctuation6-asr18-v1/"
sys.path.insert(0, str(SOURCE / CAMPAIGN))
import evidence_transport as t
import evidence_publisher as p
import transfer_recovery as r

SYNTHETIC_NOTICE = "Synthetic offline fixture; not actual ASR evidence or an actual GitHub run."
RECOVERY_STEP = "Recover exact original ASR evidence without model calls"
RESOURCE_MEMBERS = frozenset(("cpu-receipt.json", "cleanup-receipt.json", "supervision-receipt.json"))


@dataclass
class Case:
    expected: dict
    records: dict
    outer_raw: bytes
    chunks: list
    payload: bytes
    logs: dict = field(default_factory=dict)
    observations: dict = field(default_factory=dict)
    synthetic_notice: str = SYNTHETIC_NOTICE

    def arguments(self):
        return dict(expected=self.expected, records=self.records, outer_raw=self.outer_raw,
                    chunks=self.chunks, payload=self.payload, logs=self.logs)


def source_tree_response(rows):
    digest, directories = t.tree_identity(rows)
    return {"synthetic": True, "synthetic_notice": SYNTHETIC_NOTICE,
            "sha": digest, "truncated": False, "tree": copy.deepcopy(rows) + [
        {"path": name, "mode": "040000", "type": "tree", "sha": sha}
        for name, sha in sorted(directories.items())]}


def fresh_fixture(name):
    """Bind a fresh six-WAV helper to a deterministic synthetic public Git tree.

    The tree contains the helper's explicit current public source closure, with
    invented test-only release/prerequisite metadata already supplied by
    public_fixture(). It is not a claim about any repository's actual HEAD.
    """
    t.require(name in ("success18", "first_native_failure18", "all_notrun"), "fixed synthetic fixture name")
    from contextlib import ExitStack
    from unittest.mock import patch
    from test_asr18_host import CompleteSyntheticEndToEnd, public_fixture
    files, identity = public_fixture()
    import prepare_freeze
    freeze = prepare_freeze.build(SOURCE)
    source_files = {name: (files["source/" + name] if "source/" + name in files else
                          t.c.read_bytes(SOURCE / name, cap=8 * t.c.MIB))
                    for name in sorted(freeze["host_code"])}
    for role in ("host_code", "setup_code", "inference_code"):
        freeze[role] = {name: {"bytes": len(source_files[name]), "sha256": t.sha(source_files[name])}
                        for name in freeze[role]}
    freeze_raw = t.canonical(freeze)
    source_files[CAMPAIGN + "execution-freeze.json"] = freeze_raw
    files.update({"source/" + name: raw for name, raw in source_files.items()})
    files["authority/execution-freeze.json"] = freeze_raw
    rows = [{"path": path[7:], "mode": "100644", "type": "blob", "sha": t.git_sha(raw)}
            for path, raw in sorted(files.items()) if path.startswith("source/")]
    tree = source_tree_response(rows)["sha"]
    stamp = "2026-10-13T00:00:00Z"
    person = "Synthetic Fixture <fixture@example.invalid> 1791849600 +0000"
    commit_raw = ("tree " + tree + "\nauthor " + person + "\ncommitter " + person +
                  "\n\nSynthetic current public source closure; zero model fixture only\n").encode("ascii")
    head = t.git_sha(commit_raw, "commit")
    previous_head = identity["executed_head_sha"]
    def rebind(value):
        if type(value) is dict:return {key: rebind(item) for key, item in value.items()}
        if type(value) is list:return [rebind(item) for item in value]
        return head if value == previous_head else value
    identity = rebind(identity)
    files = {path: (raw if path.startswith("source/") else t.canonical(rebind(t.decode(raw))))
             for path, raw in files.items()}
    fixture = {"files": files, "identity": identity, "source_tree": tree}
    # Guard the zero-I/O test boundary even if a helper regresses later.
    with ExitStack() as stack:
        for target in ("subprocess.Popen", "socket.create_connection", "socket.socket.connect", "urllib.request.urlopen"):
            stack.enter_context(patch(target, side_effect=AssertionError("synthetic fixtures forbid network/process execution")))
        result = CompleteSyntheticEndToEnd().run_case(fail_native=name == "first_native_failure18",
            fail_input=name == "all_notrun", fixture=fixture)
    t.validate_source_binding(result["payload"], identity,
        t.sha(files["authority/execution-freeze.json"]), rows, 456, tree)
    result.update(source_rows=rows, source_commit_timestamp=stamp, source_commit_raw=commit_raw)
    return result


def verify_native_layout(payload, name):
    """Exact native-orchestration sidecars, independently of archive acceptance.

    Constructor/acquisition/kernel work is intentionally synthetic. This checks
    every file emitted by the reused _execute_family/ReceiptFixture path, not a
    real installed-model or bounded-acquisition output layout.
    """
    files = members(payload)
    shared = {"native-source-lock.json", "native-asset-lock.json", "native-config.json",
              "retained-source/invented.py", "kernel-scope.json", "cpu-process-guard.json",
              "blind-job.json", "execution-plan.json"}
    expected = set()
    if name != "all_notrun":
        for family in t.c.FAMILIES:
            expected.update("frozen/" + family + "/" + leaf for leaf in shared | {"prepared.json", "worker-complete.json"})
        families = ("qwen06",) if name == "first_native_failure18" else t.c.FAMILIES
        plan = t.decode(files["order/execution-plan.json"])
        for family in families:
            prefix = "output/" + family + "/"
            failure = name == "first_native_failure18"
            expected.update(prefix + leaf for leaf in shared | {"worker-failure.json" if failure else "worker-complete.json"})
            audio_ids = plan["audio_order"][:1] if failure else plan["audio_order"]
            for audio_id in audio_ids:
                clip = prefix + audio_id + "/"
                leaves = {"started.json"}
                if not failure:
                    leaves |= {"raw-native-output.json", "raw-transcript.json", "completed.json"}
                    if family == "whisper":
                        leaves.add("raw-output.json")
                        raw = t.decode(files[clip + "raw-native-output.json"])
                        count = len(raw["native_capture"]["termination"]["generated_token_ids"])
                        leaves.update("update-%03d.json" % index for index in range(1, count + 1))
                    else:
                        leaves.add("raw-native-row.json")
                expected.update(clip + leaf for leaf in leaves)
    observed = {path for path in files if path.startswith(("output/", "frozen/"))}
    t.require(observed == expected, "exact generated native orchestration sidecars; none omitted or filtered")
    return {"scope": "reused native orchestration with explicit fake constructor/kernel/acquisition",
            "complete_native_member_roster": sorted(expected), "native_members": len(expected),
            "all_required_generated_sidecars_present": True, "members_removed_to_fit": 0,
            "not_real_model_constructor_layout": True}


def mark_api_records(case):
    """API-shaped snapshots always carry an explicit local synthetic marker."""
    for role, value in case.records.items():
        if type(value) is dict and role != "recovery_receipt":
            value.update(synthetic=True, synthetic_notice=SYNTHETIC_NOTICE)
    return case


def members(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        return {name: archive.read(name) for name in archive.namelist() if name != t.MEMBER_MANIFEST}


def run_records(prefix, identity, numeric_job, branch, workflow, terminal, steps):
    run = {"id": identity["run_id"], "run_attempt": 1, "head_sha": identity["executed_head_sha"],
           "head_branch": branch, "event": "push", "path": workflow,
           "status": "completed", "conclusion": terminal["run_conclusion"],
           "repository": {"full_name": t.REPOSITORY, "private": False}}
    job = {"id": numeric_job, "run_id": identity["run_id"], "run_attempt": 1,
           "name": identity["job_key"], "head_sha": identity["executed_head_sha"],
           "head_branch": branch, "run_url": "https://api.github.com" + p.BASE + "/actions/runs/" + str(identity["run_id"]),
           "status": "completed", "conclusion": terminal["job_conclusion"],
           "steps": [{"name": name, "number": i + 1, "status": "completed", "conclusion": conclusion}
                     for i, (name, conclusion) in enumerate(steps)]}
    return {prefix + "_run": run,
            prefix + "_history": {"total_count": 1, "workflow_runs": [copy.deepcopy(run)]},
            prefix + "_jobs": {"total_count": 1, "jobs": [job]}}


def result_bindings(case, files, producer_rows):
    """Recompute the exact added-file Git tree and fixed-author single-parent commit."""
    producer = case.expected["producer"]
    rows = copy.deepcopy(producer_rows) + [
        {"path": name, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)}
        for name, raw in sorted(files.items())]
    tree = source_tree_response(rows)
    body, commit = p.commit_body(tree["sha"], {"source_sha": producer["identity"]["executed_head_sha"],
        "commit_timestamp": producer["source_commit_timestamp"]})
    case.expected["result"] = {"commit": commit, "tree": tree["sha"]}
    case.records.update(result_ref={"ref": "refs/heads/" + t.POLICY["result_branch"],
                                   "object": {"type": "commit", "sha": commit}},
        result_commit={"sha": commit, "message": body["message"], "tree": {"sha": body["tree"]},
                       "parents": [{"sha": sha} for sha in body["parents"]],
                       "author": body["author"], "committer": body["committer"]}, result_tree=tree)


@lru_cache(maxsize=3)
def _normal_case(name="success18"):
    if name not in ("success18", "first_native_failure18", "all_notrun"):
        raise ValueError("fixed synthetic fixture names only")
    generated = fresh_fixture(name)
    payload = generated["payload"]
    files_in_zip = members(payload)
    host = t.decode(files_in_zip["host-receipt.json"])
    preflight = t.decode(files_in_zip["run-preflight.json"])
    identity = host["identity"]
    head, tree, stamp = identity["executed_head_sha"], generated["source_tree"], generated["source_commit_timestamp"]
    rows = generated["source_rows"]
    t.require(head == t.git_sha(generated["source_commit_raw"], "commit") and
              source_tree_response(rows)["sha"] == tree, "computed synthetic Git source binding")
    freeze_sha = t.sha(files_in_zip["authority/execution-freeze.json"])
    worker_raw = files_in_zip.get("order/worker-input-binding.json")
    if worker_raw is None:
        worker_raw = t.canonical(t.decode(files_in_zip["input-freeze.json"])["actual_worker_binding"])
    worker = t.decode(worker_raw)
    complete = host["status"] == "all_planned_native_complete"
    terminal = {"run_conclusion": "success" if complete else "failure",
                "job_conclusion": "success" if complete else "failure",
                "host_step_conclusion": "success" if complete else "failure",
                "publisher_step_conclusion": "success", "host_receipt_status": host["status"]}
    expected = {"schema": "k1-asr18-offline-acceptance-v1", "mode": "normal_publication",
        "producer": {"identity": identity, "numeric_job_id": preflight["numeric_job_id"],
                     "source_tree": tree, "source_closure_sha256": freeze_sha,
                     "source_commit_timestamp": stamp, "terminal": terminal},
        "archive": {}, "result": {}, "recovery": None,
        "input_authorities": {key: worker[key] for key in
            ("tts_authority_sha256", "screening_receipt_sha256", "job_sha256", "execution_plan_sha256")}}
    expected["input_authorities"]["worker_input_binding_sha256"] = t.sha(worker_raw)
    records = run_records("producer", identity, preflight["numeric_job_id"], t.POLICY["source_branch"],
        t.POLICY["workflow"], terminal, [(r.HOST_STEP, terminal["host_step_conclusion"]), (r.PUBLISHER_STEP, "success")])
    records.update(producer_source_commit={"sha": head, "tree": {"sha": tree}, "committer": {"date": stamp}},
                   producer_source_tree=source_tree_response(rows))
    files = t.build_file_set(payload, identity, freeze_sha, preflight["numeric_job_id"])
    outer = files[t.manifest_path()]
    expected["archive"] = {"bytes": len(payload), "sha256": t.sha(payload), "outer_manifest_sha256": t.sha(outer)}
    case = Case(expected, records, outer, [(row["path"], files[row["path"]]) for row in t.decode(outer)["chunks"]], payload)
    result_bindings(case, files, rows)
    layout = verify_native_layout(payload, name)
    case.observations = {"synthetic": True, "native_layout": layout, "source_scope": "current explicit public source closure, test-only controls",
        "events": generated["events"], "staged_wavs": generated["staged_wavs"],
        "collector_member_count": generated["collector_member_count"],
        "strict_source_binding_verified": True,
        "source_file_count": len(rows), "source_commit_raw": generated["source_commit_raw"].decode("ascii"),
        "archive_generated_by": "CompleteSyntheticEndToEnd.run_case -> hosted_run.execute -> collect_public/build_package"}
    return mark_api_records(case)


def original_case(name="success18"):
    """Fresh helper output, including correct original-writer resource names."""
    return copy.deepcopy(_normal_case(name))


def normal_case(name="success18"):
    """Self-contained synthetic case; no historical input or resource rewriting."""
    return original_case(name)


def recovery_authority(case):
    producer = case.expected["producer"]
    return {"schema": "k1-asr18-transfer-recovery-authority-v1", "producer_identity": producer["identity"],
        "numeric_job_id": producer["numeric_job_id"], "source_tree": producer["source_tree"],
        "source_closure_sha256": producer["source_closure_sha256"], "source_commit_timestamp": producer["source_commit_timestamp"],
        "archive_bytes": case.expected["archive"]["bytes"], "archive_sha256": case.expected["archive"]["sha256"],
        "outer_manifest_sha256": case.expected["archive"]["outer_manifest_sha256"],
        "checkpoint_ref": "refs/heads/" + t.POLICY["result_branch"] + "-checkpoint",
        "checkpoint_commit": case.expected["result"]["commit"], "checkpoint_tree": case.expected["result"]["tree"],
        "terminal": copy.deepcopy(producer["terminal"])}


def recovery_case(name="success18", *, producer_case=None):
    case = normal_case(name) if producer_case is None else copy.deepcopy(producer_case)
    case.expected["mode"] = "recovered_checkpoint"
    producer = case.expected["producer"]
    producer["terminal"].update(run_conclusion="failure", job_conclusion="failure", publisher_step_conclusion="failure")
    case.records.update(run_records("producer", producer["identity"], producer["numeric_job_id"],
        t.POLICY["source_branch"], t.POLICY["workflow"], producer["terminal"],
        [(r.HOST_STEP, producer["terminal"]["host_step_conclusion"]), (r.PUBLISHER_STEP, "failure")]))
    authority = recovery_authority(case)
    r.validate_authority(authority)
    control = r.disabled_release()
    for key in r.APPROVALS:
        control[key] = True
    control["authority"] = authority
    control_raw = t.canonical(control)
    workflow_raw = ("# Synthetic offline fixture only; never installed or executed.\n"
        "name: Synthetic transfer recovery observation\non:\n  push:\n    branches: [" + r.SOURCE_BRANCH + "]\n"
        "permissions:\n  contents: write\njobs:\n  transfer_recovery:\n    runs-on: ubuntu-24.04\n"
        "    steps:\n      - name: " + RECOVERY_STEP + "\n"
        "        run: python3 research/k1-punctuation6-asr18-v1/transfer_recovery.py\n").encode("ascii")
    files_in_zip = members(case.payload)
    freeze = t.decode(files_in_zip["authority/execution-freeze.json"])
    new_sources = {CAMPAIGN + "transfer-recovery-release.json": control_raw, r.WORKFLOW: workflow_raw}
    for name, raw in new_sources.items():
        freeze["host_code"][name] = {"bytes": len(raw), "sha256": t.sha(raw)}
    freeze_raw = t.canonical(freeze)
    new_sources[CAMPAIGN + "execution-freeze.json"] = freeze_raw
    rows = [copy.deepcopy(row) for row in case.records["producer_source_tree"]["tree"]
            if row["type"] != "tree" and row["path"] not in new_sources]
    rows += [{"path": name, "mode": "100644", "type": "blob", "sha": t.git_sha(raw)} for name, raw in sorted(new_sources.items())]
    tree = source_tree_response(rows)
    stamp = "2026-10-13T00:01:00Z"
    raw_commit = ("tree " + tree["sha"] + "\nparent " + producer["identity"]["executed_head_sha"] +
        "\nauthor Synthetic Fixture <fixture@example.invalid> 1791849660 +0000\n"
        "committer Synthetic Fixture <fixture@example.invalid> 1791849660 +0000\n\n"
        "Synthetic offline recovery source fixture\n").encode("ascii")
    identity = {"repository": t.REPOSITORY, "run_id": 789, "run_attempt": 1,
                "job_key": r.JOB_KEY, "executed_head_sha": t.git_sha(raw_commit, "commit")}
    recovery = {"identity": identity, "numeric_job_id": 987, "source_tree": tree["sha"],
        "source_closure_sha256": t.sha(freeze_raw), "source_commit_timestamp": stamp, "step_name": RECOVERY_STEP}
    case.expected["recovery"] = recovery
    case.records.update(run_records("recovery", identity, 987, r.SOURCE_BRANCH, r.WORKFLOW,
        {"run_conclusion": "success", "job_conclusion": "success"}, [(RECOVERY_STEP, "success")]))
    case.records.update(recovery_source_commit={"sha": identity["executed_head_sha"], "tree": {"sha": tree["sha"]},
        "committer": {"date": stamp}}, recovery_source_tree=tree, recovery_source_freeze=freeze_raw,
        recovery_control=control_raw, recovery_workflow=workflow_raw,
        checkpoint_ref={"ref": authority["checkpoint_ref"], "object": {"type": "commit", "sha": authority["checkpoint_commit"]}})
    receipt = {"schema": "k1-asr18-transfer-recovery-receipt-v1", "status": "exact_original_checkpoint_final_ref_verified",
        "producer_identity": copy.deepcopy(producer["identity"]), "producer_numeric_job_id": producer["numeric_job_id"],
        "original_terminal": copy.deepcopy(producer["terminal"]), "original_run_status": "completed",
        "original_job_status": "completed", "original_publisher_reclassified": False,
        "recovery_identity": identity, "recovery_numeric_job_id": 987,
        "recovery_source_closure_sha256": recovery["source_closure_sha256"],
        "checkpoint_ref": authority["checkpoint_ref"], "checkpoint_commit": authority["checkpoint_commit"],
        "checkpoint_tree": authority["checkpoint_tree"], "final_ref": "refs/heads/" + t.POLICY["result_branch"],
        "archive_sha256": authority["archive_sha256"], "archive_bytes": authority["archive_bytes"],
        "outer_manifest_sha256": authority["outer_manifest_sha256"], "native_success_established": False,
        "new_native_calls_authorized": False, "retry_authorized": False}
    case.records["recovery_receipt"] = receipt
    case.logs["recovery"] = t.canonical(receipt) + b"\n"
    return mark_api_records(case)


def mutate_archive(case, mutation):
    """Coherent rehash for strict negative cases, never a validator bypass in accept.

Rebind ZIP/outer/file tree/checkpoint pins so rejection must inspect actual member
semantics. Original archive and source bytes on disk remain untouched.
    """
    case = copy.deepcopy(case)
    files_in_zip = members(case.payload)
    mutation(files_in_zip)
    identity = case.expected["producer"]["identity"]
    payload, manifest, expanded = t._zip_files(files_in_zip, identity)
    outer = t.decode(case.outer_raw)
    chunks = [(t.chunk_path(i // t.MAX_CHUNK), payload[i:i + t.MAX_CHUNK]) for i in range(0, len(payload), t.MAX_CHUNK)]
    outer.update(archive_bytes=len(payload), archive_sha256=t.sha(payload),
                 manifest_sha256=t.sha(t.canonical(manifest)), member_count=len(files_in_zip), uncompressed_bytes=expanded,
                 chunk_count=len(chunks), chunks=[{"index": i, "path": name, "bytes": len(raw), "sha256": t.sha(raw)}
                     for i, (name, raw) in enumerate(chunks)])
    case.payload = payload
    case.outer_raw = t.canonical(outer)
    case.chunks = chunks
    case.expected["archive"] = {"bytes": len(payload), "sha256": t.sha(payload), "outer_manifest_sha256": t.sha(case.outer_raw)}
    rows = [row for row in case.records["producer_source_tree"]["tree"] if row["type"] != "tree"]
    result_bindings(case, {t.manifest_path(): case.outer_raw, **dict(chunks)}, rows)
    return mark_api_records(case)


def writer_resource_names(files, producer_run_id):
    """Test mutation utility for three synthetic resource bodies.

The unchanged writers derive setup and family-mode names from EXPERIMENT and the
producer run. Existing CPU/cleanup stages and every other field stay untouched.
    """
    prefix = t.c.EXPERIMENT + "-" + str(producer_run_id) + "-"
    supervision = t.decode(files["supervision-receipt.json"])
    expected = [("setup", None)] + [(family, "prepare") for family in t.c.FAMILIES] + [(family, "decode") for family in t.c.FAMILIES]
    if [(row["family"], row["mode"]) for row in supervision["stages"]] != expected:
        raise ValueError("synthetic full fixture requires the original seven-stage roster")
    mapping = {row["name"]: prefix + ("setup" if row["family"] == "setup" else row["family"] + "-" + row["mode"])
               for row in supervision["stages"]}
    remap_resource_names(files, mapping)


def remap_resource_names(files, mapping):
    """Coherently rename the shared resource identities for explicit fixtures."""
    cpu = t.decode(files["cpu-receipt.json"])
    cleanup = t.decode(files["cleanup-receipt.json"])
    supervision = t.decode(files["supervision-receipt.json"])
    if set(cpu["containers"]) != set(mapping):
        raise ValueError("complete original synthetic CPU roster")
    cpu["containers"] = {mapping[name]: row for name, row in cpu["containers"].items()}
    for row in cleanup["containers"]:
        row["name"] = mapping[row["name"]]
    for row in supervision["stages"]:
        row["name"] = mapping[row["name"]]
    for name, value in (("cpu-receipt.json", cpu), ("cleanup-receipt.json", cleanup), ("supervision-receipt.json", supervision)):
        files[name] = t.canonical(value)


def started_unknown_case():
    """Synthetic partial: fresh first-slot start retained, no native completion.

Remove the family failure observation only. The independent terminal ledger is
still its original failed_no_retry report; actual consumed attempts stay unknown.
No new unsupported ledger status or model observation is invented.
    """
    return mutate_archive(normal_case("first_native_failure18"),
        lambda files: files.pop("output/qwen06/worker-failure.json"))


def diagnostic_checkpoint(identity=None, stage="entry", role="host"):
    return t.canonical({"schema": "k1-asr18-safe-checkpoint-v1", "role": role,
        "stage": stage, "sequence": 1, "identity": identity, "native_calls": "not_asserted"}) + b"\n"


def diagnostic_envelope(identity=None, stage="entry", role="host"):
    import failure_observer as observer
    record = {"schema": "k1-asr18-independent-process-diagnostic-v1", "role": role,
        "identity": identity, "stage": stage, "status": "failed_or_unconfirmed", "errors": [],
        "attachment_error": None, "checkpoint_output_error": None, "checkpoint_limit_reached": False,
        "errors_truncated": False, "claims": observer.unknown_claims(), "checkpoint_count": 0,
        "original_writer_receipt_included": False, "native_success_established": False,
        "retry_authorized": False,
        "limit": "ordinary_exception_observation; abrupt_process_or_runner_loss_may_prevent_final_record"}
    raw = t.canonical(record)
    return t.canonical({"schema": "k1-asr18-independent-process-diagnostic-envelope-v1", "record": record,
        "record_bytes": len(raw), "record_sha256": t.sha(raw),
        "persistence": {"status": "unconfigured", "error": None}}) + b"\n"


def diagnostic_fallback(stage="entry", role="host"):
    return t.canonical({"schema": "k1-asr18-independent-diagnostic-fallback-v1", "role": role,
        "stage": stage, "status": "diagnostic_failed", "consumed_attempts": None, "retry_authorized": False}) + b"\n"


if __name__ == "__main__":
    print(SYNTHETIC_NOTICE)
    for name in ("success18", "first_native_failure18", "all_notrun"):
        case = normal_case(name)
        print(json.dumps({"fixture": name, "synthetic": True, "archive": case.expected["archive"],
                          "producer": case.expected["producer"], "result": case.expected["result"]}, sort_keys=True))
    case = recovery_case()
    print(json.dumps({"fixture": "recovered_success18", "synthetic": True, "recovery": case.expected["recovery"]}, sort_keys=True))
