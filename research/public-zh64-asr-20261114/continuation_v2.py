"""First Whisper16 execution with independently checked fresh installer caches.

Both stopped parent runs, their claims and every prior source remain immutable.
Only declared cache hash variation with actual source-compile evidence may
cross the runtime identity boundary; complete within-run verification remains.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continuation as prior
import cache_verify_v2 as cache

host, inputs = prior.host, prior.inputs
need, read, parse, sha, canonical, write, put = prior.need, prior.read, prior.parse, prior.sha, prior.canonical, prior.write, prior.put
MIB = 1024 ** 2
CAMPAIGN = "public-zh64-whisper16-first-v2-20261114-v1"
REQUEST_CANONICAL_SHA = "57197c31f1fee82a5be344c7124217d3c15de951a290649d75b948745769a18d"
INVENTORY = "bindings/runtime/installed-inventory.json"
PREPARED = prior.PARENT_PREPARED
UNADMITTED = prior.UNADMITTED


def frozen_request(path=HERE / "continuation_v2_request.json"):
    raw = read(path, MIB)
    request = parse(raw)
    need(sha(canonical(request)) == REQUEST_CANONICAL_SHA and
         sha(canonical(request["cache_contract"])) == cache.CONTRACT_SHA, "frozen v2 request and cache proof contract")
    return request, sha(raw)


def verify_control(request):
    prior.verify_control()
    for name, digest in request["prior_control_source_sha256"].items():
        need(sha(read(HERE / name, MIB)) == digest, "all consumed controller/request sources remain unchanged")
    old, _ = prior.frozen_request()
    need(request["slots"] == old["slots"] and request["original_parent_artifact_sha256"] == old["parent_artifact_sha256"],
         "exact same sixteen first-audio cells and original parent")


def claim_ref(request_sha):
    need(re.fullmatch("[0-9a-f]{64}", request_sha), "request SHA256")
    return "refs/tags/" + CAMPAIGN + "-consumed"


def failed_parent_request(request):
    failed = request["failed_parent"]
    return dict(parent_run_id=failed["run_id"], parent_head_sha=failed["head_sha"], parent_claim_ref=failed["claim_ref"])


def verify_live_parents(original_claim, original_run, failed_claim, failed_run, request):
    original, _ = prior.frozen_request()
    prior.parent_claim(original_claim, original)
    prior.verify_parent_run(original_run, original)
    failed = failed_parent_request(request)
    prior.parent_claim(failed_claim, failed)
    prior.verify_parent_run(failed_run, failed)


def validate_execution(env, event, claim, request, digest):
    need(env.get("GITHUB_REPOSITORY") == "jiying2007/kws-pipeline" and env.get("GITHUB_EVENT_NAME") == "push" and
         env.get("GITHUB_REF") == "refs/heads/main" and env.get("GITHUB_RUN_ATTEMPT") == "1" and
         env.get("GITHUB_ACTIONS") == "true" and env.get("RUNNER_ENVIRONMENT") == "github-hosted" and
         env.get("RUNNER_OS") == "Linux" and env.get("RUNNER_ARCH") == "X64", "first public protected-main hosted execution only")
    head, run = env.get("GITHUB_SHA", ""), env.get("GITHUB_RUN_ID", "")
    original, _ = prior.frozen_request()
    need(re.fullmatch("[0-9a-f]{40}", head) and re.fullmatch("[1-9][0-9]{0,19}", run) and
         head not in (original["parent_head_sha"], request["failed_parent"]["head_sha"]) and
         run not in (str(original["parent_run_id"]), str(request["failed_parent"]["run_id"])), "new actual run and controller head")
    need(event.get("after") == head and event.get("ref") == "refs/heads/main" and event.get("deleted") is False and
         event.get("repository", {}).get("private") is False and event["repository"].get("full_name") == "jiying2007/kws-pipeline",
         "actual public main push")
    need(env.get("GITHUB_WORKFLOW_SHA") == head and env.get("GITHUB_WORKFLOW_REF") ==
         "jiying2007/kws-pipeline/.github/workflows/public-zh64-asr-once.yml@refs/heads/main", "actual workflow identity")
    need(claim.get("ref") == claim_ref(digest) and claim.get("object", {}).get("sha") == head and
         claim["object"].get("type") == "commit", "new permanent v2 claim")
    need(not any(key in env for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN")), "no model-host credentials")
    return head, run


def failed_archive(raw, request):
    expected = request["failed_parent"]
    need(len(raw) == expected["artifact_bytes"] and sha(raw) == expected["artifact_sha256"], "exact second failed parent ZIP")
    files, total = {}, 0
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) == expected["zip_members"] == len({row.filename for row in infos}), "203 unique failed-parent members")
        for info in infos:
            name = prior.member(info.filename)
            need(not info.is_dir() and not (info.flag_bits & 1) and
                 stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK and 0 <= info.file_size <= 32 * MIB,
                 "regular bounded failed-parent member")
            total += info.file_size
            need(total <= 32 * MIB, "bounded failed-parent expansion")
            files[name] = archive.read(info)
            need(len(files[name]) == info.file_size, "complete failed-parent bytes")
    index = parse(files["artifact-index.json"])
    need(index["schema"] == "public-zh64-whisper16-retained-evidence-index-v1" and
         set(files) == set(index["files"]) | {"artifact-index.json"} and
         len(index["files"]) == expected["indexed_members"] and index["evidence_bytes"] == expected["indexed_bytes"],
         "closed failed-parent inventory")
    for name, value in index["files"].items():
        need(cache.identity(files[prior.member(name)]) == value, "failed-parent member byte identity")
    need(sum(len(files[name]) for name in index["files"]) == index["evidence_bytes"], "failed-parent evidence byte total")
    for name, value in expected["members"].items():
        need(cache.identity(files[name]) == value, "frozen failed-parent control evidence")
    return files


def source_identity(old_identity, fresh_identity, old_source, fresh_source, old_inventory, fresh_inventory, head, contract):
    """Only actual head and the fully constrained fresh inventory may change."""
    cache.compare_inventories(parse(old_inventory), parse(fresh_inventory), contract)
    need(cache.identity(old_inventory) == contract["baseline_inventory"] and
         len(fresh_inventory) == len(old_inventory), "fixed complete runtime inventory representation size")
    expected_source = copy.deepcopy(old_source)
    expected_source["executed_head_sha"] = head
    changed = [row for row in expected_source["files"] if row["path"] == INVENTORY]
    need(len(changed) == 1 and changed[0]["sha256"] == sha(old_inventory), "one original runtime inventory binding")
    changed[0].update(cache.identity(fresh_inventory))
    need(canonical(fresh_source) == canonical(expected_source), "only actual head and declared inventory source identity may vary")
    expected_identity = copy.deepcopy(old_identity)
    need(expected_identity["native_decode_config"]["source_lock_sha256"] == sha(canonical(old_source)), "original source binding")
    expected_identity["native_decode_config"]["source_lock_sha256"] = sha(canonical(fresh_source))
    expected_identity["native_decode_config_sha256"] = sha(canonical(expected_identity["native_decode_config"]))
    need(canonical(fresh_identity) == canonical(expected_identity), "all decoder/model/assets settings remain exact")


def validate_failed_parent(files, original, rows, request):
    expected = request["failed_parent"]
    summary = parse(files["summary.json"])
    need(summary["run_id"] == str(expected["run_id"]) and summary["executed_head_sha"] == expected["head_sha"] and
         summary["status"] == "stopped_before_recognition" and summary["new_attempts_consumed"] == 0 and
         summary["combined_attempts_consumed"] == summary["combined_complete_native_observations"] == 176 and
         summary["prior_cells_reexecuted"] == 0 and summary["primary_model_loads"] == 0 and
         summary["failure"] == {"error_type": "ValueError", "message": "all original source/runtime inventory bytes unchanged"} and
         "mutable_stage_output" not in summary and "recognizer_policy_sha256" not in summary,
         "exact zero-audio prior continuation failure")
    need(set(summary["stages"]) == {"setup-whisper", "prepare-whisper"} and
         all(stage["status"] == "complete" and stage["exit_code"] == 0 for stage in summary["stages"].values()), "only setup and audio-free preparation ran")
    need(not any(name.startswith(("inference/", "policy/")) or name ==
                 "executed-native-source/runtime-policy/recognizer-policy.json" for name in files),
         "no failed-parent inference, start, claim, completion or fresh policy")
    for name in ("setup/whisper/container-setup.json", "prepare/whisper/container-tts.json"):
        diagnostic = parse(files[name])
        need(host.lifecycle_closed(diagnostic) and diagnostic.get("cleanup_failure") is None, "both failed-parent processes cleaned up")
    for name, digest in request["prior_control_source_sha256"].items():
        need(sha(files["executed-control-source/" + name]) == digest, "exact failed executed controller source")
    proof = parse(files["prepare/whisper/output/zero-forward-proof.json"])
    need(proof["family"] == "whisper" and proof["status"] == "prepared_zero_forward" and
         proof["guard_installed_before_model_loader"] is True and proof["guarded_entrypoints"] == 262 and
         proof["candidate_count"] == proof["audio_inputs"] == proof["observed_guarded_entrypoint_attempts"] == 0,
         "actual failed-run zero-forward proof")
    old_observations = parse(original["observations.json"])
    need(parse(files["combined-observations.json"])["providers"] == old_observations["providers"], "all 192 prior observation objects unchanged")
    pending = {row["audio_id"]: row for row in old_observations["providers"]["whisper"] if row["status"] == "not_run"}
    actual = parse(files["observations.json"])["rows"]
    need(len(actual) == 16 and {row["audio_id"]: row for row in actual} == pending,
         "exact same sixteen still unconsumed observations")
    for name in [prior.PARENT_JOB] + [slot["input_member"] for slot in request["slots"]]:
        need(files[name] == original[name], "second parent retained exact first input bytes")
    need(files[prior.PARENT_MAP] == original[prior.PARENT_MAP], "second parent exact blind identity map")
    for name in ("model-lock.json", "native-dependencies.json"):
        need(files[PREPARED + name] == original[PREPARED + name], "second-parent unchanged model bindings")
    old_source, new_source = parse(original[PREPARED + "source-lock.json"]), parse(files[PREPARED + "source-lock.json"])
    for entry in new_source["files"]:
        body = files[PREPARED + entry["path"]]
        need(cache.identity(body) == {"bytes": entry["bytes"], "sha256": entry["sha256"]}, "saved failed source inventory bytes")
        if entry["path"] != INVENTORY:
            need(body == original[PREPARED + entry["path"]], "38 unchanged failed-parent native source bodies")
    source_identity(parse(original["prepare/whisper/output/identity.json"]), parse(files["prepare/whisper/output/identity.json"]),
        old_source, new_source, original[PREPARED + INVENTORY], files[PREPARED + INVENTORY], expected["head_sha"], request["cache_contract"])
    need(proof["source_lock_sha256"] == sha(files[PREPARED + "source-lock.json"]) and
         proof["native_config_sha256"] == parse(files["prepare/whisper/output/identity.json"])["native_decode_config_sha256"],
         "failed zero-forward proof binds actual identity")


def verify_parents(original_path, failed_path, request):
    old_request, _ = prior.frozen_request()
    original = prior.archive_files(read(original_path, 64 * MIB), old_request)
    rows, policy, observations = prior.validate_parent(original, old_request)
    failed = failed_archive(read(failed_path, 8 * MIB), request)
    validate_failed_parent(failed, original, rows, request)
    return original, failed, rows, policy, observations


def verify_cache_proof(output, request, baseline):
    contract = request["cache_contract"]
    proof = parse(read(output / "cache-verification.json", 2 * MIB))
    fresh = read(output / "installed-inventory.json", 4 * MIB)
    need(parse(read(output / "contract.json", MIB)) == contract, "executed fixed cache contract")
    cache.compare_inventories(parse(baseline), parse(fresh), contract)
    need(proof["schema"] == "public-whisper-fresh-pip-cache-source-proof-v1" and
         proof["status"] == "verified_before_model_preparation" and proof["contract_sha256"] == cache.CONTRACT_SHA and
         proof["python_version"] == [3, 12, 14] and proof["implementation"] == "cpython" and
         proof["cache_tag"] == "cpython-312" and proof["interpreter_magic_hex"] == "cb0d0d0a" and
         proof["installed_inventory_before"] == proof["installed_inventory_after"] == cache.identity(fresh) and
         proof["installed_members_verified_twice"] == 17688 and proof["unchanged_noncache_records"] == 17277 and
         proof["verified_cache_count"] == 411 and proof["cache_code_executions"] == proof["model_imports"] ==
         proof["audio_forward_calls"] == proof["runtime_writes"] == 0 and
         proof["original_cache_payloads_available"] is False and proof["historical_cache_header_only_difference_proven"] is False,
         "complete read-only fresh-cache proof before loading models")
    scope = proof["kernel_scope"]
    need(scope["network_disabled"] is True and scope["read_only_root"] is True and scope["no_new_privileges"] is True and
         scope["limits"]["memory.max"] == "12884901888", "original offline constrained checker scope")
    records = {row["cache_path"]: row for row in proof["records"]}
    need(len(proof["records"]) == 411 and set(records) == set(contract["cache_records"]), "complete unique 411-cache proof")
    inventory = parse(fresh)["files"]
    observed = [parse(line) for line in read(output / "read-observations.jsonl", MIB).splitlines()]
    need(len(observed) == 411 and len({row["cache_path"] for row in observed}) == 411, "complete independent cache read log")
    observations = {row["cache_path"]: row for row in observed}
    need(set(observations) == set(records), "read log matches exact verified cache denominator")
    expected_members = {"contract.json", "installed-inventory.json", "cache-verification.json", "read-observations.jsonl"}
    for name, declared in contract["cache_records"].items():
        record = records[name]
        raw = read(output / "cache" / name, 8 * MIB)
        source = read(output / "source" / declared["source_path"], 8 * MIB)
        need(record["cache_identity"] == inventory[name] == cache.identity(raw) and
             record["source_identity"] == declared["source_identity"] == cache.identity(source) and
             record["source_path"] == declared["source_path"] and
             record["original_cache_identity"] == declared["original_cache"] and
             record["recursive_code_equal"] is True and record["cached_code_executed"] is False and
             record["compile_dont_inherit"] is True and record["compile_optimize"] == 0 and
             record["compile_filename"] == str(cache.SITE / declared["source_path"]) and
             record["code_fields"] == list(cache.CODE_FIELDS) and re.fullmatch("[0-9a-f]{64}", record["code_structure_sha256"]) and
             record["source_stat_before"] == record["source_stat_after"] and
             record["source_stat_before"]["st_size"] == len(source) and
             record["header"]["raw_header_hex"] == raw[:16].hex(), "actual retained cache/source and compiler proof binding")
        need(observations[name] == {key: record[key] for key in
             ("cache_path", "source_path", "cache_identity", "source_identity", "source_stat_before")},
             "raw cache/source/stat read was retained before semantic verification")
        expected_members.update(("cache/" + name, "source/" + declared["source_path"]))
    for name, expected in contract["license_records"].items():
        need(cache.identity(read(output / "licenses" / name, MIB)) == expected, "retained actual pip license text")
        expected_members.add("licenses/" + name)
    actual_members = set()
    for path in output.rglob("*"):
        need(not path.is_symlink(), "no cache evidence aliases")
        if path.is_file():
            actual_members.add(path.relative_to(output).as_posix())
    need(actual_members == expected_members, "closed source/cache/license proof output")
    return fresh


def bind_fresh_policy(original, output, head, parent_policy, fresh_inventory, request):
    proof = parse(read(output / "zero-forward-proof.json", MIB))
    identity = parse(read(output / "identity.json", MIB))
    source_raw = read(output / "prepared/source-lock.json", MIB)
    source = parse(source_raw)
    need(read(output / "prepared" / INVENTORY, 4 * MIB) == fresh_inventory, "native prepare uses exact proved fresh inventory")
    source_identity(parse(original["prepare/whisper/output/identity.json"]), identity,
        parse(original[PREPARED + "source-lock.json"]), source, original[PREPARED + INVENTORY],
        fresh_inventory, head, request["cache_contract"])
    need(source_raw == canonical(source) and proof["family"] == "whisper" and proof["status"] == "prepared_zero_forward" and
         proof["guard_installed_before_model_loader"] is True and
         proof["candidate_count"] == proof["audio_inputs"] == proof["observed_guarded_entrypoint_attempts"] == 0 and
         proof["native_config_sha256"] == identity["native_decode_config_sha256"] and
         proof["source_lock_sha256"] == sha(source_raw) and
         proof["asset_lock_sha256"] == identity["native_decode_config"]["asset_lock_sha256"], "actual fresh zero-forward native identity")
    for entry in source["files"]:
        body = read(output / "prepared" / prior.member(entry["path"]), 32 * MIB)
        need(cache.identity(body) == {"bytes": entry["bytes"], "sha256": entry["sha256"]}, "actual prepared source inventory bytes")
        need(body == (fresh_inventory if entry["path"] == INVENTORY else original[PREPARED + entry["path"]]), "only proved runtime inventory body differs")
    for name in ("model-lock.json", "native-dependencies.json"):
        need(read(output / "prepared" / name, MIB) == original[PREPARED + name], "exact model/assets/dependency bindings")
    policy = copy.deepcopy(parent_policy)
    policy["recognizers"] = [identity if row["model_family"] == "whisper" else row for row in policy["recognizers"]]
    from speech_asr_consensus import validate_recognizer_policy
    validate_recognizer_policy(policy)
    return policy, sha(canonical(policy))


def export(work, artifact, code, states, receipt, parent_observations):
    need(not artifact.exists(), "new exclusive v2 artifact")
    artifact.mkdir(parents=True)
    inventory, total = {}, 0
    def save(name, raw):
        nonlocal total
        total += len(raw)
        need(total <= 256 * MIB, "bounded public evidence without model weights")
        put(artifact / prior.member(name), raw)
        inventory[name] = cache.identity(raw)
    for name in ("continuation_v2.py", "continuation_v2_request.json", "cache_verify_v2.py", "continuation.py",
                 "continuation_request.json", "host.py", "input_adapter.py", "worker.py", "request.json", "README.md"):
        save("executed-control-source/" + name, read(HERE / name, MIB))
    for name in ("inputs", "setup", "cache-check", "prepare", "inference", "policy", "provenance"):
        root = work / name
        for path in sorted(root.rglob("*")) if root.exists() else ():
            need(not path.is_symlink(), "no evidence aliases")
            uncertain = receipt.get("mutable_stage_output")
            if uncertain and (work / uncertain) in path.parents:
                continue
            if path.is_file():
                save(path.relative_to(work).as_posix(), read(path, 64 * MIB))
    for path in sorted(code.rglob("*")) if code.exists() else ():
        need(not path.is_symlink(), "no code aliases")
        if path.is_file():
            save("executed-native-source/" + path.relative_to(code).as_posix(), read(path, 32 * MIB))
    save("observations.json", canonical(dict(schema="public-zh64-first-whisper16-v2-observations-v1", rows=states, **UNADMITTED)))
    combined = prior.combine(parent_observations, states)
    combined["lineage_run_ids"] = ["43608230908", "43614492600", receipt["run_id"]]
    combined["middle_run_audio_calls"] = 0
    save("combined-observations.json", canonical(combined))
    save("summary.json", canonical(receipt))
    write(artifact / "artifact-index.json", dict(schema="public-zh64-whisper16-v2-retained-evidence-index-v1",
        files=inventory, evidence_bytes=total, original_parent_artifact_sha256=receipt["original_parent_artifact_sha256"],
        failed_parent_artifact_sha256=receipt["failed_parent_artifact_sha256"], **UNADMITTED))


def run(args, env):
    request, digest = frozen_request()
    verify_control(request)
    event = parse(read(Path(env["GITHUB_EVENT_PATH"]), 2 * MIB))
    claim = parse(read(args.claim, MIB))
    head, run_id = validate_execution(env, event, claim, request, digest)
    need(host.git(host.ROOT, "rev-parse", "HEAD") == head, "exact actual control checkout")
    live_paths = {"original-claim": args.original_claim, "original-run": args.original_run,
                  "failed-claim": args.failed_claim, "failed-run": args.failed_run}
    live = {name: parse(read(path, MIB)) for name, path in live_paths.items()}
    verify_live_parents(live["original-claim"], live["original-run"], live["failed-claim"], live["failed-run"], request)
    original, failed, rows, parent_policy, parent_observations = verify_parents(args.original_artifact, args.failed_artifact, request)
    lock, retained = host.verify_sources(args.retained)
    work = args.work.absolute()
    need(work.resolve() == work and not work.exists(), "fresh v2 work directory")
    work.mkdir(parents=True)
    code = work / "code"
    pending = {row["audio_id"]: row for row in parent_observations["providers"]["whisper"]}
    states = [copy.deepcopy(pending[row["global_audio_id"]]) for row in rows]
    receipt = dict(schema="public-zh64-first-whisper16-v2-terminal-v1", campaign=CAMPAIGN, run_id=run_id,
        executed_head_sha=head, retained_source_commit=inputs.RETAINED_COMMIT, request_sha256=digest,
        request_canonical_sha256=REQUEST_CANONICAL_SHA, claim_ref=claim_ref(digest),
        original_parent_run_id=43608230908, original_parent_terminal_status="incomplete_no_retry",
        original_parent_artifact_sha256=request["original_parent_artifact_sha256"],
        failed_parent_run_id=43614492600, failed_parent_terminal_status="stopped_before_recognition",
        failed_parent_artifact_sha256=request["failed_parent"]["artifact_sha256"],
        original_complete_observations_preserved=176, failed_parent_audio_calls=0,
        parent_policy_sha256=sha(canonical(parent_policy)), parent_observations_rebound=False,
        prior_cells_reexecuted=0, max_new_asr_calls=16, new_tts_calls=0, new_audio_derivatives=0,
        primary_model_loads=0, new_training_runs=0, retry=False, nearphone_quarantined_rows=16,
        historical_cache_payload_equivalence_proven=False, single_run_three_model_execution=False,
        status="stopped_before_recognition", stages={}, **UNADMITTED)
    started = time.monotonic()
    deadline = started + request["inner_wall_seconds"]
    infer_output = work / "inference/batch-3/whisper/output"
    try:
        need(shutil.disk_usage(work).free >= 25 * 1024 ** 3, "unchanged initial disk reserve")
        code = retained.host.stage(Path(args.retained), work, lock)
        need(all(sha(read(code / target, 32 * MIB)) == lock["files"][source]
                 for target, source in lock["stage"].items()), "exact retained native closure")
        for name in ("worker.py", "cache_verify_v2.py"):
            put(code / "research/public-zh64-asr-20261114" / name, read(HERE / name, MIB))
        put(code / "runtime-policy/identity-map-3.json", original[prior.PARENT_MAP])
        put(code / "runtime-policy/original-installed-inventory.json", original[PREPARED + INVENTORY])
        write(code / "runtime-policy/cache-contract.json", request["cache_contract"])
        directory = work / "inputs/batch-3"
        put(directory / "job.json", original[prior.PARENT_JOB])
        for slot in request["slots"]:
            put(work / slot["input_member"], original[slot["input_member"]])
        write(work / "provenance/request.json", request)
        write(work / "provenance/claim.json", claim)
        for name, path in live_paths.items():
            put(work / "provenance" / (name + ".json"), read(path, MIB))
        for name in ("artifact-index.json", "summary.json", "observations.json", "derived/source-map.json",
                     "policy/recognizer-policy.json", "prepare/whisper/output/identity.json", PREPARED + "source-lock.json"):
            put(work / "provenance/parent" / name, original[name])
        # This small failed artifact is retained in full, including its absence
        # evidence and exact consumed control sources; the original 42 MB ZIP
        # remains independently referenced and preserved, never recompressed.
        for name, body in failed.items():
            put(work / "provenance/failed-continuation" / name, body)
        write(work / "provenance/parent-binding.json", dict(
            original_artifact_sha256=request["original_parent_artifact_sha256"], failed_parent=request["failed_parent"],
            original_complete=176, intervening_audio_calls=0, first_execution_slots=request["slots"],
            original_claim_preserved=True, failed_claim_preserved=True, **UNADMITTED))
        sys.path.insert(0, str(code / "tools"))
        readers = host.native_readers(code)
        boundary = retained.load_boundary()
        image_lock = parse(read(code / "research/source-screen32-v1/container-lock.json", MIB))
        boundary.host_capacity()
        boundary.pull_image(image_lock, work / "provenance/image-transfer.json")
        runtime = work / "runtime-whisper"
        runtime.mkdir()
        image = image_lock["image"]
        receipt["stages"]["setup-whisper"] = host.launch(boundary, code, runtime, work / "setup/whisper/output", image,
            CAMPAIGN + "-setup", ["python", "-I", "-B", "/code/research/source-screen32-whisper-v1/setup_runtime.py"], deadline, setup=True)
        cache_output = work / "cache-check/output"
        receipt["stages"]["cache-check"] = host.launch(boundary, code, runtime, cache_output, image,
            CAMPAIGN + "-cache-check", ["/runtime/venv/bin/python", "-I", "-B",
                "/code/research/public-zh64-asr-20261114/cache_verify_v2.py"], deadline, duration=300)
        fresh_inventory = verify_cache_proof(cache_output, request, original[PREPARED + INVENTORY])
        receipt["cache_proof_sha256"] = sha(read(cache_output / "cache-verification.json", 2 * MIB))
        receipt["actual_installed_inventory_sha256"] = sha(fresh_inventory)
        output = work / "prepare/whisper/output"
        command = host.shim(["/runtime/venv/bin/python", "-I", "-B",
            "/code/research/native-asr-integration-20261104/prepare_worker.py", "--family", "whisper", "--executed-head", head])
        receipt["stages"]["prepare-whisper"] = host.launch(boundary, code, runtime, output, image,
            CAMPAIGN + "-prepare", command, deadline)
        policy, policy_sha = bind_fresh_policy(original, output, head, parent_policy, fresh_inventory, request)
        write(code / "runtime-policy/recognizer-policy.json", policy)
        write(work / "policy/recognizer-policy.json", policy)
        receipt["recognizer_policy_sha256"] = policy_sha
        receipt["identity_difference"] = "actual head plus source-verified fresh pip cache inventory and dependent hashes"
        receipt["status"] = "incomplete_no_retry"
        command = host.shim(retained.host.worker_command("whisper", "infer", head, policy_sha, 3, sha(original[prior.PARENT_JOB])))
        try:
            receipt["stages"]["batch-3-whisper"] = host.launch(boundary, code, runtime, infer_output, image,
                CAMPAIGN + "-infer", command, deadline, input_dir=directory)
        except Exception as error:
            receipt["stages"]["batch-3-whisper"] = dict(status="failed_no_retry", error_type=type(error).__name__, message=str(error)[:1024])
            if isinstance(error, host.LifecycleUncertain):
                receipt["mutable_stage_output"] = str(infer_output.relative_to(work))
                raise
        finally:
            if receipt.get("mutable_stage_output") == str(infer_output.relative_to(work)):
                for row in states:
                    row.update(status="unknown_after_lifecycle_failure", attempt_consumed=True,
                               attempt_count_is_reservation=True, actual_text=None)
            else:
                states = host.collect("whisper", infer_output, rows, policy, readers)
        if all(row["status"] == "complete_native_observation" for row in states):
            receipt["status"] = "complete_first_whisper16_v2_unadmitted"
    except BaseException as error:
        receipt["failure"] = dict(error_type=type(error).__name__, message=str(error)[:1024])
        if isinstance(error, host.LifecycleUncertain):
            receipt["mutable_stage_output"] = str(error.output.relative_to(work))
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["new_attempts_consumed"] = sum(row["attempt_consumed"] for row in states)
        need(receipt["new_attempts_consumed"] <= 16, "only sixteen first audio attempts")
        receipt["combined_attempts_consumed"] = 176 + receipt["new_attempts_consumed"]
        receipt["new_status_counts"] = {status: sum(row["status"] == status for row in states) for status in
            ("complete_native_observation", "failed_no_retry", "rejected_native_evidence", "unknown_after_lifecycle_failure", "not_run")}
        receipt["combined_complete_native_observations"] = 176 + receipt["new_status_counts"]["complete_native_observation"]
        export(work, args.artifact, code, states, receipt, parent_observations)
    print(json.dumps({key: receipt[key] for key in ("status", "new_status_counts", "combined_complete_native_observations", "training_admitted")}, sort_keys=True))
    return 0 if receipt["status"] == "complete_first_whisper16_v2_unadmitted" else 2


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check-source", action="store_true")
    cli.add_argument("--verify-parents", action="store_true")
    for name in ("retained", "original-artifact", "failed-artifact", "original-claim", "original-run", "failed-claim", "failed-run", "claim", "work", "artifact"):
        cli.add_argument("--" + name, type=Path)
    args = cli.parse_args(argv)
    request, digest = frozen_request()
    verify_control(request)
    if args.check_source:
        need(args.retained is not None, "retained source required")
        lock, _ = host.verify_sources(args.retained)
        print(json.dumps(dict(request_sha256=digest, retained_files=len(lock["files"]), staged_aliases=len(lock["stage"]), fixed_caches=411, new_model_calls=0)))
        return 0
    if args.verify_parents:
        need(args.original_artifact and args.failed_artifact, "both original failed ZIPs required")
        _, _, rows, _, _ = verify_parents(args.original_artifact, args.failed_artifact, request)
        print(json.dumps(dict(original_complete=176, intervening_audio_calls=0, pending_first_whisper=len(rows), fixed_caches=411, new_model_calls=0)))
        return 0
    need(all(getattr(args, name) for name in ("retained", "original_artifact", "failed_artifact", "original_claim", "original_run",
        "failed_claim", "failed_run", "claim", "work", "artifact")), "all actual v2 execution inputs required")
    return run(args, dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
