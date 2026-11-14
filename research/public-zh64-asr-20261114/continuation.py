"""First execution of the sixteen Whisper cells proved unstarted in run 43608230908.

The first campaign, its claim, request and 176 observations stay immutable. This
controller consumes only its fixed absent batch, using original derived bytes.
All local verification is standard-library/source-only; ML runs in Actions only.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
import wave
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import host
import input_adapter as inputs

need, read, parse, sha, canonical, write = inputs.need, inputs.read, inputs.parse, inputs.sha, inputs.canonical, inputs.write
MIB = 1024 ** 2
CAMPAIGN = "public-zh64-whisper16-first-20261114-v1"
REQUEST_CANONICAL_SHA = "970bfcfdb45ac055a4db840302ce96b63ac8f4135a4f556ddee34cee3cfedcba"
OLD_CONTROL = {
    "host.py": "cbdce469f5886a0d9dff11918088bef29a3de70110c6f0eed72857d3a42108cf",
    "input_adapter.py": "70537317db8b081c5e00e22c6fc1e41b3213071e3c7ab616bc663806fea6a9a2",
    "worker.py": "6a1de9f22607b90fb682787d8c7e2e878d4dfcf51a70762fc0a35fefa6b239e1",
    "request.json": "868aa3fb60ada992888859fb3019059a75b12de3e826ce1fa7ada16caa3cace7",
}
PARENT_PREPARED = "prepare/whisper/output/prepared/"
PARENT_MAP = "executed-native-source/runtime-policy/identity-map-3.json"
PARENT_JOB = "inputs/batch-3/job.json"
UNADMITTED = dict(training_admitted=False, human_gold=False, optimizer_loss_authorized=False,
                  natural_negative_qualified=False, evaluation_or_production_use=False)


def frozen_request(path=HERE / "continuation_request.json"):
    raw = read(path, MIB)
    value = parse(raw)
    need(sha(canonical(value)) == REQUEST_CANONICAL_SHA, "fixed first-Whisper16 request changed")
    return value, sha(raw)


def claim_ref(request_sha):
    need(re.fullmatch("[0-9a-f]{64}", request_sha) is not None, "request SHA256")
    return "refs/tags/" + CAMPAIGN + "-consumed"


def member(name):
    need(type(name) is str and 0 < len(name) <= 1024 and "\\" not in name and "\x00" not in name,
         "bounded POSIX member")
    path = PurePosixPath(name)
    need(not path.is_absolute() and ".." not in path.parts and str(path) == name,
         "canonical relative member")
    return name


def verify_control():
    for name, digest in OLD_CONTROL.items():
        need(sha(read(HERE / name, MIB)) == digest, "original controller/request/worker must remain unchanged")


def parent_claim(value, request):
    need(value.get("ref") == request["parent_claim_ref"] and
         value.get("object", {}).get("type") == "commit" and
         value["object"].get("sha") == request["parent_head_sha"], "unchanged original consumed claim")


def validate_execution(env, event, claim, original_claim, request, request_sha):
    need(env.get("GITHUB_REPOSITORY") == "jiying2007/kws-pipeline" and
         env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main" and
         env.get("GITHUB_RUN_ATTEMPT") == "1" and env.get("GITHUB_ACTIONS") == "true" and
         env.get("RUNNER_ENVIRONMENT") == "github-hosted" and env.get("RUNNER_OS") == "Linux" and
         env.get("RUNNER_ARCH") == "X64", "first protected-main hosted execution only")
    head, run = env.get("GITHUB_SHA", ""), env.get("GITHUB_RUN_ID", "")
    need(re.fullmatch("[0-9a-f]{40}", head) and re.fullmatch("[1-9][0-9]{0,19}", run) and
         head != request["parent_head_sha"] and run != str(request["parent_run_id"]), "new actual controller/run")
    need(event.get("after") == head and event.get("ref") == "refs/heads/main" and
         event.get("deleted") is False and event.get("repository", {}).get("private") is False and
         event["repository"].get("full_name") == "jiying2007/kws-pipeline", "actual public main push")
    need(env.get("GITHUB_WORKFLOW_SHA") == head and env.get("GITHUB_WORKFLOW_REF") ==
         "jiying2007/kws-pipeline/.github/workflows/public-zh64-asr-once.yml@refs/heads/main",
         "actual main workflow source identity")
    need(claim.get("ref") == claim_ref(request_sha) and claim.get("object", {}).get("sha") == head and
         claim["object"].get("type") == "commit", "new permanent sixteen-slot claim")
    parent_claim(original_claim, request)
    need(not any(key in env for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN")),
         "model host cannot receive repository credentials")
    return head, run


def verify_parent_run(value, request):
    need(value.get("id") == request["parent_run_id"] and value.get("run_attempt") == 1 and
         value.get("head_sha") == request["parent_head_sha"] and value.get("status") == "completed" and
         value.get("conclusion") == "failure" and value.get("event") == "push" and
         value.get("path") == ".github/workflows/public-zh64-asr-once.yml" and
         value.get("repository", {}).get("full_name") == "jiying2007/kws-pipeline" and
         value["repository"].get("private") is False, "exact closed original public run")


def archive_files(raw, request):
    need(len(raw) == request["parent_artifact_bytes"] and sha(raw) == request["parent_artifact_sha256"],
         "exact original failed-run artifact bytes")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        need(len(infos) == request["parent_zip_members"] == len({x.filename for x in infos}),
             "exact unique parent ZIP membership")
        total = 0
        files = {}
        for info in infos:
            name = member(info.filename)
            need(not info.is_dir() and not (info.flag_bits & 1) and
                 stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK and
                 0 <= info.file_size <= 64 * MIB, "regular bounded parent member")
            total += info.file_size
            need(total <= 128 * MIB, "bounded parent evidence expansion")
            body = archive.read(info)
            need(len(body) == info.file_size, "complete parent member bytes")
            files[name] = body
    index = parse(files["artifact-index.json"])
    need(index["schema"] == "public-zh64-retained-evidence-index-v1" and
         set(files) == set(index["files"]) | {"artifact-index.json"} and
         len(index["files"]) == 3745 and index["evidence_bytes"] == request["parent_evidence_bytes"] and
         index["source_audio_regenerated"] == 0 and index["training_admitted"] is False and
         index["human_gold"] is False, "exact closed original evidence index")
    for name, value in index["files"].items():
        body = files[member(name)]
        need(value == {"bytes": len(body), "sha256": sha(body)}, "original indexed member changed")
    need(sum(len(files[name]) for name in index["files"]) == index["evidence_bytes"], "parent byte total")
    for name, expected in request["parent_members"].items():
        need(expected == {"bytes": len(files[name]), "sha256": sha(files[name])}, "frozen parent evidence changed")
    return files


def validate_parent(files, request):
    summary, observations = parse(files["summary.json"]), parse(files["observations.json"])
    need(summary["run_id"] == str(request["parent_run_id"]) and
         summary["executed_head_sha"] == request["parent_head_sha"] and
         summary["status"] == "incomplete_no_retry" and summary["attempts_consumed"] == 176 and
         summary["planned_asr_slots"] == 192 and summary["claim_ref"] == request["parent_claim_ref"] and
         "mutable_stage_output" not in summary and summary["retry"] is False,
         "closed original failed terminal with 176 consumed cells")
    for key, expected in UNADMITTED.items():
        need(summary[key] is expected, "original admission boundary")
    need(summary["stages"]["batch-3-whisper"] == {
        "error_type": "ValueError", "message": "current host memory reserve unavailable",
        "status": "failed_no_retry"}, "original pre-launch capacity refusal")
    need(not any(name == "inference/batch-3/whisper" or name.startswith("inference/batch-3/whisper/")
                 for name in files), "no original Whisper batch-three directory/diagnostic/claim/start")
    cleanup = parse(files["inference/batch-3/sensevoice/container-qwen06.json"])
    need(host.lifecycle_closed(cleanup) and cleanup.get("cleanup_failure") is None,
         "previous Sense process cleanup completed")
    for name, digest in OLD_CONTROL.items():
        need(sha(files["executed-control-source/" + name]) == digest, "exact original execution source")
    selected_ids = {row["global_audio_id"] for row in request["slots"]}
    need(selected_ids == {f"clip-{n:06d}" for n in range(49, 65)}, "only the original last batch")
    need(set(observations["providers"]) == set(host.FAMILIES), "original three-provider denominator")
    for family in host.FAMILIES:
        rows = observations["providers"][family]
        need(len(rows) == 64 and {row["audio_id"] for row in rows} ==
             {f"clip-{n:06d}" for n in range(1, 65)}, "unique original 64-slot denominator")
        for row in rows:
            pending = family == "whisper" and row["audio_id"] in selected_ids
            need(row["status"] == ("not_run" if pending else "complete_native_observation") and
                 row["attempt_consumed"] is (not pending), "exclude every consumed cell")
            if pending:
                need(row["actual_text"] is None and not any(k in row for k in
                    ("started_sha256", "completed_sha256", "native_receipt", "raw_output_sha256")),
                    "no hidden completion for pending slot")
        expected = {status: 0 for status in ("complete_native_observation", "failed_no_retry", "not_run",
                                            "rejected_native_evidence", "unknown_after_lifecycle_failure")}
        expected["complete_native_observation"] = request["already_complete_excluded"][family]
        expected["not_run"] = 64 - expected["complete_native_observation"]
        need(summary["providers"][family] == expected, "parent summary/slot count agreement")
    lock = parse(files["derived/input-lock.json"])
    sources = {row["audio_id"]: row for row in parse(files["derived/source-map.json"])["rows"]}
    job, mapping = parse(files[PARENT_JOB]), parse(files[PARENT_MAP])
    need(len(job["clips"]) == len(mapping["clips"]) == len(request["slots"]) == 16,
         "fixed blind job/map/slot denominator")
    by_global = {row["audio_id"]: row for row in observations["providers"]["whisper"]}
    selected = sorted(lock["clips"][48:64], key=lambda row: row["wav_sha256"])
    result = []
    for n, (base, slot, clip, mapped) in enumerate(zip(selected, request["slots"], job["clips"], mapping["clips"]), 1):
        local, global_id = f"clip-{n:06d}", base["audio_id"]
        source, old = sources[global_id], by_global[global_id]
        need(slot["global_audio_id"] == global_id and slot["local_audio_id"] == local and
             all(slot[k] == base[k] == old[k] for k in ("source_id", "request_id", "wav_sha256", "pcm_sha256", "frames")) and
             old["local_audio_id"] == local, "exact original pending source/PCM/WAV binding")
        need(clip == {"audio_id": local, "audio_path": "audio/" + slot["wav_sha256"] + ".wav",
                      "wav_sha256": slot["wav_sha256"]} and mapped ==
             {"audio_id": local, **{k: slot[k] for k in ("source_id", "request_id", "wav_sha256")}},
             "unchanged audio-only job and identity map")
        need(slot["input_member"] == "inputs/batch-3/" + clip["audio_path"] and
             all(slot[k] == source[k] for k in ("producer_ordinal", "source_cell_id", "preset", "research_split", "native_pcm_sha256")) and
             source["nearphone_quarantined"] is True and source["optimizer_loss_authorized"] is False and
             source["training_admitted"] is False and source["original_intended_split"] == "train_only" and
             inputs.SPLITS[slot["preset"]] == slot["research_split"], "unchanged quarantined preset role")
        raw = files[slot["input_member"]]
        need(raw == files["derived/" + base["path"]] and sha(raw) == slot["wav_sha256"],
             "copy exact parent derivative without resampling")
        with wave.open(io.BytesIO(raw), "rb") as audio:
            need((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype(), audio.getnframes()) ==
                 (1, 2, 16000, "NONE", slot["frames"]), "original PCM16 descriptor")
            pcm = audio.readframes(slot["frames"] + 1)
        need(len(pcm) == 2 * slot["frames"] and sha(pcm) == slot["pcm_sha256"], "original PCM byte identity")
        result.append(dict(base, audio_id=local, global_audio_id=global_id))
    policy = parse(files["policy/recognizer-policy.json"])
    identity = parse(files["prepare/whisper/output/identity.json"])
    need(next(row for row in policy["recognizers"] if row["model_family"] == "whisper") == identity and
         sha(canonical(policy)) == summary["recognizer_policy_sha256"], "parent Whisper/policy binding")
    return result, policy, observations


def put(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body)


def same_whisper_identity(old_identity, fresh_identity, old_source, fresh_source, head):
    """Allow actual controller provenance and its hashes only, not decoder drift."""
    need(head != old_source["executed_head_sha"] and re.fullmatch("[0-9a-f]{40}", head), "new actual source head")
    expected_source = dict(old_source, executed_head_sha=head)
    need(canonical(fresh_source) == canonical(expected_source), "all original source/runtime inventory bytes unchanged")
    need(old_identity["native_decode_config"]["source_lock_sha256"] == sha(canonical(old_source)),
         "original source-lock identity")
    expected = copy.deepcopy(old_identity)
    expected["native_decode_config"]["source_lock_sha256"] = sha(canonical(fresh_source))
    expected["native_decode_config_sha256"] = sha(canonical(expected["native_decode_config"]))
    need(canonical(fresh_identity) == canonical(expected), "only executed-source provenance may change Whisper identity")


def bind_fresh_policy(files, output, head, parent_policy):
    proof = parse(read(output / "zero-forward-proof.json", MIB))
    identity = parse(read(output / "identity.json", MIB))
    source_raw = read(output / "prepared/source-lock.json", MIB)
    source = parse(source_raw)
    old_identity = parse(files["prepare/whisper/output/identity.json"])
    old_source = parse(files[PARENT_PREPARED + "source-lock.json"])
    same_whisper_identity(old_identity, identity, old_source, source, head)
    need(source_raw == canonical(source) and proof["family"] == "whisper" and
         proof["status"] == "prepared_zero_forward" and proof["guard_installed_before_model_loader"] is True and
         proof["candidate_count"] == proof["audio_inputs"] == proof["observed_guarded_entrypoint_attempts"] == 0 and
         proof["native_config_sha256"] == identity["native_decode_config_sha256"] and
         proof["source_lock_sha256"] == sha(source_raw) and
         proof["asset_lock_sha256"] == identity["native_decode_config"]["asset_lock_sha256"],
         "fresh actual zero-forward native configuration proof")
    for entry in source["files"]:
        name = member(entry["path"])
        current = read(output / "prepared" / name, 32 * MIB)
        need(current == files[PARENT_PREPARED + name] and
             len(current) == entry["bytes"] and sha(current) == entry["sha256"],
             "actual saved fresh/parent source and installed-runtime bytes match")
    for name in ("model-lock.json", "native-dependencies.json"):
        need(read(output / "prepared" / name, MIB) == files[PARENT_PREPARED + name],
             "unchanged saved model/dependency bindings")
    policy = copy.deepcopy(parent_policy)
    policy["recognizers"] = [identity if row["model_family"] == "whisper" else row
                             for row in policy["recognizers"]]
    from speech_asr_consensus import validate_recognizer_policy
    validate_recognizer_policy(policy)
    return policy, sha(canonical(policy))


def combine(parent_observations, fresh_rows):
    expected = {f"clip-{n:06d}" for n in range(49, 65)}
    fresh = {row["audio_id"]: row for row in fresh_rows}
    need(len(fresh_rows) == 16 and set(fresh) == expected, "only fixed sixteen cells may be composed")
    combined = copy.deepcopy(parent_observations)
    old = combined["providers"]["whisper"]
    for i, row in enumerate(old):
        if row["audio_id"] in fresh:
            need(row["status"] == "not_run" and row["attempt_consumed"] is False, "never replace a consumed parent cell")
            new = fresh[row["audio_id"]]
            need(all(new[k] == row[k] for k in ("source_id", "request_id", "wav_sha256", "pcm_sha256", "frames", "local_audio_id")),
                 "unchanged composed input identity")
            old[i] = new
    combined["schema"] = "public-zh64-cross-run-native-observations-v1"
    combined["parent_observations_policy_preserved"] = True
    combined["new_whisper_policy_applies_only_to_global_ids"] = sorted(expected)
    combined["single_run_three_model_execution"] = False
    combined.update(UNADMITTED)
    return combined


def export(work, artifact, code, states, receipt, parent_observations):
    need(not artifact.exists(), "create-only continuation artifact")
    artifact.mkdir(parents=True)
    inventory = {}
    total = 0
    def save(name, raw):
        nonlocal total
        total += len(raw)
        need(total <= 256 * MIB, "bounded continuation evidence without model weights")
        put(artifact / member(name), raw)
        inventory[name] = {"bytes": len(raw), "sha256": sha(raw)}
    for name in ("continuation.py", "continuation_request.json", "host.py", "input_adapter.py", "worker.py", "request.json", "README.md"):
        save("executed-control-source/" + name, read(HERE / name, MIB))
    for name in ("inputs", "setup", "prepare", "inference", "policy", "provenance"):
        root = work / name
        for path in sorted(root.rglob("*")) if root.exists() else ():
            need(not path.is_symlink(), "no evidence aliases")
            uncertain = receipt.get("mutable_stage_output")
            if uncertain and (work / uncertain) in path.parents:
                continue
            if path.is_file():
                save(str(path.relative_to(work)), read(path, 64 * MIB))
    for path in sorted(code.rglob("*")) if code.exists() else ():
        need(not path.is_symlink(), "no staged source aliases")
        if path.is_file():
            save("executed-native-source/" + str(path.relative_to(code)), read(path, 32 * MIB))
    save("observations.json", canonical(dict(schema="public-zh64-first-whisper16-observations-v1", rows=states, **UNADMITTED)))
    save("combined-observations.json", canonical(combine(parent_observations, states)))
    save("summary.json", canonical(receipt))
    write(artifact / "artifact-index.json", dict(schema="public-zh64-whisper16-retained-evidence-index-v1",
        files=inventory, evidence_bytes=total, parent_artifact_sha256=receipt["parent_artifact_sha256"], **UNADMITTED))


def run(args, env):
    request, request_sha = frozen_request()
    verify_control()
    event = parse(read(Path(env["GITHUB_EVENT_PATH"]), 2 * MIB))
    claim, original_claim = parse(read(args.claim, MIB)), parse(read(args.parent_claim, MIB))
    head, run_id = validate_execution(env, event, claim, original_claim, request, request_sha)
    need(host.git(host.ROOT, "rev-parse", "HEAD") == head, "exact actual controller checkout")
    parent_run = parse(read(args.parent_run, MIB))
    verify_parent_run(parent_run, request)
    files = archive_files(read(args.parent_artifact, 64 * MIB), request)
    rows, parent_policy, parent_observations = validate_parent(files, request)
    lock, retained = host.verify_sources(args.retained)
    work = args.work.absolute()
    need(work.resolve() == work and not work.exists(), "fresh isolated continuation work directory")
    work.mkdir(parents=True)
    code = work / "code"
    pending = {row["audio_id"]: row for row in parent_observations["providers"]["whisper"]}
    states = [copy.deepcopy(pending[row["global_audio_id"]]) for row in rows]
    receipt = dict(schema="public-zh64-first-whisper16-terminal-v1", campaign=CAMPAIGN,
        run_id=run_id, executed_head_sha=head, retained_source_commit=inputs.RETAINED_COMMIT,
        request_sha256=request_sha, request_canonical_sha256=REQUEST_CANONICAL_SHA,
        claim_ref=claim_ref(request_sha), parent_run_id=request["parent_run_id"],
        parent_head_sha=request["parent_head_sha"], parent_artifact_sha256=request["parent_artifact_sha256"],
        parent_terminal_status="incomplete_no_retry", parent_attempts_consumed=176,
        parent_policy_sha256=sha(canonical(parent_policy)), parent_observations_rebound=False,
        prior_cells_reexecuted=0, max_new_asr_calls=16, new_tts_calls=0, new_audio_derivatives=0,
        primary_model_loads=0, new_training_runs=0, retry=False, nearphone_quarantined_rows=16,
        single_run_three_model_execution=False, status="stopped_before_recognition", stages={}, **UNADMITTED)
    started = time.monotonic()
    deadline = started + request["inner_wall_seconds"]
    policy = None
    infer_output = work / "inference/batch-3/whisper/output"
    try:
        need(shutil.disk_usage(work).free >= 25 * 1024 ** 3, "unchanged initial disk reserve")
        code = retained.host.stage(Path(args.retained), work, lock)
        need(all(sha(read(code / target, 32 * MIB)) == lock["files"][original]
                 for target, original in lock["stage"].items()), "exact unchanged 78-alias native closure")
        put(code / "research/public-zh64-asr-20261114/worker.py", read(HERE / "worker.py", MIB))
        put(code / "runtime-policy/identity-map-3.json", files[PARENT_MAP])
        directory = work / "inputs/batch-3"
        put(directory / "job.json", files[PARENT_JOB])
        for slot in request["slots"]:
            put(work / slot["input_member"], files[slot["input_member"]])
        write(work / "provenance/request.json", request)
        write(work / "provenance/claim.json", claim)
        write(work / "provenance/parent-claim.json", original_claim)
        put(work / "provenance/parent-run.json", read(args.parent_run, MIB))
        for name in ("artifact-index.json", "summary.json", "observations.json", "derived/source-map.json",
                     "policy/recognizer-policy.json", "prepare/whisper/output/identity.json",
                     "prepare/whisper/output/prepared/source-lock.json"):
            put(work / "provenance/parent" / name, files[name])
        write(work / "provenance/parent-binding.json", dict(
            parent_run_id=request["parent_run_id"], parent_artifact_id=request["parent_artifact_id"],
            parent_artifact_sha256=request["parent_artifact_sha256"], parent_members=request["parent_members"],
            absent_inference_prefix="inference/batch-3/whisper/", previous_process_cleanup_confirmed=True,
            parent_zip_members=3746, parent_complete_native_observations=176, first_execution_slots=request["slots"],
            source_audio_regenerated=0, derivatives_regenerated=0, **UNADMITTED))
        sys.path.insert(0, str(code / "tools"))
        readers = host.native_readers(code)
        boundary = retained.load_boundary()
        image_lock = parse(read(code / "research/source-screen32-v1/container-lock.json", MIB))
        boundary.host_capacity()
        boundary.pull_image(image_lock, work / "provenance/image-transfer.json")
        runtime = work / "runtime-whisper"
        runtime.mkdir()
        image = image_lock["image"]
        receipt["stages"]["setup-whisper"] = host.launch(boundary, code, runtime,
            work / "setup/whisper/output", image, CAMPAIGN + "-setup",
            ["python", "-I", "-B", "/code/research/source-screen32-whisper-v1/setup_runtime.py"],
            deadline, setup=True)
        output = work / "prepare/whisper/output"
        command = host.shim(["/runtime/venv/bin/python", "-I", "-B",
            "/code/research/native-asr-integration-20261104/prepare_worker.py",
            "--family", "whisper", "--executed-head", head])
        receipt["stages"]["prepare-whisper"] = host.launch(boundary, code, runtime, output,
            image, CAMPAIGN + "-prepare", command, deadline)
        policy, policy_sha = bind_fresh_policy(files, output, head, parent_policy)
        write(code / "runtime-policy/recognizer-policy.json", policy)
        write(work / "policy/recognizer-policy.json", policy)
        receipt["recognizer_policy_sha256"] = policy_sha
        receipt["identity_difference"] = "actual executed_head_sha and dependent source/config hashes only"
        command = host.shim(retained.host.worker_command("whisper", "infer", head, policy_sha, 3,
                                                        sha(files[PARENT_JOB])))
        receipt["status"] = "incomplete_no_retry"
        try:
            receipt["stages"]["batch-3-whisper"] = host.launch(boundary, code, runtime, infer_output,
                image, CAMPAIGN + "-infer", command, deadline, input_dir=directory)
        except Exception as error:
            receipt["stages"]["batch-3-whisper"] = dict(status="failed_no_retry",
                error_type=type(error).__name__, message=str(error)[:1024])
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
            receipt["status"] = "complete_first_whisper16_unadmitted"
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
    return 0 if receipt["status"] == "complete_first_whisper16_unadmitted" else 2


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--retained", type=Path)
    cli.add_argument("--check-source", action="store_true")
    cli.add_argument("--verify-parent", action="store_true")
    for name in ("parent-artifact", "parent-claim", "parent-run", "claim", "work", "artifact"):
        cli.add_argument("--" + name, type=Path)
    args = cli.parse_args(argv)
    request, digest = frozen_request()
    verify_control()
    if args.check_source:
        need(args.retained is not None, "retained source required")
        lock, _ = host.verify_sources(args.retained)
        print(json.dumps(dict(request_sha256=digest, retained_files=len(lock["files"]), staged_aliases=len(lock["stage"]), new_model_calls=0)))
        return 0
    if args.verify_parent:
        need(args.parent_artifact is not None, "parent archive required")
        rows, _, _ = validate_parent(archive_files(read(args.parent_artifact, 64 * MIB), request), request)
        print(json.dumps(dict(parent_members=3746, original_complete=176, pending_first_whisper=len(rows), new_model_calls=0)))
        return 0
    need(all((args.retained, args.parent_artifact, args.parent_claim, args.parent_run, args.claim, args.work, args.artifact)),
         "all first-execution inputs required")
    return run(args, dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
