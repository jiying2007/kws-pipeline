"""First bounded three-ASR observation of the existing public nonclone64 ZIP.

Fresh orchestration only: retained native source bytes are checked and staged
unchanged. Old campaign admission/main/run functions are never called. Models,
runtime locks, checkpoint verification, native capture and isolation stay pinned.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import input_adapter as inputs

CAMPAIGN = inputs.CAMPAIGN
FAMILIES = ("qwen06", "sensevoice", "whisper")
LOCK_PATH = "research/native-prepare-tokenizer-v4-20261104/source-lock.json"
LOCK_SHA = "0a7a489e94f7bc940ae8e58cc4be43a8f59873e87cd599303297c6a8b962896a"
HOST_PATH = "research/native-prepare-tokenizer-v4-20261104/host.py"
MIB = 1024 ** 2
need, read, parse, sha, canonical, write = inputs.need, inputs.read, inputs.parse, inputs.sha, inputs.canonical, inputs.write


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], timeout=30).decode().strip()


def verify_sources(retained):
    retained = Path(retained).absolute()
    need(retained.resolve() == retained and git(retained, "rev-parse", "HEAD") == inputs.RETAINED_COMMIT,
         "exact original source checkout")
    lock_raw = read(retained / LOCK_PATH, MIB)
    need(sha(lock_raw) == LOCK_SHA, "original tokenizer-v4 closure")
    lock = parse(lock_raw)
    need(len(lock["files"]) == 127 and len(lock["stage"]) == 78, "exact original source denominator")
    for relative, digest in lock["files"].items():
        path = Path(relative)
        need(not path.is_absolute() and ".." not in path.parts
             and sha(read(retained / path, 32 * MIB)) == digest, "original retained source changed")
    # Compile only authenticated bytes. Importing this module is source-only:
    # it resolves pure helpers, never invokes its old admission/run functions.
    path = retained / HOST_PATH
    module = types.ModuleType("public64_retained_primitives")
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(read(path, MIB), str(path), "exec"), module.__dict__)
    module.verify(retained, lock)
    return lock, module


def claim_ref(request_sha):
    need(re.fullmatch("[0-9a-f]{64}", request_sha) is not None, "request SHA256")
    # One atomic campaign identity. Reformatting request JSON cannot create a
    # second attempt; exact raw and canonical request hashes are receipts.
    return "refs/tags/" + CAMPAIGN + "-consumed"


class LifecycleUncertain(RuntimeError):
    """No later launch is allowed when the previous container may still run."""
    def __init__(self, message, output):
        super().__init__(message)
        self.output = Path(output)


def lifecycle_closed(diagnostic):
    operations = diagnostic.get("operations", {})
    return (diagnostic.get("created_confirmed") is True and
            operations.get("cleanup", {}).get("status") == "complete")


def validate_execution(env, event, claim, request_sha):
    need(env.get("GITHUB_REPOSITORY") == "jiying2007/kws-pipeline"
         and env.get("GITHUB_EVENT_NAME") == "push" and env.get("GITHUB_REF") == "refs/heads/main"
         and env.get("GITHUB_RUN_ATTEMPT") == "1" and env.get("GITHUB_ACTIONS") == "true"
         and env.get("RUNNER_ENVIRONMENT") == "github-hosted"
         and env.get("RUNNER_OS") == "Linux" and env.get("RUNNER_ARCH") == "X64",
         "first protected-main hosted execution only")
    head, run = env.get("GITHUB_SHA", ""), env.get("GITHUB_RUN_ID", "")
    need(re.fullmatch("[0-9a-f]{40}", head) and re.fullmatch("[1-9][0-9]{0,19}", run), "actual run/source identity")
    need(event.get("after") == head and event.get("ref") == "refs/heads/main"
         and event.get("deleted") is False and event.get("repository", {}).get("private") is False
         and event["repository"].get("full_name") == "jiying2007/kws-pipeline", "actual public main push")
    need(env.get("GITHUB_WORKFLOW_SHA") == head and env.get("GITHUB_WORKFLOW_REF") ==
         "jiying2007/kws-pipeline/.github/workflows/public-zh64-asr-once.yml@refs/heads/main",
         "actual main workflow source identity")
    need(claim.get("ref") == claim_ref(request_sha) and claim.get("object", {}).get("sha") == head
         and claim["object"].get("type") == "commit", "new immutable request claim")
    need(not any(name in env for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN")),
         "model host must not receive repository credentials")
    return head, run


def launch(boundary, code, runtime, output, image, name, command, deadline,
           *, setup=False, input_dir=None, duration=1500):
    need(deadline - time.monotonic() >= 180, "remaining job reserve")
    boundary.host_capacity()
    output.mkdir(parents=True)
    kind = "setup" if setup else ("qwen06" if input_dir is not None else "tts")
    profile = "asr" if setup or input_dir is not None else "tts"
    argv = boundary.container_args(name, image, code, runtime, output, profile, kind,
                                   code if setup else input_dir)
    argv[argv.index(image) + 1:] = command
    # No host environment or token is propagated: retained container_args uses
    # a fixed environment allowlist and network-none/read-only runtime for ASR.
    path = output.parent / ("container-" + kind + ".json")
    need(not path.exists(), "fresh stage diagnostic destination")
    try:
        result = boundary.supervise_container(argv, name, output, kind,
            min(deadline - 120, time.monotonic() + (1200 if setup else duration)))
    except BaseException as error:
        try:
            closed = lifecycle_closed(parse(read(path, MIB)))
        except (OSError, ValueError, TypeError):
            closed = False
        if not closed:
            raise LifecycleUncertain("container cleanup unconfirmed; campaign stopped", output) from error
        raise
    if not lifecycle_closed(parse(read(path, MIB))):
        raise LifecycleUncertain("successful stage did not confirm cleanup", output)
    return result


def shim(command):
    need(command[:3] == ["/runtime/venv/bin/python", "-I", "-B"], "fixed native interpreter")
    return command[:3] + ["/code/research/public-zh64-asr-20261114/worker.py"] + command[3:]


def initial_states():
    return {family: [{"audio_id": f"clip-{i:06d}", "status": "not_run",
                     "attempt_consumed": False, "actual_text": None}
                    for i in range(1, 65)] for family in FAMILIES}


def native_readers(code):
    """Pure retained text/trace readers; no model import or tokenizer download."""
    result = {}
    for family, path, function in (
        ("qwen06", "research/qwen6_asr/core/asr_stage/decoding.py", "qwen_transcript"),
        ("sensevoice", "research/qwen6_asr/sense_parser.py", "corrected_sense_extract"),
        ("whisper", "research/source-screen32-whisper-v1/adapter.py", "termination")):
        raw = read(code / path, MIB)
        namespace = {"__name__": "public64_pure_" + family, "__file__": str(code / path), "require": need}
        if family == "whisper":
            tree = ast.parse(raw)
            nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function]
            need(len(nodes) == 1, "one exact retained pure trace reader")
            source = compile(ast.Module(body=nodes, type_ignores=[]), str(code / path), "exec")
        else:
            source = compile(raw, str(code / path), "exec")
        exec(source, namespace)
        result[family] = namespace[function]
    return result


def bind_native_text(family, identity, observation, capture, row, directory, readers):
    """Independently join raw text, native tokens, termination and input bytes."""
    evidence = observation["original_native_decoder_evidence"]
    config = identity["native_decode_config"]
    term = capture["termination"]
    text = observation["actual_text"]
    if family == "qwen06":
        need(evidence["token_ids"] == term["generated_token_ids"] and
             evidence["prompt_context"] == "" and evidence["force_language"] is None and
             evidence["transcript_repair_applied"] is False, "raw Qwen native token/context binding")
        expected, metadata = readers[family](evidence["decoded_skip_special_tokens"], evidence["token_ids"])
        need(expected["raw_text"] == text and expected["completeness"] == "complete"
             and expected["quality_flags"] == [] and evidence["metadata"] == metadata,
             "raw Qwen transcript and true EOS binding")
    elif family == "sensevoice":
        raw = evidence["raw_result"]
        need(type(raw) in (tuple, list) and len(raw) == 2 and type(raw[0]) is list
             and len(raw[0]) == 1 and raw[0][0]["key"] == row["audio_id"], "raw Sense clip binding")
        decoded = raw[0][0]["text"]
        token, binding = evidence["token_evidence"], evidence["token_binding"]
        need(token["decoded"] == decoded and token["token_ids"] == term["raw_token_ids"]
             and binding["raw_decoded_sha256"] == sha(decoded.encode("utf-8"))
             and binding["metadata_boundary_verified"] is True and binding["raw_redecode_matches_capture"] is True
             and binding["lexical_unknown_present"] is False
             and binding["parser_contract_sha256"] == config["effective_config"]["parser_contract_sha256"]
             and binding["tokenizer"]["sha256"] == config["effective_config"]["tokenizer_sha256"]
             and evidence["use_itn"] is False and evidence["rich_postprocess_applied"] is False,
             "raw Sense token/text/metadata provenance")
        need(all(binding[key] == term[key] for key in ("raw_token_ids", "metadata_token_ids", "lexical_token_ids")),
             "raw Sense native termination token binding")
        expected = readers[family](decoded, term["lexical_token_ids"])
        metadata = expected.pop("metadata")
        need(expected["raw_text"] == text and expected["quality_flags"] == []
             and expected["completeness"] == "complete" and evidence["metadata"] == metadata,
             "raw Sense full transcript extraction")
    else:
        result, captured_input = evidence["decoder_result"], evidence["input_capture"]
        trace, pre = evidence["native_update_trace"], evidence["pre_finalize"]
        need(result["raw_text"] == text and result["native_capture"] == capture
             and result["decoder_completion_validated"] is True and result["full_input_processed"] is True
             and result["returned_token_ids"] == term["generated_token_ids"][:-1]
             and captured_input["pcm16_sha256"] == row["pcm_sha256"]
             and captured_input["native_input"] == capture["input"]
             and readers[family](trace, config["effective_config"]) == term
             and pre == {"token_ids": trace[-1]["token_ids"], "last_update_completed": True, "trace_count": len(trace)},
             "raw Whisper text/input/pre-finalize EOT binding")
        need(parse(read(directory / "native/raw-output.json", 8 * MIB)) == result
             and parse(read(directory / "native/input-capture.json", 8 * MIB)) == captured_input
             and parse(read(directory / "native/pre-finalize.json", 8 * MIB)) == pre,
             "saved Whisper original native member binding")
        for index, step in enumerate(trace, 1):
            need(parse(read(directory / f"native/update-{index:03d}.json", MIB)) == step,
                 "saved Whisper original update trace")
        return
    binding = evidence["input_binding"]
    need(binding["wav"]["sha256"] == row["wav_sha256"]
         and binding["raw_pcm16"]["sha256"] == row["pcm_sha256"]
         and binding["raw_pcm16"]["frame_count"] == row["frames"]
         and binding["operations"] == {"mix": False, "resample": False, "trim": False},
         "raw native original input descriptor")


def collect(family, output, rows, policy, readers):
    """Recheck exact raw native receipt bytes; incomplete attempts stay consumed."""
    from asr_native_contracts import canonical_sha256, validate_native_capture
    model_family = {"qwen06": "qwen3-asr", "sensevoice": "sensevoice", "whisper": "whisper"}[family]
    identity = next(r for r in policy["recognizers"] if r["model_family"] == model_family)
    root = output / {"qwen06": "qwen06/strict-native", "sensevoice": "sensevoice/strict-native",
                     "whisper": "whisper-infer/capture"}[family]
    native_claims = {}
    for path in output.rglob("claim-*.json") if output.exists() else ():
        try:
            value = parse(read(path, MIB))
            if value.get("schema") == "prospective-native-capture-claim-v1":
                native_claims[value.get("opaque_id")] = path
            elif value.get("schema") == "prospective-sense-capture-claim-v1":
                native_claims[value.get("binding", {}).get("opaque_id")] = path
        except (OSError, ValueError, KeyError, TypeError):
            pass  # Original malformed evidence remains in the public inventory.
    result = []
    for row in rows:
        opaque = row["audio_id"]
        directory = root / opaque
        legacy = output / family / (opaque + ".started.json")
        consumed = directory.exists() or legacy.exists() or opaque in native_claims
        record = {"audio_id": row["global_audio_id"], "local_audio_id": opaque,
            "source_id": row["source_id"], "request_id": row["request_id"],
            "wav_sha256": row["wav_sha256"], "pcm_sha256": row["pcm_sha256"],
            "frames": row["frames"], "status": "failed_no_retry" if consumed else "not_run",
            "attempt_consumed": consumed, "actual_text": None}
        if (directory / "completed.json").exists():
            try:
                start_raw = read(directory / "started.json", 8 * MIB)
                end_raw = read(directory / "completed.json", 8 * MIB)
                raw = read(directory / "raw-output.json", 8 * MIB)
                start, end, observation = map(parse, (start_raw, end_raw, raw))
                need(start["schema"] == "retained-asr-started-v1"
                     and end["schema"] == "retained-asr-completed-v1" and end["status"] == "success"
                     and end["decoder_completed"] is True and end["started_sha256"] == sha(start_raw)
                     and end["raw_output_sha256"] == sha(raw), "exact completed native receipt")
                expected = dict(source_id=row["source_id"], request_id=row["request_id"],
                    file_sha256=row["wav_sha256"], pcm_sha256=row["pcm_sha256"], audio_frames=row["frames"],
                    recognizer_policy_sha256=canonical_sha256(policy),
                    recognizer_identity_sha256=canonical_sha256(identity), recognizer_id=identity["recognizer_id"],
                    native_decode_config_sha256=identity["native_decode_config_sha256"],
                    model_lock_sha256=identity["native_decode_config"]["asset_lock_sha256"],
                    source_lock_sha256=identity["native_decode_config"]["source_lock_sha256"])
                need(all(start[k] == v and end[k] == v for k, v in expected.items()), "native source/model/policy join")
                validate_native_capture(identity["native_decode_config"], end["native_capture"])
                need(end["processed_start_frame"] == 0 and end["processed_end_frame"] == row["frames"]
                     and end["native_capture"]["input"]["audio_frames"] == row["frames"], "complete original input")
                need(observation["schema"] == "retained-asr-raw-transcript-v1"
                     and observation["request_id"] == row["request_id"]
                     and observation["output_scope"] == "complete-decoder-transcript"
                     and observation["truncated"] is False and type(observation["actual_text"]) is str
                     and 0 < len(observation["actual_text"]) <= 32768
                     and "original_native_decoder_evidence" in observation, "original native text evidence")
                bind_native_text(family, identity, observation, end["native_capture"], row, directory, readers)
                record.update(status="complete_native_observation", attempt_consumed=True,
                    actual_text=observation["actual_text"], started_sha256=sha(start_raw),
                    completed_sha256=sha(end_raw), raw_output_sha256=sha(raw),
                    native_receipt=str(directory.relative_to(output)))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                record.update(status="rejected_native_evidence", attempt_consumed=True,
                              rejection_type=type(error).__name__, rejection=str(error)[:512])
        result.append(record)
    return result


def export(work, artifact, source, states, receipt, code):
    """Closed fresh snapshot: public source/derivatives/native observations, no weights."""
    need(not artifact.exists(), "create-only public artifact")
    artifact.mkdir(parents=True)
    inventory = {}
    total = 0
    def copy_file(path, name):
        nonlocal total
        raw = read(path, 64 * MIB)
        total += len(raw)
        need(total <= 1024 * MIB, "bounded public evidence archive")
        target = artifact / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        inventory[name] = {"bytes": len(raw), "sha256": sha(raw)}
    # Keep the exact original 11 MB ZIP because its original Actions TTL is short.
    # This copies existing bytes; it performs no generation or media conversion.
    copy_file(source, "source/original-public64.zip")
    for name in ("request.json", "input_adapter.py", "host.py", "worker.py", "README.md"):
        copy_file(HERE / name, "executed-control-source/" + name)
    selected = ("derived", "inputs", "prepare", "inference", "setup", "policy", "provenance")
    for name in selected:
        root = work / name
        if root.exists():
            need(root.is_dir() and not root.is_symlink(), "regular evidence subtree")
            for path in sorted(root.rglob("*")):
                need(not path.is_symlink(), "no evidence aliases")
                uncertain = receipt.get("mutable_stage_output")
                if uncertain and (work / uncertain) in path.parents:
                    continue  # Never freeze potentially live output as completed evidence.
                if path.is_file():
                    copy_file(path, str(path.relative_to(work)))
    if code.exists():
        for path in sorted(code.rglob("*")):
            need(not path.is_symlink(), "no staged source aliases")
            if path.is_file():
                copy_file(path, "executed-native-source/" + str(path.relative_to(code)))
    write(artifact / "observations.json", {"schema": "public-zh64-three-native-observations-v1",
        "providers": states, "training_admitted": False, "human_gold": False,
        "natural_negative_qualified": False, "evaluation_or_production_use": False})
    write(artifact / "summary.json", receipt)
    for name in ("observations.json", "summary.json"):
        body = read(artifact / name, 8 * MIB)
        inventory[name] = {"bytes": len(body), "sha256": sha(body)}
    write(artifact / "artifact-index.json", {"schema": "public-zh64-retained-evidence-index-v1",
        "files": inventory, "training_admitted": False, "human_gold": False,
        "source_audio_regenerated": 0, "evidence_bytes": sum(row["bytes"] for row in inventory.values())})


def run(args, env):
    request, request_sha = inputs.frozen_request(HERE / "request.json")
    event = parse(read(Path(env["GITHUB_EVENT_PATH"]), 2 * MIB))
    claim = parse(read(args.claim, MIB))
    head, run_id = validate_execution(env, event, claim, request_sha)
    need(git(ROOT, "rev-parse", "HEAD") == head, "exact executed control checkout")
    source_raw = read(args.source, 20 * MIB)
    original_rows = inputs.verify_archive(source_raw)  # Before Docker/network/model setup.
    lock, retained = verify_sources(args.retained)
    work = args.work.absolute()
    need(work.resolve() == work and not work.exists(), "new isolated campaign work directory")
    work.mkdir(parents=True)
    deadline = time.monotonic() + request["inner_wall_seconds"]
    states = initial_states()
    receipt = {"schema": "public-zh64-first-three-asr-terminal-v1", "campaign": CAMPAIGN,
        "run_id": run_id, "executed_head_sha": head, "request_sha256": request_sha,
        "request_canonical_sha256": sha(canonical(request)),
        "claim_ref": claim_ref(request_sha), "retained_source_commit": inputs.RETAINED_COMMIT,
        "source_artifact_sha256": inputs.SOURCE_SHA, "source_rows": len(original_rows),
        "preset_split": inputs.SPLITS, "planned_asr_slots": 192,
        "new_tts_calls": 0, "new_training_runs": 0, "retry": False,
        "training_admitted": False, "human_gold": False, "natural_negative_qualified": False,
        "optimizer_loss_authorized": False, "nearphone_quarantined_rows": 48,
        "evaluation_or_production_use": False, "status": "stopped_before_recognition", "stages": {}}
    code = work / "code"
    started = time.monotonic()
    try:
        need(shutil.disk_usage(work).free >= 25 * 1024 ** 3, "same-job 25 GiB initial reserve")
        code = retained.host.stage(Path(args.retained), work, lock)
        need(all(sha(read(code / target, 32 * MIB)) == lock["files"][original]
                 for target, original in lock["stage"].items()), "exact post-stage native bytes")
        shim_path = code / "research/public-zh64-asr-20261114/worker.py"
        shim_path.parent.mkdir(parents=True)
        shutil.copyfile(HERE / "worker.py", shim_path)
        write(work / "provenance/request.json", request)
        write(work / "provenance/claim.json", claim)
        write(work / "provenance/original-input-verification.json", {"rows": original_rows,
            "source_artifact_sha256": sha(source_raw), "source_index_sha256": inputs.INDEX_SHA})
        template = parse(read(code / "research/native-asr-integration-20261104/sense-review-template.json", MIB))
        write(code / "runtime-policy/sense-review.json", {"schema": "reviewed-sense-runtime-capture-v1",
            "executed_head_sha": head, "dependencies": template["dependencies"], "sources": template["sources"]})
        sys.path.insert(0, str(code / "tools"))
        readers = native_readers(code)
        boundary = retained.load_boundary()
        image_lock = parse(read(code / "research/source-screen32-v1/container-lock.json", MIB))
        boundary.host_capacity()
        boundary.pull_image(image_lock, work / "provenance/image-transfer.json")
        image = image_lock["image"]
        runtimes = {name: work / ("runtime-" + name) for name in ("primary", "whisper")}
        for lane, runtime in runtimes.items():
            runtime.mkdir()
            script = "source-screen32-whisper-v1/setup_runtime.py" if lane == "whisper" else "source-screen32-v1/setup_adapter.py"
            command = ["python", "-I", "-B", "/code/research/" + script]
            if lane == "primary":
                command += ["--profile", "asr", "--stage", "setup", "--runtime", "/runtime"]
            receipt["stages"]["setup-" + lane] = launch(boundary, code, runtime,
                work / "setup" / lane / "output", image, CAMPAIGN + "-setup-" + lane,
                command, deadline, setup=True)
        # Derivation alone can read producer metadata. Its code and original ZIP
        # are absent from every recognition container's mounts.
        derive_code = work / "derive-code"
        shutil.copytree(code, derive_code)
        shutil.copyfile(HERE / "input_adapter.py", derive_code / "input_adapter.py")
        derive_input = work / "derive-input"
        derive_input.mkdir()
        shutil.copyfile(args.source, derive_input / "source.zip")
        receipt["stages"]["derive"] = launch(boundary, derive_code, runtimes["primary"],
            work / "derived", image, CAMPAIGN + "-derive",
            ["/runtime/venv/bin/python", "-I", "-B", "/code/input_adapter.py",
             "--derive", "/input/source.zip", "--output", "/output"], deadline,
            input_dir=derive_input, duration=300)
        input_lock = parse(read(work / "derived/input-lock.json", MIB))
        retained.host.validate_input(work / "derived", input_lock)
        jobs = retained.host.batch_inputs(work / "derived", work, input_lock)
        identities = []
        for family in FAMILIES:
            output = work / "prepare" / family / "output"
            command = shim(["/runtime/venv/bin/python", "-I", "-B",
                "/code/research/native-asr-integration-20261104/prepare_worker.py",
                "--family", family, "--executed-head", head])
            receipt["stages"]["prepare-" + family] = launch(boundary, code,
                runtimes["whisper" if family == "whisper" else "primary"], output, image,
                CAMPAIGN + "-prepare-" + family, command, deadline)
            proof = parse(read(output / "zero-forward-proof.json", MIB))
            need(proof["family"] == family and proof["status"] == "prepared_zero_forward"
                 and proof["candidate_count"] == proof["audio_inputs"] == proof["observed_guarded_entrypoint_attempts"] == 0,
                 "actual zero-forward native config capture")
            identities.append(output / "identity.json")
        policy, policy_sha = retained.host.freeze_policy(identities, code / "runtime-policy/recognizer-policy.json")
        write(work / "policy/recognizer-policy.json", policy)
        receipt["recognizer_policy_sha256"] = policy_sha
        for batch, (_, mapping, _) in enumerate(jobs):
            write(code / "runtime-policy" / (f"identity-map-{batch}.json"), mapping)
        receipt["status"] = "incomplete_no_retry"
        for batch, (directory, mapping, rows) in enumerate(jobs):
            # No worker is alive at this update. Sense reads the fixed filename;
            # the other providers read the separately frozen per-batch filename.
            target = code / "runtime-policy/identity-map.json"
            temporary = code / "runtime-policy/identity-map.next.json"
            write(temporary, mapping)
            temporary.replace(target)
            for family in FAMILIES:
                key = f"batch-{batch}-{family}"
                output = work / "inference" / f"batch-{batch}" / family / "output"
                try:
                    command = shim(retained.host.worker_command(family, "infer", head, policy_sha,
                        batch, sha(read(directory / "job.json", MIB))))
                    receipt["stages"][key] = launch(boundary, code,
                        runtimes["whisper" if family == "whisper" else "primary"], output,
                        image, CAMPAIGN + "-" + key, command, deadline, input_dir=directory)
                except Exception as error:
                    receipt["stages"][key] = {"status": "failed_no_retry", "error_type": type(error).__name__,
                        "message": str(error)[:1024]}
                    if isinstance(error, LifecycleUncertain):
                        receipt["mutable_stage_output"] = str(output.relative_to(work))
                        raise
                finally:
                    if receipt.get("mutable_stage_output") == str(output.relative_to(work)):
                        observed = [{"audio_id": row["global_audio_id"], "status": "unknown_after_lifecycle_failure",
                            "attempt_consumed": True, "attempt_count_is_reservation": True, "actual_text": None}
                            for row in rows]
                    else:
                        observed = collect(family, output, rows, policy, readers)
                    by_id = {row["audio_id"]: row for row in observed}
                    for original in states[family]:
                        if original["audio_id"] in by_id:
                            original.update(by_id[original["audio_id"]])
                if deadline - time.monotonic() < 180:
                    raise TimeoutError("remaining launch reserve exhausted; all remaining slots stay not_run")
        if all(row["status"] == "complete_native_observation" for rows in states.values() for row in rows):
            receipt["status"] = "complete_native_screen_unadmitted"
    except BaseException as error:
        receipt["failure"] = {"error_type": type(error).__name__, "message": str(error)[:1024]}
        if isinstance(error, LifecycleUncertain):
            receipt["mutable_stage_output"] = str(error.output.relative_to(work))
    finally:
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["providers"] = {family: {status: sum(row["status"] == status for row in rows)
            for status in ("complete_native_observation", "failed_no_retry", "rejected_native_evidence",
                           "unknown_after_lifecycle_failure", "not_run")}
            for family, rows in states.items()}
        receipt["attempts_consumed"] = sum(row["attempt_consumed"] for rows in states.values() for row in rows)
        need(receipt["attempts_consumed"] <= 192, "global one-attempt denominator")
        export(work, args.artifact, args.source, states, receipt, code)
    print(json.dumps({k: receipt[k] for k in ("status", "providers", "attempts_consumed", "training_admitted")}, sort_keys=True))
    return 0 if receipt["status"] == "complete_native_screen_unadmitted" else 2


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--retained", type=Path, required=True)
    cli.add_argument("--check-source", action="store_true")
    cli.add_argument("--source", type=Path)
    cli.add_argument("--claim", type=Path)
    cli.add_argument("--work", type=Path)
    cli.add_argument("--artifact", type=Path)
    args = cli.parse_args(argv)
    if args.check_source:
        _, digest = inputs.frozen_request(HERE / "request.json")
        lock, _ = verify_sources(args.retained)
        print(json.dumps({"request_sha256": digest, "retained_files": len(lock["files"]),
                          "staged_aliases": len(lock["stage"]), "new_model_calls": 0}))
        return 0
    need(all((args.source, args.claim, args.work, args.artifact)), "all actual execution inputs required")
    return run(args, dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
