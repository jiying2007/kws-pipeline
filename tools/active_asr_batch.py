#!/usr/bin/env python3
"""Private, resumable local ASR over existing audio; no generation or publication.

The decoder manifest contains audio identities only. Results are weak ASR
evidence, never human labels or proof of acoustic completeness.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import research_runtime
from research_job_lock import heavy_model_slot
from setup_asr_runtime import ROOT, canonical, safe_relative, save, sha_file

MODELS = ("qwen06", "sensevoice", "whisper")
CONFIGS = {
    "qwen06": {"device": "cpu", "dtype": "float32", "attention": "eager", "context": "",
               "force_language": None, "max_new_tokens": 256, "do_sample": False,
               "eos_token_id": [151645, 151643], "batch_size": 1},
    "sensevoice": {"device": "cpu", "dtype": "float32", "language": "auto", "use_itn": False,
                   "text_norm": "woitn", "fs": 16000, "data_type": "sound", "output_timestamp": False,
                   "ban_emo_unk": False, "frontend": "kaldi-native-fbank", "dither": 0.0,
                   "vad": None, "batch_size": 1},
    "whisper": {"task": "transcribe", "language": "zh", "temperature": 0.0, "sample_len": None,
                "beam_size": None, "best_of": None, "patience": None, "length_penalty": None,
                "prompt": None, "prefix": None, "suppress_tokens": "-1", "suppress_blank": True,
                "without_timestamps": True, "max_initial_timestamp": None, "fp16": False},
}
BOUNDARY = {"human_gold": False, "training_admitted": False, "shipping_approved": False,
            "target_text_received": False, "peer_transcript_received": False,
            "acoustic_verbatim_completeness": "unknown",
            "language_independent_discovery": False,
            "whisper_language_condition": "Chinese (zh), explicitly supplied, not detected"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def archive_audio_names(archive, expected_count):
    rows = archive.infolist()
    names = [row.filename for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate archive member")
    for row in rows:
        if row.is_dir():
            continue
        safe_relative(row.filename)
    audio = sorted((row for row in rows if row.filename.lower().endswith(".wav")), key=lambda row: row.filename)
    if len(audio) != expected_count or not 1 <= len(audio) <= 64:
        raise ValueError("audio count differs from the declared bounded batch")
    if any(not 44 <= row.file_size <= 16_000_000 for row in audio):
        raise ValueError("audio member size outside bounds")
    return audio


def verify_inputs(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("schema") != "active-asr-inputs-v1" or not 1 <= len(manifest["clips"]) <= 64:
        raise ValueError("invalid input manifest")
    ids = [row["id"] for row in manifest["clips"]]
    if len(set(ids)) != len(ids) or any(re.fullmatch(r"clip-[0-9]{3}", x) is None for x in ids):
        raise ValueError("opaque input IDs required")
    for row in manifest["clips"]:
        if set(row) != {"id", "original_member", "original_wav", "original_pcm", "derived_pcm"}:
            raise ValueError("decoder manifest must contain audio identities only")
        derived = row["derived_pcm"]
        pcm_path = path.parent / safe_relative(derived["path"])
        if (derived["sample_rate"] != 16000 or derived["channels"] != 1 or derived["dtype"] != "<f4"
                or pcm_path.stat().st_size != derived["frames"] * 4 or sha_file(pcm_path) != derived["sha256"]):
            raise ValueError("derived audio identity changed")
    return manifest


def prepare(args):
    research_runtime.check_interpreter(research_runtime.load_manifest())
    if sha_file(args.archive) != args.source_sha256:
        raise ValueError("source ZIP SHA256 mismatch")
    output = args.output.resolve()
    manifest_path = output / "inputs.json"
    if manifest_path.exists():
        manifest = verify_inputs(manifest_path)
        if manifest["source_archive_sha256"] != args.source_sha256:
            raise ValueError("existing inputs belong to a different source archive")
        print(json.dumps({"reused": True, "manifest": str(manifest_path), "clips": len(manifest["clips"])}))
        return 0
    import numpy as np
    import scipy
    from scipy.signal import resample_poly
    import soundfile as sf
    import io
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "active-asr-inputs-v1", "source_archive_sha256": args.source_sha256,
                "created_at": utc(), "preparation_source_sha256": sha_file(__file__),
                "derivation": {"source": "original FLOAT WAV, unmodified", "operation": "scipy.signal.resample_poly",
                               "up": 2, "down": 3, "window": ["kaiser", 5.0], "padtype": "constant",
                               "output": "mono 16000 Hz little-endian float32 PCM", "clipping": False,
                               "normalization": False, "trimming": False, "numpy": np.__version__,
                               "scipy": scipy.__version__, "soundfile": sf.__version__}, "clips": []}
    with zipfile.ZipFile(args.archive) as archive:
        for index, member in enumerate(archive_audio_names(archive, args.expected_count)):
            raw = archive.read(member)
            identifier = f"clip-{index:03d}"
            info = sf.info(io.BytesIO(raw))
            if info.samplerate != 24000 or info.channels != 1 or info.subtype != "FLOAT":
                raise ValueError("expected original mono FLOAT 24 kHz WAV")
            pcm, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if pcm.ndim != 1 or not 1 <= len(pcm) <= 24000 * 30 or not np.isfinite(pcm).all():
                raise ValueError("invalid bounded original waveform")
            derived = np.asarray(resample_poly(pcm, 2, 3, window=("kaiser", 5.0), padtype="constant"), dtype="<f4")
            if not np.isfinite(derived).all():
                raise ValueError("nonfinite derived audio")
            original_path = output / "original" / (identifier + ".wav")
            pcm_path = output / "pcm16k" / (identifier + ".f32")
            for path, body in ((original_path, raw), (pcm_path, derived.tobytes())):
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes() != body:
                    raise ValueError("existing prepared audio differs")
                if not path.exists():
                    with path.open("xb") as stream:
                        stream.write(body)
                        stream.flush()
                        os.fsync(stream.fileno())
            manifest["clips"].append({"id": identifier, "original_member": member.filename,
                "original_wav": {"path": str(original_path.relative_to(output)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)},
                "original_pcm": {"sha256": hashlib.sha256(pcm.astype("<f4", copy=False).tobytes()).hexdigest(),
                                 "sample_rate": rate, "channels": 1, "frames": len(pcm), "dtype": "<f4"},
                "derived_pcm": {"path": str(pcm_path.relative_to(output)), "sha256": sha_file(pcm_path),
                                "sample_rate": 16000, "channels": 1, "frames": len(derived), "dtype": "<f4"}})
    write_once(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "clips": len(manifest["clips"]), "sha256": sha_file(manifest_path)}))
    return 0


def model_identity(model, runtime, input_manifest, threads):
    from setup_asr_runtime import MODEL_LOCK, WHISPER_SOURCE, WHISPER_SHA
    setup = read_json(runtime / "setup.json")
    if setup.get("status") != "complete":
        raise ValueError("ASR installation is incomplete")
    locks = read_json(MODEL_LOCK)
    assets = locks[model]["asset_lock"] if model != "whisper" else {"model": "openai/whisper-small", "sha256": WHISPER_SHA,
                                                                 "source_sha256": sha_file(WHISPER_SOURCE)}
    return {"model": model, "model_assets": assets, "runtime_packages_sha256": digest(setup["installed_packages"]),
            "runtime_lock_sha256": setup["lock_sha256"], "python": setup["active_python"],
            "input_manifest_sha256": sha_file(input_manifest), "decoder_config": CONFIGS[model],
            "threads": threads, "interop_threads": 1, "recipe": "active-asr-native-v1"}


def existing_result(output, model, clip, identity):
    result_path = output / "results" / model / (clip["id"] + ".json")
    claim_path = output / "claims" / model / (clip["id"] + ".json")
    expected = digest({"model_identity": identity, "input": clip})
    if result_path.exists():
        result = read_json(result_path)
        if result.get("identity_sha256") != expected or result.get("status") != "complete":
            raise ValueError("existing result has a different identity or is incomplete")
        if not claim_path.exists() or read_json(claim_path).get("identity_sha256") != expected:
            raise ValueError("completed result has no matching durable call claim")
        claim = read_json(claim_path)
        for row, schema in ((result, "active-asr-result-v1"), (claim, "active-asr-call-claim-v1")):
            if (row.get("schema") != schema or row.get("model") != model or row.get("clip_id") != clip["id"]
                    or row.get("input") != clip or row.get("model_identity") != identity):
                raise ValueError("completed result/claim metadata differs from the requested recognition")
        if claim.get("attempt") != result.get("attempt"):
            raise ValueError("completed result/claim attempts differ")
        native_path = output / "native" / model / (clip["id"] + ".json")
        if not native_path.is_file():
            raise ValueError("completed result native output is missing")
        native_bytes = native_path.read_bytes()
        if hashlib.sha256(native_bytes).hexdigest() != result.get("native_output_sha256"):
            raise ValueError("completed result native output SHA256 differs")
        native = json.loads(native_bytes)
        if (native.get("schema") != "active-asr-native-output-v1" or native.get("identity_sha256") != expected
                or native.get("model") != model or native.get("clip_id") != clip["id"] or not isinstance(native.get("native"), dict)):
            raise ValueError("completed result native output metadata differs")
        return "complete", expected
    if claim_path.exists():
        if read_json(claim_path).get("identity_sha256") != expected:
            raise ValueError("existing call claim has a different identity")
        return "consumed_without_complete_result", expected
    return "pending", expected


def run_worker(command, environment, log, runtime, remaining):
    """Keep a supervised decoder in its queue's group and inherited model slot."""
    with heavy_model_slot(Path(runtime) / "heavy-model.lock", environment=environment) as lease:
        process = subprocess.Popen(command, env=lease.child_environment(environment),
                                   stdout=log, stderr=subprocess.STDOUT,
                                   pass_fds=lease.pass_fds, start_new_session=not lease.supervised)
        try:
            return process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            # The queue owns the supervised process group. Do not signal the
            # batch/queue itself here; its final cleanup removes descendants.
            import signal
            if lease.supervised:
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if lease.supervised:
                    process.kill()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def verify_completed_pilot(path, expected_sha256, runtime, threads):
    """Authenticate a saved completed pilot without loading a model or editing it.

    This opt-in authorizes a fresh input matrix, not reuse of a previous call
    allowance. Every old native cell and its actual queue/audit/save lineage is
    checked. Only the input manifest may differ in the model identity.
    """
    import research_d2_checkpoints as checkpoints
    from research_queue import require
    path = Path(path)
    require(sha_file(path) == expected_sha256, "COMPLETED_PILOT_DECLARATION_CHANGED")
    proof = checkpoints.read(path)
    require(set(proof) == {"schema", "inputs", "output", "source_root", "queue_result",
                          "audit_receipt", "checkpoint", "checkpoint_descriptor", "checkpoint_ack"} and
            proof["schema"] == "active-asr-completed-pilot-v1", "COMPLETED_PILOT_SCHEMA")
    for field in ("inputs", "queue_result", "audit_receipt", "checkpoint_descriptor", "checkpoint_ack"):
        require(checkpoints.pin(proof[field]["path"]) == proof[field], "COMPLETED_PILOT_PIN:" + field)
    folder = Path(proof["checkpoint"])
    require(Path(proof["checkpoint_descriptor"]["path"]) == folder / "capsule.json" and
            Path(proof["checkpoint_ack"]["path"]) == folder / "ack.json", "COMPLETED_PILOT_SAVE_BINDING")
    ack = checkpoints.verify_ack(folder)
    descriptor = checkpoints.verify_capsule(folder, current_sources=True)
    with zipfile.ZipFile(descriptor["capsule"]["path"]) as archive:
        members = {row["path"]: row for row in json.loads(archive.read("MANIFEST.json"))["files"].values()}

    def retained(file):
        actual = checkpoints.pin(file)
        require(members.get(actual["path"]) == actual, "COMPLETED_PILOT_NOT_SAVED:" + actual["path"])
        return actual

    inputs_path, output, old_source = Path(proof["inputs"]["path"]), Path(proof["output"]), Path(proof["source_root"])
    for field in ("inputs", "queue_result", "audit_receipt"):
        retained(proof[field]["path"])
    inputs = verify_inputs(inputs_path)
    require(2 <= len(inputs["clips"]) <= 64, "COMPLETED_PILOT_BOUNDED_INPUTS")
    job = read_json(output / "job.json")
    job_pin = retained(output / "job.json")
    expected = {(model, clip["id"]) for model in MODELS for clip in inputs["clips"]}
    require(job["identity"] == {"input_manifest_sha256": sha_file(inputs_path), "models": list(MODELS),
            "max_calls": len(expected), "heavy_model_concurrency": 1, "threads": threads} and
            job["boundary"] == BOUNDARY, "COMPLETED_PILOT_JOB_SCOPE")
    terminal = read_json(proof["queue_result"]["path"])
    queue_output = Path(proof["queue_result"]["path"]).parent / "outputs"
    relative_job = (output / "job.json").relative_to(queue_output).as_posix()
    require(terminal.get("schema") == "kws-local-research-job-result-v1" and
            terminal.get("status") == "completed" and terminal.get("exit_code") == 0 and
            terminal.get("actual_duration_known") is True and terminal.get("automatic_retry") is False and
            terminal["outputs"].get(relative_job) == {k: job_pin[k] for k in ("bytes", "sha256")},
            "COMPLETED_PILOT_QUEUE_RECEIPT")
    audit_receipt = read_json(proof["audit_receipt"]["path"])
    require(audit_receipt.get("returncode") == 0 and audit_receipt.get("timed_out") is False and
            audit_receipt.get("launch_error") is None, "COMPLETED_PILOT_AUDIT_RECEIPT")
    for field in ("command", "stdout", "stderr"):
        require(retained(audit_receipt[field]["path"]) == audit_receipt[field], "COMPLETED_PILOT_AUDIT_BYTES")
    command = read_json(audit_receipt["command"]["path"])["argv"]
    require(str(old_source / "tools/summarize_active_asr_batch.py") in command and
            all(command[command.index(flag) + 1] == value for flag, value in
                (("--runtime", str(runtime)), ("--inputs", str(inputs_path)), ("--output", str(output)))),
            "COMPLETED_PILOT_AUDIT_COMMAND")
    audit = read_json(output / "audit/final-summary.json")
    retained(output / "audit/final-summary.json")
    inventory = retained(output / "audit/file-hashes.json")
    require(audit.get("schema") == "active-asr-batch-final-audit-v1" and audit.get("status") == "complete" and
            not audit["errors"] and audit["input_manifest_sha256"] == sha_file(inputs_path) and
            audit["expected_matrix_cells"] == audit["verified_matrix_cells"] == audit["original_recognition_calls"] == len(expected) and
            audit["duplicate_model_clip_pairs"] == 0 and audit["finished_before_deadline"] is True and
            audit["persisted_deadline_unix"] == job["deadline_unix"] and
            audit["inventory_sha256"] == inventory["sha256"], "COMPLETED_PILOT_AUDIT_SCOPE")
    hashes = read_json(output / "audit/file-hashes.json")
    require(hashes["roots"] == {"inputs": str(inputs_path.parent), "results": str(output)} and
            hashes["file_count"] == len(hashes["files"]) and hashes["files_sha256"] == digest(hashes["files"]),
            "COMPLETED_PILOT_INVENTORY_SCOPE")
    for name, row in hashes["files"].items():
        relative = safe_relative(name)
        file = inputs_path.parent.joinpath(*relative.parts[1:]) if relative.parts[0] == "inputs" else output / relative
        actual = retained(file)
        require({k: actual[k] for k in ("bytes", "sha256")} == row, "COMPLETED_PILOT_INVENTORY_BYTES")
    sources = ["tools/active_asr_worker.py", "tools/setup_asr_runtime.py", "tools/research_runtime.py",
               "tools/research_job_lock.py", "tools/summarize_active_asr_batch.py", "configs/runtime/research-runtime.json",
               "research/qwen6_asr/model-locks.json", "research/source-screen32-whisper-v1/source-lock.json"]
    sources += ["research/qwen6_asr/core/" + name for name in (
        "asr_stage/__init__.py", "asr_stage/architecture.py", "asr_stage/decoding.py", "asr_stage/runtime_support.py",
        "vendor/__init__.py", "vendor/qwen_loading_gate_v2.py", "vendor/checkpoint_guard/__init__.py",
        "vendor/checkpoint_guard/checkpoint_contract.py", "vendor/checkpoint_guard/exact_apply.py")]
    source_hashes = {}
    for name in sources:
        prior = retained(old_source / name)
        require(sha_file(ROOT / name) == prior["sha256"], "COMPLETED_PILOT_NUMERICAL_SOURCE_CHANGED:" + name)
        source_hashes[name] = prior["sha256"]
    retained(old_source / "tools/active_asr_batch.py")
    old_tree, new_tree = ast.parse((old_source / "tools/active_asr_batch.py").read_bytes()), ast.parse(Path(__file__).read_bytes())
    for name in ("verify_inputs", "prepare", "model_identity", "existing_result", "run_worker"):
        a = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        b = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        require(ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False),
                "COMPLETED_PILOT_BATCH_NUMERICS_CHANGED:" + name)
    worker_sha = source_hashes["tools/active_asr_worker.py"]
    require(audit["decoder_source_sha256"] == worker_sha and
            audit["auditor_source_sha256"] == source_hashes["tools/summarize_active_asr_batch.py"], "COMPLETED_PILOT_AUDITOR_SOURCE")
    for kind in ("claims", "native", "results"):
        paths = list((output / kind).glob("*/*.json"))
        require({(p.parent.name, p.stem) for p in paths} == expected, "COMPLETED_PILOT_MATRIX")
        for file in paths:
            retained(file)
    identities = {}
    for model in MODELS:
        identity = model_identity(model, runtime, inputs_path, threads)
        identities[model] = {k: v for k, v in identity.items() if k != "input_manifest_sha256"}
        for clip in inputs["clips"]:
            require(existing_result(output, model, clip, identity)[0] == "complete" and
                    read_json(output / "results" / model / (clip["id"] + ".json"))["decoder_source_sha256"] == worker_sha,
                    "COMPLETED_PILOT_MODEL_OR_RESULT_CHANGED")
    return {"proof_sha256": expected_sha256, "input_manifest_sha256": sha_file(inputs_path),
            "job_sha256": job_pin["sha256"], "verified_calls": len(expected),
            "checkpoint_sha256": descriptor["capsule"]["sha256"], "object_id": ack["object_id"],
            "version": ack["version"], "compatible_source_sha256": digest(source_hashes),
            "original_pcm_sha256": sorted({clip["original_pcm"]["sha256"] for clip in inputs["clips"]}),
            "derived_pcm_sha256": sorted({clip["derived_pcm"]["sha256"] for clip in inputs["clips"]}),
            "model_identities_without_input": identities}


def run(args):
    research_runtime.check_interpreter(research_runtime.load_manifest())
    manifest = verify_inputs(args.inputs)
    runtime, output = args.runtime.resolve(), args.output.resolve()
    pilot_path, pilot_sha = getattr(args, "completed_pilot", None), getattr(args, "completed_pilot_sha256", None)
    if (pilot_path is None) != (pilot_sha is None):
        raise ValueError("completed pilot path and SHA256 must be supplied together")
    pilot = None
    if pilot_path is not None:
        if args.phase != "full" or args.models != list(MODELS) or args.threads != 4 or args.budget_minutes != 9:
            raise ValueError("verified pilot extension requires full three-model order, four threads, and nine minutes")
        pilot = verify_completed_pilot(pilot_path, pilot_sha, runtime, args.threads)
        if pilot["input_manifest_sha256"] == sha_file(args.inputs):
            raise ValueError("completed pilot audio must not be recognized again")
        if any({clip[kind]["sha256"] for clip in manifest["clips"]} & set(pilot[kind + "_sha256"])
               for kind in ("original_pcm", "derived_pcm")):
            raise ValueError("completed pilot PCM must not be recognized again under new IDs or a reordered manifest")
        with heavy_model_slot(runtime / "heavy-model.lock") as lease:
            if not lease.inherited or not lease.supervised:
                raise ValueError("verified pilot extension requires the supervised inherited heavy-model slot")
        if output.exists() and any(output.iterdir()):
            raise ValueError("verified pilot extension requires unused output; no automatic restart")
    output.mkdir(parents=True, exist_ok=True)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.attempt) is None:
        raise ValueError("simple bounded attempt ID required")
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job_path = output / "job.json"
        identity = {"input_manifest_sha256": sha_file(args.inputs), "models": list(MODELS),
                    "max_calls": 3 * len(manifest["clips"]), "heavy_model_concurrency": 1, "threads": args.threads}
        if pilot is not None:
            identity["completed_pilot"] = pilot
        if job_path.exists():
            job = read_json(job_path)
            if job["identity"] != identity:
                raise ValueError("existing job scope differs; use a separate output directory")
        else:
            job = {"schema": "active-asr-job-v1", "identity": identity, "created_at": utc(),
                   "deadline_unix": time.time() + args.budget_minutes * 60,
                   "budget_minutes": args.budget_minutes, "per_item_minutes_cap": 999, "boundary": BOUNDARY}
            write_once(job_path, job)
        if time.time() >= job["deadline_unix"]:
            raise TimeoutError("persisted job wall-clock budget exhausted")
        clips = manifest["clips"]
        ids = [clips[0]["id"], clips[-1]["id"]] if args.phase == "smoke" and len(clips) > 1 else [x["id"] for x in clips]
        if args.phase == "full" and pilot is None:
            smoke_ids = {clips[0]["id"], clips[-1]["id"]}
            for model in MODELS:
                current = model_identity(model, runtime, args.inputs, args.threads)
                for clip in clips:
                    if clip["id"] in smoke_ids and existing_result(output, model, clip, current)[0] != "complete":
                        raise ValueError("each model must complete the same bounded smoke samples before the full batch")
        attempt_dir = output / "attempts" / args.attempt
        attempt_dir.mkdir(parents=True, exist_ok=False)
        summary = {"schema": "active-asr-attempt-v1", "attempt": args.attempt, "phase": args.phase,
                   "started_at": utc(), "models": {}, "status": "running", "boundary": BOUNDARY}
        save(attempt_dir / "summary.json", summary)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                   OPENBLAS_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads),
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                   HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES="")
        for model in args.models:
            remaining = job["deadline_unix"] - time.time()
            if remaining <= 0:
                summary["models"][model] = {"status": "budget_exhausted"}
                break
            command = [sys.executable, "-B", str(ROOT / "tools/active_asr_worker.py"), "--model", model,
                       "--runtime", str(runtime), "--inputs", str(args.inputs.resolve()), "--output", str(output),
                       "--attempt", args.attempt, "--threads", str(args.threads), "--ids", ",".join(ids)]
            with (attempt_dir / (model + ".log")).open("x") as log:
                started = time.monotonic()
                code = run_worker(command, env, log, runtime, remaining)
                summary["models"][model] = {"exit_code": code, "elapsed_seconds": round(time.monotonic() - started, 3)}
            save(attempt_dir / "summary.json", summary)
            print(json.dumps({"model": model, **summary["models"][model]}), flush=True)
        summary.update(status="complete" if len(summary["models"]) == len(args.models) and all(x.get("exit_code") == 0 for x in summary["models"].values()) else "incomplete",
                       finished_at=utc(), calls_claimed=len(list((output / "claims").glob("*/*.json"))),
                       results_saved=len(list((output / "results").glob("*/*.json"))))
        save(attempt_dir / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "boundary"}), flush=True)
        return 0 if summary["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--source-sha256", required=True)
    prep.add_argument("--expected-count", type=int, default=64)
    prep.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run")
    execute.add_argument("--runtime", type=Path, required=True)
    execute.add_argument("--inputs", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--attempt", required=True)
    execute.add_argument("--phase", choices=("smoke", "full"), required=True)
    execute.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    execute.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    execute.add_argument("--budget-minutes", type=int, default=90)
    execute.add_argument("--completed-pilot", type=Path,
                         help="Explicit saved completed-pilot proof for one fresh full batch; requires its SHA256.")
    execute.add_argument("--completed-pilot-sha256")
    verify = sub.add_parser("verify-pilot", help="Read-only completed-pilot authentication; no model imports or queue.")
    verify.add_argument("--completed-pilot", type=Path, required=True)
    verify.add_argument("--completed-pilot-sha256", required=True)
    verify.add_argument("--runtime", type=Path, required=True)
    verify.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args(argv)
    if args.command == "run" and (not 1 <= args.budget_minutes <= 999 or len(set(args.models)) != len(args.models)):
        parser.error("budget must be 1..999 minutes and model IDs distinct")
    if args.command == "verify-pilot":
        value = verify_completed_pilot(args.completed_pilot, args.completed_pilot_sha256, args.runtime.resolve(), args.threads)
        print(json.dumps({"status": "verified-completed-pilot", "binding": value, "new_model_calls": 0}, sort_keys=True))
        return 0
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
