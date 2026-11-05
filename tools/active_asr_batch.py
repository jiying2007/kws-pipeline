#!/usr/bin/env python3
"""Private, resumable local ASR over existing audio; no generation or publication.

The decoder manifest contains audio identities only. Results are weak ASR
evidence, never human labels or proof of acoustic completeness.
"""
from __future__ import annotations

import argparse
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


def run(args):
    research_runtime.check_interpreter(research_runtime.load_manifest())
    manifest = verify_inputs(args.inputs)
    runtime, output = args.runtime.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.attempt) is None:
        raise ValueError("simple bounded attempt ID required")
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job_path = output / "job.json"
        identity = {"input_manifest_sha256": sha_file(args.inputs), "models": list(MODELS),
                    "max_calls": 3 * len(manifest["clips"]), "heavy_model_concurrency": 1, "threads": args.threads}
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
        if args.phase == "full":
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
    args = parser.parse_args(argv)
    if args.command == "run" and (not 1 <= args.budget_minutes <= 999 or len(set(args.models)) != len(args.models)):
        parser.error("budget must be 1..999 minutes and model IDs distinct")
    return prepare(args) if args.command == "prepare" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
