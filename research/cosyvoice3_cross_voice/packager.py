#!/usr/bin/env python3
"""Stdlib-only, no-generation packager for the fixed CosyVoice30 batch.

Only canonical row filenames are admitted. NPY/WAV validation inspects headers,
lengths and hashes, never sample values. No source logs or raw receipts are
published. A generated status means a validated completed API return, not EOS
proof, intelligibility, a human label, or permission to train.

pack(job_root, config, output_root, exposed_pcm_sha256=(), technical_receipts=None)
creates exactly two new artifact directories below a new output_root. config
is a mapping or JSON filename; JOB/outputs is the only audio input directory.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import struct

LIMIT = 128 * 1024**2
JSON_LIMIT = 64 * 1024
FRAMING_ALLOWANCE = 1024**2
SUFFIXES = (".native.npy", ".wav", ".16k.wav")
ARTIFACTS = ("cosy30-train-dev-24", "cosy30-sealed-six")
INTERNAL_JSON = frozenset(("input-identities.json", "token-contract.json", "startup.json", "outcomes.json"))
VOICES = ("Eric", "Serena", "Vivian", "Uncle_Fu", "Dylan")
PHRASES = (("k1", "你好小窝", 1), ("k2", "小窝小窝", 2),
           ("repeat_nihao", "你好你好", 0), ("single_xiaowo", "小窝", 0),
           ("k1_wu_foil", "你好小屋", 0), ("k2_wu_foil", "小屋小屋", 0))
IDS = (
    "963e8a82e801f800 a27d486bf3011cce 9bf0442a59aa1ed9 bf5567786efe5d37 e40742c3b7807808 ea9ec4f1914de7a1 "
    "44f42bfe7e6c8cff 63e439ef22c44da3 d47372ae6c8d55a3 702fdac8f0cd2a55 6d96f1531378529f f0fb4689ef999ad7 "
    "5afb9ad97dbd1fd6 63941b3d09d895e7 dbe5f5f6b95818c3 0f8e3a0e467d9149 9bc4956f2e87322e e147dcf374fa660c "
    "a408689e8c523e2e 90cd72fd652f9bef 94765e1dc9767a97 399f2a742c267a74 dbcdf9beb8cf511c bf92bed164ccb28f "
    "6d9d7324ce7f7079 adb94d5f7fb77b29 33ffdcc4db24cf90 d73c2339b5d9546d 1ffd2254512e9ff6 b08c9850b1d5757f"
).split()
REFERENCE_PCM = dict(zip(VOICES, (
    "614df5de255d0609ef1894f1ed085d4a5c32100ce3919dee6320e4ba5d7f20e8",
    "d3c7fa79aaeb273a93379637cdf1d93b8ce4f5af7d7983bdf974dfd966b94fa4",
    "4d5db9285d3bd10fae13345b4ccd3ad081b748257eff03d1c01e211865b7926d",
    "61dcec8fc60753686a363a3ace40109a26df37cf556733c046b8c6cbc757ce02",
    "f2c212d2a54f124535a6ac7ce6e2fc272a99dd878030c8765064dca8f30ec037",
)))
EXPECTED_ROWS = tuple({
    "clip_id": "cv30-" + clip, "stock_reference_voice": VOICES[i // 6],
    "split": "train" if i < 18 else "development" if i < 24 else "sealed_recording_holdout",
    "phrase_id": PHRASES[i % 6][0], "intended_text": PHRASES[i % 6][1],
    "intended_keyword_id": PHRASES[i % 6][2], "seed": 610401 + i,
    "attempt_limit": 1, "artifact_group": "train-dev-24" if i < 24 else "sealed-six",
} for i, clip in enumerate(IDS))
INITIAL_FOUR = tuple(EXPECTED_ROWS[i]["clip_id"] for i in (0, 1, 6, 7))
LISTENING_ORDER = tuple(sorted(INITIAL_FOUR, key=lambda clip: hashlib.sha256(("initial-listening-v1:" + clip).encode("ascii")).hexdigest()))

# These files may be read, but only explicit scalar projections are published.
RECEIPT_PATHS = {
    "controller": "controller-state.json",
    "generate_supervisor": "receipts/generation/generate-supervisor.json",
    "generate_worker": "receipts/generation/generate-worker.json",
    "runtime_install": "receipts/runtime/install-receipt.json",
    "runtime_qualification": "receipts/runtime/runtime-qualification.json",
}
STATUS_VALUES = frozenset(("complete", "completed", "incomplete", "passed", "failed", "qualified", "stopped", "running", "not_attempted", "installed_pending_offline_qualification"))
PHASE_VALUES = frozenset(("acquire-source", "install-runtime", "qualify-runtime", "qualify-source", "acquire-model", "generate-fixed30"))
NUMERIC_FIELDS = frozenset((
    "returncode", "peak_process_tree_rss_bytes", "elapsed_seconds", "cpu_seconds",
    "aggregate_cpu_seconds", "output_bytes", "retained_bytes", "log_bytes",
    "log_limit_bytes", "log_observed_bytes", "generated_clips", "attempted_clips",
    "wall_seconds", "peak_rss_bytes", "cpu_limit_seconds", "wall_limit_seconds",
    "new_job_bytes_max", "max_rss_bytes", "peak_job_bytes", "package_input_count",
    "peak_observed_consumed_bytes", "runtime_download_bytes",
))
BOOLEAN_FIELDS = frozenset(("no_retry", "log_truncated", "log_eof", "log_drain_timed_out", "no_training_admission", "cleanup_verified"))
RECEIPT_FIELDS = {
    "controller": {"status", "run_id", "head_sha", "elapsed_seconds", "new_job_bytes_max", "steps"},
    "runtime_install": {"status", "lock_sha256", "python_version", "package_input_count",
                        "peak_observed_consumed_bytes", "runtime_download_bytes"},
    "runtime_qualification": {"status", "lock_sha256", "elapsed_seconds"},
    "generate_supervisor": {"status", "returncode", "peak_process_tree_rss_bytes", "elapsed_seconds",
                            "no_retry", "log_bytes", "log_truncated", "log_eof"},
    "generate_worker": {"status", "generated_clips", "network_attempts"},
}


class GateError(RuntimeError):
    """A bounded code only; never include source paths or raw exception text."""


def require(ok, code):
    if not ok:
        raise GateError(code)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_links(path):
    """Check lexical components before resolve; a resolved symlink is too late."""
    path = Path(path).absolute()
    require(".." not in path.parts, "unsafe_path")
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            require(not item.is_symlink(), "symlink")
    return path


def regular(path, maximum=LIMIT):
    no_links(path)
    info = Path(path).lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1, "nonregular_file")
    require(info.st_size <= maximum, "oversize_file")
    return info.st_size


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key")
        result[key] = value
    return result


def read_json(path, maximum=JSON_LIMIT):
    regular(path, maximum)
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_json_pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(GateError("nonfinite_json")))
    except (ValueError, UnicodeError, RecursionError):
        raise GateError("invalid_json") from None


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def config_rows(config):
    cfg = read_json(config, 1024**2) if isinstance(config, (str, os.PathLike)) else config
    require(isinstance(cfg, dict), "invalid_config")
    rows = cfg.get("rows")
    require(isinstance(rows, list) and len(rows) == 30, "fixed30_required")
    # Unknown config fields are not copied; every fixed identity field is pinned.
    for actual, expected in zip(rows, EXPECTED_ROWS):
        require(isinstance(actual, dict) and all(type(actual.get(k)) is type(v) and actual[k] == v
                                               for k, v in expected.items()), "frozen_row_mismatch")
    refs = cfg.get("references", [])
    require(isinstance(refs, list) and len(refs) == 5, "reference_matrix_mismatch")
    require(all(isinstance(ref, dict) for ref in refs), "reference_matrix_mismatch")
    require({(ref.get("stock_voice"), ref.get("pcm_sha256")) for ref in refs} == set(REFERENCE_PCM.items()),
            "reference_matrix_mismatch")
    return EXPECTED_ROWS


def native_header(path):
    size = regular(path, 480000 * 4 + 4108)
    try:
        with open(path, "rb") as stream:
            require(stream.read(6) == b"\x93NUMPY", "native_magic")
            version = stream.read(2)
            require(version in (b"\x01\x00", b"\x02\x00"), "native_version")
            width = 2 if version[0] == 1 else 4
            length_bytes = stream.read(width)
            require(len(length_bytes) == width, "native_header")
            length = int.from_bytes(length_bytes, "little")
            require(0 < length <= 4096, "native_header")
            header = ast.literal_eval(stream.read(length).decode("ascii"))
            require(isinstance(header, dict) and set(header) == {"descr", "fortran_order", "shape"}, "native_header")
            require(header["descr"] == "<f4" and header["fortran_order"] is False, "native_format")
            shape = header["shape"]
            require(isinstance(shape, tuple) and len(shape) == 1 and type(shape[0]) is int, "native_shape")
            frames = shape[0]
            require(0 < frames <= 480000, "duration_limit")
            require(size == 8 + width + length + 4 * frames, "native_length")
    except (ValueError, SyntaxError, UnicodeError, RecursionError):
        raise GateError("native_header") from None
    return {"dtype": "float32_le", "frames": frames, "channels": 1}


def wav_header(path, rate, expected_frames):
    """Canonical PCM16 WAV: fmt + data only, no metadata or sample decoding."""
    size = regular(path, rate * 20 * 2 + 44)
    with open(path, "rb") as stream:
        header = stream.read(44)
        require(len(header) == 44, "wav_header")
        (riff, riff_size, wave, fmt, fmt_size, encoding, channels, sample_rate,
         byte_rate, align, bits, data, data_size) = struct.unpack("<4sI4s4sIHHIIHH4sI", header)
        require((riff, wave, fmt, fmt_size, encoding, channels, sample_rate, byte_rate, align, bits, data)
                == (b"RIFF", b"WAVE", b"fmt ", 16, 1, 1, rate, rate * 2, 2, 16, b"data"), "wav_format")
        require(0 < expected_frames <= rate * 20, "duration_limit")
        require(data_size == expected_frames * 2 and size == data_size + 44 and riff_size == size - 8,
                "wav_length")
        digest = hashlib.sha256()
        # Hash opaque PCM bytes for exact duplicate checks only. No waveform metrics.
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sample_rate": rate, "channels": 1, "sample_width_bytes": 2,
            "frames": expected_frames, "pcm_sha256": digest.hexdigest()}


def validate_row(output, row, result):
    clip = row["clip_id"]
    require(isinstance(result, dict) and result.get("clip_id") == clip, "result_identity")
    require(result.get("status") == "generated", "result_status")
    require(result.get("api_return_completed") is True, "api_return_unconfirmed")
    require(result.get("eos_proved") is False, "unsupported_eos_claim")
    for key in ("seed", "phrase_id", "split", "stock_reference_voice", "intended_text", "intended_keyword_id"):
        if key in result:
            require(type(result[key]) is type(row[key]) and result[key] == row[key], "result_identity")
    files = result.get("files")
    require(isinstance(files, dict) and set(files) == set(SUFFIXES), "file_hash_contract")
    identities = {}
    for suffix in SUFFIXES:
        path = output / (clip + suffix)
        require(path.exists(), "missing_audio")
        size = regular(path, 2 * 1024**2)
        expected_hash = files[suffix]
        require(isinstance(expected_hash, str) and re.fullmatch(r"[0-9a-f]{64}", expected_hash), "file_hash_contract")
        require(sha256(path) == expected_hash, "file_hash_mismatch")
        identities[suffix] = {"bytes": size, "sha256": expected_hash}
    native = native_header(output / (clip + ".native.npy"))
    frames = native["frames"]
    require(type(result.get("frames24k")) is int and result["frames24k"] == frames, "duration_receipt")
    if "frames" in result:
        require(type(result["frames"]) is int and result["frames"] == frames, "duration_receipt")
    require(type(result.get("sample_rate")) is int and result["sample_rate"] == 24000, "duration_receipt")
    require(type(result.get("duration_seconds")) in (int, float)
            and result["duration_seconds"] == frames / 24000, "duration_receipt")
    headers = {".native.npy": native,
               ".wav": wav_header(output / (clip + ".wav"), 24000, frames),
               ".16k.wav": wav_header(output / (clip + ".16k.wav"), 16000, (frames * 2 + 2) // 3)}
    return {"clip_id": clip, "split": row["split"], "status": "generated", "attempt": 1,
            "api_return_completed": True, "eos_proved": False, "frames24k": frames,
            "sample_rate": 24000, "duration_seconds": frames / 24000,
            "files": identities, "headers": headers}


def scalar_projection(receipt):
    projected = {}
    if isinstance(receipt.get("status"), str) and receipt["status"] in STATUS_VALUES:
        projected["status"] = receipt["status"]
    for key in NUMERIC_FIELDS:
        value = receipt.get(key)
        lower = -255 if key == "returncode" else 0
        if type(value) in (int, float) and lower <= value <= 10**15 and math.isfinite(value):
            projected[key] = value
    for key in BOOLEAN_FIELDS:
        if type(receipt.get(key)) is bool:
            projected[key] = receipt[key]
    for key, pattern in (("head_sha", r"[0-9a-f]{40}"), ("lock_sha256", r"[0-9a-f]{64}"),
                         ("receipt_sha256", r"[0-9a-f]{64}"), ("log_sha256", r"[0-9a-f]{64}"),
                         ("run_id", r"[0-9]{1,24}"), ("python_version", r"[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,3}")):
        if isinstance(receipt.get(key), str) and re.fullmatch(pattern, receipt[key]):
            projected[key] = receipt[key]
    return projected


def sanitize_receipts(receipts):
    require(isinstance(receipts, dict) and set(receipts) <= set(RECEIPT_PATHS), "receipt_name")
    clean = {}
    for name, receipt in receipts.items():
        require(isinstance(receipt, dict), "receipt_format")
        projected = scalar_projection(receipt)
        if name == "controller" and isinstance(receipt.get("steps"), list):
            require(len(receipt["steps"]) <= 6, "controller_step_limit")
            projected["steps"] = []
            for step in receipt["steps"]:
                require(isinstance(step, dict), "controller_step_format")
                if not isinstance(step.get("phase"), str) or step["phase"] not in PHASE_VALUES:
                    continue
                cleaned = scalar_projection(step)
                step_keys = {"status", "elapsed_seconds", "max_rss_bytes", "peak_job_bytes", "returncode",
                             "cleanup_verified", "log_bytes", "log_truncated", "log_sha256"}
                projected["steps"].append(dict({k: v for k, v in cleaned.items() if k in step_keys}, phase=step["phase"]))
        if name == "runtime_install" and isinstance(receipt.get("events"), list):
            require(len(receipt["events"]) <= 4096, "runtime_event_limit")
            sizes = [event.get("bytes") for event in receipt["events"]
                     if isinstance(event, dict) and event.get("event") == "verified_download"]
            if all(type(size) is int and 0 <= size <= 10**12 for size in sizes):
                projected["runtime_download_bytes"] = sum(sizes)
        if name == "generate_worker" and receipt.get("network_attempts") == []:
            projected["network_attempts"] = []
        clean[name] = {k: v for k, v in projected.items() if k in RECEIPT_FIELDS[name] | {"receipt_sha256"}}
    return clean


def pack(job_root, config, output_root, *, exposed_pcm_sha256=(), technical_receipts=None):
    """Package finished valid cells; return a public-safe aggregate summary.

    Global path/allowlist/cap failures raise GateError before creating artifacts.
    Individual missing, incomplete or invalid cells remain in outcomes but their
    audio is excluded. Exact duplicates invalidate all matching generated cells.
    Existing destinations are never overwritten. Input files are never changed.
    """
    rows = config_rows(config)
    job = no_links(job_root)
    require(job.is_dir(), "missing_job")
    output = no_links(job / "outputs")
    destination = no_links(output_root)
    require(not destination.exists(), "destination_exists")
    require(destination.parent.is_dir(), "destination_parent_missing")
    require(not destination.is_relative_to(job) and not job.is_relative_to(destination), "overlapping_roots")
    exposed = set(REFERENCE_PCM.values()) | set(exposed_pcm_sha256)
    require(all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in exposed), "exposed_hash_format")
    allowed = set(INTERNAL_JSON)
    for row in rows:
        allowed.update(row["clip_id"] + suffix for suffix in (*SUFFIXES, ".attempt.json", ".result.json"))
    retained = 0
    if output.exists():
        require(output.is_dir(), "output_not_directory")
        for path in output.iterdir():
            require(path.name in allowed, "unexpected_output_entry")
            retained += regular(path, JSON_LIMIT if path.suffix == ".json" else LIMIT)
    # Receipt and phase-log retention also consumes the same publication
    # envelope, even though their raw contents are never published.
    for directory_name in ("receipts", "logs"):
        directory = no_links(job / directory_name)
        if directory.exists():
            require(directory.is_dir(), "retained_tree_not_directory")
            for path in directory.rglob("*"):
                no_links(path)
                if path.is_dir():
                    continue
                retained += regular(path, LIMIT)
    require(retained <= LIMIT, "retained_output_limit")
    outcomes = []
    gap_seen = False
    for row in rows:
        clip = row["clip_id"]
        outcome = {"clip_id": clip, "split": row["split"], "status": "not_attempted"}
        attempt_path = output / (clip + ".attempt.json")
        result_path = output / (clip + ".result.json")
        any_files = any((output / (clip + suffix)).exists() for suffix in (*SUFFIXES, ".result.json", ".attempt.json"))
        if not any_files:
            gap_seen = True
        else:
            outcome["status"] = "invalid"
            try:
                require(attempt_path.exists(), "missing_attempt")
                attempt = read_json(attempt_path)
                require(attempt == {"clip_id": clip, "seed": row["seed"], "attempt": 1}
                        and type(attempt.get("attempt")) is int and type(attempt.get("seed")) is int, "attempt_contract")
                require(not gap_seen, "attempt_after_gap")
                require(result_path.exists(), "missing_result")
                outcome = validate_row(output, row, read_json(result_path))
            except (GateError, FileNotFoundError) as error:
                outcome["invalid_reason"] = str(error) if isinstance(error, GateError) else "missing_file"
        outcomes.append(outcome)
    owners = defaultdict(set)
    for outcome in outcomes:
        if outcome["status"] == "generated":
            for suffix in (".wav", ".16k.wav"):
                owners[outcome["headers"][suffix]["pcm_sha256"]].add(outcome["clip_id"])
    duplicates = {clip for digest, clips in owners.items() if len(clips) > 1 or digest in exposed for clip in clips}
    for index, outcome in enumerate(outcomes):
        if outcome["clip_id"] in duplicates:
            outcomes[index] = {"clip_id": outcome["clip_id"], "split": outcome["split"],
                               "status": "invalid", "invalid_reason": "duplicate_or_exposed_pcm"}
    if technical_receipts is None:
        technical_receipts = {}
        for name, relative in RECEIPT_PATHS.items():
            source = no_links(job / relative)
            if source.exists():
                receipt = read_json(source, 2 * 1024**2)
                require(isinstance(receipt, dict), "receipt_format")
                technical_receipts[name] = dict(receipt, receipt_sha256=sha256(source))
    clean_receipts = sanitize_receipts(technical_receipts)
    counts = {status: sum(row["status"] == status for row in outcomes)
              for status in ("generated", "invalid", "not_attempted")}
    controller = clean_receipts.get("controller", {})
    supervisor = clean_receipts.get("generate_supervisor", {})
    worker = clean_receipts.get("generate_worker", {})
    execution_ok = (controller.get("status") == "completed"
                    and supervisor.get("status") == "passed" and supervisor.get("returncode") == 0
                    and worker.get("status") == "complete"
                    and type(worker.get("generated_clips")) is int and worker["generated_clips"] == 30
                    and worker.get("network_attempts") == [])
    complete = counts["generated"] == 30 and execution_ok
    # Global outcomes contain only opaque IDs, split, status and bounded codes.
    public_outcomes = [{k: value for k, value in row.items() if k in ("clip_id", "split", "status", "invalid_reason")}
                       for row in outcomes]
    summary = {"schema": "cosy30.two-artifact.v1", "status": "complete" if complete else "incomplete",
               "maximum_clips": 30, "counts": counts, "outcomes": public_outcomes,
               "successful_execution_receipts": execution_ok,
               "no_retry": True, "no_training_admission": True, "heldout_semantic_inspection": False,
               "api_return_is_eos_proof": False, "initial_listening_enabled": complete,
               "reference_audio_included": False, "raw_receipts_or_logs_included": False}
    # Build a complete explicit staging plan before any destination writes.
    plans = {name: {} for name in ARTIFACTS}
    source_facts = {}
    for group_index, artifact in enumerate(ARTIFACTS):
        selected = outcomes[:24] if group_index == 0 else outcomes[24:]
        plan = plans[artifact]
        plan["technical/summary.json"] = json_bytes(dict(summary, artifact=artifact, artifact_row_count=len(selected)))
        plan["technical/row-qc.json"] = json_bytes({"rows": selected, "inspection": "headers_lengths_hashes_only"})
        for outcome in selected:
            if outcome["status"] == "generated":
                for suffix in SUFFIXES:
                    name = outcome["clip_id"] + suffix
                    plan["audio/" + name] = output / name
                    source_facts[output / name] = outcome["files"][suffix]
    plans[ARTIFACTS[0]]["technical/controller-resources.json"] = json_bytes(clean_receipts)
    plans[ARTIFACTS[0]]["intended-rows.json"] = json_bytes({"rows": list(rows[:24]), "human_reviewed": False,
                                                          "intended_text_is_verified_label": False})
    listening = {"enabled": complete, "requires_all30_generated": True, "requires_successful_execution": True, "heldout_opened": False,
                 "distinct_audio_cap_seconds": 80, "total_playback_cap_seconds": 160,
                 "at_most_one_replay": True, "rows": []}
    if complete:
        for number, clip in enumerate(LISTENING_ORDER, 1):
            alias = "initial-listening/%02d.wav" % number
            source = output / (clip + ".16k.wav")
            plans[ARTIFACTS[0]][alias] = source
            listening["rows"].append({"clip_id": clip, "path": alias, "sample_rate": 16000,
                                      "sha256": source_facts[source]["sha256"], "human_reviewed": False})
    plans[ARTIFACTS[0]]["technical/initial-listening-manifest.json"] = json_bytes(listening)
    for artifact, plan in plans.items():
        members = []
        for name, value in sorted(plan.items()):
            size = len(value) if isinstance(value, bytes) else source_facts[value]["bytes"]
            digest = hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else source_facts[value]["sha256"]
            members.append({"path": name, "bytes": size, "sha256": digest})
        plan["artifact-manifest.json"] = json_bytes({"schema": "cosy30.artifact.v1", "artifact": artifact,
                                                    "files": members, "combined_limit_bytes": LIMIT})
    staged_bytes = sum(len(value) if isinstance(value, bytes) else source_facts[value]["bytes"]
                       for plan in plans.values() for value in plan.values())
    require(staged_bytes <= LIMIT, "combined_artifact_limit")
    envelope = retained + 2 * staged_bytes + FRAMING_ALLOWANCE
    require(envelope <= LIMIT, "combined_publication_envelope")
    destination.mkdir()
    try:
        for artifact, plan in plans.items():
            base = destination / artifact
            base.mkdir()
            # Write manifest last, so only a fully staged artifact is publishable.
            names = sorted(name for name in plan if name != "artifact-manifest.json") + ["artifact-manifest.json"]
            for name in names:
                target = base / name
                target.parent.mkdir(exist_ok=True)
                value = plan[name]
                with target.open("xb") as stream:
                    if isinstance(value, bytes):
                        stream.write(value)
                    else:
                        require(regular(value) == source_facts[value]["bytes"], "source_changed")
                        with value.open("rb") as source:
                            shutil.copyfileobj(source, stream)
                expected = hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else source_facts[value]["sha256"]
                require(sha256(target) == expected, "staged_hash_mismatch")
        # Final allowlist check; no wildcard or recursive source copy is used.
        require({p.name for p in destination.iterdir()} == set(ARTIFACTS), "artifact_partition")
    except Exception:
        # Leave partial stage for diagnosis, but remove completion manifests.
        for artifact in ARTIFACTS:
            marker = destination / artifact / "artifact-manifest.json"
            if marker.exists():
                marker.unlink()
        raise
    return dict(summary, staged_bytes=staged_bytes, publication_envelope_bytes=envelope,
                artifacts=list(ARTIFACTS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--exposed-pcm-sha256", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        summary = pack(args.job_root, args.config, args.output_root, exposed_pcm_sha256=args.exposed_pcm_sha256)
    except (GateError, OSError, ValueError, TypeError) as error:
        # CLI logs must not expose absolute paths or raw receipt contents.
        print(json.dumps({"status": "packaging_failed", "reason": str(error) if isinstance(error, GateError) else "input_or_io_error"}))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
