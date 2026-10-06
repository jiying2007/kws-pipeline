#!/usr/bin/env python3
"""Offline, standard-library-only audio triage; never invokes an ASR model."""
from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import math
from pathlib import Path
import re
import struct
import sys
import tempfile
import unicodedata
import wave

VERSION = "1.0.0"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_WAV_BYTES = 32 * 1024 * 1024
MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_CLIPS = 1000
MAX_TEXT = 1024
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


class ReviewError(ValueError):
    """Invalid input; do not silently repair evidence."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ReviewError(message)


def exact_keys(value, required, optional=()):
    require(type(value) is dict, "Expected JSON object")
    require(set(required) <= set(value) <= set(required) | set(optional),
            f"Unexpected or missing fields; required={sorted(required)}, optional={sorted(optional)}")


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_json_float(value):
    result = float(value)
    require(math.isfinite(result), "Nonfinite JSON number, including exponent overflow")
    return result


def validate_json_values(value):
    """Validate all keys and nested values before any output directory is created."""
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        require(depth <= 128, "JSON nesting exceeds 128 levels")
        if type(item) is str:
            try:
                item.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ReviewError("JSON string/key contains an unpaired surrogate; not UTF-8 safe") from exc
        elif type(item) is float:
            require(math.isfinite(item), "Nonfinite JSON number")
        elif type(item) is dict:
            pending.extend((key, depth + 1) for key in item)
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)


def read_bytes(path: Path, limit: int) -> bytes:
    require(path.is_file(), f"Missing input file: {path}")
    require(path.stat().st_size <= limit, f"Input exceeds byte limit: {path}")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, f"Input exceeds byte limit: {path}")
    return data


def read_json(path):
    data = read_bytes(Path(path), MAX_JSON_BYTES)
    try:
        value = json.loads(data, object_pairs_hook=no_duplicate_keys,
                           parse_float=finite_json_float,
                           parse_constant=lambda value: (_ for _ in ()).throw(ReviewError("Nonfinite JSON number")))
    except ReviewError:
        raise
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ReviewError(f"Invalid JSON: {path}: {exc}") from exc
    validate_json_values(value)
    return value, data


def encode_json(value):
    validate_json_values(value)
    try:
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise ReviewError(f"JSON output is not finite/UTF-8 safe: {exc}") from exc


def write_json(path, value):
    Path(path).write_bytes(encode_json(value))


def valid_id(value):
    require(type(value) is str and ID_RE.fullmatch(value), "Invalid anonymous audio ID")
    return value


def valid_hash(value):
    require(type(value) is str and HASH_RE.fullmatch(value), "Invalid SHA256")
    return value


def valid_text(value):
    require(type(value) is str and len(value) <= MAX_TEXT and
            not any(0xD800 <= ord(c) <= 0xDFFF for c in value), "Invalid or oversized transcript")
    return value


def indexed(rows):
    require(type(rows) is list and len(rows) <= MAX_CLIPS, "Invalid clip list")
    result = {}
    for row in rows:
        require(type(row) is dict and "audio_id" in row, "Missing audio_id")
        key = valid_id(row["audio_id"])
        require(key not in result, f"Duplicate audio ID: {key}")
        result[key] = row
    return result


def normalize(text):
    """Comparison only: no homophone, spelling, Unicode-form or repetition repair."""
    return "".join(c for c in valid_text(text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def repetition(text):
    if not text:
        return {"unit": "", "count": 0}
    for width in range(1, len(text) // 2 + 1):
        if len(text) % width == 0 and text == text[:width] * (len(text) // width):
            return {"unit": text[:width], "count": len(text) // width}
    return {"unit": text, "count": 1}


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def dbfs(value):
    # JSON null denotes zero amplitude; never write -Infinity into evidence.
    return 20 * math.log10(value / 32768.0) if value > 0 else None


def amplitude(samples):
    n = len(samples)
    peak = max((abs(x) for x in samples), default=0)
    squares = sum(x * x for x in samples)
    rms = math.sqrt(squares / n) if n else 0.0
    return {"samples": n, "peak_abs_pcm16": peak, "peak_dbfs": dbfs(peak),
            "sum_squares_integer": squares, "rms_pcm16": rms, "rms_dbfs": dbfs(rms)}


def flag_config(near_quiet_rms_dbfs=None, near_quiet_peak_dbfs=None,
                activity_dbfs=None):
    require((near_quiet_rms_dbfs is None) == (near_quiet_peak_dbfs is None),
            "Near-quiet flag requires both RMS and peak thresholds")
    for value in (near_quiet_rms_dbfs, near_quiet_peak_dbfs, activity_dbfs):
        require(value is None or (type(value) in (float, int) and math.isfinite(value)
                                  and -200 <= value <= 0), "Invalid dBFS threshold")
    return {"near_quiet_rms_dbfs": near_quiet_rms_dbfs,
            "near_quiet_peak_dbfs": near_quiet_peak_dbfs,
            "activity_dbfs": activity_dbfs,
            "calibration_status": "uncalibrated_user_configuration"}


def at_or_below(value, threshold):
    return value is None or value <= threshold


def wav_metrics(data, config):
    require(len(data) <= MAX_WAV_BYTES, "WAV exceeds byte limit")
    require(len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE", "Expected RIFF WAVE")
    # Refuse files whose stated RIFF extent does not cover exactly the original bytes.
    require(int.from_bytes(data[4:8], "little") + 8 == len(data), "RIFF length mismatch")
    # Byte-preserving blinding cannot copy LIST/INFO, iXML, bext, ID3, cue labels,
    # arbitrary JUNK, or unknown chunks that may disclose text/speaker identity.
    # Accept only the minimal PCM layout. Do not silently strip metadata.
    chunks = []
    position = 12
    while position < len(data):
        require(position + 8 <= len(data), "Truncated RIFF chunk header")
        name = data[position:position + 4]
        length = int.from_bytes(data[position + 4:position + 8], "little")
        end = position + 8 + length
        require(end + (length & 1) <= len(data), "Truncated RIFF chunk payload")
        require(name in (b"fmt ", b"data"),
                f"Non-audio/unknown WAV chunk {name!r} rejected for blind input; original bytes unchanged")
        chunks.append((name, position + 8, length))
        position = end + (length & 1)
    require([chunk[0] for chunk in chunks] == [b"fmt ", b"data"],
            "Blind input requires exactly one fmt chunk followed by one data chunk")
    require(chunks[0][2] == 16, "Blind input requires a minimal 16-byte PCM fmt chunk; no extension metadata")
    pcm_format, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<HHIIHH", data, chunks[0][1])
    require((pcm_format, channels, block_align, bits) == (1, 1, 2, 16) and byte_rate == rate * 2,
            "Blind input requires consistent mono PCM16 format fields")
    require(chunks[1][2] % 2 == 0, "Odd-length PCM16 data")
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            header = {"channels": stream.getnchannels(), "sample_width_bytes": stream.getsampwidth(),
                      "sample_rate": stream.getframerate(), "frames": stream.getnframes(),
                      "compression": stream.getcomptype()}
            require(header["channels"] == 1 and header["sample_width_bytes"] == 2 and
                    header["compression"] == "NONE", "Only mono uncompressed PCM16 WAV is supported")
            require(8000 <= header["sample_rate"] <= 192000, "Supported sample rate range is 8–192 kHz")
            require(header["frames"] <= header["sample_rate"] * 300, "Clip exceeds 300 seconds")
            pcm = stream.readframes(header["frames"] + 1)
    except (wave.Error, EOFError) as exc:
        raise ReviewError(f"Invalid WAV header: {exc}") from exc
    require(len(pcm) == header["frames"] * 2, "Truncated or odd-length PCM data")
    values = array.array("h")
    require(values.itemsize == 2, "Host signed-short size is unsupported")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    whole = amplitude(values)
    width = max(1, round(header["sample_rate"] * .020))
    frames = []
    for start in range(0, len(values), width):
        frames.append({"start_sample": start, **amplitude(values[start:start + width])})
    tails = {str(ms): amplitude(values[-max(1, round(header["sample_rate"] * ms / 1000)):])
             for ms in (20, 50)}
    clipped = sum(x in (-32768, 32767) for x in values)
    flags = []
    if not values:
        flags.append("empty_audio")
    elif not whole["peak_abs_pcm16"]:
        flags.append("digital_all_zero")
    if values and config["near_quiet_rms_dbfs"] is not None and at_or_below(
            whole["rms_dbfs"], config["near_quiet_rms_dbfs"]) and at_or_below(
            whole["peak_dbfs"], config["near_quiet_peak_dbfs"]):
        flags.append("near_quiet_candidate_uncalibrated")
    activity = None
    if config["activity_dbfs"] is not None:
        threshold = config["activity_dbfs"]
        active = [f["rms_dbfs"] is not None and f["rms_dbfs"] >= threshold for f in frames]
        tail_active = tails["20"]["rms_dbfs"] is not None and tails["20"]["rms_dbfs"] >= threshold
        activity = {"threshold_dbfs": threshold, "active_frame_count": sum(active),
                    "frame_count": len(active), "tail_20ms_active": tail_active}
        if tail_active:
            flags.append("tail_activity_ambiguous_uncalibrated")
    if clipped:
        flags.append("pcm_rail_samples_present")
    return {"header": header, "duration_seconds": header["frames"] / header["sample_rate"],
            "pcm_sha256": digest(pcm), "whole": whole, "nonzero_samples": sum(x != 0 for x in values),
            "digital_all_zero": bool(values) and whole["peak_abs_pcm16"] == 0,
            "rail_sample_count": clipped, "rail_sample_fraction": clipped / len(values) if values else None,
            "frame_window_nominal_ms": 20, "frame_samples": width, "frames": frames, "tails_ms": tails,
            "last_sample_pcm16": values[-1] if values else None,
            "last_sample_step_pcm16": values[-1] - values[-2] if len(values) > 1 else None,
            "activity": activity, "flags": flags,
            "interpretation": "Signal measurements and uncalibrated flags only; no speech, lexical or cutoff truth",
            "original_audio_bytes_unchanged": True}


def prepare(audio_manifest, out_dir, config=None):
    config = config or flag_config()
    # Revalidate caller-provided configuration rather than trusting arbitrary dictionaries.
    exact_keys(config, ("near_quiet_rms_dbfs", "near_quiet_peak_dbfs", "activity_dbfs", "calibration_status"))
    require(config == flag_config(config["near_quiet_rms_dbfs"], config["near_quiet_peak_dbfs"], config["activity_dbfs"]),
            "Configuration claims unsupported calibration")
    manifest_path = Path(audio_manifest).resolve()
    manifest, manifest_bytes = read_json(manifest_path)
    exact_keys(manifest, ("schema", "clips"))
    require(manifest["schema"] == "audio-inputs-v1", "Wrong input schema")
    clips = indexed(manifest["clips"])
    require(clips, "No audio inputs")
    prepared, batch_bytes = [], 0
    for audio_id, row in clips.items():
        exact_keys(row, ("audio_id", "audio_path", "wav_sha256"))
        require(type(row["audio_path"]) is str and row["audio_path"], "Invalid audio path")
        valid_hash(row["wav_sha256"])
        source = (manifest_path.parent / row["audio_path"]).resolve()
        data = read_bytes(source, MAX_WAV_BYTES)
        batch_bytes += len(data)
        require(batch_bytes <= MAX_BATCH_BYTES, "Audio batch exceeds 64 MiB")
        require(digest(data) == row["wav_sha256"], f"Audio hash mismatch: {audio_id}")
        prepared.append((audio_id, row, data, wav_metrics(data, config)))
    out = Path(out_dir)
    require(not out.exists(), "Output directory already exists; refusing to overwrite")
    out.mkdir(parents=True)
    blind = out / "blind"
    (blind / "audio").mkdir(parents=True)
    (out / "private").mkdir()
    (out / "private/audio-inputs.json").write_bytes(manifest_bytes)
    jobs = []
    metrics = []
    for number, (audio_id, row, data, measurement) in enumerate(prepared, 1):
        blind_id = f"clip-{number:06d}"
        # Hash-only filename prevents the original filename leaking target words.
        relative = "audio/" + row["wav_sha256"] + ".wav"
        destination = blind / relative
        if not destination.exists():
            destination.write_bytes(data)
        require(digest(destination.read_bytes()) == row["wav_sha256"], "Copied WAV hash mismatch")
        jobs.append({"audio_id": blind_id, "audio_path": relative, "wav_sha256": row["wav_sha256"]})
        metrics.append({"audio_id": audio_id, "blind_audio_id": blind_id,
                        "wav_sha256": row["wav_sha256"], "metrics": measurement})
    job = {"schema": "blind-asr-job-v1", "clips": jobs}
    write_json(blind / "job.json", job)
    receipt = {"schema": "audio-review-prepared-v1", "tool_version": VERSION,
               "input_manifest_sha256": digest(manifest_bytes),
               "blind_job_sha256": digest((blind / "job.json").read_bytes()),
               "config": config, "clips": metrics, "asr_executed": False,
               "network_used": False, "models_loaded": False,
               "note": "Only blind/ belongs in a decoder input bundle. Plan and human labels remain separate."}
    write_json(out / "measurements.json", receipt)
    return receipt


def load_asr(path, jobs, job_hash):
    source, raw = read_json(path)
    exact_keys(source, ("schema", "model", "blind_job_sha256", "clips"))
    require(source["schema"] == "saved-asr-transcripts-v1", "Wrong ASR import schema")
    require(source["blind_job_sha256"] == job_hash, "ASR job hash mismatch")
    model = source["model"]
    exact_keys(model, ("model_id", "family", "revision"))
    for value in model.values():
        require(type(value) is str and 0 < len(value.strip()) <= 256, "Invalid model metadata")
    rows = indexed(source["clips"])
    for audio_id, row in rows.items():
        exact_keys(row, ("audio_id", "wav_sha256", "raw_text", "status"), ("quality_flags", "raw_output"))
        require(audio_id in jobs, "ASR contains an unrequested audio ID")
        require(row["wav_sha256"] == jobs[audio_id]["wav_sha256"], "ASR audio hash mismatch")
        valid_text(row["raw_text"])
        require(row["status"] in ("complete", "incomplete", "error"), "Invalid ASR status")
        flags = row.get("quality_flags", [])
        require(type(flags) is list and len(flags) <= 64 and
                all(type(f) is str and len(f) <= 256 for f in flags), "Invalid ASR quality flags")
    return source, raw, rows


def compare(prepared_dir, plan_path, asr_a_path, asr_b_path, out_dir, human_path=None):
    prepared_dir = Path(prepared_dir)
    prep, prep_bytes = read_json(prepared_dir / "measurements.json")
    exact_keys(prep, ("schema", "tool_version", "input_manifest_sha256", "blind_job_sha256", "config",
                      "clips", "asr_executed", "network_used", "models_loaded", "note"))
    require(prep["schema"] == "audio-review-prepared-v1", "Wrong preparation schema")
    job, job_bytes = read_json(prepared_dir / "blind/job.json")
    exact_keys(job, ("schema", "clips"))
    require(job["schema"] == "blind-asr-job-v1", "Wrong blind job schema")
    require(digest(job_bytes) == prep["blind_job_sha256"], "Blind manifest changed after preparation")
    jobs, measurements = indexed(job["clips"]), indexed(prep["clips"])
    for row in measurements.values():
        exact_keys(row, ("audio_id", "blind_audio_id", "wav_sha256", "metrics"))
        valid_id(row["blind_audio_id"])
        require(type(row["metrics"]) is dict and type(row["metrics"].get("flags")) is list and
                all(type(flag) is str for flag in row["metrics"]["flags"]), "Invalid measurement flags")
    blind_to_input = {r["blind_audio_id"]: key for key, r in measurements.items()}
    require(jobs and len(blind_to_input) == len(measurements) and set(jobs) == set(blind_to_input),
            "Preparation IDs differ")
    require(list(jobs) == [f"clip-{i:06d}" for i in range(1, len(jobs) + 1)], "Nonopaque blind job IDs")
    for key, row in jobs.items():
        exact_keys(row, ("audio_id", "audio_path", "wav_sha256"))
        valid_hash(row["wav_sha256"])
        require(row["audio_path"] == "audio/" + row["wav_sha256"] + ".wav", "Nonopaque blind audio path")
        require(measurements[blind_to_input[key]]["wav_sha256"] == row["wav_sha256"], "Measurement identity mismatch")
    plan, plan_bytes = read_json(plan_path)
    exact_keys(plan, ("schema", "clips"))
    require(plan["schema"] == "intended-text-v1", "Wrong plan schema")
    planned = indexed(plan["clips"])
    require(set(planned) == set(measurements), "Plan must cover exactly the prepared clips")
    for key, row in planned.items():
        exact_keys(row, ("audio_id", "wav_sha256", "intended_text"))
        require(row["wav_sha256"] == measurements[key]["wav_sha256"], "Plan audio hash mismatch")
        require(normalize(row["intended_text"]), "Intended text is empty after normalization")
    a, a_bytes, a_rows = load_asr(asr_a_path, jobs, digest(job_bytes))
    b, b_bytes, b_rows = load_asr(asr_b_path, jobs, digest(job_bytes))
    require(a["model"]["model_id"].strip().casefold() != b["model"]["model_id"].strip().casefold(),
            "Two copies of one model are not two model families")
    require(a["model"]["family"].strip().casefold() != b["model"]["family"].strip().casefold(),
            "Two variants of one model family do not meet this two-family scheme")
    human_rows, human_bytes = {}, None
    if human_path is not None:
        human, human_bytes = read_json(human_path)
        exact_keys(human, ("schema", "clips"))
        require(human["schema"] == "existing-human-labels-v1", "Wrong human schema")
        human_rows = indexed(human["clips"])
        for key, row in human_rows.items():
            exact_keys(row, ("audio_id", "wav_sha256", "record"))
            require(key in measurements and row["wav_sha256"] == measurements[key]["wav_sha256"], "Human audio identity mismatch")
            require(type(row["record"]) is dict, "Human record must be a JSON object")
    rows = []
    for blind_id, job_row in jobs.items():
        key = blind_to_input[blind_id]
        intent = normalize(planned[key]["intended_text"])
        evidence = []
        for name, model_rows in (("a", a_rows), ("b", b_rows)):
            row = model_rows.get(blind_id)
            text = normalize(row["raw_text"]) if row else None
            evidence.append({"source": name, "raw_record": row, "normalized_text": text,
                             "character_count": len(text) if text is not None else None,
                             "repetition": repetition(text) if text is not None else None,
                             "edit_distance_to_intended": edit_distance(text, intent) if text is not None else None})
        valid = all(e["raw_record"] is not None and e["raw_record"]["status"] == "complete"
                    and e["normalized_text"] for e in evidence)
        texts = [e["normalized_text"] for e in evidence]
        flags = list(measurements[key]["metrics"]["flags"])
        for e in evidence:
            flags += ["asr_" + e["source"] + ":" + f for f in (e["raw_record"] or {}).get("quality_flags", [])]
        if not valid:
            state = "machine_unresolved"
        elif texts[0] != texts[1]:
            state = "machine_dispute"
        elif texts[0] == intent:
            state = "machine_agreement_match"
        else:
            state = "machine_agreement_mismatch"
        candidate = state == "machine_agreement_match" and not flags
        human_row = human_rows.get(key)
        rows.append({"audio_id": key, "blind_audio_id": blind_id, "wav_sha256": job_row["wav_sha256"],
                     "intended": {"raw_text": planned[key]["intended_text"], "normalized_text": intent,
                                  "character_count": len(intent), "repetition": repetition(intent)},
                     "signal_measurements": measurements[key]["metrics"],
                     "machine": {"state": state, "evidence": evidence, "flags": flags,
                                 "candidate_only": candidate, "human_truth_established": False,
                                 "confidence_probability": None, "training_admitted": False},
                     "existing_human": human_row,
                     "label_authority": "existing_human_record_unchanged" if human_row else "no_human_label",
                     "automatic_relabel": False,
                     "suggested_handling": "keep_existing_human_record" if human_row else
                         ("machine_candidate_only" if candidate else "quarantine_or_review_only_if_needed_for_use")})
    source_hashes = {"measurements.json": digest(prep_bytes), "blind-job.json": digest(job_bytes),
                     "plan.json": digest(plan_bytes), "asr-a.json": digest(a_bytes), "asr-b.json": digest(b_bytes)}
    if human_bytes is not None:
        source_hashes["human.json"] = digest(human_bytes)
    report = {"schema": "audio-review-comparison-v1", "tool_version": VERSION,
              "sources_sha256": source_hashes, "models": {"a": a["model"], "b": b["model"]},
              "normalization": "Remove Unicode P* punctuation and whitespace only",
              "unicode_database_version": unicodedata.unidata_version,
              "model_family_metadata": "Caller-declared; not proof of independent training or errors",
              "blinding": "Job schema checked; imported historical decoder behavior is not independently proven",
              "decoder_complete_means": "Execution returned complete, not proof of acoustic completeness",
              "asr_executed": False, "audio_redecoded_or_gain_normalized": False,
              "statistical_independence_assumed": False, "automatic_human_label_override": False,
              "clips": rows}
    out = Path(out_dir)
    require(not out.exists(), "Output directory already exists; refusing to overwrite")
    # Serialize before creating any comparison output. All input strings/numbers
    # have already passed validation, including nested raw outputs and human notes.
    report_bytes = encode_json(report)
    imports = {"measurements.json": prep_bytes, "blind-job.json": job_bytes,
               "plan.json": plan_bytes, "asr-a.json": a_bytes, "asr-b.json": b_bytes}
    if human_bytes is not None:
        imports["human.json"] = human_bytes
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".audio-review-staging-", dir=out.parent) as temporary:
        staging = Path(temporary) / "complete-result"
        (staging / "source-imports").mkdir(parents=True)
        for name, raw in imports.items():
            target = staging / "source-imports" / name
            target.write_bytes(raw)
            require(digest(target.read_bytes()) == source_hashes[name], "Evidence copy hash mismatch")
        (staging / "report.json").write_bytes(report_bytes)
        require(not out.exists(), "Output directory appeared during write; refusing to overwrite")
        staging.rename(out)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Measure PCM16 WAV and create target-free job files; does not run ASR")
    p.add_argument("--audio-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--near-quiet-rms-dbfs", type=float)
    p.add_argument("--near-quiet-peak-dbfs", type=float)
    p.add_argument("--activity-dbfs", type=float)
    p = commands.add_parser("compare", help="Import two already-saved transcript files; does not run ASR")
    p.add_argument("--prepared-dir", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--asr-a", required=True)
    p.add_argument("--asr-b", required=True)
    p.add_argument("--human")
    p.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.audio_manifest, args.out_dir,
                             flag_config(args.near_quiet_rms_dbfs, args.near_quiet_peak_dbfs, args.activity_dbfs))
        else:
            result = compare(args.prepared_dir, args.plan, args.asr_a, args.asr_b, args.out_dir, args.human)
    except (ReviewError, OSError) as exc:
        print(f"Input/output error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": result["schema"], "clips": len(result["clips"]),
                      "asr_executed": False, "output": str(Path(args.out_dir) / (
                          "measurements.json" if args.command == "prepare" else "report.json"))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
