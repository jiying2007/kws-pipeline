#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import struct
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from synthetic_audio import clamp16  # noqa: E402
from corpus_identity import canonical_audio_path, rebind_audio_path  # noqa: E402

SCHEMA_VERSION = 1
EVIDENCE_CLASS = "frozen-far-source-replay-request-v1"
RESULT_CLASS = "frozen-far-source-replay-evidence-v1"
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,111}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def normalize_spec(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("frozen FAR replay spec must be an object")
    required = {
        "schema_version",
        "evidence_class",
        "experiment_id",
        "development_only",
        "selection_feedback_allowed",
        "protected_evidence_used",
        "historical_run_id",
        "historical_artifact_id",
        "historical_head_sha",
        "expected_model_sha256",
        "expected_keyword_pack_sha256",
        "cases",
    }
    if set(value) != required:
        raise ValueError(
            "frozen FAR replay spec fields mismatch: "
            f"missing={sorted(required-set(value))} extra={sorted(set(value)-required)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("frozen FAR replay schema_version must be 1")
    if value["evidence_class"] != EVIDENCE_CLASS:
        raise ValueError("frozen FAR replay evidence_class mismatch")
    experiment_id = value["experiment_id"]
    if not isinstance(experiment_id, str) or ID_RE.fullmatch(experiment_id) is None:
        raise ValueError("frozen FAR replay experiment_id is invalid")
    if value["development_only"] is not True:
        raise ValueError("frozen FAR replay must be development_only")
    if value["selection_feedback_allowed"] is not False:
        raise ValueError("frozen FAR replay must not feed selection")
    if value["protected_evidence_used"] is not False:
        raise ValueError("frozen FAR replay must not use protected evidence")
    for key in ("historical_run_id", "historical_artifact_id"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    historical_head = value["historical_head_sha"]
    if (
        not isinstance(historical_head, str)
        or re.fullmatch(r"[0-9a-f]{40}", historical_head) is None
    ):
        raise ValueError("historical_head_sha must be a lowercase git SHA")
    for key in ("expected_model_sha256", "expected_keyword_pack_sha256"):
        if not isinstance(value[key], str) or SHA256_RE.fullmatch(value[key]) is None:
            raise ValueError(f"{key} must be lowercase SHA256")
    cases = value["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("frozen FAR replay cases must be non-empty")
    normalized_cases: list[dict] = []
    seen: set[str] = set()
    for index, row in enumerate(cases):
        required_case = {
            "case_id",
            "seed",
            "detection_time_s",
            "historical_confidence",
            "detected_keyword_id",
            "rendered_wav_sha256",
            "historical_gain",
            "expected_tokens",
        }
        if not isinstance(row, dict) or set(row) != required_case:
            raise ValueError(f"case[{index}] fields mismatch")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or ID_RE.fullmatch(case_id) is None:
            raise ValueError(f"case[{index}].case_id is invalid")
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        seed = row["seed"]
        keyword_id = row["detected_keyword_id"]
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in (seed, keyword_id)
        ):
            raise ValueError(f"case[{index}] integer identity is invalid")
        detection_time = float(row["detection_time_s"])
        confidence = float(row["historical_confidence"])
        gain = float(row["historical_gain"])
        if not math.isfinite(detection_time) or detection_time < 0.0:
            raise ValueError(f"case[{index}].detection_time_s is invalid")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"case[{index}].historical_confidence is invalid")
        if not math.isfinite(gain) or not 0.0 < gain <= 1.0:
            raise ValueError(f"case[{index}].historical_gain is invalid")
        rendered_sha = row["rendered_wav_sha256"]
        if not isinstance(rendered_sha, str) or SHA256_RE.fullmatch(rendered_sha) is None:
            raise ValueError(f"case[{index}].rendered_wav_sha256 is invalid")
        tokens = row["expected_tokens"]
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(not isinstance(token, str) or not token for token in tokens)
        ):
            raise ValueError(f"case[{index}].expected_tokens is invalid")
        normalized_cases.append(
            {
                "case_id": case_id,
                "seed": seed,
                "detection_time_s": detection_time,
                "historical_confidence": confidence,
                "detected_keyword_id": keyword_id,
                "rendered_wav_sha256": rendered_sha,
                "historical_gain": gain,
                "expected_tokens": list(tokens),
            }
        )
    result = dict(value)
    result["cases"] = normalized_cases
    return result


def scale_wav(source: pathlib.Path, gain: float, output: pathlib.Path) -> dict:
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"source WAV must be mono 16-kHz PCM16: {source}")
        frames = reader.getnframes()
        raw = reader.readframes(frames)
    if len(raw) != frames * 2:
        raise ValueError("source WAV is truncated")
    import struct

    values = struct.unpack("<" + "h" * frames, raw)
    scaled = [clamp16(value * gain) for value in values]
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<" + "h" * len(scaled), *scaled))
    return {
        "frames": frames,
        "seconds": frames / 16000.0,
        "source_sha256": sha256_file(source),
        "scaled_sha256": sha256_file(output),
        "gain": gain,
    }


def run_json_lines(command: list[str]) -> list[dict]:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            + completed.stderr[-4000:]
        )
    rows: list[dict] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("diagnostic command emitted non-object JSON")
        rows.append(value)
    return rows


def detection_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if isinstance(row.get("keyword_id"), int)
        and isinstance(row.get("time_s"), (int, float))
        and isinstance(row.get("confidence"), (int, float))
    ]


def canonical_detections(rows: list[dict]) -> list[tuple[int, float, float]]:
    return [
        (
            int(row["keyword_id"]),
            round(float(row["time_s"]), 6),
            round(float(row["confidence"]), 6),
        )
        for row in detection_rows(rows)
    ]


def find_domain_row(rows: list[dict], case: dict) -> dict:
    matches = [
        row for row in rows
        if row.get("wav_sha256") == case["rendered_wav_sha256"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{case['case_id']}: expected exactly one rendered domain row, got {len(matches)}"
        )
    row = matches[0]
    if row.get("tokens") != case["expected_tokens"]:
        raise ValueError(f"{case['case_id']}: domain tokens drifted")
    path = pathlib.Path(canonical_audio_path(row, str(case["case_id"]))).resolve()
    if not path.is_file():
        raise ValueError(f"{case['case_id']}: rendered WAV is missing: {path}")
    if sha256_file(path) != case["rendered_wav_sha256"]:
        raise ValueError(f"{case['case_id']}: rendered WAV SHA drifted")
    return rebind_audio_path(row, str(path))


def target_path_summary(rows: list[dict], keyword_id: int) -> dict:
    if len(rows) != 1:
        raise ValueError("decoder path replay must emit exactly one JSON object")
    value = rows[0]
    keywords = value.get("keywords")
    if not isinstance(keywords, list):
        raise ValueError("decoder path replay lacks keyword rows")
    matches = [
        row for row in keywords
        if isinstance(row, dict) and int(row.get("keyword_id", -1)) == keyword_id
    ]
    if len(matches) != 1:
        raise ValueError("decoder path replay target keyword is missing or duplicated")
    return matches[0]


def analyze_wav(
    *,
    wav: pathlib.Path,
    recording: str,
    keyword_id: int,
    model: pathlib.Path,
    pack: pathlib.Path,
    runner: pathlib.Path,
    posterior_dump: pathlib.Path,
    decoder_replay: pathlib.Path,
    decoder_path_replay: pathlib.Path,
    output_dir: pathlib.Path,
    trace_stem: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_rows = run_json_lines(
        [str(runner), str(model), str(pack), str(wav), recording]
    )
    trace = output_dir / f"{trace_stem}.kwtr"
    posterior_rows = run_json_lines(
        [str(posterior_dump), str(model), str(wav), str(trace)]
    )
    replay_rows = run_json_lines(
        [
            str(decoder_replay),
            str(model),
            str(pack),
            str(trace),
            recording,
        ]
    )
    path_rows = run_json_lines(
        [
            str(decoder_path_replay),
            str(model),
            str(pack),
            str(trace),
            recording,
        ]
    )
    direct = canonical_detections(direct_rows)
    replay = canonical_detections(replay_rows)
    if direct != replay:
        raise ValueError(
            f"{recording}: direct runtime and fixed-posterior replay differ: "
            f"{direct} != {replay}"
        )
    target_hits = [row for row in direct if row[0] == keyword_id]
    return {
        "audio_path": str(wav),
        "audio_sha256": sha256_file(wav),
        "direct_detections": direct_rows,
        "posterior_dump_receipts": posterior_rows,
        "replay_detections": replay_rows,
        "direct_replay_exact_parity": True,
        "target_keyword_hits": len(target_hits),
        "target_keyword_path": target_path_summary(path_rows, keyword_id),
    }


def historical_detection(rows: list[dict], case: dict) -> dict:
    matches = [
        row
        for row in rows
        if int(row.get("keyword_id", -1)) == int(case["detected_keyword_id"])
        and math.isclose(
            float(row.get("time_s", -1.0)),
            float(case["detection_time_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
        and math.isclose(
            float(row.get("confidence", -1.0)),
            float(case["historical_confidence"]),
            rel_tol=0.0,
            abs_tol=1.0e-6,
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{case['case_id']}: historical detection missing/duplicated"
        )
    return matches[0]


def historical_active_injection(rows: list[dict], case: dict) -> dict:
    time_s = float(case["detection_time_s"])
    active = [
        row
        for row in rows
        if float(row["start_second"]) <= time_s
        < float(row["start_second"]) + float(row["source_seconds"])
    ]
    if len(active) != 1:
        raise ValueError(
            f"{case['case_id']}: historical active injection is not unique"
        )
    row = active[0]
    if row.get("source_sha256") != case["rendered_wav_sha256"]:
        raise ValueError(
            f"{case['case_id']}: historical injected source SHA drifted"
        )
    if not math.isclose(
        float(row.get("gain", -1.0)),
        float(case["historical_gain"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{case['case_id']}: historical injected gain drifted"
        )
    return row


def extract_pcm_capture(
    *,
    spool: pathlib.Path,
    output: pathlib.Path,
    center_time_s: float,
    before_seconds: float,
    after_seconds: float,
) -> dict:
    if before_seconds < 0.0 or after_seconds < 0.0:
        raise ValueError("capture context must be non-negative")
    raw_bytes = spool.stat().st_size
    if raw_bytes <= 0 or raw_bytes % 2:
        raise ValueError("stream PCM spool must contain non-empty PCM16LE")
    total_samples = raw_bytes // 2
    center = int(round(center_time_s * 16000.0))
    before = int(round(before_seconds * 16000.0))
    after = int(round(after_seconds * 16000.0))
    start = max(0, center - before)
    end = min(total_samples, center + after)
    if end <= start:
        raise ValueError("stream-context capture interval is empty")
    with spool.open("rb") as stream:
        stream.seek(start * 2)
        pcm = stream.read((end - start) * 2)
    if len(pcm) != (end - start) * 2:
        raise ValueError("stream-context PCM spool is truncated")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(pcm)
    return {
        "policy": "historical-mixed-stream-window-v1",
        "spool_bytes": raw_bytes,
        "total_samples": total_samples,
        "center_time_s": center_time_s,
        "before_seconds": before_seconds,
        "after_seconds": after_seconds,
        "start_sample": start,
        "end_sample": end,
        "frames": end - start,
        "wav_sha256": sha256_file(output),
    }


def run_stream_context(
    *,
    spec: dict,
    model: pathlib.Path,
    pack: pathlib.Path,
    stream_root: pathlib.Path,
    spool_root: pathlib.Path,
    runner: pathlib.Path,
    posterior_dump: pathlib.Path,
    decoder_replay: pathlib.Path,
    decoder_path_replay: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    if sha256_file(model) != spec["expected_model_sha256"]:
        raise ValueError("frozen stream-context model SHA mismatch")
    if sha256_file(pack) != spec["expected_keyword_pack_sha256"]:
        raise ValueError("frozen stream-context keyword-pack SHA mismatch")
    results: list[dict] = []
    for case in spec["cases"]:
        far_root = stream_root / case["case_id"] / "far"
        summary_path = far_root / "summary.json"
        if not summary_path.is_file():
            raise ValueError(
                f"{case['case_id']}: regenerated FAR summary is missing"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("seed", -1)) != int(case["seed"]):
            raise ValueError(f"{case['case_id']}: regenerated seed drifted")
        if int(summary.get("seconds", -1)) != 7200:
            raise ValueError(
                f"{case['case_id']}: regenerated stream duration drifted"
            )
        if summary.get("model_sha256") != spec["expected_model_sha256"]:
            raise ValueError(
                f"{case['case_id']}: regenerated model SHA drifted"
            )
        if summary.get("keyword_pack_sha256") != spec["expected_keyword_pack_sha256"]:
            raise ValueError(
                f"{case['case_id']}: regenerated keyword-pack SHA drifted"
            )
        detections = load_jsonl(far_root / "detections.jsonl")
        detection = historical_detection(detections, case)
        injections = load_jsonl(far_root / "hard-negative-injections.jsonl")
        injection = historical_active_injection(injections, case)
        spool = spool_root / f"{case['case_id']}.pcm16le"
        expected_spool_bytes = 7200 * 16000 * 2
        if not spool.is_file() or spool.stat().st_size != expected_spool_bytes:
            raise ValueError(
                f"{case['case_id']}: exact stream spool size mismatch"
            )
        case_root = output_dir / case["case_id"]
        capture_wav = case_root / "mixed-stream-context.wav"
        capture = extract_pcm_capture(
            spool=spool,
            output=capture_wav,
            center_time_s=float(case["detection_time_s"]),
            before_seconds=10.0,
            after_seconds=2.0,
        )
        analysis = analyze_wav(
            wav=capture_wav,
            recording=f"{case['case_id']}-mixed-context",
            keyword_id=int(case["detected_keyword_id"]),
            model=model,
            pack=pack,
            runner=runner,
            posterior_dump=posterior_dump,
            decoder_replay=decoder_replay,
            decoder_path_replay=decoder_path_replay,
            output_dir=case_root,
            trace_stem="mixed-stream-context",
        )
        classification = (
            "mixed-stream-context-sufficient"
            if int(analysis["target_keyword_hits"]) > 0
            else "longer-stream-state-or-history-required"
        )
        result = {
            "case": case,
            "historical_stream_detection": detection,
            "historical_active_injection": injection,
            "capture": capture,
            "analysis": analysis,
            "classification": classification,
        }
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "stream-context-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)
    output = {
        "schema_version": 1,
        "evidence_class": "frozen-far-stream-context-replay-evidence-v1",
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "training_changed": False,
        "decoder_math_changed": False,
        "thresholds_changed": False,
        "historical_stream_generator_used": True,
        "capture_context_before_seconds": 10.0,
        "capture_context_after_seconds": 2.0,
        "model_sha256": sha256_file(model),
        "keyword_pack_sha256": sha256_file(pack),
        "cases": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stream-context-summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def pcm16le_to_wav(source: pathlib.Path, output: pathlib.Path) -> dict:
    size = source.stat().st_size
    if size <= 0 or size % 2:
        raise ValueError("PCM16LE spool must be non-empty and sample-aligned")
    frames = size // 2
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as stream, wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            writer.writeframesraw(chunk)
    return {
        "frames": frames,
        "seconds": frames / 16000.0,
        "pcm_sha256": sha256_file(source),
        "wav_sha256": sha256_file(output),
    }


def target_detection_matches(rows: list[dict], case: dict, tolerance_s: float = 0.10) -> list[dict]:
    target_time = float(case["detection_time_s"])
    keyword_id = int(case["detected_keyword_id"])
    return [
        row
        for row in detection_rows(rows)
        if int(row["keyword_id"]) == keyword_id
        and abs(float(row["time_s"]) - target_time) <= tolerance_s
    ]


def read_phase_trace(path: pathlib.Path) -> tuple[dict, bytes, list[tuple]]:
    """Read the fixed KWTRACE1 contract without changing logits or VAD bits."""
    data = path.read_bytes()
    if len(data) < 120 or data[:8] != b"KWTRACE1":
        raise ValueError("phase control requires a KWTRACE1 header")
    version, vocab, frontend, rate, frame, hop, reserved = struct.unpack_from("<IHHIIII", data, 8)
    count = struct.unpack_from("<Q", data, 40)[0]
    model_sha = data[48:112].decode("ascii")
    stride = 16 + 4 * vocab
    if (version != 1 or not 2 <= vocab <= 256 or rate != 16000
            or frame <= 0 or hop <= 0 or frame < hop or reserved != 0
            or struct.unpack_from("<Q", data, 32)[0] == 0
            or data[112:120] != bytes(8) or count <= 0
            or len(data) != 120 + count * stride or not SHA256_RE.fullmatch(model_sha)):
        raise ValueError("invalid phase-control trace contract")
    rows = []
    for index in range(count):
        offset = 120 + index * stride
        end = struct.unpack_from("<Q", data, offset)[0]
        flags = data[offset + 8:offset + 16]
        logits = struct.unpack_from("<" + "f" * vocab, data, offset + 16)
        if (end <= 0 or (rows and end - rows[-1][0] != hop)
                or flags[0] not in (0, 1) or flags[1:] != bytes(7)
                or any(not math.isfinite(value) for value in logits)):
            raise ValueError("invalid phase-control trace frames")
        rows.append((end, flags[0], logits))
    return {"sample_rate_hz": rate, "frame_length_samples": frame,
            "frame_hop_samples": hop, "vocab_size": vocab,
            "frontend_kind": frontend, "model_sha256": model_sha,
            "vocab_fingerprint": struct.unpack_from("<Q", data, 32)[0],
            "frames": count}, data, rows


def phase_aligned_interval(trace: pathlib.Path, end_sample: int) -> dict:
    header, _, rows = read_phase_trace(trace)
    # Cold frontend emits its first frame after frame_length input samples.
    # Use the actual first frozen endpoint; aligning crop start alone is insufficient.
    start = rows[0][0] - header["frame_length_samples"]
    hop = header["frame_hop_samples"]
    if (start < 0 or end_sample < rows[-1][0] or end_sample >= rows[-1][0] + hop):
        raise ValueError("phase-control exposure does not match frozen trace")
    return {**header, "start_sample": start, "end_sample": end_sample,
            "first_absolute_frame_end": rows[0][0],
            "last_absolute_frame_end": rows[-1][0]}


def extract_phase_capture(spool: pathlib.Path, output: pathlib.Path, interval: dict) -> dict:
    start, end = interval["start_sample"], interval["end_sample"]
    size = spool.stat().st_size
    if (size <= 0 or size % 2 or start < 0 or not start < end <= size // 2):
        raise ValueError("phase-control PCM exposure outside exact spool")
    with spool.open("rb") as source:
        source.seek(2 * start)
        pcm = source.read(2 * (end - start))
    if len(pcm) != 2 * (end - start):
        raise ValueError("phase-control PCM spool truncated")
    with wave.open(str(output), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(interval["sample_rate_hz"])
        writer.writeframes(pcm)
    return {**interval, "pcm_samples": end - start, "spool_bytes": size,
            "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "wav_sha256": sha256_file(output)}


def align_cold_trace(cold: pathlib.Path, frozen: pathlib.Path,
                     output: pathlib.Path, offset_samples: int) -> dict:
    cold_header, cold_bytes, cold_rows = read_phase_trace(cold)
    frozen_header, _, frozen_rows = read_phase_trace(frozen)
    if cold_header != frozen_header or offset_samples < 0:
        raise ValueError("phase-control acoustic trace identity or frame-count mismatch")
    if [row[0] + offset_samples for row in cold_rows] != [row[0] for row in frozen_rows]:
        raise ValueError("phase-control absolute frame grid mismatch")
    data = bytearray(cold_bytes)
    stride = 16 + 4 * cold_header["vocab_size"]
    for index, row in enumerate(cold_rows):
        struct.pack_into("<Q", data, 120 + index * stride, row[0] + offset_samples)
    output.write_bytes(data)
    return {"absolute_frame_grid_equal": True, "frames": len(cold_rows),
            "timestamp_offset_samples": offset_samples,
            "cold_relative_trace_sha256": sha256_file(cold),
            "cold_absolute_trace_sha256": sha256_file(output),
            "frozen_trace_sha256": sha256_file(frozen),
            "speech_active_mismatch_frames": sum(a[1] != b[1] for a, b in zip(cold_rows, frozen_rows)),
            "max_abs_logit_difference": max(abs(x - y) for a, b in zip(cold_rows, frozen_rows)
                                            for x, y in zip(a[2], b[2]))}


def run_phase_aligned_control(*, case: dict, spool: pathlib.Path, frozen: pathlib.Path,
                              end_sample: int, model: pathlib.Path, pack: pathlib.Path,
                              posterior_dump: pathlib.Path, decoder_replay: pathlib.Path,
                              output_dir: pathlib.Path) -> dict:
    interval = phase_aligned_interval(frozen, end_sample)
    if interval["model_sha256"] != sha256_file(model):
        raise ValueError("phase-control model SHA mismatch")
    wav = output_dir / "phase-aligned-cold-context.wav"
    receipt = extract_phase_capture(spool, wav, interval)
    cold = output_dir / "phase-aligned-cold-relative.kwtr"
    dump = run_json_lines([str(posterior_dump), str(model), str(wav), str(cold)])
    if (len(dump) != 1 or dump[0].get("trace_sha256") != sha256_file(cold)
            or dump[0].get("model_sha256") != interval["model_sha256"]):
        raise ValueError("phase-control posterior receipt mismatch")
    absolute = output_dir / "phase-aligned-cold-absolute.kwtr"
    comparison = align_cold_trace(cold, frozen, absolute, interval["start_sample"])
    detections = run_json_lines([str(decoder_replay), str(model), str(pack), str(absolute),
                                case["case_id"] + "-phase-aligned-cold"])
    hits = target_detection_matches(detections, case)
    return {"policy": "frozen-frame-grid-aligned-cold-acoustic-control-v1",
            "model_sha256": sha256_file(model), "keyword_pack_sha256": sha256_file(pack),
            "capture": receipt, "trace_comparison": comparison,
            "posterior_dump_receipts": dump, "decoder_detections": detections,
            "target_window_hits": len(hits), "target_time_tolerance_seconds": 0.10,
            "historical_event_exactly_reproduced": any(
                abs(float(row["time_s"]) - float(case["detection_time_s"])) <= 1e-6
                and abs(float(row["confidence"]) - float(case["historical_confidence"])) <= 1e-6
                for row in hits),
            "classification": ("phase-aligned-cold-context-sufficient" if hits else
                               "phase-aligned-cold-context-insufficient-retained-acoustic-state-required"),
            "interpretation_limit": "One fixed event; sufficiency is not a general history bound or shipping qualification."}


def classify_history_windows(
    *,
    fresh_context_hits: int,
    windows: list[dict],
) -> dict:
    reproduced = [row for row in windows if int(row["target_window_hits"]) > 0]
    if not reproduced:
        raise ValueError("full acoustic-history replay never reproduced target detection")
    first = reproduced[0]
    if first["label"] == "pre-10s" and fresh_context_hits == 0:
        classification = "acoustic-history-or-frame-phase-decoder-10s-sufficient"
        bracket = {"lower_no_hit_seconds": None, "upper_hit_seconds": 10}
    elif first["label"] == "full":
        previous = windows[-2] if len(windows) >= 2 else None
        classification = "decoder-history-required-with-full-acoustic-state"
        bracket = {
            "lower_no_hit_seconds": (
                None if previous is None else previous["pre_roll_seconds"]
            ),
            "upper_hit_seconds": "full",
        }
    else:
        index = windows.index(first)
        previous = windows[index - 1] if index > 0 else None
        classification = "decoder-history-required-with-full-acoustic-state"
        bracket = {
            "lower_no_hit_seconds": (
                None if previous is None else previous["pre_roll_seconds"]
            ),
            "upper_hit_seconds": first["pre_roll_seconds"],
        }
    return {
        "classification": classification,
        "first_reproducing_window": first["label"],
        "history_bracket": bracket,
    }


def run_history_decomposition(
    *,
    spec: dict,
    model: pathlib.Path,
    pack: pathlib.Path,
    context_root: pathlib.Path,
    spool_root: pathlib.Path,
    posterior_dump: pathlib.Path,
    trace_slice: pathlib.Path,
    decoder_replay: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    if sha256_file(model) != spec["expected_model_sha256"]:
        raise ValueError("history decomposition model SHA mismatch")
    if sha256_file(pack) != spec["expected_keyword_pack_sha256"]:
        raise ValueError("history decomposition keyword-pack SHA mismatch")
    pre_rolls = [10, 60, 600, 1800, 3600]
    results: list[dict] = []
    for case in spec["cases"]:
        case_context = context_root / case["case_id"] / "stream-context-result.json"
        if not case_context.is_file():
            raise ValueError(f"{case['case_id']}: stream-context result is missing")
        context = json.loads(case_context.read_text(encoding="utf-8"))
        if context.get("classification") != "longer-stream-state-or-history-required":
            results.append(
                {
                    "case_id": case["case_id"],
                    "status": "not-needed",
                    "reason": context.get("classification"),
                }
            )
            continue
        fresh_hits = int(
            ((context.get("analysis") or {}).get("target_keyword_hits", -1))
        )
        if fresh_hits != 0:
            raise ValueError(
                f"{case['case_id']}: longer-history case unexpectedly has fresh-context target hit"
            )
        spool = spool_root / f"{case['case_id']}.pcm16le"
        expected_bytes = 7200 * 16000 * 2
        if not spool.is_file() or spool.stat().st_size != expected_bytes:
            raise ValueError(f"{case['case_id']}: exact 2h PCM spool is missing or wrong size")

        case_root = output_dir / case["case_id"]
        full_wav = case_root / "full-stream.wav"
        wav_receipt = pcm16le_to_wav(spool, full_wav)
        full_trace = case_root / "full-stream.kwtr"
        posterior_receipts = run_json_lines(
            [str(posterior_dump), str(model), str(full_wav), str(full_trace)]
        )
        if len(posterior_receipts) != 1:
            raise ValueError(f"{case['case_id']}: posterior dump receipt count mismatch")
        full_wav.unlink()

        total_samples = 7200 * 16000
        end_sample = min(
            total_samples,
            int(round((float(case["detection_time_s"]) + 2.0) * 16000.0)),
        )
        windows: list[dict] = []
        window_specs = [(f"pre-{seconds}s", seconds) for seconds in pre_rolls]
        window_specs.append(("full", None))
        for label, pre_roll in window_specs:
            start_sample = (
                0
                if pre_roll is None
                else max(
                    0,
                    int(
                        round(
                            (float(case["detection_time_s"]) - float(pre_roll))
                            * 16000.0
                        )
                    ),
                )
            )
            sliced = case_root / f"{label}.kwtr"
            slice_receipts = run_json_lines(
                [
                    str(trace_slice),
                    str(full_trace),
                    str(sliced),
                    str(start_sample),
                    str(end_sample),
                ]
            )
            if len(slice_receipts) != 1:
                raise ValueError(f"{case['case_id']}:{label}: trace-slice receipt mismatch")
            replay_rows = run_json_lines(
                [
                    str(decoder_replay),
                    str(model),
                    str(pack),
                    str(sliced),
                    f"{case['case_id']}-{label}",
                ]
            )
            target_hits = target_detection_matches(replay_rows, case)
            window = {
                "label": label,
                "pre_roll_seconds": pre_roll,
                "start_sample_exclusive": start_sample,
                "end_sample_inclusive": end_sample,
                "trace_sha256": sha256_file(sliced),
                "trace_frames": int(slice_receipts[0]["frames"]),
                "decoder_detections": replay_rows,
                "target_window_hits": len(target_hits),
            }
            windows.append(window)

        full_rows = windows[-1]["decoder_detections"]
        historical_detection(full_rows, case)
        verdict = classify_history_windows(
            fresh_context_hits=fresh_hits,
            windows=windows,
        )
        phase_control = (
            run_phase_aligned_control(
                case=case, spool=spool, frozen=case_root / "pre-10s.kwtr",
                end_sample=end_sample, model=model, pack=pack,
                posterior_dump=posterior_dump, decoder_replay=decoder_replay,
                output_dir=case_root,
            )
            if windows[0]["target_window_hits"] > 0 else
            {"status": "not-applicable", "reason": "frozen-10s-decoder-baseline-does-not-reproduce"}
        )
        result = {
            "case": case,
            "fresh_context_target_hits": fresh_hits,
            "phase_aligned_control": phase_control,
            "legacy_context_comparison_limit": "Cold unaligned PCM resets both acoustic state and frame phase; its miss alone cannot isolate recurrent history.",
            "full_stream_pcm": wav_receipt,
            "full_posterior_receipts": posterior_receipts,
            "full_trace_sha256": sha256_file(full_trace),
            "windows": windows,
            **verdict,
        }
        (case_root / "history-decomposition-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)

    output = {
        "schema_version": 1,
        "evidence_class": "frozen-far-history-decomposition-v1",
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "training_changed": False,
        "decoder_math_changed": False,
        "thresholds_changed": False,
        "acoustic_posterior_history": "full-stream-from-zero-v1",
        "decoder_history_windows_seconds": pre_rolls + ["full"],
        "target_time_tolerance_seconds": 0.10,
        "model_sha256": sha256_file(model),
        "keyword_pack_sha256": sha256_file(pack),
        "cases": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history-decomposition-summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def run_replay(
    *,
    spec: dict,
    model: pathlib.Path,
    pack: pathlib.Path,
    domain_index: pathlib.Path,
    runner: pathlib.Path,
    posterior_dump: pathlib.Path,
    decoder_replay: pathlib.Path,
    decoder_path_replay: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict:
    if sha256_file(model) != spec["expected_model_sha256"]:
        raise ValueError("frozen replay model SHA mismatch")
    if sha256_file(pack) != spec["expected_keyword_pack_sha256"]:
        raise ValueError("frozen replay keyword-pack SHA mismatch")
    domain_rows = load_jsonl(domain_index)
    results: list[dict] = []
    for case in spec["cases"]:
        domain = find_domain_row(domain_rows, case)
        case_root = output_dir / case["case_id"]
        scaled_wav = case_root / "historical-gain-source.wav"
        scale = scale_wav(
            pathlib.Path(domain["path"]),
            case["historical_gain"],
            scaled_wav,
        )
        analysis = analyze_wav(
            wav=scaled_wav,
            recording=case["case_id"],
            keyword_id=int(case["detected_keyword_id"]),
            model=model,
            pack=pack,
            runner=runner,
            posterior_dump=posterior_dump,
            decoder_replay=decoder_replay,
            decoder_path_replay=decoder_path_replay,
            output_dir=case_root,
            trace_stem="historical-gain-source",
        )
        classification = (
            "standalone-source-sufficient-under-historical-gain"
            if int(analysis["target_keyword_hits"]) > 0
            else "standalone-source-not-sufficient-mixture-or-stream-context-required"
        )
        result = {
            "case": case,
            "domain": {
                key: domain[key]
                for key in (
                    "kind",
                    "family_id",
                    "keyword_id",
                    "tokens",
                    "target_ids",
                    "wav_sha256",
                    "source_wav_sha256",
                    "scene_seed",
                    "scene",
                    "domain_id",
                    "split",
                )
                if key in domain
            },
            "scale": scale,
            "direct_detections": analysis["direct_detections"],
            "posterior_dump_receipts": analysis["posterior_dump_receipts"],
            "replay_detections": analysis["replay_detections"],
            "direct_replay_exact_parity": analysis["direct_replay_exact_parity"],
            "target_keyword_standalone_hits": analysis["target_keyword_hits"],
            "target_keyword_path": analysis["target_keyword_path"],
            "classification": classification,
        }
        case_root.mkdir(parents=True, exist_ok=True)
        (case_root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)
    output = {
        "schema_version": 1,
        "evidence_class": RESULT_CLASS,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "training_changed": False,
        "decoder_math_changed": False,
        "thresholds_changed": False,
        "model_sha256": sha256_file(model),
        "keyword_pack_sha256": sha256_file(pack),
        "cases": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="frozen-far-source-replay-") as td:
        root = pathlib.Path(td)
        wav = root / "source.wav"
        import struct

        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(
                struct.pack("<hhhh", 1000, -1000, 32767, -32768)
            )
        scaled = root / "scaled.wav"
        receipt = scale_wav(wav, 0.5, scaled)
        assert receipt["frames"] == 4
        with wave.open(str(scaled), "rb") as reader:
            values = struct.unpack("<hhhh", reader.readframes(4))
        assert values == (500, -500, 16384, -16384)

        spec_path = root / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_class": EVIDENCE_CLASS,
                    "experiment_id": "fixture-frozen-far-source-replay",
                    "development_only": True,
                    "selection_feedback_allowed": False,
                    "protected_evidence_used": False,
                    "historical_run_id": 1,
                    "historical_artifact_id": 2,
                    "historical_head_sha": "d" * 40,
                    "expected_model_sha256": "a" * 64,
                    "expected_keyword_pack_sha256": "b" * 64,
                    "cases": [
                        {
                            "case_id": "seed-1103-fixture",
                            "seed": 1103,
                            "detection_time_s": 1.25,
                            "historical_confidence": 0.5,
                            "detected_keyword_id": 2,
                            "rendered_wav_sha256": "c" * 64,
                            "historical_gain": 0.9,
                            "expected_tokens": ["xiao3", "wo1"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        normalized = normalize_spec(spec_path)
        assert normalized["cases"][0]["historical_gain"] == 0.9

        domain_wav = root / "domain.wav"
        domain_wav.write_bytes(b"domain")
        domain_rows = [
            {
                "path": str(domain_wav),
                "wav_sha256": sha256_file(domain_wav),
                "family_id": "qualification-negative-0-3",
                "tokens": ["xiao3", "wo1"],
            }
        ]
        domain_case = dict(normalized["cases"][0])
        domain_case["rendered_wav_sha256"] = sha256_file(domain_wav)
        matched = find_domain_row(domain_rows, domain_case)
        assert matched["family_id"] == "qualification-negative-0-3"

        assert canonical_detections(
            [{"keyword_id": 2, "time_s": 1.2345678, "confidence": 0.8765432}]
        ) == [(2, 1.234568, 0.876543)]

        fixture_case = dict(normalized["cases"][0])
        fixture_case["detection_time_s"] = 1.25
        fixture_case["historical_confidence"] = 0.5
        fixture_case["rendered_wav_sha256"] = "e" * 64
        assert historical_detection(
            [{"keyword_id": 2, "time_s": 1.25, "confidence": 0.5}],
            fixture_case,
        )["keyword_id"] == 2
        assert historical_active_injection(
            [
                {
                    "start_second": 1,
                    "source_seconds": 1.0,
                    "source_sha256": "e" * 64,
                    "gain": 0.9,
                }
            ],
            fixture_case,
        )["source_sha256"] == "e" * 64

        spool = root / "stream.pcm16le"
        spool.write_bytes(
            struct.pack("<" + "h" * 32000, *([123] * 32000))
        )
        capture = root / "capture.wav"
        capture_receipt = extract_pcm_capture(
            spool=spool,
            output=capture,
            center_time_s=1.0,
            before_seconds=0.5,
            after_seconds=0.25,
        )
        assert capture_receipt["frames"] == 12000
        with wave.open(str(capture), "rb") as reader:
            assert reader.getnframes() == 12000

        # Explicit KWTRACE1 fixtures verify sample-grid derivation, not trim guesses.
        def phase_fixture(path, endpoints, *, model_sha="a" * 64, logit_delta=0.0):
            header = bytearray(120)
            header[:8] = b"KWTRACE1"
            struct.pack_into("<IHHIIIIQQ", header, 8, 1, 2, 0, 16000, 400, 320, 0, 7, len(endpoints))
            header[48:112] = model_sha.encode("ascii")
            frames = b"".join(struct.pack("<QB7xff", end, 1, 0.5 + logit_delta, -0.5)
                              for end in endpoints)
            path.write_bytes(header + frames)

        frozen_phase = root / "frozen-phase.kwtr"
        cold_phase = root / "cold-phase.kwtr"
        absolute_phase = root / "absolute-phase.kwtr"
        phase_fixture(frozen_phase, [720, 1040, 1360])
        phase_fixture(cold_phase, [400, 720, 1040])
        interval = phase_aligned_interval(frozen_phase, 1400)
        assert interval["start_sample"] == 320
        assert interval["end_sample"] == 1400
        assert interval["frames"] == 3
        phase_capture = extract_phase_capture(spool, root / "phase.wav", interval)
        assert phase_capture["pcm_samples"] == 1080
        assert phase_capture["pcm_sha256"] == hashlib.sha256(spool.read_bytes()[640:2800]).hexdigest()
        comparison = align_cold_trace(cold_phase, frozen_phase, absolute_phase, 320)
        assert comparison["absolute_frame_grid_equal"]
        assert comparison["max_abs_logit_difference"] == 0.0
        assert absolute_phase.read_bytes() == frozen_phase.read_bytes()
        original_frozen_hash = sha256_file(frozen_phase)
        phase_fixture(cold_phase, [400, 720, 1040], logit_delta=0.25)
        comparison = align_cold_trace(cold_phase, frozen_phase, absolute_phase, 320)
        assert comparison["max_abs_logit_difference"] == 0.25
        assert sha256_file(frozen_phase) == original_frozen_hash
        for bad_offset in (0, 240, 400):
            try:
                align_cold_trace(cold_phase, frozen_phase, absolute_phase, bad_offset)
            except ValueError:
                pass
            else:
                raise AssertionError("misaligned cold grid accepted")
        for bad_end in (1359, 1680):
            try:
                phase_aligned_interval(frozen_phase, bad_end)
            except ValueError:
                pass
            else:
                raise AssertionError("different frozen exposure accepted")
        phase_fixture(cold_phase, [400, 720])
        try:
            align_cold_trace(cold_phase, frozen_phase, absolute_phase, 320)
        except ValueError:
            pass
        else:
            raise AssertionError("different frame count accepted")

        # Malformed external trace bytes must fail before creating replay evidence.
        phase_fixture(cold_phase, [400, 720, 1040])
        valid_trace = cold_phase.read_bytes()
        malformed = [valid_trace[:-1], valid_trace + b"x"]
        for offset, fmt, value in ((28, "<I", 1), (112, "<B", 1),
                                   (136, "<f", float("nan")), (128, "<B", 2),
                                   (144, "<Q", 721)):
            damaged = bytearray(valid_trace)
            struct.pack_into(fmt, damaged, offset, value)
            malformed.append(damaged)
        for damaged in malformed:
            cold_phase.write_bytes(damaged)
            try:
                read_phase_trace(cold_phase)
            except ValueError:
                pass
            else:
                raise AssertionError("malformed phase trace accepted")
        phase_fixture(cold_phase, [400, 720, 1040], model_sha="b" * 64)
        try:
            align_cold_trace(cold_phase, frozen_phase, absolute_phase, 320)
        except ValueError:
            pass
        else:
            raise AssertionError("different model identity accepted")

        # Mock only tool execution; the control still checks real trace/WAV bytes,
        # hashes, exposure and converts local timestamps before event matching.
        from unittest import mock
        model_fixture = root / "phase-model.kwm"
        model_fixture.write_bytes(b"phase-model")
        (root / "pack-fixture").write_bytes(b"phase-pack")
        phase_fixture(frozen_phase, [720, 1040, 1360], model_sha=sha256_file(model_fixture))
        phase_case = {"case_id": "phase-unit-case", "detected_keyword_id": 2,
                      "detection_time_s": 0.065, "historical_confidence": 0.6}
        calls = []
        fake_detections = [{"keyword_id": 2, "time_s": 0.065, "confidence": 0.6}]
        def fake_phase_run(argv):
            calls.append(argv)
            if argv[0] == "posterior-fixture":
                destination = pathlib.Path(argv[3])
                phase_fixture(destination, [400, 720, 1040], model_sha=sha256_file(model_fixture))
                return [{"model_sha256": sha256_file(model_fixture), "trace_sha256": sha256_file(destination)}]
            assert argv[0] == "decoder-fixture"
            _, _, actual_rows = read_phase_trace(pathlib.Path(argv[3]))
            assert [row[0] for row in actual_rows] == [720, 1040, 1360]
            return fake_detections
        with mock.patch.dict(run_phase_aligned_control.__globals__, {"run_json_lines": fake_phase_run}):
            result = run_phase_aligned_control(
                case=phase_case, spool=spool, frozen=frozen_phase, end_sample=1400,
                model=model_fixture, pack=root / "pack-fixture", posterior_dump=pathlib.Path("posterior-fixture"),
                decoder_replay=pathlib.Path("decoder-fixture"), output_dir=root,
            )
        assert len(calls) == 2
        assert result["target_window_hits"] == 1
        assert result["historical_event_exactly_reproduced"]
        assert result["classification"] == "phase-aligned-cold-context-sufficient"
        # Events outside the original target window cannot satisfy the control.
        fake_detections[:] = [{"keyword_id": 2, "time_s": 1.065, "confidence": 0.6}]
        with mock.patch.dict(run_phase_aligned_control.__globals__, {"run_json_lines": fake_phase_run}):
            missed = run_phase_aligned_control(
                case=phase_case, spool=spool, frozen=frozen_phase, end_sample=1400,
                model=model_fixture, pack=root / "pack-fixture", posterior_dump=pathlib.Path("posterior-fixture"),
                decoder_replay=pathlib.Path("decoder-fixture"), output_dir=root,
            )
        assert missed["target_window_hits"] == 0
        assert not missed["historical_event_exactly_reproduced"]
        assert missed["classification"] == "phase-aligned-cold-context-insufficient-retained-acoustic-state-required"

        assert classify_history_windows(
            fresh_context_hits=0,
            windows=[
                {"label": "pre-10s", "pre_roll_seconds": 10, "target_window_hits": 1},
                {"label": "full", "pre_roll_seconds": None, "target_window_hits": 1},
            ],
        )["classification"] == "acoustic-history-or-frame-phase-decoder-10s-sufficient"
        history = classify_history_windows(
            fresh_context_hits=0,
            windows=[
                {"label": "pre-10s", "pre_roll_seconds": 10, "target_window_hits": 0},
                {"label": "pre-60s", "pre_roll_seconds": 60, "target_window_hits": 1},
                {"label": "full", "pre_roll_seconds": None, "target_window_hits": 1},
            ],
        )
        assert history["classification"] == "decoder-history-required-with-full-acoustic-state"
        assert history["history_bracket"] == {
            "lower_no_hit_seconds": 10,
            "upper_hit_seconds": 60,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    verify = sub.add_parser("verify")
    verify.add_argument("--spec", required=True, type=pathlib.Path)

    run = sub.add_parser("run")
    run.add_argument("--spec", required=True, type=pathlib.Path)
    run.add_argument("--model", required=True, type=pathlib.Path)
    run.add_argument("--pack", required=True, type=pathlib.Path)
    run.add_argument("--domain-index", required=True, type=pathlib.Path)
    run.add_argument("--runner", required=True, type=pathlib.Path)
    run.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    run.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    run.add_argument("--decoder-path-replay", required=True, type=pathlib.Path)
    run.add_argument("--output-dir", required=True, type=pathlib.Path)

    context = sub.add_parser("context")
    context.add_argument("--spec", required=True, type=pathlib.Path)
    context.add_argument("--model", required=True, type=pathlib.Path)
    context.add_argument("--pack", required=True, type=pathlib.Path)
    context.add_argument("--stream-root", required=True, type=pathlib.Path)
    context.add_argument("--spool-root", required=True, type=pathlib.Path)
    context.add_argument("--runner", required=True, type=pathlib.Path)
    context.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    context.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    context.add_argument("--decoder-path-replay", required=True, type=pathlib.Path)
    context.add_argument("--output-dir", required=True, type=pathlib.Path)

    history = sub.add_parser("history")
    history.add_argument("--spec", required=True, type=pathlib.Path)
    history.add_argument("--model", required=True, type=pathlib.Path)
    history.add_argument("--pack", required=True, type=pathlib.Path)
    history.add_argument("--context-root", required=True, type=pathlib.Path)
    history.add_argument("--spool-root", required=True, type=pathlib.Path)
    history.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    history.add_argument("--trace-slice", required=True, type=pathlib.Path)
    history.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    history.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("frozen FAR source replay self-test: PASS")
        return 0
    if args.command == "verify":
        value = normalize_spec(args.spec.resolve())
        print(json.dumps(value, sort_keys=True, allow_nan=False))
        return 0
    if args.command == "run":
        spec = normalize_spec(args.spec.resolve())
        value = run_replay(
            spec=spec,
            model=args.model.resolve(),
            pack=args.pack.resolve(),
            domain_index=args.domain_index.resolve(),
            runner=args.runner.resolve(),
            posterior_dump=args.posterior_dump.resolve(),
            decoder_replay=args.decoder_replay.resolve(),
            decoder_path_replay=args.decoder_path_replay.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": row["case"]["case_id"],
                            "classification": row["classification"],
                            "target_keyword_standalone_hits": row[
                                "target_keyword_standalone_hits"
                            ],
                        }
                        for row in value["cases"]
                    ]
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "context":
        spec = normalize_spec(args.spec.resolve())
        value = run_stream_context(
            spec=spec,
            model=args.model.resolve(),
            pack=args.pack.resolve(),
            stream_root=args.stream_root.resolve(),
            spool_root=args.spool_root.resolve(),
            runner=args.runner.resolve(),
            posterior_dump=args.posterior_dump.resolve(),
            decoder_replay=args.decoder_replay.resolve(),
            decoder_path_replay=args.decoder_path_replay.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": row["case"]["case_id"],
                            "classification": row["classification"],
                            "target_keyword_context_hits": row["analysis"][
                                "target_keyword_hits"
                            ],
                        }
                        for row in value["cases"]
                    ]
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "history":
        spec = normalize_spec(args.spec.resolve())
        value = run_history_decomposition(
            spec=spec,
            model=args.model.resolve(),
            pack=args.pack.resolve(),
            context_root=args.context_root.resolve(),
            spool_root=args.spool_root.resolve(),
            posterior_dump=args.posterior_dump.resolve(),
            trace_slice=args.trace_slice.resolve(),
            decoder_replay=args.decoder_replay.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": (
                                row["case"]["case_id"]
                                if "case" in row
                                else row["case_id"]
                            ),
                            "status": row.get("status", "analyzed"),
                            "classification": row.get("classification"),
                            "first_reproducing_window": row.get(
                                "first_reproducing_window"
                            ),
                            "history_bracket": row.get("history_bracket"),
                        }
                        for row in value["cases"]
                    ]
                },
                sort_keys=True,
            )
        )
        return 0
    parser.error("a command or --self-test is required")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
