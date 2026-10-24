#!/usr/bin/env python3
"""Verify saved per-clip event-diagnostic bindings, without executing a replay.

A consistent receipt is not proof that its claimed producer ran. This narrow
contract never issues acoustic metrics, historical-equivalence or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import wave

from gen_parameter_limits import load_contract, render_header
from qualification_common import read_model, read_pack, PACK_RECORD

SCHEMA = "kws-event-readback-identity-v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
# CLI events print six decimals; allow only the corresponding rounding error.
PRINT_TOLERANCE = 0.500001e-6
SOURCE_FILES = frozenset({
    "src/kws.c", "src/decoder.c", "src/frontend.c", "src/model.c",
    "src/keyword_pack.c", "src/kws_debug.h", "src/decoder.h",
    "src/frontend.h", "src/model.h", "include/kws_pipeline/kws.h",
    "tools/kws_trace_io.c", "tools/kws_trace_io.h", "tools/tool_io.c",
    "tools/tool_io.h", "tools/sha256.c", "tools/sha256.h",
    "tools/kws_posterior_dump.c", "tools/kws_decoder_replay.c",
    "configs/parameter-contract.json",
})
ARTIFACTS = frozenset({"model", "keyword_pack", "trace", "source_wav",
                      "capture_binary", "replay_binary", "source_manifest",
                      "generated_header", "parameter_contract", "events", "completion"})
CONTEXT = {"mode": "per-clip-fresh-engine", "sample_origin": 0,
           "vad": "saved-frame-flags", "preprocessing": "unchanged-mono-16k-pcm16",
           "padding": "none", "flush": False, "reset_records": [],
           "discontinuities": [], "overrides": [], "capture_keyword_pack": None}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def unique(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw: str) -> object:
    def bad(value: str) -> None:
        raise ValueError("non-finite JSON: " + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=bad)


def read_json(path: Path) -> dict:
    require(path.stat().st_size <= 1024 * 1024, "JSON too large")
    obj = parse(path.read_text(encoding="utf-8"))
    require(isinstance(obj, dict), "JSON object required")
    return obj


def integer(value: object, label: str) -> int:
    require(type(value) is int and value >= 0, label + " must be nonnegative integer")
    return value


def bound_file(root: Path, binding: dict) -> Path:
    require(isinstance(binding, dict) and set(binding) == {"path", "bytes", "sha256"},
            "file binding fields")
    name = binding["path"]
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "relative file path required")
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), "file outside receipt root or missing")
    size = integer(binding["bytes"], "bytes")
    require(size <= MAX_FILE_BYTES and path.stat().st_size == size, "file size mismatch/cap")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(binding["sha256"] == digest, "file hash mismatch")
    return path


def verify_completion(value: dict, identities: dict, runtime: dict,
                      source_id: str) -> dict:
    """Check a native execution record's bindings, not its authenticity.

    Native float values are serialized with nine significant digits; compare
    binary32 values exactly rather than inventing a numerical tolerance.
    """
    require(isinstance(value, dict) and set(value) == {
        "schema", "execution_kind", "production_qualified", "completion",
        "identity", "context", "runtime"}, "execution completion fields")
    require(value["schema"] == "kws-decoder-replay-completion-v1" and
            value["execution_kind"] == "native-linux-decoder-replay" and
            value["production_qualified"] is False, "execution completion scope")
    expected_identity = {"model_sha256", "keyword_pack_sha256", "trace_sha256",
                         "tool_sha256", "parameter_contract_sha256"}
    require(isinstance(identities, dict) and set(identities) == expected_identity and
            isinstance(value["identity"], dict) and
            value["identity"] == identities, "execution identity mismatch")
    require(all(isinstance(v, str) and len(v) == 64 and
                all(c in "0123456789abcdef" for c in v)
                for v in identities.values()), "execution identity digest")
    require(canonical(value["context"]) == canonical({
        "recording_id": source_id, "mode": "per-clip-fresh-engine",
        "vad": "saved-frame-flags",
        "reset": "fresh-engine-and-decoder-inactive-boundary",
        "clock": "saved-end-sample"}), "execution context mismatch")
    runtime_fields = {"min_speech_dbfs", "token_boost", "state_retention",
                      "refractory_ms", "external_vad_threshold", "blank_retention",
                      "fuzzy_child_cost_log", "search_policy_override"}
    require(isinstance(runtime, dict) and set(runtime) == runtime_fields and
            isinstance(value["runtime"], dict) and
            set(value["runtime"]) == runtime_fields, "execution runtime fields")
    for key, expected in runtime.items():
        actual = value["runtime"][key]
        if key == "search_policy_override":
            require(type(actual) is bool and actual is expected,
                    "execution runtime override mismatch")
        elif key == "refractory_ms":
            require(type(actual) is int and actual == expected,
                    "execution runtime refractory mismatch")
        else:
            require(type(actual) in (int, float) and math.isfinite(actual),
                    "execution runtime finite numeric value required")
            try:
                equal = struct.pack("<f", actual) == struct.pack("<f", expected)
            except (OverflowError, struct.error):
                equal = False
            require(equal, "execution runtime binary32 mismatch")
    counts = value["completion"]
    require(isinstance(counts, dict) and set(counts) == {
        "exit_code", "frames_consumed", "decoder_hits", "detections",
        "refractory_suppressed"}, "execution completion counts")
    for key, item in counts.items():
        integer(item, key)
    require(counts["exit_code"] == 0 and counts["decoder_hits"] ==
            counts["detections"] + counts["refractory_suppressed"] and
            counts["decoder_hits"] <= counts["frames_consumed"] and
            (counts["detections"] > 0 or counts["refractory_suppressed"] == 0),
            "execution completion accounting")
    return counts


def verify(path: Path) -> dict:
    path = path.resolve(strict=True)
    receipt = read_json(path)
    require(set(receipt) == {"schema", "scope", "source_id", "data_role", "label_grade",
                             "context", "runtime", "keyword_policy", "artifacts"},
            "receipt fields; metrics/qualification claims are not accepted")
    require(receipt["schema"] == SCHEMA and receipt["scope"] == "observed-per-clip-diagnostic",
            "only observed per-clip diagnostic scope is supported")
    require(isinstance(receipt["source_id"], str) and receipt["source_id"].strip(), "source id required")
    require(receipt["data_role"] in ("observed-development", "research-training", "synthetic-control"),
            "unsupported or sealed data role")
    require(receipt["label_grade"] in ("unknown", "weak-asr", "human-receipt", "constructed"),
            "unsupported label grade")
    require(canonical(receipt["context"]) == canonical(CONTEXT), "unsupported context/reset/VAD policy")
    bindings = receipt["artifacts"]
    require(isinstance(bindings, dict) and set(bindings) == ARTIFACTS, "artifact set incomplete")
    files = {key: bound_file(path.parent, binding) for key, binding in bindings.items()}
    sources = read_json(files["source_manifest"])
    require(set(sources) == SOURCE_FILES, "source manifest must cover the complete supported capture/replay closure")
    source_paths = {key: bound_file(files["source_manifest"].parent, value) for key, value in sources.items()}
    require(source_paths["configs/parameter-contract.json"].read_bytes() == files["parameter_contract"].read_bytes(),
            "parameter contract/source mismatch")
    read_json(files["parameter_contract"])  # Reject duplicate/non-finite JSON before rendering.
    contract, digest = load_contract(files["parameter_contract"])
    require(files["generated_header"].read_text() == render_header(contract, digest), "generated header drift")
    defaults = {key: value["default"] for key, value in contract["runtime"].items()}
    require(canonical(receipt["runtime"]) == canonical(defaults), "only bound default runtime/no overrides supported")
    model = read_model(files["model"])
    pack = read_pack(files["keyword_pack"])
    require((model["vocab_size"], model["vocab_fingerprint"]) ==
            (pack["vocab_size"], pack["vocab_fingerprint"]), "model/pack vocabulary mismatch")
    blob = files["keyword_pack"].read_bytes()
    keywords = []
    for i in range(pack["keyword_count"]):
        key, threshold, count, blanks, priority, policy, grace, _, *tokens = PACK_RECORD.unpack_from(blob, 24 + i * 48)
        require(blanks <= 8 and priority <= 15 and grace <= 32, "keyword policy range")
        keywords.append(dict(id=key, threshold=threshold, tokens=tokens[:count],
                             min_trailing_blanks=blanks, priority=priority,
                             prefix_policy=policy, grace_frames=grace))
    require(canonical(receipt["keyword_policy"]) == canonical(keywords), "keyword policy differs from bound pack")
    blob = files["trace"].read_bytes()
    require(len(blob) >= 120, "trace header missing (raw transcripts/logits are not event input)")
    h = struct.unpack_from("<8sIHHIIIIQQ64s8s", blob)
    require(h[:9] == (b"KWTRACE1", 1, model["vocab_size"], model["frontend_kind"],
                      16000, 400, 320, 0, model["vocab_fingerprint"]), "trace identity/geometry")
    require(h[10] == bindings["model"]["sha256"].encode() and h[11] == bytes(8), "trace model/reserved identity")
    frames, width = h[9], 16 + model["vocab_size"] * 4
    require(frames > 0, "supported capture requires a non-empty trace")
    require(len(blob) == 120 + frames * width, "trace length mismatch")
    for index in range(frames):
        end, flags = struct.unpack_from("<Q8s", blob, 120 + index * width)
        require(end == 400 + index * 320, "gap/reset/noncanonical clock is unsupported")
        require(flags[0] in (0, 1) and flags[1:] == bytes(7), "saved VAD/reserved flags")
        logits = struct.unpack_from("<" + "f" * model["vocab_size"], blob, 136 + index * width)
        require(all(math.isfinite(x) for x in logits), "non-finite logits")
    with wave.open(str(files["source_wav"]), "rb") as audio:
        require((audio.getnchannels(), audio.getsampwidth(), audio.getframerate(), audio.getcomptype()) ==
                (1, 2, 16000, "NONE"), "source PCM geometry")
        samples = audio.getnframes()
        require(len(audio.readframes(samples)) == samples * 2, "source WAV truncated")
    require(frames == max(0, 1 + (samples - 400) // 320), "source/trace complete-frame coverage")
    completion = read_json(files["completion"])
    if "schema" in completion:
        def f32(number: float) -> float:
            return struct.unpack("<f", struct.pack("<f", number))[0]
        constants = contract["algorithm_constants"]
        execution_runtime = dict(defaults,
            blank_retention=f32(math.exp(f32(constants["KWS_SILENCE_RETENTION_LOG"]["default"]))),
            fuzzy_child_cost_log=constants["KWS_FUZZY_CHILD_RETENTION_COST_LOG"]["default"],
            search_policy_override=False)
        completion = verify_completion(completion, {
            "model_sha256": bindings["model"]["sha256"],
            "keyword_pack_sha256": bindings["keyword_pack"]["sha256"],
            "trace_sha256": bindings["trace"]["sha256"],
            "tool_sha256": bindings["replay_binary"]["sha256"],
            "parameter_contract_sha256": digest,
        }, execution_runtime, receipt["source_id"])
    require(set(completion) == {"exit_code", "frames_consumed", "decoder_hits", "detections", "refractory_suppressed"},
            "completion fields")
    for key, value in completion.items():
        integer(value, key)
    require(completion["exit_code"] == 0 and completion["frames_consumed"] == frames, "incomplete/failed producer claim")
    require(completion["decoder_hits"] == completion["detections"] + completion["refractory_suppressed"]
            and completion["decoder_hits"] <= frames, "hit/detection/suppression accounting")
    require(completion["detections"] > 0 or completion["refractory_suppressed"] == 0,
            "fresh engine cannot suppress its first hit")
    event_rows = [parse(line) for line in files["events"].read_text().splitlines() if line.strip()]
    require(len(event_rows) == completion["detections"], "event count/completion mismatch")
    previous_sample = None
    thresholds = {k["id"]: k["threshold"] for k in keywords}
    refractory_samples = defaults["refractory_ms"] * 16000 // 1000
    for row in event_rows:
        require(isinstance(row, dict) and set(row) == {"recording", "keyword_id", "time_s", "confidence"}, "event fields")
        integer(row["keyword_id"], "keyword_id")
        require(row["recording"] == receipt["source_id"] and row["keyword_id"] in thresholds, "event recording/keyword identity")
        for key in ("time_s", "confidence"):
            require(type(row[key]) in (float, int) and math.isfinite(row[key]), "event numeric field")
        require(0 <= row["time_s"] <= samples / 16000 and
                0 <= row["confidence"] <= 1, "event time/confidence range")
        frame_index = round((row["time_s"] * 16000 - 400) / 320)
        end_sample = 400 + frame_index * 320
        require(0 <= frame_index < frames and
                abs(row["time_s"] - end_sample / 16000) <= PRINT_TOLERANCE,
                "event does not match saved frame clock")
        require(row["confidence"] + PRINT_TOLERANCE >= thresholds[row["keyword_id"]],
                "event confidence below keyword threshold")
        require(previous_sample is None or (end_sample > previous_sample and
                end_sample - previous_sample >= refractory_samples),
                "event violates frame uniqueness/refractory policy")
        previous_sample = end_sample
    return {"schema": SCHEMA, "status": "VERIFIED_BINDINGS_ONLY", "frames": frames,
            "event_count_claimed": len(event_rows), "execution_observed": False,
            "completion_authenticity_verified": False, "historical_event_equivalence": False,
            "production_qualified": False, "promotion_allowed": False,
            "frr": None, "far_per_hour": None, "word_end_latency_ms": None,
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.receipt), sort_keys=True))


if __name__ == "__main__":
    main()
