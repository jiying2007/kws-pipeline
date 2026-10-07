"""Strict, standard-library-only scoring of the three-stream constructed-noise collector log.

This module never executes audio, a frontend, a model, or a decoder.  It validates
the complete trace before producing descriptive counts.  A malformed or missing
observation raises ValidationError; it is never converted to a negative result.

Paths bind metadata to the SHA-256 of the exact file bytes.  In-memory mappings
bind to canonical_json_bytes(), intended for pure-data toy tests.  The caller
must independently supply expected protocol/model/library/decoder-config hashes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


RAW_SCHEMA = "a20-n0-deterministic-negative-raw-v1"
SCORE_SCHEMA = "a20-n0-deterministic-negative-score-v1"
HASH_KEYS = ("protocol_sha256", "model_sha256", "library_sha256",
             "manifest_sha256", "geometry_sha256", "decoder_config_sha256")
EXPECTED_IDS = ("n0-low-triangular", "n0-colored-ma32", "n0-sparse-transients")
DOMAINS = ("a20-n0-v1/low", "a20-n0-v1/colored", "a20-n0-v1/transient-base")
PLAN_FIELDS = ("call_index", "available_samples", "call_samples", "fbank_rows",
               "selected_rows", "centers")
COUNT_FIELDS = ("frames", "feed_calls", "callbacks", "fbank_rows", "model_rows")


class ValidationError(ValueError):
    """The evidence cannot be safely scored."""


def _require(condition, message):
    if not condition:
        raise ValidationError(message)


def _int(value, name, minimum=0):
    _require(type(value) is int and value >= minimum,
             f"{name}: expected integer >= {minimum}")
    return value


def _number(value, name):
    _require(type(value) in (int, float), f"{name}: expected finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{name}: expected finite number")
    return value


def _finite_tree(value, name):
    # Also inspect supplemental timing/diagnostic fields, including inactive rows.
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_tree(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_tree(item, f"{name}[{index}]")
    elif type(value) is float:
        _number(value, name)


def _hash(value, name, length=64):
    _require(isinstance(value, str) and
             re.fullmatch(r"[0-9a-f]{%d}" % length, value) is not None,
             f"{name}: expected lowercase {length}-digit hash")
    _require(value != "0" * length, f"{name}: all-zero hash is not an identity")
    return value


def canonical_json_bytes(value):
    """Stable bytes used for in-memory metadata identity, not for file paths."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_json(data, source):
    try:
        return json.loads(data, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValidationError(f"{source}: nonfinite {value}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValidationError(f"{source}: malformed JSON: {error}") from error


def _load_document(value, name):
    if isinstance(value, (str, os.PathLike)):
        data = Path(value).read_bytes()
        document = _parse_json(data, name)
    else:
        document = copy.deepcopy(value)
        _finite_tree(document, name)
        try:
            data = canonical_json_bytes(document)
        except (TypeError, ValueError) as error:
            raise ValidationError(f"{name}: not JSON data") from error
    _require(isinstance(document, dict), f"{name}: expected object")
    _finite_tree(document, name)
    return document, hashlib.sha256(data).hexdigest()


def _load_raw(raw):
    if isinstance(raw, (str, os.PathLike)):
        data = Path(raw).read_bytes()
        # The collector writes one newline-terminated JSON object per record.
        _require(data.endswith(b"\n"), "raw: missing final newline / truncated EOF")
        lines = data.splitlines()
        _require(lines and all(line.strip() for line in lines),
                 "raw: empty log or blank record")
        records = [_parse_json(line, f"raw line {i + 1}")
                   for i, line in enumerate(lines)]
        digest = hashlib.sha256(data).hexdigest()
    else:
        _require(isinstance(raw, (list, tuple)), "raw: expected path or record list")
        records = copy.deepcopy(list(raw))
        _finite_tree(records, "raw")
        try:
            data = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
        except (TypeError, ValueError) as error:
            raise ValidationError("raw: not JSON data") from error
        digest = hashlib.sha256(data).hexdigest()
    _require(records, "raw: empty log")
    for i, row in enumerate(records):
        _require(isinstance(row, dict), f"raw record {i}: expected object")
        _finite_tree(row, f"raw record {i}")
    return records, digest


def _rows(document, name):
    rows = document.get("rows")
    _require(isinstance(rows, list) and len(rows) == 3,
             f"{name}: requires exactly the three frozen complete streams")
    _require(all(isinstance(row, dict) for row in rows), f"{name}: invalid row")
    ids = [row.get("recording") for row in rows]
    _require(all(isinstance(key, str) for key in ids), f"{name}: invalid recording")
    _require(len(set(ids)) == len(ids), f"{name}: duplicate recording")
    _require(set(ids) == set(EXPECTED_IDS), f"{name}: missing or unexpected recording")
    return rows


def _validate_manifest(manifest):
    rows = _rows(manifest, "manifest")
    _require([r["recording"] for r in rows] == list(EXPECTED_IDS), "manifest: fixed input order required")
    for row, domain in zip(rows, DOMAINS):
        name = row["recording"]
        for key, value in (("kind", "constructed_nonspeech_negative_control"), ("domain", domain),
                           ("sample_rate_hz", 16000), ("channels", 1), ("sample_width_bytes", 2),
                           ("frames", 4800000), ("duration_s", 300), ("verified", True),
                           ("formal_far_qualification_allowed", False)):
            _require(type(row.get(key)) is type(value) and row[key] == value, name + ": inconsistent " + key)
        for key in ("wav_sha256", "pcm_sha256", "recipe_sha256", "seed_hex"):
            _hash(row.get(key), name + "." + key)
    for key, value in (("audio_count", 3), ("frames", 14400000), ("duration_s", 900)):
        _require(type(manifest.get(key)) is type(value) and manifest[key] == value, "manifest: inconsistent " + key)
    _require(len(set(r["seed_hex"] for r in rows)) == 3, "manifest: seeds must differ")
    return rows


def _validate_geometry(geometry, manifest_rows):
    rows = _rows(geometry, "geometry")
    _require([r["recording"] for r in rows] == [r["recording"] for r in manifest_rows],
             "geometry: row order differs from manifest")
    for row, manifest_row in zip(rows, manifest_rows):
        name = "geometry " + row["recording"]
        for key in COUNT_FIELDS:
            _int(row.get(key), name + "." + key)
        _require(row["frames"] == manifest_row["frames"], name + ": frame mismatch")
        _require(row["feed_calls"] > 0, name + ": no feed calls")
        _require(row["feed_calls"] == (row["frames"] + 4799) // 4800,
                 name + ": feed count differs from 4800-sample chunk protocol")
        for key, expected in (("finish_calls", 1), ("decoder_input_rows", row["model_rows"])):
            if key in row:
                _require(type(row[key]) is int and row[key] == expected, name + ": invalid " + key)
        plan = row.get("callback_plan")
        _require(isinstance(plan, list) and len(plan) == row["callbacks"],
                 name + ": callback plan/count mismatch")
        previous_available = 0
        fbank_sum = model_sum = 0
        for index, call in enumerate(plan):
            _require(isinstance(call, dict), name + ": invalid callback plan")
            for key in PLAN_FIELDS[:-1]:
                _int(call.get(key), name + "." + key)
            _require(call["call_index"] == index, name + ": callback index/order mismatch")
            _require(previous_available <= call["available_samples"] <= row["frames"],
                     name + ": available samples outside clip or nonmonotonic")
            previous_available = call["available_samples"]
            centers = call.get("centers")
            _require(isinstance(centers, list) and len(centers) == call["selected_rows"],
                     name + ": center/selected row mismatch")
            for center in centers:
                _int(center, name + ".center")
            _require(all(a < b for a, b in zip(centers, centers[1:])),
                     name + ": centers not strictly increasing")
            fbank_sum += call["fbank_rows"]
            model_sum += call["selected_rows"]
        _require((fbank_sum, model_sum) == (row["fbank_rows"], row["model_rows"]),
                 name + ": plan totals inconsistent")
    totals = geometry.get("totals")
    _require(isinstance(totals, dict), "geometry: missing totals")
    expected = {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
    expected["clips"] = len(rows)
    # Geometry may have additional derived totals; all shared counts must agree.
    for key in COUNT_FIELDS:
        _int(totals.get(key), "geometry.totals." + key)
        _require(totals[key] == expected[key], "geometry: inconsistent total " + key)
    if "clips" in totals:
        _require(type(totals["clips"]) is int and totals["clips"] == len(rows),
                 "geometry: inconsistent clip total")
    for key, value in (("finish_calls", len(rows)), ("decoder_input_rows", expected["model_rows"])):
        if key in totals:
            _require(type(totals[key]) is int and totals[key] == value,
                     "geometry: inconsistent total " + key)
    return rows, expected


def _check_identity(record, row, name):
    _require(record.get("recording") == row["recording"], name + ": recording/order mismatch")
    _require(type(record.get("frames")) is int and record["frames"] == row["frames"],
             name + ": frame mismatch")
    for key in ("wav_sha256", "pcm_sha256"):
        _hash(record.get(key), name + "." + key)
        _require(record[key] == row[key], name + ": input hash mismatch: " + key)
    # Reject contradictory labels if a producer includes them as diagnostics.
    for key in ("domain", "seed_hex", "recipe_sha256"):
        if key in record:
            _require(type(record[key]) is type(row[key]) and record[key] == row[key],
                     name + ": label metadata mismatch: " + key)


def _validate_callback(call, plan, model_rows_before, where):
    for key in PLAN_FIELDS:
        _require(key in call, where + ": missing " + key)
    # Every supplied plan field is an independently expected callback value.
    for key, expected in plan.items():
        _require(key in call and type(call[key]) is type(expected) and call[key] == expected,
                 where + ": geometry mismatch: " + key)
    for key in PLAN_FIELDS[:-1]:
        _int(call[key], where + "." + key)
    _require(isinstance(call["centers"], list), where + ": invalid centers")
    for center in call["centers"]:
        _int(center, where + ".center")
    rows = call["selected_rows"]
    expected_centers = [3 * (model_rows_before + index) for index in range(rows)]
    _require(call["centers"] == expected_centers,
             where + ": selected centers disagree with decoder clock")
    logits = call.get("logits")
    _require(isinstance(logits, list) and len(logits) == rows,
             where + ": logits row mismatch")
    for index, values in enumerate(logits):
        _require(isinstance(values, list) and len(values) == 6,
                 where + f": logits[{index}] must have six classes")
        for value in values:
            _number(value, where + ".logits")
    for key in ("valid", "state"):
        _require(type(call.get(key)) is int and call[key] in (0, 1),
                 where + ": invalid " + key)
    keyword = _int(call.get("keyword"), where + ".keyword")
    _require(keyword in (0, 1, 2), where + ": unknown decoder keyword")
    decoded = _int(call.get("decoder_rows_decoded"), where + ".decoder_rows_decoded")
    total = _int(call.get("decoder_total_frames"), where + ".decoder_total_frames")
    _require(total == (model_rows_before + rows) * 3,
             where + ": decoder clock mismatch")
    start = _int(call.get("start_frame"), where + ".start_frame", -1)
    end = _int(call.get("end_frame"), where + ".end_frame", -1)
    score_value = _number(call.get("score"), where + ".score")
    _require(0 <= score_value <= 1, where + ": score outside [0,1]")
    if rows == 0:
        _require((call["valid"], call["state"], keyword, decoded, start, end, score_value)
                 == (0, 0, 0, 0, 0, 0, 0), where + ": invalid empty callback result")
    elif call["state"] == 0:
        _require((call["valid"], keyword, decoded, start, end, score_value)
                 == (1, 0, rows, -1, -1, 0), where + ": invalid inactive callback result")
    else:
        _require(call["valid"] == 1 and keyword in (1, 2) and 1 <= decoded <= rows,
                 where + ": invalid activation result")
        _require(0 <= start <= end < model_rows_before * 3 + decoded * 3,
                 where + ": invalid activation frame interval")
        _require(start % 3 == 0 and end % 3 == 0,
                 where + ": activation frames outside decoder grid")
        _require(5 <= end - start <= 250,
                 where + ": activation duration violates fixed decoder gate [5,250]")
    return decoded


def _validate_timing_evidence(records):
    required = {
        "model_loaded": ("wall_ns_since_run_start", "cpu_ns_since_run_start"),
        "callback": ("callback_entry_wall_ns_since_run_start", "callback_entry_cpu_ns_since_run_start"),
        "feed": ("service_wall_ns_including_callback_output", "service_cpu_ns_including_callback_output"),
        "finish": ("service_wall_ns_including_callback_output", "service_cpu_ns_including_callback_output"),
        "run_end": ("wall_ns", "process_cpu_ns", "maxrss_kib", "minor_faults", "major_faults", "block_input_ops", "block_output_ops"),
    }
    previous_wall = previous_cpu = 0
    for index, record in enumerate(records):
        kind = record.get("kind")
        for key in required.get(kind, ()):
            _int(record.get(key), f"raw[{index}].{key}")
        if kind in ("model_loaded", "callback", "run_end"):
            keys = required[kind][:2]
            wall, cpu = (record[k] for k in keys)
            _require(wall >= previous_wall and cpu >= previous_cpu,
                     f"raw[{index}]: nonmonotonic cumulative wall/CPU timing")
            previous_wall, previous_cpu = wall, cpu


def score(raw, manifest, expected_geometry, bindings=None):
    """Return a fully validated descriptive report, or raise ValidationError.

    bindings must include expected protocol_sha256, model_sha256,
    library_sha256 and decoder_config_sha256.  Optional manifest_sha256 and
    geometry_sha256 values must agree with the actual metadata supplied.
    """
    records, raw_hash = _load_raw(raw)
    _validate_timing_evidence(records)
    manifest, manifest_hash = _load_document(manifest, "manifest")
    geometry, geometry_hash = _load_document(expected_geometry, "geometry")
    manifest_rows = _validate_manifest(manifest)
    geometry_rows, geometry_totals = _validate_geometry(geometry, manifest_rows)
    if isinstance(bindings, (str, os.PathLike)):
        bindings, _ = _load_document(bindings, "bindings")
    _require(isinstance(bindings, dict), "independent expected hash bindings are required")
    expected_hashes = dict(bindings)
    for key, actual in (("manifest_sha256", manifest_hash), ("geometry_sha256", geometry_hash)):
        if key in expected_hashes:
            _require(expected_hashes[key] == actual, "bindings mismatch: " + key)
        expected_hashes[key] = actual
    first = records[0]
    _require(first.get("kind") == "run_start" and first.get("schema") == RAW_SCHEMA,
             "raw: missing/invalid initial run_start")
    for key in HASH_KEYS:
        _hash(expected_hashes.get(key), "expected " + key)
        _hash(first.get(key), "run_start." + key)
        _require(first[key] == expected_hashes[key], "run_start hash mismatch: " + key)
    position = 1
    model_load_record = None
    if position < len(records) and records[position].get("kind") == "model_loaded":
        model_load_record = copy.deepcopy(records[position])
        position += 1
    _require(model_load_record is not None, "raw: mandatory model_loaded missing")
    _require(model_load_record.get("weights_bytes") == 1565280 and model_load_record.get("state_bytes") == 333704,
             "model_loaded: frozen storage size mismatch")
    clips = []
    total_decoded = 0
    for row, geometry_row in zip(manifest_rows, geometry_rows):
        name = row["recording"]
        _require(position < len(records) and records[position].get("kind") == "clip_start",
                 name + ": missing clip_start")
        _check_identity(records[position], row, name + " clip_start")
        position += 1
        callbacks = []
        events = []
        pending = []
        model_rows = fbank_rows = decoded_rows = cumulative_samples = feed_count = 0
        finish_seen = False
        instrumented = False
        last_available = 0
        decoded_since_activation = set()
        last_active_end = None
        while position < len(records) and records[position].get("kind") != "clip_end":
            record = records[position]
            kind = record.get("kind")
            _require(record.get("recording") == name, name + ": record outside current clip")
            _require(not finish_seen, name + ": record after finish")
            if kind == "callback":
                index = len(callbacks)
                _require(index < len(geometry_row["callback_plan"]), name + ": extra callback")
                decoded = _validate_callback(record, geometry_row["callback_plan"][index],
                                             model_rows, f"{name} callback {index}")
                decoded_since_activation.update(record["centers"][:decoded])
                _require(last_available <= record["available_samples"] <= row["frames"],
                         name + ": callback available-audio order mismatch")
                last_available = record["available_samples"]
                if "phase" in record:
                    _require(record["phase"] in ("feed", "finish"), name + ": invalid callback phase")
                callbacks.append(record)
                pending.append(record)
                fbank_rows += record["fbank_rows"]
                model_rows += record["selected_rows"]
                decoded_rows += decoded
                if record["state"] == 1:
                    _require(record["start_frame"] in decoded_since_activation and
                             record["end_frame"] in decoded_since_activation,
                             name + ": activation endpoint not in actually decoded centers since reset")
                    _require(last_active_end is None or record["end_frame"] - last_active_end >= 50,
                             name + ": activation violates fixed decoder cooldown of 50 frames")
                    event = copy.deepcopy(record)
                    event["available_audio_s"] = record["available_samples"] / 16000
                    events.append(event)
                    last_active_end = record["end_frame"]
                    # The runtime clears hypotheses immediately upon activation.
                    # Skipped model tail rows never enter decoder history.
                    decoded_since_activation.clear()
            elif kind == "feed":
                instrumented = True
                _require(_int(record.get("feed_index"), name + ".feed_index") == feed_count,
                         name + ": missing/duplicate/out-of-order feed")
                input_samples = _int(record.get("input_samples"), name + ".input_samples", 1)
                _require(input_samples == min(4800, row["frames"] - cumulative_samples),
                         name + ": feed size differs from fixed 4800-sample chunk protocol")
                cumulative_samples += input_samples
                _require(cumulative_samples <= row["frames"] and
                         _int(record.get("cumulative_samples"), name + ".cumulative_samples")
                         == cumulative_samples, name + ": feed sample count mismatch")
                _require(_int(record.get("callbacks_delta"), name + ".callbacks_delta") == len(pending),
                         name + ": feed callback delta mismatch")
                _require(all(call.get("phase", "feed") == "feed" and
                             call["available_samples"] <= cumulative_samples for call in pending),
                         name + ": callback assigned to wrong feed/phase")
                pending.clear()
                feed_count += 1
            elif kind == "finish":
                instrumented = True
                _require(feed_count == geometry_row["feed_calls"] and cumulative_samples == row["frames"],
                         name + ": finish before all feeds/samples")
                _require(type(record.get("finish_calls")) is int and record["finish_calls"] == 1,
                         name + ": finish_calls must be one")
                _require(_int(record.get("callbacks_delta"), name + ".callbacks_delta") == len(pending),
                         name + ": finish callback delta mismatch")
                _require(all(call.get("phase", "finish") == "finish" and
                             call["available_samples"] == row["frames"] for call in pending),
                         name + ": callback assigned to wrong finish/phase")
                pending.clear()
                finish_seen = True
            else:
                raise ValidationError(name + ": unexpected record kind " + repr(kind))
            position += 1
        _require(position < len(records), name + ": missing clip_end / truncated EOF")
        end = records[position]
        _check_identity(end, row, name + " clip_end")
        _require(end.get("complete") is True, name + ": clip incomplete")
        _require(type(end.get("finish_calls")) is int and end["finish_calls"] == 1,
                 name + ": end finish_calls must be one")
        for key in COUNT_FIELDS:
            _require(_int(end.get(key), name + " clip_end." + key) == geometry_row[key],
                     name + ": end/geometry mismatch: " + key)
        observed = {"callbacks": len(callbacks), "fbank_rows": fbank_rows,
                    "model_rows": model_rows, "decoder_rows_decoded": decoded_rows}
        for key, value in observed.items():
            _require(_int(end.get(key), name + " clip_end." + key) == value,
                     name + ": raw callback/end mismatch: " + key)
        if "event_count" in end:
            _require(_int(end["event_count"], name + ".event_count") == len(events),
                     name + ": clip_end event_count mismatch")
        if "decoder_total_frames" in end:
            _require(_int(end["decoder_total_frames"], name + ".decoder_total_frames") == model_rows * 3,
                     name + ": clip_end decoder clock mismatch")
        _require(instrumented and finish_seen and not pending,
                 name + ": missing mandatory feed/finish instrumentation")
        _require(feed_count == end["feed_calls"] and cumulative_samples == row["frames"],
                 name + ": incomplete feed instrumentation")
        clip = {key: row[key] for key in ("recording", "kind", "domain", "seed_hex", "frames", "wav_sha256", "pcm_sha256")}
        clip.update(observed)
        clip["events"] = events
        clip["complete"] = True
        clip["feed_finish_records_validated"] = instrumented
        clip["duration_s"] = row["frames"] / 16000
        clip["event_count"] = len(events)
        clip["event_rate_per_constructed_hour"] = len(events) * 3600 / clip["duration_s"]
        clip["probabilities"] = "NOT_CAPTURED"
        clips.append(clip)
        total_decoded += decoded_rows
        position += 1
    _require(position == len(records) - 1, "raw: missing, duplicate or trailing run records")
    end = records[position]
    _require(end.get("kind") == "run_end" and end.get("complete") is True,
             "raw: missing/incomplete run_end")
    _require(type(end.get("clips")) is int and end["clips"] == 3, "run_end: clip count mismatch")
    # Support totals either nested or top-level; validate all required counts.
    totals = end.get("totals", end)
    _require(isinstance(totals, dict), "run_end: invalid totals")
    for key in COUNT_FIELDS:
        _require(_int(totals.get(key), "run_end." + key) == geometry_totals[key],
                 "run_end: total mismatch: " + key)
    for key, expected in geometry["totals"].items():
        if key in totals:
            _require(type(totals[key]) is type(expected) and totals[key] == expected,
                     "run_end: geometry total mismatch: " + key)
    if "finish_calls" in totals:
        _require(_int(totals["finish_calls"], "run_end.finish_calls") == 3,
                 "run_end: finish total mismatch")
    _require(_int(totals.get("decoder_rows_decoded"), "run_end.decoder_rows_decoded") == total_decoded,
             "run_end: decoded total mismatch")
    event_count = sum(len(clip["events"]) for clip in clips)
    for location, record in (("run_end", end), ("run_end.totals", totals)):
        if "event_count" in record:
            _require(_int(record["event_count"], location + ".event_count") == event_count,
                     location + ": event_count mismatch")
    duration_s = sum(row["frames"] for row in manifest_rows) / 16000
    return {
        "schema": SCORE_SCHEMA,
        "status": "complete_constructed_domain_counts_only",
        "source_hashes": {**{key: expected_hashes[key] for key in HASH_KEYS}, "raw_sha256": raw_hash},
        "scope": "exactly three deterministic constructed-nonspeech streams; one pass each, 900 seconds total; no real-world FAR/product/board qualification",
        "event_time_semantics": "available_audio_s = available_samples / 16000; available input time, not a word endpoint",
        "decoder_frame_semantics": "raw start/end frame IDs on original 10ms grid; no true word endpoints",
        "probabilities": "NOT_CAPTURED",
        "aggregate": {"streams": 3, "duration_s": duration_s, "activation_events": event_count,
                      "event_rate_per_constructed_hour": event_count * 3600 / duration_s,
                      "by_keyword": {str(k): sum(e["keyword"] == k for c in clips for e in c["events"]) for k in (1,2)}},
        "conditional_zero_count_poisson95_upper_per_hour": (-math.log(0.05) * 3600 / duration_s if event_count == 0 else None),
        "poisson_caveat": "If and only if zero events, 11.9829/h is a conditional arithmetic upper bound assuming qualifying negative exposure and independent stationary Poisson counts. Deterministic constructed domains do not establish these assumptions or product low FAR; no independence of real-world sessions claimed.",
        "resource_and_identity_admission": "SEPARATE_SUPERVISOR_AND_INDEPENDENT_AUDIT_REQUIRED; this scorer alone cannot PASS them",
        "observed_totals": {**geometry_totals, "decoder_rows_decoded": total_decoded},
        "model_load_record": model_load_record,
        "streams": clips,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path)
    parser.add_argument("--bindings", required=True, type=Path,
                        help="independently verified expected protocol/model/library/config SHA-256 mapping")
    parser.add_argument("--protocol", type=Path, help="also verify exact protocol file bytes")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        bindings, _ = _load_document(args.bindings, "bindings")
        if args.protocol is not None:
            protocol_hash = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
            _require(bindings.get("protocol_sha256") == protocol_hash, "protocol file/hash mismatch")
        report = score(args.raw, args.manifest, args.geometry, bindings)
        text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        # Never leave a partial report or replace an earlier result after validation failure.
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(args.output)
    except (ValidationError, OSError) as error:
        print("SCORING FAILED: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
