"""Fail-closed, stdlib-only contracts for saved paired decoder evidence.

This module reads local bytes only. It does not import run_corpus, fetch data,
run a decoder, create traces, or admit data to training.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from eval import score_events  # noqa: E402; existing dependency-free event semantics
from speech_asr_consensus import require_independent_reference_labels  # noqa: E402

ARMS = ("strict_children", "single_fuzzy_wo")
MODEL_SHA = "ece44b47bd378c20dd254220b368e41143ec678cbab9dc56901513026ed8d402"
PACK_SHA = "370ee3eeba27b1d62b38f32f53d8302b47c2b762392101e6ca7eb0ccf8dfb723"
FINGERPRINT = 0x316AEA1A94B303BF
HEADER = struct.Struct("<8sIHHIIIIQQ64s8s")
FRAME = struct.Struct("<QII5f")


class ContractError(ValueError):
    pass


def require(condition, message):
    # Never use assert: the guards must survive python -O and -OO.
    if not condition:
        raise ContractError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"missing/unreadable saved input: {path}") from exc
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def parse_json(text, label):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"{label}: duplicate JSON key: {key}")
            result[key] = value
        return result
    def reject_constant(value):
        raise ContractError(f"{label}: non-finite JSON number: {value}")
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label}: invalid JSON") from exc


def load_json(path):
    try:
        return parse_json(Path(path).read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read saved JSON: {path}") from exc


def load_detections(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read saved detections: {path}") from exc
    result = []
    for index, line in enumerate(lines, 1):
        if line.strip():
            event = parse_json(line, f"{path}:{index}")
            require(isinstance(event, dict), "saved detections must be JSON objects")
            result.append(event)
    return result


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def number(value, label, minimum=0.0):
    require(type(value) in (int, float), f"{label}: finite number >= {minimum} required")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ContractError(f"{label}: finite number required") from exc
    require(math.isfinite(result) and result >= minimum, f"{label}: finite number >= {minimum} required")
    return result


def saved_path(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
            "saved file path must be relative")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "saved path escapes supplied root")
    require(path.is_file(), f"missing saved file: {relative}; STOP, no fallback")
    return path


def checked_file(root, record):
    require(isinstance(record, dict) and is_sha(record.get("sha256")), "file SHA256 required")
    path = saved_path(root, record.get("path"))
    require(sha256_file(path) == record["sha256"], f"saved file hash mismatch: {record['path']}")
    if "bytes" in record:
        require(type(record["bytes"]) is int and path.stat().st_size == record["bytes"],
                f"saved file size mismatch: {record['path']}")
    return path


def validate_plan(plan):
    require(isinstance(plan, dict), "plan must be an object")
    require(plan.get("schema") == "kws-near-pronunciation-plan-v1", "unsupported plan")
    require(plan.get("purpose") == "saved_synthetic_mechanism_regression", "current plan is mechanism-only")
    require(plan.get("arms") == list(ARMS), "exact two arm names/order required")
    require(plan.get("thresholds") == [0.55], "current plan permits only fixed threshold 0.55")
    model, pack = plan.get("model", {}), plan.get("keyword_pack", {})
    require(isinstance(model, dict) and isinstance(pack, dict), "model and keyword pack must be objects")
    require(model.get("sha256") == MODEL_SHA and model.get("bytes") == 6812,
            "wrong pinned model")
    require([model.get(k) for k in ("feature_dim", "hidden_dim", "vocab_size")] == [32, 64, 5],
            "wrong F32/H64/V5 model geometry")
    require(model.get("tokens") == ["<blk>", "ni3", "hao3", "xiao3", "wo1"] and
            model.get("vocab_fingerprint") == f"0x{FINGERPRINT:016x}", "wrong vocabulary; wu is absent")
    require(pack.get("sha256") == PACK_SHA and
            pack.get("paths") == {"1": [1, 2, 3, 4], "2": [3, 4, 3, 4]}, "wrong full keyword paths")
    controls = plan.get("shared_controls", {})
    require(controls == {
        "root_nonblank_margin": 0.5, "fuzzy_child_cost_log": -8.25,
        "retention_budget_log": -16.0, "vad": "saved speech_active unchanged",
        "blank": "unchanged production rule", "ctc_repeat": "unchanged production rule",
        "state_retention": "unchanged production default", "refractory": "unchanged production default",
        "initial_decoder_state": "reset at each trace start", "threshold": 0.55},
        "shared decoder controls changed")
    execution = plan.get("execution", {})
    require(isinstance(execution, dict), "execution policy must be an object")
    for key, expected in (("saved_only", True), ("acoustic_forward_allowed", False),
                          ("decoder_execution_by_comparator", False), ("threshold_sweep_allowed", False),
                          ("full_decoder_replay_per_threshold_required", True), ("cache_miss", "STOP")):
        require(execution.get(key) is expected if isinstance(expected, bool) else
                execution.get(key) == expected, f"execution policy changed: {key}")
    return plan


def protected_member_identities():
    summary = load_json(REPO / "research/diagnostic-readiness-2026-10-09/PUBLIC-READINESS-SUMMARY.json")
    return summary["shipping_saved_evidence"]["member_identities"]


def protected_trace_hashes():
    return {entry["sha256"] for name, entry in protected_member_identities().items() if name.endswith(".kwtr")}


def validate_detection_values(events):
    require(isinstance(events, list), "detections must be a list")
    for event in events:
        require(isinstance(event, dict) and isinstance(event.get("recording"), str), "detection object/recording required")
        require(type(event.get("keyword_id")) is int and event["keyword_id"] in (1, 2), "pinned integer keyword ID required")
        number(event.get("time_s"), "detection time")
        require(number(event.get("confidence"), "detection confidence") <= 1.0, "confidence must be <= 1")


def require_independent_truth(value):
    """Reject known weak/forbidden label evidence, including nested truth sources.

    This is a negative authority check, not authentication of a human or source.
    The existing shared guard owns machine-consensus marker semantics.
    """
    if isinstance(value, dict):
        try:
            require_independent_reference_labels([value])
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        for flag in ("evaluation_gold_allowed", "qualification_allowed", "human_gold"):
            require(flag not in value or value[flag] is True,
                    f"independent reference labels forbid false/unknown {flag}")
        require(value.get("source_screen_only") is not True,
                "screen-only source cannot become independent reference labels")
        for nested in value.values():
            require_independent_truth(nested)
    elif isinstance(value, list):
        for nested in value:
            require_independent_truth(nested)


def validate_manifest(manifest):
    require(isinstance(manifest, dict), "manifest must be an object")
    require(manifest.get("schema") == "kws-near-pronunciation-data-v1", "unsupported data manifest")
    require(manifest.get("model_sha256") == MODEL_SHA and
            manifest.get("keyword_pack_sha256") == PACK_SHA, "manifest model/keyword identity mismatch")
    require(manifest.get("purpose") in ("saved_synthetic_mechanism_regression",
            "qualified_event_evaluation", "qualified_event_evaluation_fixture"), "unsupported manifest purpose")
    records = manifest.get("recordings")
    require(isinstance(records, list) and records, "nonempty recordings required")
    require(all(isinstance(row, dict) and isinstance(row.get("trace"), dict) for row in records),
            "recordings and trace descriptors must be objects")
    members = protected_member_identities()
    protected = {entry["sha256"] for path, entry in members.items() if path.endswith(".kwtr")}
    if manifest["purpose"] == "saved_synthetic_mechanism_regression":
        require(len(records) == len(protected) and {row.get("trace", {}).get("sha256") for row in records} == protected,
                "current mechanism manifest must retain the exact 13 protected traces")
        archive = manifest.get("source_archive", {})
        require(isinstance(archive, dict), "source archive must be an object")
        pointer = load_json(REPO / "research/diagnostic-readiness-2026-10-09/DURABLE-TRACE.json")
        require(archive == pointer["original_artifact"] and manifest.get("source_url") == pointer["source_url"],
                "historical archive identity mismatch")
        require(manifest.get("source_member_identity_sha256") == canonical_sha(members), "historical member catalog identity mismatch")
        coverage = manifest.get("coverage")
        require(isinstance(coverage, dict) and coverage.get("ordinary_real_negative_hours") == 0.0 and
                coverage.get("independent_accent_positives") == 0 and coverage.get("independent_farfield_positives") == 0 and
                coverage.get("total_exposure_hours") is None and
                coverage.get("historical_initial_state_equivalence_established") is False,
                "historical coverage cannot become qualified exposure")
        cases = manifest.get("source_cases")
        require(isinstance(cases, list) and len(cases) == 2, "historical source cases required")
        for case, seed, tokens, kind in zip(cases, (1103, 3301), ([3, 3, 3, 4], [3, 4, 3, 1]),
                                          ("synthetic_random_negative", "synthetic_confusable_negative")):
            path = f"build/frozen-far-source-replay/seed-{seed}-{'random-negative' if seed == 1103 else 'kw2-confusable'}-source/result.json"
            require(case == {"seed": seed, "kind": kind, "expected_source_token_ids": tokens,
                    "member": path, "sha256": members[path]["sha256"], "human_event_annotations": False},
                    "historical source-case metadata mismatch")
    names, groups = set(), {key: {} for key in ("source_family", "speaker", "session", "overlap_group", "source_recording_sha256")}
    intervals, qualified_traces = {}, set()
    for row in records:
        require(isinstance(row, dict), "recording must be an object")
        name = row.get("recording")
        require(isinstance(name, str) and name and name not in names, "unique recording IDs required")
        names.add(name)
        trace = row.get("trace", {})
        require(is_sha(trace.get("sha256")), f"{name}: trace hash required")
        for key in ("source_family", "speaker", "session", "initial_state", "overlap_group",
                    "annotation_role", "annotation_status", "expected", "split", "origin",
                    "independent_exposure", "ordinary_negative", "duration_s", "truth_source"):
            require(key in row, f"{name}: missing {key}; unknown must be explicit null")
        state = row["initial_state"]
        require(isinstance(state, dict) and set(state) == {"decoder", "acoustic", "original_nightly_equivalence"},
                f"{name}: initial state contract required")
        require(state["decoder"] == "reset_at_trace_start", f"{name}: paired replay must reset decoder")
        require(row["annotation_role"] in ("diagnostic_only", "screen_only", "gold_dev", "gold_test"),
                f"{name}: training/qualification role not admitted")
        historical = trace["sha256"] in protected or row.get("source_artifact_id") == 11085476672
        if historical:
            require(row["expected"] is None and row["annotation_status"] == "unknown" and
                    row["truth_source"] is None and row["duration_s"] is None and
                    row["speaker"] is None and row["session"] is None,
                    f"{name}: protected historical unknown truth and identity must remain unknown")
            require(trace.get("path") in members and trace["sha256"] == members[trace["path"]]["sha256"]
                    and trace.get("bytes") == members[trace["path"]]["bytes"], f"{name}: original trace path/size/hash mismatch")
            require(row.get("source_run_id") == 36689641752 and row.get("source_artifact_id") == 11085476672,
                    f"{name}: protected historical trace run/artifact identity mismatch")
            require(row["split"] == "historical_diagnostic" and row["annotation_role"] == "diagnostic_only"
                    and row["independent_exposure"] is False and row["origin"] == "synthetic"
                    and row["ordinary_negative"] is False
                    and state["original_nightly_equivalence"] == "unknown",
                    f"{name}: protected historical evidence cannot be repurposed or qualified")
        if row["annotation_role"] == "screen_only":
            require(row["split"] == "screen" and row["expected"] is None and
                    row["independent_exposure"] is False, f"{name}: screen is not gold/train evidence")
        if row["expected"] is None:
            require(row["annotation_status"] == "unknown", f"{name}: missing truth must stay unknown")
        else:
            require(isinstance(row["expected"], list) and row["annotation_status"] == "complete" and
                    isinstance(row["truth_source"], dict) and is_sha(row["truth_source"].get("sha256")),
                    f"{name}: complete event truth with source hash required")
            require(number(row["duration_s"], f"{name}: duration_s") > 0, f"{name}: duration must be positive")
            for event in row["expected"]:
                require(isinstance(event, dict) and type(event.get("keyword_id")) is int and
                        event["keyword_id"] in (1, 2), f"{name}: truth must use the two pinned integer keyword IDs")
                for key in ("start_s", "end_s"):
                    number(event.get(key), f"{name}: {key}")
                for key in ("match_not_before_s", "raw_start_s", "raw_end_s"):
                    if key in event:
                        number(event[key], f"{name}: {key}")
            score_events.validate_recordings([row])
        if row["split"] in ("dev", "test"):
            require_independent_truth(row)
            require(row["annotation_role"] == f"gold_{row['split']}" and row["expected"] is not None,
                    f"{name}: qualified split needs complete gold truth")
            require(row["origin"] == "real_human" and row["independent_exposure"] is True,
                    f"{name}: only independent real-human evidence qualifies")
            require(all(isinstance(state[k], str) and state[k] not in ("", "unknown")
                        for k in ("decoder", "acoustic")), f"{name}: qualified initial state unknown")
            require(trace["sha256"] not in qualified_traces, f"{name}: duplicate qualified trace content hash")
            qualified_traces.add(trace["sha256"])
            require(is_sha(row.get("source_recording_sha256")), f"{name}: stable source-recording content hash required")
            for key, seen in groups.items():
                value = row[key]
                require(isinstance(value, str) and value and value != "unknown", f"{name}: qualified {key} unknown")
                require(value not in seen or seen[value] == row["split"], f"{name}: {key} dev/test group leakage")
                seen[value] = row["split"]
            window = row.get("source_interval_s")
            require(isinstance(window, list) and len(window) == 2, f"{name}: source interval required")
            start, end = (number(value, "source interval") for value in window)
            require(end > start and math.isclose(end - start, row["duration_s"], abs_tol=1e-9),
                    f"{name}: duration/source interval mismatch")
            for old_start, old_end in intervals.setdefault(row["source_recording_sha256"], []):
                require(end <= old_start or start >= old_end, f"{name}: overlapping qualified exposure")
            intervals[row["source_recording_sha256"]].append((start, end))
            if not row["expected"]:
                require(row["ordinary_negative"] is True, f"{name}: ordinary negative qualification required")
        else:
            require(row["split"] in ("historical_diagnostic", "screen"), f"{name}: unsupported split")
    return manifest


def inspect_trace(path, declared):
    """Validate the exact KWTR v1 format, pinned acoustic identity and every frame."""
    with Path(path).open("rb") as stream:
        header = stream.read(HEADER.size)
        require(len(header) == HEADER.size, "short KWTR header")
        magic, version, vocab, frontend, rate, length, hop, reserved, fp, count, model, tail = HEADER.unpack(header)
        require((magic, version, vocab, frontend, rate, length, hop, reserved, fp, model, tail) ==
                (b"KWTRACE1", 1, 5, 0, 16000, 400, 320, 0, FINGERPRINT, MODEL_SHA.encode(), b"\0" * 8),
                "KWTR geometry/model/vocabulary mismatch")
        require(type(declared.get("frames")) is int and count == declared["frames"] and count > 0,
                "KWTR frame count mismatch")
        require(Path(path).stat().st_size == HEADER.size + count * FRAME.size, "KWTR byte count mismatch")
        first = last = None
        for _ in range(count):
            sample, active, zero, *logits = FRAME.unpack(stream.read(FRAME.size))
            require(sample > 0 and active in (0, 1) and zero == 0 and all(math.isfinite(x) for x in logits),
                    "invalid KWTR frame")
            require(last is None or sample - last == 320, "non-contiguous KWTR frame grid")
            if first is None:
                first = sample
            last = sample
        require((first, last) == (declared.get("first_end_sample"), declared.get("last_end_sample")),
                "KWTR timestamp bounds mismatch")
    return {"frames": count, "first_end_sample": first, "last_end_sample": last}


def validate_saved_inputs(plan, manifest, trace_root, repo_root=REPO):
    validate_plan(plan)
    validate_manifest(manifest)
    require(manifest["purpose"] == plan["purpose"], "current input verifier only admits the fixed mechanism plan")
    checked_file(repo_root, plan["model"])
    checked_file(repo_root, plan["keyword_pack"])
    for row in manifest["recordings"]:
        inspect_trace(checked_file(trace_root, row["trace"]), row["trace"])
    return {"status": "PASS_SAVED_INPUT_CONTRACT", "recordings": len(manifest["recordings"]),
            "model_sha256": MODEL_SHA, "keyword_pack_sha256": PACK_SHA,
            "acoustic_forward_calls": 0, "decoder_replay_calls": 0}


def input_binding(plan, manifest):
    """Canonical byte-independent binding of all semantic input and truth fields."""
    return {"plan_canonical_sha256": canonical_sha(plan),
            "data_manifest_canonical_sha256": canonical_sha(manifest),
            "model_sha256": MODEL_SHA, "keyword_pack_sha256": PACK_SHA,
            "shared_controls_sha256": canonical_sha(plan["shared_controls"])}


def validate_arm_receipt(receipt, arm, plan, manifest, results_root, *, reference=False):
    require(isinstance(receipt, dict), "receipt must be an object")
    schema = "kws-saved-production-reference-v1" if reference else "kws-saved-paired-arm-v1"
    require(arm == "production_reference" if reference else arm in ARMS, "unsupported arm")
    if reference:
        require(receipt.get("reference_only") is True and receipt.get("source_kind") == "unmodified_production",
                "production baseline must be an unmodified, fresh reference-only replay")
    require(receipt.get("schema") == schema and receipt.get("arm") == arm,
            "wrong arm receipt schema/name")
    require(receipt.get("threshold") == 0.55, "saved arm threshold must be 0.55")
    require(receipt.get("input_binding") == input_binding(plan, manifest), "two-arm input/denominator binding mismatch")
    require(receipt.get("replay_kind") == "full_decoder_replay" and
            type(receipt.get("acoustic_forward_calls")) is int and receipt["acoustic_forward_calls"] == 0,
            "full saved-trace decoder replay required; no detection filtering or acoustic forward")
    require(is_sha(receipt.get("decoder_executable_sha256")), "research decoder executable identity required")
    replays = receipt.get("replays")
    require(isinstance(replays, list) and all(isinstance(item, dict) for item in replays), "saved replay object list required")
    require([x.get("recording") for x in replays] == [x["recording"] for x in manifest["recordings"]],
            "every manifest trace must be replayed once in manifest order")
    detections = []
    for replay, row in zip(replays, manifest["recordings"]):
        require(replay.get("trace_sha256") == row["trace"]["sha256"] and
                replay.get("initial_state") == row["initial_state"], "replay trace/initial-state mismatch")
        require(type(replay.get("returncode")) is int and replay["returncode"] == 0 and
                replay.get("complete_trace_replay") is True, "incomplete/failed saved replay")
        argv = replay.get("argv")
        expected_tail = [row["recording"]] if reference else [row["recording"], "--child-admission", arm]
        require(isinstance(argv, list) and len(argv) == (5 if reference else 7) and
                all(isinstance(x, str) and x for x in argv) and argv[-len(expected_tail):] == expected_tail,
                "wrong research/reference replay invocation")
        require(replay.get("invocation_input_hashes") == {
            "executable": receipt["decoder_executable_sha256"], "model": MODEL_SHA,
            "keywords": PACK_SHA, "trace": row["trace"]["sha256"]}, "invocation identities mismatch")
        observed = replay.get("observed_invocation_inputs")
        labels = ("executable", "model", "keywords", "trace")
        require(isinstance(observed, list) and len(observed) == 4 and
                all(isinstance(item, dict) for item in observed), "observed invocation path/hash/size descriptors required")
        for item, label, arg in zip(observed, labels, argv[:4]):
            require(item.get("path") == arg and item.get("sha256") == replay["invocation_input_hashes"][label]
                    and type(item.get("bytes")) is int and item["bytes"] > 0,
                    "invocation argv path/identity binding mismatch")
        require(observed[1]["bytes"] == 6812 and observed[3]["bytes"] == row["trace"]["bytes"],
                "invocation model/trace size mismatch")
        path = checked_file(results_root, replay.get("stdout", {}))
        records = load_detections(path)
        validate_detection_values(records)
        maximum = row["trace"]["last_end_sample"] / 16000
        minimum = row["trace"]["first_end_sample"] / 16000
        for event in records:
            require(event.get("recording") == row["recording"] and type(event.get("keyword_id")) is int
                    and event["keyword_id"] in (1, 2), "unexpected recording/keyword in saved replay")
            time = number(event.get("time_s"), "detection time")
            confidence = number(event.get("confidence"), "detection confidence")
            require(minimum - .000000500001 <= time <= maximum + .000000500001 and
                    .55 - .000000500001 <= confidence <= 1.0, "saved detection outside trace/threshold/confidence range")
            frame_index = round((time * 16000 - row["trace"]["first_end_sample"]) / 320)
            grid_time = (row["trace"]["first_end_sample"] + frame_index * 320) / 16000
            require(abs(time - grid_time) <= .000000500001, "saved detection time not on exact trace frame grid")
        detections.extend(records)
    return detections
