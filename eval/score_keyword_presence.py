#!/usr/bin/env python3
"""Score explicit per-keyword development labels; never infer event truth.

This consumes label declarations, not ASR text or intended prompts. It does not
verify the declared upstream evidence, admit training data, estimate natural
FAR/FRR, rank models, or manufacture word timestamps. Unknowns stay unscored.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

KEYWORD_IDS = (1, 2)
STATES = ("present", "absent", "unknown")
SHA = re.compile(r"[0-9a-f]{64}\Z")
ARM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
MAX_BYTES = 16 * 1024 * 1024
MAX_RECORDINGS = 10000
MAX_EVENTS = 100000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value, label):
    require(type(value) in (int, float), label + " must be numeric, not boolean/string")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(label + " must be finite") from exc
    require(math.isfinite(result), label + " must be finite")
    return result


def text(value, label):
    require(type(value) is str and 0 < len(value) <= 256
            and value == value.strip() and not any(ord(c) < 32 for c in value),
            label + " must be nonempty canonical text")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_references(rows):
    require(type(rows) is list and 1 <= len(rows) <= MAX_RECORDINGS,
            "bounded nonempty reference list required")
    recordings, identities = {}, set()
    for row in rows:
        require(type(row) is dict and set(row) == {
            "recording", "duration_s", "pcm_sha256", "label_evidence_sha256",
            "split", "keyword_states"}, "keyword reference fields")
        name = text(row["recording"], "recording")
        require(name not in recordings, "duplicate recording")
        duration = finite_number(row["duration_s"], "duration_s")
        require(duration > 0, "duration_s must be positive")
        require(row["split"] == "development", "development declarations only")
        for key in ("pcm_sha256", "label_evidence_sha256"):
            value = row[key]
            require(type(value) is str and SHA.fullmatch(value) is not None
                    and value != "0" * 64, key + " must be a nonzero SHA-256")
        require(row["pcm_sha256"] not in identities,
                "duplicate PCM: repeated conditions must be scored separately")
        states = row["keyword_states"]
        require(type(states) is dict and set(states) == {"1", "2"},
                "both keyword states must be explicit")
        require(all(type(v) is str and v in STATES for v in states.values()),
                "keyword state must be present, absent or unknown")
        recordings[name] = dict(duration_s=duration, states=dict(states))
        identities.add(row["pcm_sha256"])
    return recordings


def score_arm(recordings, events):
    require(type(events) is list and len(events) <= MAX_EVENTS, "bounded event list required")
    counts = Counter()
    for event in events:
        require(type(event) is dict and set(event) == {
            "recording", "keyword_id", "time_s", "confidence"}, "event fields")
        name = text(event["recording"], "event recording")
        require(name in recordings, "unknown event recording")
        kid = event["keyword_id"]
        require(type(kid) is int and kid in KEYWORD_IDS, "keyword_id must be 1 or 2")
        when = finite_number(event["time_s"], "time_s")
        require(0 <= when <= recordings[name]["duration_s"], "event outside recording")
        confidence = finite_number(event["confidence"], "confidence")
        require(0 <= confidence <= 1, "confidence must lie in [0,1]")
        counts[name, kid] += 1
    result = {}
    for kid in KEYWORD_IDS:
        summary = dict(known_positive_clips=0, hit_positive_clips=0,
            missed_positive_clips=0, known_negative_clips=0,
            negative_clips_with_events=0, negative_events=0,
            unknown_clips=0, unknown_events=0, total_events=0,
            repeat_events=0)
        for name, row in recordings.items():
            n = counts[name, kid]
            summary["total_events"] += n
            summary["repeat_events"] += max(0, n - 1)
            state = row["states"][str(kid)]
            if state == "unknown":
                summary["unknown_clips"] += 1
                summary["unknown_events"] += n
            elif state == "present":
                summary["known_positive_clips"] += 1
                summary["hit_positive_clips"] += int(n > 0)
                summary["missed_positive_clips"] += int(n == 0)
            else:
                summary["known_negative_clips"] += 1
                summary["negative_clips_with_events"] += int(n > 0)
                summary["negative_events"] += n
        summary["known_label_coverage"] = (
            summary["known_positive_clips"] + summary["known_negative_clips"]
        ) / len(recordings)
        summary["status"] = "UNSCORED" if summary["known_label_coverage"] == 0 else "DIAGNOSTIC_ONLY"
        result[str(kid)] = summary
    return result


def compare_keyword_presence(references, arm_events):
    """Counts only: label hashes are declarations, never authentication."""
    require(type(arm_events) is dict and 1 <= len(arm_events) <= 8,
            "one to eight explicit model event streams required")
    require(all(type(name) is str and ARM.fullmatch(name) for name in arm_events),
            "invalid model name")
    recordings = validate_references(references)
    arms = {name: score_arm(recordings, arm_events[name]) for name in sorted(arm_events)}
    known = sum(s != "unknown" for row in recordings.values() for s in row["states"].values())
    return {
        "schema": "kws-keyword-presence-diagnostic-v1",
        "status": "UNSCORED" if known == 0 else "DIAGNOSTIC_ONLY",
        "evidence_scope": "declared-observed-development-keyword-labels",
        "recordings": len(recordings), "known_keyword_labels": known,
        "unknown_keyword_labels": 2 * len(recordings) - known,
        "references_sha256": digest(references),
        "event_rows_sha256": {name: digest(arm_events[name]) for name in sorted(arm_events)},
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arms": arms, "winner": None,
        "upstream_evidence_authenticated": False,
        "acoustic_labels_verified": False,
        "training_admitted": False, "independent_evaluation": False,
        "qualification_allowed": False, "event_annotations_available": False,
        "natural_far_available": False, "accepted_negative_seconds": 0,
        "context_comparability_verified": False,
        "note": "Counts consume explicit labels only; no ASR inference, automatic selection, "
                "FAR/FRR, word timestamps or latency. Repeat events are not independent samples.",
    }


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key: " + key)
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def load_jsonl(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, "JSONL input exceeds byte budget")
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line, object_pairs_hook=unique_pairs,
                                   parse_constant=reject_constant))
            require(len(rows) <= limit, "JSONL input exceeds row budget")
    return rows, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--events", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        references, ref_sha = load_jsonl(args.references, MAX_RECORDINGS)
        arms, input_sha = {}, {}
        for value in args.events:
            name, separator, path = value.partition("=")
            require(separator == "=" and path and name not in arms,
                    "events require unique MODEL=PATH bindings")
            arms[name], input_sha[name] = load_jsonl(Path(path), MAX_EVENTS)
        result = compare_keyword_presence(references, arms)
        result["input_file_sha256"] = dict(references=ref_sha, events=input_sha)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, OSError, UnicodeError) as exc:
        parser.exit(2, "keyword-presence: " + str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
