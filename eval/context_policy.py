"""Saved-input context checks for score_events; no execution or attestation.

Matching means complete, equal declarations bound to the scorer's input bytes.
It cannot prove the declarations describe an actual execution or qualify audio.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re


POLICY_FIELDS = {
    "sample_rate_hz", "feed_chunk_samples", "appended_context_samples",
    "appended_context_kind", "reset_frontend", "reset_model", "reset_decoder",
    "reset_clocks", "eof_partial_chunk", "eof_flush", "eof_padding",
}


def require(condition, message):
    if not condition:
        raise ValueError("context: " + message)


def sha256_file(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def valid_sha(value):
    return (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            and value != "0" * 64)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def policy_unknowns(policy):
    """Missing and explicitly unknown values stay unknown; never fill defaults."""
    if policy is None:
        return sorted(POLICY_FIELDS)
    require(isinstance(policy, dict), "policy must be an object")
    require(set(policy) <= POLICY_FIELDS, "unrecognized policy field")
    unknown = [key for key in POLICY_FIELDS if policy.get(key) in (None, "unknown")]
    for key, value in policy.items():
        if key in unknown:
            continue
        if key in {"sample_rate_hz", "feed_chunk_samples", "appended_context_samples"}:
            require(type(value) is int and value >= (0 if key == "appended_context_samples" else 1),
                    key + " must be an explicit integer in range")
        elif key.startswith("reset_"):
            require(value in ("per-recording", "continuous"), "unsupported " + key)
        elif key == "eof_flush":
            require(type(value) is bool, "eof_flush must be boolean")
        else:
            choices = {
                "appended_context_kind": {"none", "digital-zero"},
                "eof_partial_chunk": {"process-retained", "drop"},
                "eof_padding": {"none", "zero", "replicate-right"},
            }
            require(value in choices[key], "unsupported " + key)
    if not {"appended_context_kind", "appended_context_samples"} & set(unknown):
        require((policy["appended_context_kind"] == "none") ==
                (policy["appended_context_samples"] == 0), "inconsistent appended context")
    return sorted(unknown)


def assess_context(path, references_path, detections_path, rows, *, clip_presence=False):
    report = {
        "status": "CONTEXT_UNVERIFIED", "matched_context": False,
        "scope": "saved declarations and input bindings only; no execution attestation",
        "qualification_allowed": False, "unknown": [], "differences": [],
    }
    if path is None:
        report["unknown"] = ["context manifest missing"]
        return report
    manifest = json.loads(pathlib.Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object)
    require(isinstance(manifest, dict) and manifest.get("schema") == "eval-context-v1",
            "unsupported manifest schema")
    for key, input_path in (("references_sha256", references_path),
                            ("detections_sha256", detections_path)):
        require(manifest.get(key) == sha256_file(input_path), key + " binding mismatch")
    entries = manifest.get("recordings")
    require(isinstance(entries, dict), "recordings must be an object")
    require(set(entries) == {row["recording"] for row in rows}, "recording set mismatch")
    policies, roles = {}, set()
    for row in rows:
        name = row["recording"]
        entry = entries[name]
        require(isinstance(entry, dict), name + ": entry must be an object")
        expected = row["expected_keywords" if clip_presence else "expected"]
        role = "positive" if expected else "negative"
        roles.add(role)
        require(entry.get("role") == role, name + ": positive/negative role mismatch")
        audio = row.get("audio_sha256")
        require(valid_sha(audio) and entry.get("audio_sha256") == audio,
                name + ": audio identity binding mismatch")
        missing = policy_unknowns(entry.get("policy"))
        report["unknown"].extend(name + ": " + key for key in missing)
        policies[name] = entry.get("policy") or {}
    if roles != {"positive", "negative"}:
        report["unknown"].append("both positive and negative recordings required")
    for key in sorted(POLICY_FIELDS):
        observed = {json.dumps(policy[key], sort_keys=True) for policy in policies.values()
                    if policy.get(key) not in (None, "unknown")}
        if len(observed) > 1:
            report["differences"].append(key)
    report["manifest_sha256"] = sha256_file(path)
    report["recordings"] = entries
    if report["differences"]:
        report["status"] = "CONTEXT_MISMATCH"
    elif not report["unknown"]:
        report["status"] = "MATCHED_DECLARED_CONTEXT"
        report["matched_context"] = True
    return report
