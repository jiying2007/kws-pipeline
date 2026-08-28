#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
FINGERPRINT_RE = re.compile(r"0x[0-9a-f]{16}")


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def finite(value, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is out of range or non-finite")
    return result


def validate_sha(value, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA256 hex")
    return value


def validate_policy(policy: dict) -> dict:
    if integer(policy.get("schema_version"), "policy.schema_version") != 1:
        raise ValueError("policy schema_version must be 1")
    name = policy.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be non-empty text")
    result = {
        "name": name.strip(),
        "min_audio_hours": finite(policy["min_audio_hours"], "policy.min_audio_hours", 0.0),
        "min_expected_wakes": integer(policy["min_expected_wakes"], "policy.min_expected_wakes", 0),
        "max_frr": finite(policy["max_frr"], "policy.max_frr", 0.0),
        "max_far_per_hour": finite(policy["max_far_per_hour"], "policy.max_far_per_hour", 0.0),
        "max_p95_latency_ms": finite(policy["max_p95_latency_ms"], "policy.max_p95_latency_ms", 0.0),
        "max_p99_process_us": finite(policy["max_p99_process_us"], "policy.max_p99_process_us", 0.0),
        "max_rtf": finite(policy["max_rtf"], "policy.max_rtf", 0.0),
        "min_p99_headroom": finite(policy["min_p99_headroom"], "policy.min_p99_headroom", 0.0),
        "min_soak_hours": finite(policy["min_soak_hours"], "policy.min_soak_hours", 0.0),
        "max_cpu_percent": finite(policy["max_cpu_percent"], "policy.max_cpu_percent", 0.0),
        "max_rss_kib": finite(policy["max_rss_kib"], "policy.max_rss_kib", 0.0),
        "max_stack_high_water_bytes": finite(policy["max_stack_high_water_bytes"], "policy.max_stack_high_water_bytes", 0.0),
        "max_temp_c": finite(policy["max_temp_c"], "policy.max_temp_c"),
        "max_average_power_mw": finite(policy["max_average_power_mw"], "policy.max_average_power_mw", 0.0),
    }
    if result["max_frr"] > 1.0 or result["max_cpu_percent"] > 100.0:
        raise ValueError("policy FRR/CPU limits exceed valid ranges")
    return result


def validate_manifest(manifest: dict) -> dict:
    if integer(manifest.get("schema_version"), "manifest.schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    runtime = manifest.get("runtime")
    vocabulary = manifest.get("vocabulary")
    evaluation = manifest.get("evaluation")
    board = manifest.get("board")
    evidence = manifest.get("evidence")
    artifacts = manifest.get("artifacts")
    if not all(isinstance(item, dict) for item in (runtime, vocabulary, evaluation, board, evidence, artifacts)):
        raise ValueError("manifest is missing required object sections")
    if integer(runtime["model_abi"], "manifest.runtime.model_abi") != 2 or integer(runtime["keyword_pack_abi"], "manifest.runtime.keyword_pack_abi") != 2:
        raise ValueError("qualification gate requires ABI v2 model and keyword pack")
    if (integer(runtime["sample_rate_hz"], "manifest.runtime.sample_rate_hz"), integer(runtime["frame_length_samples"], "manifest.runtime.frame_length_samples"), integer(runtime["frame_hop_samples"], "manifest.runtime.frame_hop_samples")) != (16000, 400, 320):
        raise ValueError("manifest runtime geometry is not ABI-v2")

    source_sha = manifest.get("source_sha")
    fingerprint = vocabulary.get("fingerprint")
    if not isinstance(source_sha, str) or SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("manifest source_sha is invalid")
    if not isinstance(fingerprint, str) or FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError("manifest vocabulary fingerprint is invalid")

    artifact_hashes: dict[str, str] = {}
    for name in ("model", "keyword_pack", "tokens", "config", "eval_runner", "references", "detections", "board_runner", "board_audio"):
        item = artifacts.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"manifest artifact {name} is missing")
        artifact_hashes[name] = validate_sha(item["sha256"], f"manifest.artifacts.{name}.sha256")
        integer(item["bytes"], f"manifest.artifacts.{name}.bytes", 1)
    if validate_sha(vocabulary["sha256"], "manifest.vocabulary.sha256") != artifact_hashes["tokens"]:
        raise ValueError("manifest vocabulary hash does not match token artifact")

    for key in ("summary_sha256", "provenance_sha256"):
        validate_sha(evaluation[key], f"manifest.evaluation.{key}")
    validate_sha(board["summary_sha256"], "manifest.board.summary_sha256")
    validate_sha(evidence["sha256"], "manifest.evidence.sha256")

    eval_links = {
        "runner_sha256": "eval_runner",
        "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack",
        "references_sha256": "references",
        "detections_sha256": "detections",
    }
    for key, artifact_name in eval_links.items():
        if validate_sha(evaluation[key], f"manifest.evaluation.{key}") != artifact_hashes[artifact_name]:
            raise ValueError(f"manifest evaluation {key} cross-link is inconsistent")
    board_links = {
        "runner_sha256": "board_runner",
        "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack",
        "audio_sha256": "board_audio",
    }
    for key, artifact_name in board_links.items():
        if validate_sha(board[key], f"manifest.board.{key}") != artifact_hashes[artifact_name]:
            raise ValueError(f"manifest board {key} cross-link is inconsistent")

    result = {
        "source_sha": source_sha,
        "vocab_fingerprint": fingerprint,
        "audio_hours": finite(evaluation["audio_hours"], "manifest.evaluation.audio_hours", 0.0),
        "expected": integer(evaluation["expected"], "manifest.evaluation.expected", 0),
        "frr": finite(evaluation["frr"], "manifest.evaluation.frr", 0.0),
        "far_per_hour": finite(evaluation["far_per_hour"], "manifest.evaluation.far_per_hour", 0.0),
        "p95_latency_ms": finite(evaluation["p95_post_end_latency_ms"], "manifest.evaluation.p95_post_end_latency_ms", 0.0),
        "p99_process_us": finite(board["p99_process_us"], "manifest.board.p99_process_us", 0.0),
        "rtf": finite(board["rtf"], "manifest.board.rtf", 0.0),
        "p99_headroom": finite(board["p99_headroom"], "manifest.board.p99_headroom", 0.0),
        "soak_hours": finite(evidence["soak_hours"], "manifest.evidence.soak_hours", 0.0),
        "cpu_percent": finite(evidence["cpu_percent"], "manifest.evidence.cpu_percent", 0.0),
        "rss_kib": finite(evidence["rss_kib"], "manifest.evidence.rss_kib", 0.0),
        "stack_high_water_bytes": finite(evidence["stack_high_water_bytes"], "manifest.evidence.stack_high_water_bytes", 0.0),
        "max_temp_c": finite(evidence["max_temp_c"], "manifest.evidence.max_temp_c"),
        "average_power_mw": finite(evidence["average_power_mw"], "manifest.evidence.average_power_mw", 0.0),
    }
    if result["frr"] > 1.0 or result["cpu_percent"] > 100.0:
        raise ValueError("manifest contains impossible FRR/CPU values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    policy = validate_policy(load_json(args.policy))
    measured = validate_manifest(manifest)
    checks = (
        (measured["audio_hours"] < policy["min_audio_hours"], "evaluation.audio_hours below minimum"),
        (measured["expected"] < policy["min_expected_wakes"], "evaluation.expected below minimum"),
        (measured["frr"] > policy["max_frr"], "evaluation.frr above maximum"),
        (measured["far_per_hour"] > policy["max_far_per_hour"], "evaluation.far_per_hour above maximum"),
        (measured["p95_latency_ms"] > policy["max_p95_latency_ms"], "evaluation.p95_post_end_latency_ms above maximum"),
        (measured["p99_process_us"] > policy["max_p99_process_us"], "board.p99_process_us above maximum"),
        (measured["rtf"] > policy["max_rtf"], "board.rtf above maximum"),
        (measured["p99_headroom"] < policy["min_p99_headroom"], "board.p99_headroom below minimum"),
        (measured["soak_hours"] < policy["min_soak_hours"], "evidence.soak_hours below minimum"),
        (measured["cpu_percent"] > policy["max_cpu_percent"], "evidence.cpu_percent above maximum"),
        (measured["rss_kib"] > policy["max_rss_kib"], "evidence.rss_kib above maximum"),
        (measured["stack_high_water_bytes"] > policy["max_stack_high_water_bytes"], "evidence.stack_high_water_bytes above maximum"),
        (measured["max_temp_c"] > policy["max_temp_c"], "evidence.max_temp_c above maximum"),
        (measured["average_power_mw"] > policy["max_average_power_mw"], "evidence.average_power_mw above maximum"),
    )
    violations = [message for failed, message in checks if failed]

    result = {
        "schema_version": 1,
        "qualified": not violations,
        "policy": policy["name"],
        "manifest_sha256": sha256_file(args.manifest),
        "policy_sha256": sha256_file(args.policy),
        "source_sha": measured["source_sha"],
        "vocab_fingerprint": measured["vocab_fingerprint"],
        "violations": violations,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not violations else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
