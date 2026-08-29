#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
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
FRONTEND_NAMES = {0: "logmel", 1: "pcen-lite"}


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
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
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


def validate_artifact(item, label: str) -> str:
    if not isinstance(item, dict):
        raise ValueError(f"{label} is missing")
    digest = validate_sha(item.get("sha256"), f"{label}.sha256")
    integer(item.get("bytes"), f"{label}.bytes", 1)
    return digest


def validate_manifest(manifest: dict) -> dict:
    if integer(manifest.get("schema_version"), "manifest.schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    runtime = manifest.get("runtime")
    vocabulary = manifest.get("vocabulary")
    lineage = manifest.get("model_lineage")
    evaluation = manifest.get("evaluation")
    board = manifest.get("board")
    evidence = manifest.get("evidence")
    artifacts = manifest.get("artifacts")
    if not all(isinstance(item, dict) for item in (runtime, vocabulary, lineage, evaluation, board, evidence, artifacts)):
        raise ValueError("manifest is missing required object sections")
    if integer(runtime.get("model_abi"), "runtime.model_abi") != 2:
        raise ValueError("qualification gate requires model ABI v2")
    if integer(runtime.get("keyword_pack_abi"), "runtime.keyword_pack_abi") != 3:
        raise ValueError("qualification gate requires keyword-pack ABI v3")
    if (
        integer(runtime.get("sample_rate_hz"), "runtime.sample_rate_hz"),
        integer(runtime.get("frame_length_samples"), "runtime.frame_length_samples"),
        integer(runtime.get("frame_hop_samples"), "runtime.frame_hop_samples"),
    ) != (16000, 400, 320):
        raise ValueError("manifest runtime geometry is unsupported")

    source_sha = manifest.get("source_sha")
    fingerprint = vocabulary.get("fingerprint")
    if not isinstance(source_sha, str) or SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("manifest source_sha is invalid")
    if not isinstance(fingerprint, str) or FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError("manifest vocabulary fingerprint is invalid")

    names = (
        "model", "model_provenance", "model_checkpoint", "training_tokens",
        "keyword_pack", "tokens", "config", "eval_runner", "references",
        "detections", "board_runner", "board_audio",
    )
    hashes = {name: validate_artifact(artifacts.get(name), f"artifacts.{name}") for name in names}
    training_artifacts = artifacts.get("training_manifests")
    if not isinstance(training_artifacts, list) or not training_artifacts:
        raise ValueError("artifacts.training_manifests must be non-empty")
    training_hashes = [validate_artifact(item, f"training_manifests[{i}]") for i, item in enumerate(training_artifacts)]

    if validate_sha(vocabulary.get("sha256"), "vocabulary.sha256") != hashes["tokens"]:
        raise ValueError("manifest vocabulary hash does not match token artifact")
    cross = {
        "provenance_sha256": "model_provenance",
        "model_sha256": "model",
        "tokens_sha256": "tokens",
        "checkpoint_sha256": "model_checkpoint",
        "training_tokens_sha256": "training_tokens",
    }
    for key, artifact_name in cross.items():
        if validate_sha(lineage.get(key), f"model_lineage.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest model-lineage {key} is inconsistent")
    identical = lineage.get("token_bytes_identical_to_training")
    if not isinstance(identical, bool) or identical != (hashes["tokens"] == hashes["training_tokens"]):
        raise ValueError("manifest model-lineage token-byte identity is inconsistent")
    if integer(lineage.get("frontend_spec_version"), "model_lineage.frontend_spec_version") != 2:
        raise ValueError("manifest model-lineage frontend spec is unsupported")
    frontend_kind = integer(lineage.get("frontend_kind"), "model_lineage.frontend_kind")
    frontend_name = lineage.get("frontend_name")
    if frontend_kind not in FRONTEND_NAMES or frontend_name != FRONTEND_NAMES[frontend_kind]:
        raise ValueError("manifest model-lineage frontend identity is invalid")

    training = lineage.get("training")
    quantization = lineage.get("quantization")
    if not isinstance(training, dict) or not isinstance(quantization, dict):
        raise ValueError("manifest model-lineage training/quantization is missing")
    recorded = training.get("manifests")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("manifest model-lineage training manifests are missing")
    recorded_hashes = []
    for index, item in enumerate(recorded):
        if not isinstance(item, dict):
            raise ValueError("manifest model-lineage training manifest must be an object")
        recorded_hashes.append(validate_sha(item.get("sha256"), f"training.manifests[{index}].sha256"))
    if Counter(recorded_hashes) != Counter(training_hashes):
        raise ValueError("manifest model-lineage training-manifest hashes are inconsistent")

    eval_links = {
        "runner_sha256": "eval_runner", "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack", "references_sha256": "references",
        "detections_sha256": "detections",
    }
    for key, artifact_name in eval_links.items():
        if validate_sha(evaluation.get(key), f"evaluation.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest evaluation {key} cross-link is inconsistent")
    board_links = {
        "runner_sha256": "board_runner", "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack", "audio_sha256": "board_audio",
    }
    for key, artifact_name in board_links.items():
        if validate_sha(board.get(key), f"board.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest board {key} cross-link is inconsistent")
    validate_sha(evaluation.get("summary_sha256"), "evaluation.summary_sha256")
    validate_sha(evaluation.get("provenance_sha256"), "evaluation.provenance_sha256")
    validate_sha(board.get("summary_sha256"), "board.summary_sha256")
    validate_sha(evidence.get("sha256"), "evidence.sha256")

    result = {
        "source_sha": source_sha,
        "vocab_fingerprint": fingerprint,
        "model_checkpoint_sha256": hashes["model_checkpoint"],
        "audio_hours": finite(evaluation.get("audio_hours"), "evaluation.audio_hours", 0.0),
        "expected": integer(evaluation.get("expected"), "evaluation.expected", 0),
        "frr": finite(evaluation.get("frr"), "evaluation.frr", 0.0),
        "far_per_hour": finite(evaluation.get("far_per_hour"), "evaluation.far_per_hour", 0.0),
        "p95_latency_ms": finite(evaluation.get("p95_post_end_latency_ms"), "evaluation.p95_post_end_latency_ms", 0.0),
        "p99_process_us": finite(board.get("p99_process_us"), "board.p99_process_us", 0.0),
        "rtf": finite(board.get("rtf"), "board.rtf", 0.0),
        "p99_headroom": finite(board.get("p99_headroom"), "board.p99_headroom", 0.0),
        "soak_hours": finite(evidence.get("soak_hours"), "evidence.soak_hours", 0.0),
        "cpu_percent": finite(evidence.get("cpu_percent"), "evidence.cpu_percent", 0.0),
        "rss_kib": finite(evidence.get("rss_kib"), "evidence.rss_kib", 0.0),
        "stack_high_water_bytes": finite(evidence.get("stack_high_water_bytes"), "evidence.stack_high_water_bytes", 0.0),
        "max_temp_c": finite(evidence.get("max_temp_c"), "evidence.max_temp_c"),
        "average_power_mw": finite(evidence.get("average_power_mw"), "evidence.average_power_mw", 0.0),
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
        "model_checkpoint_sha256": measured["model_checkpoint_sha256"],
        "violations": violations,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if not violations else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
