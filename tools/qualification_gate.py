#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def finite(value, label: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def integer(value, label: str, minimum: int = 0) -> int:
    result = int(value)
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def validate_policy(policy: dict) -> dict:
    if int(policy.get("schema_version", 0)) != 1:
        raise ValueError("policy schema_version must be 1")
    name = str(policy.get("name", "")).strip()
    if not name:
        raise ValueError("policy name must be non-empty")
    result = {
        "name": name,
        "min_audio_hours": finite(policy["min_audio_hours"], "policy.min_audio_hours", 0.0),
        "min_expected_wakes": integer(
            policy["min_expected_wakes"], "policy.min_expected_wakes", 0
        ),
        "max_frr": finite(policy["max_frr"], "policy.max_frr", 0.0),
        "max_far_per_hour": finite(
            policy["max_far_per_hour"], "policy.max_far_per_hour", 0.0
        ),
        "max_p95_latency_ms": finite(
            policy["max_p95_latency_ms"], "policy.max_p95_latency_ms", 0.0
        ),
        "max_p99_process_us": finite(
            policy["max_p99_process_us"], "policy.max_p99_process_us", 0.0
        ),
        "max_rtf": finite(policy["max_rtf"], "policy.max_rtf", 0.0),
        "min_p99_headroom": finite(
            policy["min_p99_headroom"], "policy.min_p99_headroom", 0.0
        ),
        "min_soak_hours": finite(
            policy["min_soak_hours"], "policy.min_soak_hours", 0.0
        ),
    }
    if result["max_frr"] > 1.0:
        raise ValueError("policy.max_frr must be <= 1")
    return result


def validate_manifest(manifest: dict) -> None:
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("manifest schema_version must be 1")
    if int(manifest["runtime"]["model_abi"]) != 2 or int(
        manifest["runtime"]["keyword_pack_abi"]
    ) != 2:
        raise ValueError("qualification gate requires ABI v2 model and keyword pack")
    fingerprint = str(manifest["vocabulary"]["fingerprint"])
    if not fingerprint.startswith("0x") or len(fingerprint) != 18:
        raise ValueError("manifest vocabulary fingerprint is invalid")
    finite(manifest["evaluation"]["frr"], "manifest.evaluation.frr", 0.0)
    finite(
        manifest["evaluation"]["far_per_hour"],
        "manifest.evaluation.far_per_hour",
        0.0,
    )
    finite(
        manifest["evaluation"]["p95_post_end_latency_ms"],
        "manifest.evaluation.p95_post_end_latency_ms",
        0.0,
    )
    finite(manifest["board"]["p99_process_us"], "manifest.board.p99_process_us", 0.0)
    finite(manifest["board"]["rtf"], "manifest.board.rtf", 0.0)
    finite(
        manifest["board"]["p99_headroom"], "manifest.board.p99_headroom", 0.0
    )
    finite(manifest["evidence"]["soak_hours"], "manifest.evidence.soak_hours", 0.0)
    finite(manifest["evidence"]["cpu_percent"], "manifest.evidence.cpu_percent", 0.0)
    finite(manifest["evidence"]["rss_kib"], "manifest.evidence.rss_kib", 0.0)
    finite(
        manifest["evidence"]["stack_high_water_bytes"],
        "manifest.evidence.stack_high_water_bytes",
        0.0,
    )
    finite(manifest["evidence"]["max_temp_c"], "manifest.evidence.max_temp_c")
    finite(
        manifest["evidence"]["average_power_mw"],
        "manifest.evidence.average_power_mw",
        0.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    policy = validate_policy(load_json(args.policy))
    validate_manifest(manifest)

    evaluation = manifest["evaluation"]
    board = manifest["board"]
    evidence = manifest["evidence"]
    violations: list[str] = []

    if float(evaluation["audio_hours"]) < policy["min_audio_hours"]:
        violations.append("evaluation.audio_hours below minimum")
    if int(evaluation["expected"]) < policy["min_expected_wakes"]:
        violations.append("evaluation.expected below minimum")
    if float(evaluation["frr"]) > policy["max_frr"]:
        violations.append("evaluation.frr above maximum")
    if float(evaluation["far_per_hour"]) > policy["max_far_per_hour"]:
        violations.append("evaluation.far_per_hour above maximum")
    if (
        float(evaluation["p95_post_end_latency_ms"])
        > policy["max_p95_latency_ms"]
    ):
        violations.append("evaluation.p95_post_end_latency_ms above maximum")
    if float(board["p99_process_us"]) > policy["max_p99_process_us"]:
        violations.append("board.p99_process_us above maximum")
    if float(board["rtf"]) > policy["max_rtf"]:
        violations.append("board.rtf above maximum")
    if float(board["p99_headroom"]) < policy["min_p99_headroom"]:
        violations.append("board.p99_headroom below minimum")
    if float(evidence["soak_hours"]) < policy["min_soak_hours"]:
        violations.append("evidence.soak_hours below minimum")

    result = {
        "schema_version": 1,
        "qualified": not violations,
        "policy": policy["name"],
        "source_sha": manifest["source_sha"],
        "vocab_fingerprint": manifest["vocabulary"]["fingerprint"],
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
