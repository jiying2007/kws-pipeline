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

from corpus_identity import corpus_digest
from statistical_bounds import qualification_bounds

SHA256_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
FINGERPRINT_RE = re.compile(r"0x[0-9a-f]{16}")
FRONTEND_NAMES = {0: "logmel", 1: "pcen-lite"}
REQUIRED_HUMAN_IDENTITY = {"speaker_id", "session_id", "source_id"}


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
    if integer(policy.get("schema_version"), "policy.schema_version") != 2:
        raise ValueError("policy schema_version must be 2")
    name = policy.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("policy name must be non-empty text")
    policy_id = policy.get("policy_id")
    sku = policy.get("sku")
    shipping_approved = policy.get("shipping_approved")
    if not isinstance(policy_id, str) or not policy_id.strip():
        raise ValueError("policy_id must be non-empty text")
    if not isinstance(sku, str) or not sku.strip():
        raise ValueError("policy sku must be non-empty text")
    if shipping_approved is not True:
        raise ValueError("qualification gate requires shipping_approved=true")
    result = {
        "policy_id": policy_id.strip(),
        "sku": sku.strip(),
        "shipping_approved": True,
        "name": name.strip(),
        "confidence_level": finite(policy["confidence_level"], "policy.confidence_level"),
        "min_audio_hours": finite(policy["min_audio_hours"], "policy.min_audio_hours", 0.0),
        "min_expected_wakes": integer(policy["min_expected_wakes"], "policy.min_expected_wakes", 1),
        "max_frr": finite(policy["max_frr"], "policy.max_frr", 0.0),
        "max_frr_upper_bound": finite(policy["max_frr_upper_bound"], "policy.max_frr_upper_bound", 0.0),
        "max_far_per_hour": finite(policy["max_far_per_hour"], "policy.max_far_per_hour", 0.0),
        "max_far_upper_bound_per_hour": finite(policy["max_far_upper_bound_per_hour"], "policy.max_far_upper_bound_per_hour", 0.0),
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
    if not 0.5 < result["confidence_level"] < 1.0:
        raise ValueError("policy confidence_level must be in (0.5,1)")
    if result["max_frr"] > 1.0 or result["max_frr_upper_bound"] > 1.0 or result["max_cpu_percent"] > 100.0:
        raise ValueError("policy FRR/CPU limits exceed valid ranges")
    if result["max_frr_upper_bound"] < result["max_frr"]:
        raise ValueError("statistical FRR upper-bound limit cannot be below point limit")
    if result["max_far_upper_bound_per_hour"] < result["max_far_per_hour"]:
        raise ValueError("statistical FAR upper-bound limit cannot be below point limit")
    return result


def validate_artifact(item, label: str) -> str:
    if not isinstance(item, dict):
        raise ValueError(f"{label} is missing")
    digest = validate_sha(item.get("sha256"), f"{label}.sha256")
    integer(item.get("bytes"), f"{label}.bytes", 1)
    return digest


def validate_corpus(value: object, label: str) -> str:
    if not isinstance(value, dict) or integer(value.get("schema_version"), f"{label}.schema_version") != 1:
        raise ValueError(f"{label} must be schema_version 1")
    rows = value.get("recordings")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label}.recordings must be non-empty")
    digest = validate_sha(value.get("corpus_sha256"), f"{label}.corpus_sha256")
    if corpus_digest(rows) != digest:
        raise ValueError(f"{label} canonical digest is inconsistent")
    return digest


def validate_manifest(manifest: dict) -> dict:
    if integer(manifest.get("schema_version"), "manifest.schema_version") != 2:
        raise ValueError("manifest schema_version must be 2")
    runtime = manifest.get("runtime")
    vocabulary = manifest.get("vocabulary")
    lineage = manifest.get("model_lineage")
    evaluation = manifest.get("evaluation")
    board = manifest.get("board")
    evidence = manifest.get("evidence")
    artifacts = manifest.get("artifacts")
    dataset_audit = manifest.get("dataset_audit")
    if not all(isinstance(item, dict) for item in (runtime, vocabulary, lineage, evaluation, board, evidence, artifacts, dataset_audit)):
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
    sku = manifest.get("sku")
    fingerprint = vocabulary.get("fingerprint")
    if not isinstance(source_sha, str) or SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("manifest source_sha is invalid")
    if not isinstance(sku, str) or not sku.strip():
        raise ValueError("manifest sku is invalid")
    if not isinstance(fingerprint, str) or FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError("manifest vocabulary fingerprint is invalid")

    names = (
        "model",
        "model_provenance",
        "model_checkpoint",
        "training_tokens",
        "dataset_audit",
        "keyword_pack",
        "tokens",
        "config",
        "eval_runner",
        "references",
        "detections",
        "board_runner",
        "board_audio",
        "evidence_collector",
    )
    hashes = {name: validate_artifact(artifacts.get(name), f"artifacts.{name}") for name in names}
    training_artifacts = artifacts.get("training_manifests")
    raw_artifacts = artifacts.get("raw_evidence")
    if not isinstance(training_artifacts, list) or not training_artifacts:
        raise ValueError("artifacts.training_manifests must be non-empty")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("artifacts.raw_evidence must be non-empty")
    training_hashes = [validate_artifact(item, f"training_manifests[{i}]") for i, item in enumerate(training_artifacts)]
    raw_hashes = [validate_artifact(item, f"raw_evidence[{i}]") for i, item in enumerate(raw_artifacts)]

    if validate_sha(vocabulary.get("sha256"), "vocabulary.sha256") != hashes["tokens"]:
        raise ValueError("manifest vocabulary hash does not match token artifact")
    if validate_sha(dataset_audit.get("sha256"), "dataset_audit.sha256") != hashes["dataset_audit"]:
        raise ValueError("dataset audit cross-link is inconsistent")
    required_metadata = dataset_audit.get("required_metadata")
    if not isinstance(required_metadata, list) or not REQUIRED_HUMAN_IDENTITY.issubset(set(required_metadata)):
        raise ValueError("dataset audit does not enforce human identity isolation")

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
    runtime_frontend_kind = integer(runtime.get("frontend_kind"), "runtime.frontend_kind")
    runtime_frontend_name = runtime.get("frontend_name")
    if runtime_frontend_kind != frontend_kind or runtime_frontend_name != frontend_name or runtime_frontend_name != FRONTEND_NAMES.get(runtime_frontend_kind):
        raise ValueError("manifest runtime frontend does not match model lineage")

    training = lineage.get("training")
    quantization = lineage.get("quantization")
    if not isinstance(training, dict) or not isinstance(quantization, dict):
        raise ValueError("manifest model-lineage training/quantization is missing")
    recorded = training.get("manifests")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError("manifest model-lineage training manifests are missing")
    recorded_hashes = [validate_sha(item.get("sha256"), f"training.manifests[{i}].sha256") for i, item in enumerate(recorded) if isinstance(item, dict)]
    if len(recorded_hashes) != len(recorded) or Counter(recorded_hashes) != Counter(training_hashes):
        raise ValueError("manifest model-lineage training-manifest hashes are inconsistent")
    training_corpus_sha = validate_corpus(training.get("corpus_identity"), "model_lineage.training.corpus_identity")
    if validate_sha(lineage.get("training_corpus_sha256"), "model_lineage.training_corpus_sha256") != training_corpus_sha:
        raise ValueError("model-lineage training corpus cross-link is inconsistent")

    eval_links = {
        "runner_sha256": "eval_runner",
        "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack",
        "references_sha256": "references",
        "detections_sha256": "detections",
    }
    for key, artifact_name in eval_links.items():
        if validate_sha(evaluation.get(key), f"evaluation.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest evaluation {key} cross-link is inconsistent")
    eval_rows = evaluation.get("audio_files")
    if not isinstance(eval_rows, list) or not eval_rows:
        raise ValueError("manifest evaluation audio_files are missing")
    eval_corpus_sha = validate_sha(evaluation.get("audio_corpus_sha256"), "evaluation.audio_corpus_sha256")
    if corpus_digest(eval_rows) != eval_corpus_sha:
        raise ValueError("manifest evaluation audio corpus digest is inconsistent")

    board_links = {
        "runner_sha256": "board_runner",
        "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack",
        "audio_sha256": "board_audio",
    }
    for key, artifact_name in board_links.items():
        if validate_sha(board.get(key), f"board.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest board {key} cross-link is inconsistent")
    if validate_sha(evidence.get("collector_sha256"), "evidence.collector_sha256") != hashes["evidence_collector"]:
        raise ValueError("manifest evidence collector cross-link is inconsistent")
    declared_raw = evidence.get("raw_evidence")
    if not isinstance(declared_raw, list) or not declared_raw:
        raise ValueError("manifest evidence raw_evidence is missing")
    declared_raw_hashes = [validate_sha(item.get("sha256"), f"evidence.raw_evidence[{i}].sha256") for i, item in enumerate(declared_raw) if isinstance(item, dict)]
    if len(declared_raw_hashes) != len(declared_raw) or Counter(declared_raw_hashes) != Counter(raw_hashes):
        raise ValueError("manifest raw evidence artifacts are inconsistent")

    validate_sha(evaluation.get("summary_sha256"), "evaluation.summary_sha256")
    validate_sha(evaluation.get("provenance_sha256"), "evaluation.provenance_sha256")
    validate_sha(board.get("summary_sha256"), "board.summary_sha256")
    if validate_sha(evidence.get("sha256"), "evidence.sha256") != hashes["evidence"]:
        raise ValueError("manifest evidence hash does not match evidence artifact")
    if integer(evidence.get("schema_version"), "evidence.schema_version") != 2:
        raise ValueError("manifest evidence schema_version must be 2")
    if evidence.get("evidence_class") != "product-board":
        raise ValueError("manifest evidence_class must be product-board")
    if evidence.get("sku") != sku or evidence.get("source_sha") != source_sha:
        raise ValueError("manifest evidence identity does not match SKU/source")
    builder_id = evidence.get("builder_id")
    dut_id = evidence.get("dut_id")
    collector_id = evidence.get("collector_id")
    if not all(isinstance(value, str) and value.strip()
               for value in (builder_id, dut_id, collector_id)):
        raise ValueError("manifest evidence runner identities are invalid")
    if builder_id == dut_id:
        raise ValueError("manifest evidence builder and DUT must be distinct")
    collected_at = evidence.get("collected_at_utc")
    if not isinstance(collected_at, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", collected_at
    ) is None:
        raise ValueError("manifest evidence timestamp is invalid")
    evidence_links = {
        "collector_sha256": "collector",
        "raw_evidence_sha256": "evidence_raw",
        "attestation_verification_sha256": "attestation_verification",
        "board_runner_sha256": "board_runner",
        "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack",
        "board_audio_sha256": "board_audio",
    }
    for key, artifact_name in evidence_links.items():
        if validate_sha(evidence.get(key), f"evidence.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest evidence {key} cross-link is inconsistent")
    attestation = evidence.get("attestation")
    if not isinstance(attestation, dict) or attestation.get("verified") is not True or \
       attestation.get("subject_kind") != "kws-target-evidence":
        raise ValueError("manifest attestation verification is invalid")
    if not all(isinstance(attestation.get(key), str) and attestation.get(key).strip()
               for key in ("issuer", "trust_policy", "verified_at_utc")):
        raise ValueError("manifest attestation trust identity is incomplete")
    attestation_links = {
        "subject_sha256": "evidence_raw",
        "collector_sha256": "collector",
        "board_runner_sha256": "board_runner",
        "model_sha256": "model",
        "keyword_pack_sha256": "keyword_pack",
    }
    for key, artifact_name in attestation_links.items():
        if validate_sha(attestation.get(key), f"attestation.{key}") != hashes[artifact_name]:
            raise ValueError(f"manifest attestation {key} cross-link is inconsistent")

    expected = integer(evaluation.get("expected"), "evaluation.expected", 1)
    matched = integer(evaluation.get("matched"), "evaluation.matched", 0)
    false_rejects = integer(evaluation.get("false_rejects"), "evaluation.false_rejects", 0)
    false_accepts = integer(evaluation.get("false_accepts"), "evaluation.false_accepts", 0)
    if matched + false_rejects != expected:
        raise ValueError("manifest evaluation matched/false-reject counts disagree")

    result = {
        "sku": sku.strip(),
        "source_sha": source_sha,
        "vocab_fingerprint": fingerprint,
        "model_checkpoint_sha256": hashes["model_checkpoint"],
        "training_corpus_sha256": training_corpus_sha,
        "evaluation_corpus_sha256": eval_corpus_sha,
        "frontend_kind": frontend_kind,
        "frontend_name": frontend_name,
        "audio_hours": finite(evaluation.get("audio_hours"), "evaluation.audio_hours", 0.0),
        "expected": expected,
        "matched": matched,
        "false_rejects": false_rejects,
        "false_accepts": false_accepts,
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
    if result["audio_hours"] <= 0.0 or result["frr"] > 1.0 or result["cpu_percent"] > 100.0:
        raise ValueError("manifest contains impossible evaluation/resource values")
    expected_frr = false_rejects / expected
    expected_far = false_accepts / result["audio_hours"]
    if not math.isclose(result["frr"], expected_frr, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("manifest evaluation FRR disagrees with counts")
    if not math.isclose(result["far_per_hour"], expected_far, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("manifest evaluation FAR disagrees with count/exposure")
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
    if measured["sku"] != policy["sku"]:
        raise ValueError("manifest SKU does not match shipping policy SKU")
    statistics = qualification_bounds(
        false_rejects=measured["false_rejects"],
        expected_wakes=measured["expected"],
        false_accepts=measured["false_accepts"],
        audio_hours=measured["audio_hours"],
        confidence=policy["confidence_level"],
    )
    checks = (
        (measured["audio_hours"] < policy["min_audio_hours"], "evaluation.audio_hours below minimum"),
        (measured["expected"] < policy["min_expected_wakes"], "evaluation.expected below minimum"),
        (measured["frr"] > policy["max_frr"], "evaluation.frr above maximum"),
        (statistics["frr_upper_bound"] > policy["max_frr_upper_bound"], "evaluation.frr statistical upper bound above maximum"),
        (measured["far_per_hour"] > policy["max_far_per_hour"], "evaluation.far_per_hour above maximum"),
        (statistics["far_upper_bound_per_hour"] > policy["max_far_upper_bound_per_hour"], "evaluation.far statistical upper bound above maximum"),
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
        "schema_version": 3,
        "qualified": not violations,
        "policy": policy["name"],
        "policy_id": policy["policy_id"],
        "sku": policy["sku"],
        "qualification_level": "product-certified",
        "manifest_sha256": sha256_file(args.manifest),
        "policy_sha256": sha256_file(args.policy),
        "source_sha": measured["source_sha"],
        "vocab_fingerprint": measured["vocab_fingerprint"],
        "model_checkpoint_sha256": measured["model_checkpoint_sha256"],
        "training_corpus_sha256": measured["training_corpus_sha256"],
        "evaluation_corpus_sha256": measured["evaluation_corpus_sha256"],
        "frontend_kind": measured["frontend_kind"],
        "frontend_name": measured["frontend_name"],
        "statistics": statistics,
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
