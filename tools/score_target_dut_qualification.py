#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys

from qualification_metrics import validate_board, validate_evidence

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    return rows


def require_bundle_file(root: pathlib.Path, relative: str) -> pathlib.Path:
    base = root.resolve(strict=True)
    path = (base / relative).resolve(strict=True)
    if path != base and base not in path.parents:
        raise ValueError(f"bundle path escapes root: {relative}")
    if not path.is_file():
        raise ValueError(f"bundle file is missing: {relative}")
    return path


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA256 hex")
    return value


def require_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def require_number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"{label} is out of range")
    return result


def validate_resource_budget(value: dict, required_fields: list[str]) -> dict:
    if value.get("schema_version") != 1 or value.get("status") != "approved":
        raise ValueError("resource budget must be schema_version 1 and status=approved")
    budget_id = require_text(value.get("budget_id"), "resource budget budget_id")
    sku = require_text(value.get("sku"), "resource budget sku")
    board_revision = require_text(
        value.get("board_revision"), "resource budget board_revision"
    )
    measurement_contract_id = require_text(
        value.get("measurement_contract_id"),
        "resource budget measurement_contract_id",
    )
    authority = require_text(value.get("authority"), "resource budget authority")
    approved_at = require_text(
        value.get("approved_at_utc"), "resource budget approved_at_utc"
    )
    if not approved_at.endswith("Z"):
        raise ValueError("resource budget approved_at_utc must be UTC")
    limits = value.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("resource budget limits must be an object")
    missing = [field for field in required_fields if field not in limits]
    if missing:
        raise ValueError(f"resource budget is missing limits: {missing}")
    normalized = {
        "max_cpu_percent": require_number(
            limits["max_cpu_percent"], "resource budget max_cpu_percent", positive=True
        ),
        "max_rss_kib": require_number(
            limits["max_rss_kib"], "resource budget max_rss_kib", positive=True
        ),
        "max_stack_high_water_bytes": require_number(
            limits["max_stack_high_water_bytes"],
            "resource budget max_stack_high_water_bytes",
            positive=True,
        ),
        "max_temp_c": require_number(
            limits["max_temp_c"], "resource budget max_temp_c"
        ),
        "max_average_power_mw": require_number(
            limits["max_average_power_mw"],
            "resource budget max_average_power_mw",
            positive=True,
        ),
    }
    if normalized["max_cpu_percent"] > 100.0:
        raise ValueError("resource budget max_cpu_percent must be <=100")
    return {
        "budget_id": budget_id,
        "sku": sku,
        "board_revision": board_revision,
        "measurement_contract_id": measurement_contract_id,
        "authority": authority,
        "approved_at_utc": approved_at,
        "limits": normalized,
    }


def verify_external_attestation(
    attestation: dict,
    *,
    evidence_raw_sha256: str,
    collector_sha256: str,
    board_runner_sha256: str,
    model_sha256: str,
    keyword_pack_sha256: str,
    final_afe_identity_sha256: str,
    resource_budget_sha256: str,
) -> dict:
    if attestation.get("schema_version") != 1:
        raise ValueError("attestation verification schema_version must be 1")
    if attestation.get("verified") is not True:
        raise ValueError("attestation verification must report verified=true")
    if attestation.get("subject_kind") != "kws-target-evidence":
        raise ValueError("attestation subject_kind must be kws-target-evidence")
    issuer = require_text(attestation.get("issuer"), "attestation.issuer")
    trust_policy = require_text(attestation.get("trust_policy"), "attestation.trust_policy")
    verified_at = require_text(attestation.get("verified_at_utc"), "attestation.verified_at_utc")
    if not verified_at.endswith("Z"):
        raise ValueError("attestation verified_at_utc must be UTC")
    expected = {
        "subject_sha256": evidence_raw_sha256,
        "collector_sha256": collector_sha256,
        "board_runner_sha256": board_runner_sha256,
        "model_sha256": model_sha256,
        "keyword_pack_sha256": keyword_pack_sha256,
        "audio_frontend_identity_sha256": final_afe_identity_sha256,
        "resource_budget_sha256": resource_budget_sha256,
    }
    for field, digest in expected.items():
        measured = require_sha256(attestation.get(field), f"attestation.{field}")
        if measured != digest:
            raise ValueError(f"attestation {field} does not match selected qualification tuple")
    return {
        "issuer": issuer,
        "trust_policy": trust_policy,
        "verified_at_utc": verified_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--phase-a-receipt", required=True, type=pathlib.Path)
    parser.add_argument("--phase-a-summary", required=True, type=pathlib.Path)
    parser.add_argument("--deployment-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    bundle = args.bundle.resolve(strict=True)
    profile = load_json(require_bundle_file(bundle, "target-profile.json"))
    policy = load_json(args.policy)
    phase_a_receipt = load_json(args.phase_a_receipt)
    phase_a = load_json(args.phase_a_summary)
    deployment = load_json(args.deployment_manifest)
    target_raw = load_json(require_bundle_file(bundle, "target-evidence.json"))
    board_raw = load_json(require_bundle_file(bundle, "board-summary.json"))

    if profile.get("schema_version") != 1:
        raise ValueError("target profile schema_version must be 1")
    if policy.get("schema_version") != 2:
        raise ValueError("target policy schema_version must be 2")
    deployment_tag = str(policy["deployment_tag"])
    human_tag = str(profile.get("human_qualification_tag", ""))
    if profile.get("deployment_tag") != deployment_tag:
        raise ValueError("target profile deployment differs from target policy")
    if profile.get("board_audio_class") != policy["board_audio_policy"]["required_class"]:
        raise ValueError("board benchmark audio is not declared non-human-public-safe")
    if phase_a_receipt.get("qualified") is not True or phase_a_receipt.get("shipping_approved") is not False:
        raise ValueError("Phase-A receipt must be qualified with shipping_approved=false")
    if phase_a.get("qualified") is not True or phase_a.get("shipping_approved") is not False:
        raise ValueError("Phase-A summary must be qualified with shipping_approved=false")
    if phase_a_receipt.get("deployment_tag") != deployment_tag or phase_a.get("deployment_tag") != deployment_tag:
        raise ValueError("Phase-A evidence is not bound to target deployment")
    if human_tag != str(phase_a_receipt.get("release_tag", human_tag)) and phase_a_receipt.get("release_tag") is not None:
        raise ValueError("target profile Phase-A tag differs from Phase-A receipt")
    if deployment.get("deployment_tag") != deployment_tag or deployment.get("status") != "commercial-candidate":
        raise ValueError("deployment manifest identity/status is invalid")
    if deployment.get("shipping_approved") is not False:
        raise ValueError("target qualification requires pre-shipping deployment")

    source_sha = str(deployment["source_sha"])
    deployment_target = str(phase_a_receipt.get("deployment_target", ""))
    if deployment_target != source_sha:
        raise ValueError("Phase-A receipt deployment target differs from deployment manifest")
    if str(target_raw.get("source_sha", "")) != source_sha:
        raise ValueError("target evidence source differs from frozen deployment")

    phase_a_afe = phase_a.get("afe")
    if not isinstance(phase_a_afe, dict):
        raise ValueError("Phase-A summary is missing final AFE identity")
    final_afe_identity = require_sha256(
        phase_a_receipt.get("final_afe_identity_sha256"),
        "Phase-A receipt final_afe_identity_sha256",
    )
    if final_afe_identity != require_sha256(
        phase_a_afe.get("identity_sha256"), "Phase-A summary AFE identity_sha256"
    ):
        raise ValueError("Phase-A receipt/final-AFE identity mismatch")
    if require_sha256(
        target_raw.get("audio_frontend_identity_sha256"),
        "target evidence audio_frontend_identity_sha256",
    ) != final_afe_identity:
        raise ValueError("physical target evidence uses a different full final-AFE identity")
    if require_sha256(
        target_raw.get("audio_frontend_sha256"),
        "target evidence audio_frontend_sha256",
    ) != require_sha256(
        phase_a_afe.get("executable_sha256"), "Phase-A summary AFE executable_sha256"
    ):
        raise ValueError("physical target evidence uses a different final-AFE executable")
    if str(target_raw.get("sku", "")) != str(phase_a_afe.get("sku", "")):
        raise ValueError("physical target SKU differs from Phase-A final-AFE SKU")

    budget_cfg = policy.get("resource_budget")
    if not isinstance(budget_cfg, dict) or budget_cfg.get("required") is not True:
        raise ValueError("target policy must require an approved resource budget")
    budget_name = str(budget_cfg.get("fixed_bundle_file", ""))
    if pathlib.Path(budget_name).name != budget_name or not budget_name:
        raise ValueError("target policy resource budget bundle file is invalid")
    budget_path = require_bundle_file(bundle, budget_name)
    budget_sha256 = sha256_file(budget_path)
    budget = validate_resource_budget(
        load_json(budget_path),
        [str(item) for item in budget_cfg.get("required_limit_fields", [])],
    )
    if str(profile.get("resource_budget_id", "")) != budget["budget_id"]:
        raise ValueError("target profile resource budget ID differs from approved budget")
    if budget["sku"] != str(phase_a_afe["sku"]):
        raise ValueError("approved resource budget SKU differs from Phase-A SKU")
    if require_sha256(
        target_raw.get("resource_budget_sha256"),
        "target evidence resource_budget_sha256",
    ) != budget_sha256:
        raise ValueError("target evidence resource budget hash differs from selected budget")

    board_runner = require_bundle_file(bundle, "board-runner")
    board_audio = require_bundle_file(bundle, "board-audio.wav")
    evidence_raw = require_bundle_file(bundle, "evidence-raw.jsonl")
    attestation_path = require_bundle_file(bundle, "attestation-verification.json")
    model = args.model.resolve(strict=True)
    keywords = args.keywords.resolve(strict=True)
    collector = ROOT / "tools" / "collect_target_evidence.py"

    board_runner_sha256 = sha256_file(board_runner)
    model_sha256 = sha256_file(model)
    keyword_pack_sha256 = sha256_file(keywords)
    evidence_raw_sha256 = sha256_file(evidence_raw)
    external_attestation = verify_external_attestation(
        load_json(attestation_path),
        evidence_raw_sha256=evidence_raw_sha256,
        collector_sha256=sha256_file(collector),
        board_runner_sha256=board_runner_sha256,
        model_sha256=model_sha256,
        keyword_pack_sha256=keyword_pack_sha256,
        final_afe_identity_sha256=final_afe_identity,
        resource_budget_sha256=budget_sha256,
    )

    board = validate_board(
        board_raw,
        model.stat().st_size,
        keywords.stat().st_size,
        source_sha,
        {
            "runner_sha256": board_runner_sha256,
            "model_sha256": model_sha256,
            "keyword_pack_sha256": keyword_pack_sha256,
            "audio_sha256": sha256_file(board_audio),
        },
    )
    evidence = validate_evidence(
        target_raw,
        sku=str(phase_a_afe["sku"]),
        source_sha=source_sha,
        actual_hashes={
            "collector_sha256": sha256_file(collector),
            "raw_evidence_sha256": evidence_raw_sha256,
            "attestation_verification_sha256": sha256_file(attestation_path),
            "board_runner_sha256": board_runner_sha256,
            "model_sha256": model_sha256,
            "keyword_pack_sha256": keyword_pack_sha256,
            "board_audio_sha256": sha256_file(board_audio),
        },
    )
    if budget["board_revision"] != evidence["board_revision"]:
        raise ValueError("approved resource budget board revision differs from DUT evidence")

    manifest_rows = load_jsonl(evidence_raw)
    required_names = set(policy["required_raw_evidence"])
    actual_names = {str(row.get("name", "")) for row in manifest_rows}
    if not required_names.issubset(actual_names):
        raise ValueError(f"target raw evidence is missing required files: {sorted(required_names - actual_names)}")
    raw_hashes: dict[str, str] = {}
    for index, row in enumerate(manifest_rows):
        name = str(row.get("name", ""))
        if not name or pathlib.Path(name).name != name:
            raise ValueError(f"evidence-raw[{index}] name must be a basename")
        path = require_bundle_file(bundle, f"raw/{name}")
        digest = sha256_file(path)
        if digest != row.get("sha256") or path.stat().st_size != row.get("bytes"):
            raise ValueError(f"raw evidence bytes do not match manifest: {name}")
        raw_hashes[name] = digest

    continuity_path = require_bundle_file(bundle, "raw/audio-continuity.json")
    continuity = load_json(continuity_path)
    if continuity.get("schema_version") != 1:
        raise ValueError("audio-continuity schema_version must be 1")
    for key in ("xrun_count", "discontinuity_count", "lost_samples", "backpressure_count"):
        value = continuity.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"audio-continuity {key} must be a non-negative integer")

    hard = policy["per_dut_hard_gates"]
    failures: list[str] = []
    if evidence["soak_hours"] < float(hard["min_soak_hours"]):
        failures.append("soak-hours")
    if hard.get("require_p99_below_block_deadline") is True and not (
        board["p99_process_us"] < board["block_deadline_us"]
    ):
        failures.append("p99-block-deadline")
    if hard.get("require_rtf_below_realtime") is True and not board["rtf"] < 1.0:
        failures.append("rtf-realtime")
    if hard.get("require_p99_headroom_above_one") is True and not (
        board["p99_headroom"] > 1.0
    ):
        failures.append("p99-headroom")
    continuity_limits = {
        "xrun_count": "max_xrun_count",
        "discontinuity_count": "max_discontinuity_count",
        "lost_samples": "max_lost_samples",
        "backpressure_count": "max_backpressure_count",
    }
    for measured, limit in continuity_limits.items():
        if continuity[measured] > int(hard[limit]):
            failures.append(measured.replace("_count", ""))

    limits = budget["limits"]
    resource_comparisons = (
        (evidence["cpu_percent"] <= limits["max_cpu_percent"], "cpu-budget"),
        (evidence["rss_kib"] <= limits["max_rss_kib"], "rss-budget"),
        (
            evidence["stack_high_water_bytes"]
            <= limits["max_stack_high_water_bytes"],
            "stack-budget",
        ),
        (evidence["max_temp_c"] <= limits["max_temp_c"], "temperature-budget"),
        (
            evidence["average_power_mw"] <= limits["max_average_power_mw"],
            "power-budget",
        ),
    )
    failures.extend(label for passed, label in resource_comparisons if not passed)

    result = {
        "schema_version": 1,
        "phase": "physical-target-dut-qualification",
        "qualified": not failures,
        "shipping_approved": False,
        "deployment_tag": deployment_tag,
        "deployment_target": source_sha,
        "human_qualification_tag": human_tag,
        "human_corpus_sha256": phase_a["corpus_sha256"],
        "final_afe_identity_sha256": final_afe_identity,
        "sku": evidence["sku"],
        "board_revision": evidence["board_revision"],
        "dut_id": evidence["dut_id"],
        "builder_id": evidence["builder_id"],
        "collector_id": evidence["collector_id"],
        "board_audio_class": profile["board_audio_class"],
        "external_attestation": external_attestation,
        "resource_budget": {
            "budget_id": budget["budget_id"],
            "sha256": budget_sha256,
            "measurement_contract_id": budget["measurement_contract_id"],
            "authority": budget["authority"],
            "approved_at_utc": budget["approved_at_utc"],
            "limits": limits,
        },
        "metrics": {
            "p99_process_us": board["p99_process_us"],
            "block_deadline_us": board["block_deadline_us"],
            "rtf": board["rtf"],
            "p99_headroom": board["p99_headroom"],
            "soak_hours": evidence["soak_hours"],
            "cpu_percent": evidence["cpu_percent"],
            "rss_kib": evidence["rss_kib"],
            "stack_high_water_bytes": evidence["stack_high_water_bytes"],
            "max_temp_c": evidence["max_temp_c"],
            "average_power_mw": evidence["average_power_mw"],
        },
        "continuity": continuity,
        "evidence_sha256": {
            "target_profile": sha256_file(require_bundle_file(bundle, "target-profile.json")),
            "target_evidence": sha256_file(require_bundle_file(bundle, "target-evidence.json")),
            "board_summary": sha256_file(require_bundle_file(bundle, "board-summary.json")),
            "board_runner": board_runner_sha256,
            "board_audio": sha256_file(board_audio),
            "evidence_raw": evidence_raw_sha256,
            "attestation_verification": sha256_file(attestation_path),
            "audio_continuity": sha256_file(continuity_path),
            "resource_budget": budget_sha256,
        },
        "raw_evidence_sha256": raw_hashes,
        "failures": sorted(failures),
        "next_gate": "physical-target-cohort-qualification" if not failures else "physical-target-dut-failed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
