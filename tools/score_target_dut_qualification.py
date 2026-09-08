#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

from qualification_metrics import validate_board, validate_evidence

ROOT = pathlib.Path(__file__).resolve().parents[1]


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
    if policy.get("schema_version") != 1:
        raise ValueError("target policy schema_version must be 1")
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
    final_afe_identity = str(phase_a_receipt.get("final_afe_identity_sha256", ""))
    if final_afe_identity != str(phase_a_afe.get("identity_sha256", "")):
        raise ValueError("Phase-A receipt/final-AFE identity mismatch")
    if str(target_raw.get("audio_frontend_sha256", "")) != str(phase_a_afe.get("executable_sha256", "")):
        raise ValueError("physical target evidence uses a different final-AFE executable")
    if str(target_raw.get("sku", "")) != str(phase_a_afe.get("sku", "")):
        raise ValueError("physical target SKU differs from Phase-A final-AFE SKU")

    board_runner = require_bundle_file(bundle, "board-runner")
    board_audio = require_bundle_file(bundle, "board-audio.wav")
    evidence_raw = require_bundle_file(bundle, "evidence-raw.jsonl")
    attestation = require_bundle_file(bundle, "attestation-verification.json")
    model = args.model.resolve(strict=True)
    keywords = args.keywords.resolve(strict=True)
    collector = ROOT / "tools" / "collect_target_evidence.py"

    board = validate_board(
        board_raw,
        model.stat().st_size,
        keywords.stat().st_size,
        source_sha,
        {
            "runner_sha256": sha256_file(board_runner),
            "model_sha256": sha256_file(model),
            "keyword_pack_sha256": sha256_file(keywords),
            "audio_sha256": sha256_file(board_audio),
        },
    )
    evidence = validate_evidence(
        target_raw,
        sku=str(phase_a_afe["sku"]),
        source_sha=source_sha,
        actual_hashes={
            "collector_sha256": sha256_file(collector),
            "raw_evidence_sha256": sha256_file(evidence_raw),
            "attestation_verification_sha256": sha256_file(attestation),
            "board_runner_sha256": sha256_file(board_runner),
            "model_sha256": sha256_file(model),
            "keyword_pack_sha256": sha256_file(keywords),
            "board_audio_sha256": sha256_file(board_audio),
        },
    )

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

    gates = policy["per_dut_gates"]
    failures: list[str] = []
    comparisons = (
        (board["p99_process_us"] <= float(gates["max_p99_process_us"]), "p99-process"),
        (board["rtf"] <= float(gates["max_rtf"]), "rtf"),
        (board["p99_headroom"] >= float(gates["min_p99_headroom"]), "p99-headroom"),
        (evidence["soak_hours"] >= float(gates["min_soak_hours"]), "soak-hours"),
        (evidence["cpu_percent"] <= float(gates["max_cpu_percent"]), "cpu"),
        (evidence["rss_kib"] <= float(gates["max_rss_kib"]), "rss"),
        (evidence["stack_high_water_bytes"] <= float(gates["max_stack_high_water_bytes"]), "stack"),
        (evidence["max_temp_c"] <= float(gates["max_temp_c"]), "temperature"),
        (evidence["average_power_mw"] <= float(gates["max_average_power_mw"]), "power"),
        (continuity["xrun_count"] <= int(gates["max_xrun_count"]), "xrun"),
        (continuity["discontinuity_count"] <= int(gates["max_discontinuity_count"]), "discontinuity"),
        (continuity["lost_samples"] <= int(gates["max_lost_samples"]), "lost-samples"),
        (continuity["backpressure_count"] <= int(gates["max_backpressure_count"]), "backpressure"),
    )
    failures.extend(label for passed, label in comparisons if not passed)

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
        "metrics": {
            "p99_process_us": board["p99_process_us"],
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
            "board_runner": sha256_file(board_runner),
            "board_audio": sha256_file(board_audio),
            "evidence_raw": sha256_file(evidence_raw),
            "attestation_verification": sha256_file(attestation),
            "audio_continuity": sha256_file(continuity_path),
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
