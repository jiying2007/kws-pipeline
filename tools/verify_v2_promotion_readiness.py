#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

POLICY = "v2-promotion-readiness-v1"


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_policy(path: pathlib.Path) -> dict:
    value = load_json(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != POLICY:
        raise ValueError("v2 promotion readiness policy identity mismatch")
    if value.get("required_generalization_tier") != "freeze":
        raise ValueError("v2 requires freeze-tier development generalization")
    if int(value.get("minimum_independent_seeds", 0)) < 8:
        raise ValueError("v2 requires at least 8 independent generalization seeds")
    if value.get("required_statistical_gate_mode") != "enforced-v1":
        raise ValueError("v2 readiness requires enforced-v1 statistical gate mode")
    groups = value.get("required_speech_groups")
    if groups != ["train", "generalization-search", "generalization-freeze"]:
        raise ValueError("v2 requires train/search/freeze speech-like groups")
    return value


def verify_generalization(policy: dict, value: dict, candidate_id: str, family: str) -> list[str]:
    blockers: list[str] = []
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("generalization report schema_version must be 1")
    if value.get("evidence_class") != "development-generalization-report-v1":
        raise ValueError("generalization evidence class mismatch")
    if value.get("evidence_scope") != "development-only":
        raise ValueError("generalization evidence scope must be development-only")
    if value.get("protected_evidence_used") is not False:
        raise ValueError("generalization report must not consume protected evidence")
    if value.get("tier") != policy["required_generalization_tier"]:
        blockers.append("generalization-tier")
    if value.get("candidate_id") != candidate_id:
        raise ValueError("generalization candidate_id mismatch")
    if value.get("model_family") != family:
        raise ValueError("generalization model_family mismatch")
    if int(value.get("independent_seed_count", 0)) < int(policy["minimum_independent_seeds"]):
        blockers.append("generalization-seed-count")
    if policy.get("require_coverage_passed") is True and value.get("coverage_passed") is not True:
        blockers.append("generalization-coverage")
    if value.get("statistical_gate_mode") != policy["required_statistical_gate_mode"]:
        blockers.append("generalization-statistical-gate-not-enforced")
    if policy.get("require_statistical_gate_passed") is True and value.get("statistical_gate_passed") is not True:
        blockers.append("generalization-statistical-gate-not-passed")
    return blockers


def verify_speech(policy: dict, value: dict) -> list[str]:
    blockers: list[str] = []
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("speech-like audit schema_version must be 1")
    if value.get("evidence_class") != "speech-like-synthetic-audit-v1":
        raise ValueError("speech-like audit evidence class mismatch")
    if policy.get("require_tone_backend_forbidden") is True and value.get("tone_backend_allowed") is not False:
        blockers.append("speech-like-tone-backend")
    groups = value.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("speech-like audit groups must be an object")
    missing = [name for name in policy["required_speech_groups"] if name not in groups]
    if missing:
        blockers.append("speech-like-missing-groups")
    isolation = value.get("cross_group_isolation")
    if not isinstance(isolation, dict):
        raise ValueError("speech-like audit isolation must be an object")
    for field in ("voice_id", "source_id"):
        overlaps = isolation.get(field)
        if overlaps is None:
            raise ValueError(f"speech-like audit missing isolation field {field}")
        if list(overlaps):
            blockers.append(f"speech-like-{field}-overlap")
    return blockers


def verify_resource(policy: dict, value: dict, candidate_id: str, family: str, vocabulary: str) -> tuple[list[str], dict]:
    blockers: list[str] = []
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("resource report schema_version must be 1")
    if value.get("evidence_class") != policy["required_resource_evidence_class"]:
        raise ValueError("resource evidence class mismatch")
    if value.get("hosted_timing_is_shipping_evidence") is not False:
        raise ValueError("hosted timing must not be shipping evidence")
    if policy.get("require_target_board_measurement_for_shipping") is True and value.get("requires_target_board_measurement_for_shipping") is not True:
        raise ValueError("resource report must require target-board measurement for shipping")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise ValueError("resource report rows must be a list")
    matches = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("candidate") == candidate_id
        and row.get("family") == family
        and row.get("vocabulary") == vocabulary
    ]
    if len(matches) != 1:
        blockers.append("resource-candidate-row")
        return blockers, {}
    row = matches[0]
    if int(row.get("model_bytes_estimate", 0)) <= 0 or float(row.get("dense_mmac_per_s", 0.0)) <= 0.0:
        raise ValueError("resource candidate row contains invalid estimates")
    return blockers, row


def build_report(*, policy: dict, generalization: dict, speech: dict, resource: dict,
                 candidate_id: str, family: str, vocabulary: str) -> dict:
    if family not in {"rnn", "gru"}:
        raise ValueError("model family must be rnn or gru")
    if not candidate_id or not vocabulary:
        raise ValueError("candidate_id and vocabulary must be non-empty")
    blockers: list[str] = []
    blockers += verify_generalization(policy, generalization, candidate_id, family)
    blockers += verify_speech(policy, speech)
    resource_blockers, resource_row = verify_resource(policy, resource, candidate_id, family, vocabulary)
    blockers += resource_blockers
    blockers = sorted(set(blockers))
    ready = not blockers
    return {
        "schema_version": 1,
        "policy": POLICY,
        "evidence_class": "v2-promotion-readiness-report-v1",
        "candidate_id": candidate_id,
        "model_family": family,
        "resource_vocabulary": vocabulary,
        "protected_promotion_ready": ready,
        "blockers": blockers,
        "generalization": {
            "tier": generalization.get("tier"),
            "independent_seed_count": generalization.get("independent_seed_count"),
            "coverage_passed": generalization.get("coverage_passed"),
            "statistical_gate_mode": generalization.get("statistical_gate_mode"),
            "statistical_gate_passed": generalization.get("statistical_gate_passed"),
        },
        "speech_like": {
            "tone_backend_allowed": speech.get("tone_backend_allowed"),
            "groups": sorted((speech.get("groups") or {}).keys()),
        },
        "resource": resource_row,
        "note": "This is a development-to-protected preflight only; existing frozen/source/model-SHA/board gates remain mandatory."
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed v2 preflight before protected KWS qualification.")
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--generalization", required=True, type=pathlib.Path)
    parser.add_argument("--speech-like-audit", required=True, type=pathlib.Path)
    parser.add_argument("--resource-report", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--model-family", required=True, choices=("rnn", "gru"))
    parser.add_argument("--resource-vocabulary", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    report = build_report(
        policy=load_policy(args.policy.resolve()),
        generalization=load_json(args.generalization.resolve()),
        speech=load_json(args.speech_like_audit.resolve()),
        resource=load_json(args.resource_report.resolve()),
        candidate_id=args.candidate_id,
        family=args.model_family,
        vocabulary=args.resource_vocabulary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"v2-promotion-readiness: candidate={args.candidate_id} ready={str(report['protected_promotion_ready']).lower()} blockers={','.join(report['blockers']) or 'none'}")
    if args.require_ready and not report["protected_promotion_ready"]:
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"v2-promotion-readiness: invalid evidence: {exc}", file=sys.stderr)
        raise SystemExit(2)
