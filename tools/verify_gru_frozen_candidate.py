#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys

FREEZE_POLICY = "gru-frozen-candidate-v1"
SOURCE_POLICY = "gru-development-curriculum-loop-v1"
SELECTION_POLICY = "best-strict-development-objective-round"
MEMBERS = {
    "model_sha256": "model.kwm",
    "checkpoint_sha256": "model.pt",
    "pack_sha256": "keywords.kwk",
    "keywords_sha256": "keywords.tsv",
    "provenance_sha256": "model-provenance.json",
}
EVIDENCE_MEMBERS = {
    "selection_evidence_sha256": "selection-evidence.json",
    "development_wav_identities_sha256": "development-wav-sha256.json",
    "source_config_snapshot_sha256": "source-config.json",
    "source_development_policy_snapshot_sha256": "source-development-policy.json",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def gate_values(config: dict) -> dict[str, float]:
    raw = config.get("domain_gates")
    if not isinstance(raw, dict):
        raise ValueError("source config is missing domain_gates")
    fields = ("max_frr", "max_far_per_hour", "max_p95_latency_ms", "max_far_frr")
    result = {field: float(raw[field]) for field in fields}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("source domain gates are invalid")
    return result


def strict(base: dict, domains: dict, gates: dict[str, float]) -> bool:
    far = domains.get("domains", {}).get("distance:far")
    return (
        float(base["frr"]) <= gates["max_frr"]
        and float(base["far_per_hour"]) <= gates["max_far_per_hour"]
        and float(base["p95_post_end_latency_ms"]) <= gates["max_p95_latency_ms"]
        and isinstance(far, dict)
        and float(far["frr"]) <= gates["max_far_frr"]
    )


def verify(root: pathlib.Path) -> dict:
    root = root.resolve()
    manifest_path = root / "freeze-manifest.json"
    value = load_object(manifest_path)
    if value.get("policy") != FREEZE_POLICY or value.get("source_policy") != SOURCE_POLICY:
        raise ValueError("freeze policy identity mismatch")
    if value.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("freeze selection policy mismatch")
    if value.get("evidence_scope") != "development-only":
        raise ValueError("frozen candidate source scope must be development-only")
    if value.get("selection_evidence") != ["development-calibration", "development-test"]:
        raise ValueError("candidate was selected with non-development evidence")
    for field in (
        "qualification_used_for_selection",
        "shadow_used_for_selection",
        "formal_qualification_used_for_selection",
    ):
        if value.get(field) is not False:
            raise ValueError(f"freeze manifest requires {field}=false")
    stage = value.get("candidate_stage")
    if not isinstance(stage, dict):
        raise ValueError("candidate_stage contract is missing")
    for field in (
        "fresh_validation_required",
        "shadow_required",
        "formal_qualification_required",
        "bounded_repair_only",
    ):
        if stage.get(field) is not True:
            raise ValueError(f"candidate stage requires {field}=true")
    for field in (
        "validation_feedback_allowed",
        "threshold_feedback_allowed",
        "training_rule_feedback_allowed",
    ):
        if stage.get(field) is not False:
            raise ValueError(f"candidate stage requires {field}=false")

    for digest_field, name in {**MEMBERS, **EVIDENCE_MEMBERS}.items():
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"frozen candidate member missing: {name}")
        actual = sha256_file(path)
        if actual != str(value.get(digest_field, "")):
            raise ValueError(f"frozen candidate digest mismatch: {name}")

    config = load_object(root / "source-config.json")
    policy = load_object(root / "source-development-policy.json")
    if sha256_file(root / "source-config.json") != str(value.get("config_sha256", "")):
        raise ValueError("source config snapshot does not match frozen config SHA")
    if sha256_file(root / "source-development-policy.json") != str(
        value.get("development_policy_sha256", "")
    ):
        raise ValueError("source policy snapshot does not match frozen policy SHA")
    if policy.get("policy") != SOURCE_POLICY:
        raise ValueError("source development policy identity mismatch")
    candidate_freeze = policy.get("candidate_freeze")
    if not isinstance(candidate_freeze, dict) or candidate_freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("source development selection policy mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if policy.get(field) is not False:
            raise ValueError(f"source policy requires {field}=false")

    evidence = load_object(root / "selection-evidence.json")
    if evidence.get("evidence_class") != "gru-frozen-development-selection":
        raise ValueError("selection evidence class mismatch")
    if evidence.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("selection evidence policy mismatch")
    if int(evidence.get("selected_round", -1)) != int(value.get("selected_round", -2)):
        raise ValueError("selection evidence round mismatch")
    if float(evidence.get("selected_score")) != float(value.get("selected_score")):
        raise ValueError("selection evidence score mismatch")
    if str(evidence.get("model_sha256")) != str(value.get("model_sha256")):
        raise ValueError("selection evidence model identity mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if evidence.get(field) is not False:
            raise ValueError(f"selection evidence requires {field}=false")
    gates = gate_values(config)
    cal_gate = strict(evidence["calibration"], evidence["calibration_domains"], gates)
    test_gate = strict(evidence["test"], evidence["test_domains"], gates)
    if not cal_gate or not test_gate:
        raise ValueError("selection evidence does not recompute to strict development pass")
    if evidence.get("calibration_gate") is not True or evidence.get("test_gate") is not True:
        raise ValueError("selection evidence gate flags are not strict-pass")

    corpus = load_object(root / "development-wav-sha256.json")
    if corpus.get("evidence_class") != "gru-frozen-development-wav-identities":
        raise ValueError("development WAV identity evidence class mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if corpus.get(field) is not False:
            raise ValueError(f"development WAV evidence requires {field}=false")
    hashes = corpus.get("wav_sha256")
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("development WAV identity list is empty")
    normalized = [str(item) for item in hashes]
    if normalized != sorted(set(normalized)):
        raise ValueError("development WAV identities must be sorted and unique")
    if any(len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item) for item in normalized):
        raise ValueError("development WAV identity contains invalid SHA256")
    if int(corpus.get("wav_sha256_count", -1)) != len(normalized):
        raise ValueError("development WAV identity count mismatch")

    if value.get("selected_model_matches_selection_evidence") is not True:
        raise ValueError("frozen model/selection-evidence binding is not asserted")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    args = parser.parse_args()
    value = verify(args.candidate)
    print(
        json.dumps(
            {
                "verified": True,
                "policy": value["policy"],
                "selection_policy": value["selection_policy"],
                "selected_round": value["selected_round"],
                "model_sha256": value["model_sha256"],
                "candidate_stage_feedback_allowed": False,
                "development_wav_identities_sha256": value[
                    "development_wav_identities_sha256"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
