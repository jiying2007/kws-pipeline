#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

FREEZE_POLICY = "gru-frozen-candidate-v1"
SOURCE_POLICY = "gru-development-curriculum-loop-v1"
MEMBERS = {
    "model_sha256": "model.kwm",
    "checkpoint_sha256": "model.pt",
    "pack_sha256": "keywords.kwk",
    "keywords_sha256": "keywords.tsv",
    "provenance_sha256": "model-provenance.json",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: pathlib.Path) -> dict:
    root = root.resolve()
    manifest_path = root / "freeze-manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("freeze manifest must be a JSON object")
    if value.get("policy") != FREEZE_POLICY or value.get("source_policy") != SOURCE_POLICY:
        raise ValueError("freeze policy identity mismatch")
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
    for digest_field, name in MEMBERS.items():
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"frozen candidate member missing: {name}")
        actual = sha256_file(path)
        if actual != str(value.get(digest_field, "")):
            raise ValueError(f"frozen candidate digest mismatch: {name}")
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
                "selected_round": value["selected_round"],
                "model_sha256": value["model_sha256"],
                "candidate_stage_feedback_allowed": False,
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
