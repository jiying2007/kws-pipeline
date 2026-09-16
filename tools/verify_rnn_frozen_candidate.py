#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
from rnn_development_gate import evaluate_development_split  # noqa: E402

FREEZE_POLICY = "rnn-frozen-candidate-v1"
SOURCE_POLICY = "rnn-development-curriculum-loop-v1"
SELECTION_POLICY = "best-strict-development-objective-round"
ARCHITECTURE = "tiny-streaming-rnn-v1"
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


def verify(root: pathlib.Path) -> dict:
    root = root.resolve()
    freeze = load_object(root / "freeze-manifest.json")
    if freeze.get("policy") != FREEZE_POLICY or freeze.get("source_policy") != SOURCE_POLICY:
        raise ValueError("RNN freeze policy identity mismatch")
    if freeze.get("model_family") != "rnn" or freeze.get("architecture") != ARCHITECTURE:
        raise ValueError("RNN frozen model identity mismatch")
    if freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("RNN freeze selection policy mismatch")
    if freeze.get("evidence_scope") != "development-only":
        raise ValueError("RNN candidate source scope must be development-only")
    if freeze.get("selection_evidence") != ["development-calibration", "development-test"]:
        raise ValueError("RNN candidate selected with non-development evidence")
    for field in ("qualification_used_for_selection", "shadow_used_for_selection", "formal_qualification_used_for_selection"):
        if freeze.get(field) is not False:
            raise ValueError(f"RNN freeze requires {field}=false")
    stage = freeze.get("candidate_stage")
    if not isinstance(stage, dict):
        raise ValueError("RNN candidate_stage contract missing")
    for field in ("fresh_validation_required", "shadow_required", "formal_qualification_required", "bounded_repair_only"):
        if stage.get(field) is not True:
            raise ValueError(f"RNN candidate stage requires {field}=true")
    for field in ("validation_feedback_allowed", "threshold_feedback_allowed", "training_rule_feedback_allowed"):
        if stage.get(field) is not False:
            raise ValueError(f"RNN candidate stage requires {field}=false")

    for digest_field, name in {**MEMBERS, **EVIDENCE_MEMBERS}.items():
        path = root / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"frozen RNN candidate member missing: {name}")
        if sha256_file(path) != str(freeze.get(digest_field, "")):
            raise ValueError(f"frozen RNN candidate digest mismatch: {name}")

    config = load_object(root / "source-config.json")
    policy = load_object(root / "source-development-policy.json")
    if policy.get("policy") != SOURCE_POLICY or policy.get("model_family") != "rnn":
        raise ValueError("RNN source development policy identity mismatch")
    if sha256_file(root / "source-config.json") != str(freeze.get("config_sha256", "")):
        raise ValueError("RNN config snapshot mismatch")
    if sha256_file(root / "source-development-policy.json") != str(freeze.get("development_policy_sha256", "")):
        raise ValueError("RNN policy snapshot mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if policy.get(field) is not False:
            raise ValueError(f"RNN source policy requires {field}=false")

    evidence = load_object(root / "selection-evidence.json")
    if evidence.get("evidence_class") != "rnn-frozen-development-selection":
        raise ValueError("RNN selection evidence class mismatch")
    if evidence.get("model_family") != "rnn" or evidence.get("architecture") != ARCHITECTURE:
        raise ValueError("RNN selection evidence model identity mismatch")
    if evidence.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("RNN selection evidence policy mismatch")
    if int(evidence.get("selected_round", -1)) != int(freeze.get("selected_round", -2)):
        raise ValueError("RNN selection round mismatch")
    if float(evidence.get("selected_score")) != float(freeze.get("selected_score")):
        raise ValueError("RNN selection score mismatch")
    if str(evidence.get("model_sha256")) != str(freeze.get("model_sha256")):
        raise ValueError("RNN selection model identity mismatch")
    for field in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if evidence.get(field) is not False:
            raise ValueError(f"RNN selection evidence requires {field}=false")
    calibration = evaluate_development_split(evidence["calibration"], evidence["calibration_domains"], config)
    test = evaluate_development_split(evidence["test"], evidence["test_domains"], config)
    if not calibration["qualified"] or not test["qualified"]:
        raise ValueError("RNN selection evidence fails full development robustness")
    if evidence.get("calibration_gate") is not True or evidence.get("test_gate") is not True:
        raise ValueError("RNN selection gate flags are not strict-pass")

    corpus = load_object(root / "development-wav-sha256.json")
    if corpus.get("evidence_class") != "rnn-frozen-development-wav-identities" or corpus.get("model_family") != "rnn":
        raise ValueError("RNN development WAV identity evidence mismatch")
    hashes = corpus.get("wav_sha256")
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("RNN development WAV identity list empty")
    normalized = [str(item) for item in hashes]
    if normalized != sorted(set(normalized)) or int(corpus.get("wav_sha256_count", -1)) != len(normalized):
        raise ValueError("RNN development WAV identities must be sorted, unique and counted")
    if any(len(item) != 64 or any(ch not in "0123456789abcdef" for ch in item) for item in normalized):
        raise ValueError("RNN development WAV identity contains invalid SHA256")
    if freeze.get("selected_model_matches_selection_evidence") is not True:
        raise ValueError("RNN model/selection binding not asserted")
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidate", required=True, type=pathlib.Path)
    args = parser.parse_args(); value = verify(args.candidate)
    print(json.dumps({"verified": True, "model_family": "rnn", "policy": value["policy"], "selected_round": value["selected_round"], "model_sha256": value["model_sha256"]}, sort_keys=True)); return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(2)
