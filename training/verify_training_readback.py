#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import torch

from ctc_primary import normalize_label_prior_contract
from objective_config import (
    auxiliary_loss_weights,
    ctc_primary_policy_setting,
    sequence_margin_negative_policy_setting,
    sequence_margin_positive_policy_setting,
    verify_auxiliary_loss_readback,
)
from objective_contract import CTC_PRIMARY_POLICY_LABEL_PRIOR
from training_state import state_identity


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_candidate(train: dict, checkpoint: pathlib.Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = checkpoint.with_suffix(".kwm")
    provenance_path = pathlib.Path(str(model) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["checkpoint"]["sha256"] != sha256_file(checkpoint):
        raise ValueError("provenance checkpoint bytes mismatch")
    if provenance["model"]["sha256"] != sha256_file(model):
        raise ValueError("provenance deployment model bytes mismatch")
    identity = state_identity(payload["state_dict"])
    if payload.get("float_state_identity") != identity:
        raise ValueError("checkpoint float state identity missing or mismatched")
    training = provenance["training"]
    if training.get("float_state_identity") != identity:
        raise ValueError("provenance float state identity missing or mismatched")
    weights = verify_auxiliary_loss_readback(train, payload.get("auxiliary_loss_weights"))
    verify_auxiliary_loss_readback(weights, training.get("auxiliary_loss_weights"))
    verify_auxiliary_loss_readback(weights, auxiliary_loss_weights(payload))
    primary_policy, _ = ctc_primary_policy_setting(train)
    if payload.get("ctc_primary_policy", primary_policy) != primary_policy:
        raise ValueError("checkpoint primary CTC policy mismatch")
    if training.get("ctc_primary_policy", primary_policy) != primary_policy:
        raise ValueError("provenance primary CTC policy mismatch")
    payload_prior = payload.get("ctc_label_prior")
    training_prior = training.get("ctc_label_prior")
    if primary_policy == CTC_PRIMARY_POLICY_LABEL_PRIOR:
        normalized_prior = normalize_label_prior_contract(payload_prior)
        if normalize_label_prior_contract(training_prior) != normalized_prior:
            raise ValueError("checkpoint/provenance label-prior mismatch")
        if (
            normalized_prior["training_corpus_sha256"]
            != payload["training_corpus_identity"]["corpus_sha256"]
        ):
            raise ValueError("label-prior training corpus identity mismatch")
    else:
        normalized_prior = None
        if payload_prior is not None or training_prior is not None:
            raise ValueError("standard primary CTC carries label-prior metadata")

    negative_policy, _ = sequence_margin_negative_policy_setting(train)
    positive_policy, _ = sequence_margin_positive_policy_setting(train)
    if payload.get("sequence_margin_negative_policy", negative_policy) != negative_policy:
        raise ValueError("checkpoint negative sequence-margin policy mismatch")
    if training.get("sequence_margin_negative_policy", negative_policy) != negative_policy:
        raise ValueError("provenance negative sequence-margin policy mismatch")
    if payload.get("sequence_margin_positive_policy", positive_policy) != positive_policy:
        raise ValueError("checkpoint positive sequence-margin policy mismatch")
    if training.get("sequence_margin_positive_policy", positive_policy) != positive_policy:
        raise ValueError("provenance positive sequence-margin policy mismatch")
    for key in ("cpu_runtime", "torch_runtime"):
        expected = payload["training_environment"].get(key)
        if not isinstance(expected, dict) or not expected or training["environment"].get(key) != expected:
            raise ValueError(f"training runtime identity missing or mismatched: {key}")
    if payload["training_corpus_identity"] != training["corpus_identity"]:
        raise ValueError("training corpus identity mismatch")
    return {
        "candidate": checkpoint.parent.name,
        "float_state_sha256": identity["sha256"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_sha256": sha256_file(model),
        "provenance_sha256": sha256_file(provenance_path),
        "auxiliary_loss_weights": weights,
        "ctc_primary_policy": primary_policy,
        **(
            {"ctc_label_prior": normalized_prior}
            if normalized_prior is not None
            else {}
        ),
        "sequence_margin_negative_policy": negative_policy,
        "sequence_margin_positive_policy": positive_policy,
        "training_corpus_sha256": payload["training_corpus_identity"]["corpus_sha256"],
        "resume_authority": "weights-only-not-optimizer-continuous",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = sorted((args.work_dir / "candidates").glob("*/model.pt"))
    if not candidates:
        raise ValueError("training readback requires at least one checkpoint")
    report = {
        "schema_version": 1,
        "evidence_class": "training-objective-state-readback-v1",
        "config_sha256": sha256_file(args.config),
        "candidates": [verify_candidate(config.get("train", {}), path) for path in candidates],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"training objective/state readback: {len(candidates)} candidate(s) verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
