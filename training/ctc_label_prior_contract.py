"""Torch-independent validation for the CTC label-prior metadata contract."""
from __future__ import annotations

import hashlib
import json
import math


LABEL_PRIOR_ALPHA = 0.3
LABEL_PRIOR_ESTIMATION_POLICY = (
    "initial-model-training-corpus-posterior-marginal-v1"
)
LABEL_PRIOR_LOSS_POLICY = "posterior-marginal-label-prior-ctc-v1"
LABEL_PRIOR_POSTERIOR_SOURCE = "raw-initial-model-before-vad-alignment-v1"


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def normalize_label_prior_contract(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("CTC label-prior contract must be an object")
    required = {
        "schema_version",
        "policy",
        "loss_policy",
        "posterior_source",
        "alpha",
        "frame_count",
        "training_corpus_sha256",
        "source_float_state_sha256",
        "values",
        "values_sha256",
    }
    if set(value) != required:
        raise ValueError("CTC label-prior contract fields mismatch")
    if int(value["schema_version"]) != 1:
        raise ValueError("CTC label-prior schema version must be 1")
    if value["policy"] != LABEL_PRIOR_ESTIMATION_POLICY:
        raise ValueError("CTC label-prior estimation policy mismatch")
    if value["loss_policy"] != LABEL_PRIOR_LOSS_POLICY:
        raise ValueError("CTC label-prior loss policy mismatch")
    if value["posterior_source"] != LABEL_PRIOR_POSTERIOR_SOURCE:
        raise ValueError("CTC label-prior posterior source mismatch")
    alpha = float(value["alpha"])
    if alpha != LABEL_PRIOR_ALPHA:
        raise ValueError("CTC label-prior alpha mismatch")
    frame_count = int(value["frame_count"])
    if frame_count <= 0:
        raise ValueError("CTC label-prior frame count must be positive")
    values = value["values"]
    if not isinstance(values, list) or len(values) <= 1:
        raise ValueError("CTC label-prior values must be a vocabulary vector")
    normalized_values = [float(item) for item in values]
    if (
        any(not math.isfinite(item) or item <= 0.0 for item in normalized_values)
        or abs(sum(normalized_values) - 1.0) > 1.0e-5
    ):
        raise ValueError("CTC label-prior values are invalid")
    if value["values_sha256"] != canonical_sha256(normalized_values):
        raise ValueError("CTC label-prior values SHA mismatch")
    for key in ("training_corpus_sha256", "source_float_state_sha256"):
        raw = str(value[key])
        if (
            len(raw) != 64
            or any(ch not in "0123456789abcdef" for ch in raw)
        ):
            raise ValueError(f"CTC label-prior {key} is invalid")
    return {
        "schema_version": 1,
        "policy": LABEL_PRIOR_ESTIMATION_POLICY,
        "loss_policy": LABEL_PRIOR_LOSS_POLICY,
        "posterior_source": LABEL_PRIOR_POSTERIOR_SOURCE,
        "alpha": LABEL_PRIOR_ALPHA,
        "frame_count": frame_count,
        "training_corpus_sha256": str(value["training_corpus_sha256"]),
        "source_float_state_sha256": str(value["source_float_state_sha256"]),
        "values": normalized_values,
        "values_sha256": str(value["values_sha256"]),
    }
