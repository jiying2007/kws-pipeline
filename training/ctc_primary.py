from __future__ import annotations

import hashlib
import json
import math
from typing import Callable

import torch
from torch.utils.data import DataLoader

from objective_contract import (
    CTC_PRIMARY_POLICIES,
    CTC_PRIMARY_POLICY_LABEL_PRIOR,
    CTC_PRIMARY_POLICY_STANDARD,
)


LABEL_PRIOR_ALPHA = 0.3
LABEL_PRIOR_ESTIMATION_POLICY = (
    "initial-model-training-corpus-posterior-marginal-v1"
)
LABEL_PRIOR_LOSS_POLICY = "posterior-marginal-label-prior-ctc-v1"
LABEL_PRIOR_POSTERIOR_SOURCE = "raw-initial-model-before-vad-alignment-v1"
NEGATIVE_SENTINEL = -1.0e4


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


def validate_ctc_primary_policy(policy: str) -> str:
    value = str(policy)
    if value not in CTC_PRIMARY_POLICIES:
        raise ValueError(
            "CTC primary policy must be one of "
            + ", ".join(sorted(CTC_PRIMARY_POLICIES))
        )
    return value


def _extended_target(
    sequence: tuple[int, ...],
    *,
    blank: int,
) -> list[int]:
    states = [blank]
    for token in sequence:
        states.extend((int(token), blank))
    return states


def _allowed_predecessors(
    states: list[int],
    index: int,
    *,
    blank: int,
) -> list[int]:
    result = [index]
    if index > 0:
        result.append(index - 1)
    if (
        index > 1
        and states[index] != blank
        and states[index] != states[index - 2]
    ):
        result.append(index - 2)
    return result


def label_prior_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    label_priors: torch.Tensor,
    *,
    blank: int = 0,
    alpha: float = LABEL_PRIOR_ALPHA,
) -> torch.Tensor:
    """Return per-sample label-prior CTC NLL with CTCLoss-compatible scale."""
    if log_probs.ndim != 3:
        raise ValueError("log_probs must be [T,B,V]")
    steps_available, batch, vocab = (
        int(log_probs.shape[0]),
        int(log_probs.shape[1]),
        int(log_probs.shape[2]),
    )
    if input_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("CTC lengths must contain one value per batch sample")
    if label_priors.shape != (vocab,):
        raise ValueError("label priors must match vocabulary")
    if not torch.isfinite(label_priors).all() or bool((label_priors <= 0).any()):
        raise ValueError("label priors must be finite and positive")
    if abs(float(label_priors.sum()) - 1.0) > 1.0e-5:
        raise ValueError("label priors must sum to one")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("label-prior alpha must be finite and >= 0")
    if blank < 0 or blank >= vocab:
        raise ValueError("blank id is outside vocabulary")

    flat_targets = targets.detach().cpu().tolist()
    offset = 0
    prior_adjustment = (
        float(alpha)
        * label_priors.to(dtype=log_probs.dtype, device=log_probs.device).log()
    )
    losses: list[torch.Tensor] = []

    for batch_index in range(batch):
        steps = int(input_lengths[batch_index])
        count = int(target_lengths[batch_index])
        if steps <= 0 or steps > steps_available:
            raise ValueError("CTC input length is outside model output")
        if count < 0:
            raise ValueError("CTC target length must be non-negative")
        sequence = tuple(
            int(value)
            for value in flat_targets[offset : offset + count]
        )
        offset += count
        if any(token <= blank or token >= vocab for token in sequence):
            raise ValueError("CTC target token is outside vocabulary")

        adjusted = (
            log_probs[:steps, batch_index, :]
            - prior_adjustment.unsqueeze(0)
        )
        if not sequence:
            losses.append(-adjusted[:, blank].sum())
            continue

        states = _extended_target(sequence, blank=blank)
        state_count = len(states)
        previous: list[torch.Tensor] = [
            adjusted.new_tensor(NEGATIVE_SENTINEL)
            for _ in range(state_count)
        ]
        previous[0] = adjusted[0, blank]
        if state_count > 1:
            previous[1] = adjusted[0, states[1]]

        for frame in range(1, steps):
            current: list[torch.Tensor] = []
            for state, token in enumerate(states):
                values = torch.stack(
                    [
                        previous[prior]
                        for prior in _allowed_predecessors(
                            states,
                            state,
                            blank=blank,
                        )
                    ]
                )
                current.append(
                    torch.logsumexp(values, dim=0)
                    + adjusted[frame, token]
                )
            previous = current

        log_probability = torch.logsumexp(
            torch.stack(previous[-2:]),
            dim=0,
        )
        if not bool(torch.isfinite(log_probability)):
            raise ValueError("label-prior CTC sequence probability is non-finite")
        losses.append(-log_probability)

    if offset != len(flat_targets):
        raise ValueError("flattened CTC targets do not match target lengths")
    return torch.stack(losses)


@torch.no_grad()
def estimate_training_label_priors(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    collate_fn: Callable,
    vocab_size: int,
) -> tuple[torch.Tensor, int]:
    """Estimate one fixed prior from raw initial-model posteriors on training rows."""
    if batch_size <= 0 or vocab_size <= 1:
        raise ValueError("label-prior estimation arguments are invalid")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    total = torch.zeros(vocab_size, dtype=torch.float64)
    frame_count = 0
    was_training = bool(model.training)
    model.eval()
    try:
        for batch in loader:
            if len(batch) not in (4, 5):
                raise ValueError("unexpected training batch shape")
            x = batch[0]
            input_lengths = batch[2]
            raw_log_probs = model(x).log_softmax(dim=2)
            if int(raw_log_probs.shape[2]) != vocab_size:
                raise ValueError("model posterior vocabulary drifted")
            for batch_index, raw_steps in enumerate(input_lengths.tolist()):
                steps = int(raw_steps)
                if steps <= 0 or steps > int(raw_log_probs.shape[0]):
                    raise ValueError("training prior input length is invalid")
                total += (
                    raw_log_probs[:steps, batch_index, :]
                    .exp()
                    .to(dtype=torch.float64)
                    .sum(dim=0)
                    .cpu()
                )
                frame_count += steps
    finally:
        model.train(was_training)

    if frame_count <= 0:
        raise ValueError("label-prior estimation requires training frames")
    priors = (total / float(frame_count)).to(dtype=torch.float32)
    if not torch.isfinite(priors).all() or bool((priors <= 0).any()):
        raise ValueError("estimated label priors are invalid")
    priors = priors / priors.sum()
    return priors, frame_count


def build_label_prior_contract(
    priors: torch.Tensor,
    *,
    frame_count: int,
    training_corpus_sha256: str,
    source_float_state_sha256: str,
) -> dict:
    if priors.ndim != 1 or int(priors.numel()) <= 1:
        raise ValueError("label priors must be a vocabulary vector")
    values = [float(value) for value in priors.detach().cpu().tolist()]
    if (
        frame_count <= 0
        or len(training_corpus_sha256) != 64
        or len(source_float_state_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in training_corpus_sha256)
        or any(ch not in "0123456789abcdef" for ch in source_float_state_sha256)
    ):
        raise ValueError("label-prior contract identity is invalid")
    payload = {
        "schema_version": 1,
        "policy": LABEL_PRIOR_ESTIMATION_POLICY,
        "loss_policy": LABEL_PRIOR_LOSS_POLICY,
        "posterior_source": LABEL_PRIOR_POSTERIOR_SOURCE,
        "alpha": LABEL_PRIOR_ALPHA,
        "frame_count": int(frame_count),
        "training_corpus_sha256": training_corpus_sha256,
        "source_float_state_sha256": source_float_state_sha256,
        "values": values,
    }
    payload["values_sha256"] = canonical_sha256(values)
    return normalize_label_prior_contract(payload)


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


def primary_ctc_per_sample(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    *,
    policy: str,
    standard_loss: torch.nn.Module,
    label_prior_contract: dict | None,
    blank: int = 0,
) -> torch.Tensor:
    policy = validate_ctc_primary_policy(policy)
    if policy == CTC_PRIMARY_POLICY_STANDARD:
        if label_prior_contract is not None:
            raise ValueError("standard CTC must not carry a label-prior contract")
        return standard_loss(log_probs, targets, input_lengths, target_lengths)
    if policy != CTC_PRIMARY_POLICY_LABEL_PRIOR:
        raise ValueError("unsupported CTC primary policy")
    contract = normalize_label_prior_contract(label_prior_contract)
    priors = log_probs.new_tensor(contract["values"])
    return label_prior_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        priors,
        blank=blank,
        alpha=float(contract["alpha"]),
    )
