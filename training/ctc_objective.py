from __future__ import annotations

import math

import torch

from objective_contract import (
    CTC_OBJECTIVE_POLICY_DEFAULT,
    CTC_OBJECTIVE_POLICY_LABEL_PRIOR,
    CTC_OBJECTIVE_POLICY_STANDARD,
)

LABEL_PRIOR_ALPHA = 0.3
LABEL_PRIOR_STATE_POLICY = "train-ctc-posterior-marginal-epoch-boundary-v1"
LABEL_PRIOR_STATE_SCHEMA_VERSION = 1


def _target_rows(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
) -> list[tuple[int, ...]]:
    flat = targets.detach().cpu().tolist()
    rows: list[tuple[int, ...]] = []
    offset = 0
    for raw_length in target_lengths.detach().cpu().tolist():
        length = int(raw_length)
        if length < 0:
            raise ValueError("CTC target length must be non-negative")
        rows.append(tuple(int(value) for value in flat[offset : offset + length]))
        offset += length
    if offset != len(flat):
        raise ValueError("flattened CTC targets do not match target lengths")
    return rows


def _extended_target(sequence: tuple[int, ...], blank: int) -> list[int]:
    states = [blank]
    for token in sequence:
        states.extend((token, blank))
    return states


def _predecessors(states: list[int], index: int, blank: int) -> list[int]:
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


def validate_label_priors(
    priors: torch.Tensor,
    *,
    vocab_size: int,
) -> torch.Tensor:
    if (
        priors.ndim != 1
        or int(priors.numel()) != vocab_size
        or vocab_size < 2
    ):
        raise ValueError("label-prior vector must match vocabulary")
    value = priors.detach().cpu().to(dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(value).all()) or bool((value <= 0.0).any()):
        raise ValueError("label priors must be finite and > 0")
    total = float(value.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("label-prior sum must be finite and > 0")
    value = value / total
    if abs(float(value.sum()) - 1.0) > 1.0e-12:
        raise ValueError("normalized label priors must sum to one")
    return value


def uniform_label_priors(vocab_size: int) -> torch.Tensor:
    if vocab_size < 2:
        raise ValueError("label-prior vocabulary must contain at least two labels")
    return torch.full(
        (vocab_size,),
        1.0 / float(vocab_size),
        dtype=torch.float64,
    )


def label_prior_ctc_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    priors: torch.Tensor,
    blank: int = 0,
    alpha: float = LABEL_PRIOR_ALPHA,
) -> torch.Tensor:
    """Return per-sample CTC NLL with fixed label-prior path correction.

    Path scores use log p(label|frame) - alpha * log prior(label). Priors are
    treated as fixed state for the current epoch; no gradient flows through the
    prior vector. At alpha=0 this is mathematically standard CTC.
    """
    if log_probs.ndim != 3:
        raise ValueError("CTC log_probs must be [T,B,V]")
    steps_total, batch, vocab_size = map(int, log_probs.shape)
    if not 0 <= blank < vocab_size:
        raise ValueError("CTC blank id is outside vocabulary")
    if input_lengths.shape != (batch,) or target_lengths.shape != (batch,):
        raise ValueError("CTC lengths must contain one value per sample")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("label-prior alpha must be finite and >= 0")

    normalized_priors = validate_label_priors(
        priors,
        vocab_size=vocab_size,
    ).to(device=log_probs.device, dtype=log_probs.dtype)
    prior_log = normalized_priors.log().detach()
    rows = _target_rows(targets, target_lengths)
    losses: list[torch.Tensor] = []

    for batch_index, sequence in enumerate(rows):
        steps = int(input_lengths[batch_index])
        if steps <= 0 or steps > steps_total:
            raise ValueError("CTC input length is outside model output")
        if any(token == blank or token < 0 or token >= vocab_size for token in sequence):
            raise ValueError("CTC target contains invalid token id")

        adjusted = (
            log_probs[:steps, batch_index, :]
            - float(alpha) * prior_log.unsqueeze(0)
        )
        if not sequence:
            losses.append(-adjusted[:, blank].sum())
            continue

        states = _extended_target(sequence, blank)
        count = len(states)
        negative_inf = adjusted.new_tensor(float("-inf"))
        previous: list[torch.Tensor] = [negative_inf for _ in range(count)]
        previous[0] = adjusted[0, blank]
        previous[1] = adjusted[0, states[1]]

        for frame in range(1, steps):
            current: list[torch.Tensor] = []
            for state, token in enumerate(states):
                values = torch.stack(
                    [
                        previous[prior]
                        for prior in _predecessors(states, state, blank)
                    ]
                )
                current.append(
                    torch.logsumexp(values, dim=0) + adjusted[frame, token]
                )
            previous = current

        terminal = torch.logsumexp(torch.stack(previous[-2:]), dim=0)
        if not bool(torch.isfinite(terminal.detach())):
            # Match the trainer's existing zero_infinity=True behavior.
            losses.append(log_probs[:steps, batch_index, :].sum() * 0.0)
        else:
            losses.append(-terminal)

    return torch.stack(losses)


def empty_label_prior_accumulator(vocab_size: int) -> tuple[torch.Tensor, int]:
    if vocab_size < 2:
        raise ValueError("label-prior accumulator requires a valid vocabulary")
    return torch.zeros(vocab_size, dtype=torch.float64), 0


def accumulate_label_prior_statistics(
    total: torch.Tensor,
    frame_count: int,
    *,
    ctc_log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Deterministically accumulate train-only posterior marginals.

    Only frames participating in CTC (0..input_length-1) are included. Padding
    and recurrent-release tail frames are excluded. The caller updates priors
    only at the epoch boundary so the objective is fixed within each epoch.
    """
    if ctc_log_probs.ndim != 3:
        raise ValueError("CTC log_probs must be [T,B,V]")
    steps_total, batch, vocab_size = map(int, ctc_log_probs.shape)
    if input_lengths.shape != (batch,):
        raise ValueError("label-prior input lengths must match batch")
    if total.shape != (vocab_size,) or total.dtype != torch.float64:
        raise ValueError("label-prior accumulator has invalid shape/dtype")
    if frame_count < 0:
        raise ValueError("label-prior frame count must be non-negative")

    result = total.detach().cpu().clone()
    count = int(frame_count)
    with torch.no_grad():
        probabilities = ctc_log_probs.detach().exp().to(dtype=torch.float64)
        for batch_index in range(batch):
            steps = int(input_lengths[batch_index])
            if steps <= 0 or steps > steps_total:
                raise ValueError("label-prior input length is outside model output")
            result += probabilities[:steps, batch_index, :].sum(dim=0).cpu()
            count += steps
    if not bool(torch.isfinite(result).all()) or bool((result <= 0.0).any()):
        raise ValueError("label-prior posterior marginal accumulator is invalid")
    return result, count


def priors_from_accumulator(
    total: torch.Tensor,
    frame_count: int,
) -> torch.Tensor:
    if total.ndim != 1 or total.dtype != torch.float64:
        raise ValueError("label-prior accumulator must be float64 vector")
    if frame_count <= 0:
        raise ValueError("label-prior epoch must contain positive frame count")
    mean = total / float(frame_count)
    return validate_label_priors(mean, vocab_size=int(total.numel()))


def label_prior_state(
    priors: torch.Tensor,
    *,
    epoch_count: int,
    frame_count_last_epoch: int,
) -> dict:
    normalized = validate_label_priors(
        priors,
        vocab_size=int(priors.numel()),
    )
    if epoch_count < 0:
        raise ValueError("label-prior epoch count must be non-negative")
    if frame_count_last_epoch < 0:
        raise ValueError("label-prior frame count must be non-negative")
    return {
        "schema_version": LABEL_PRIOR_STATE_SCHEMA_VERSION,
        "policy": LABEL_PRIOR_STATE_POLICY,
        "alpha": LABEL_PRIOR_ALPHA,
        "epoch_count": int(epoch_count),
        "frame_count_last_epoch": int(frame_count_last_epoch),
        "priors": [float(value) for value in normalized.tolist()],
    }


def validate_label_prior_state(
    value: object,
    *,
    vocab_size: int,
) -> dict:
    required = {
        "schema_version",
        "policy",
        "alpha",
        "epoch_count",
        "frame_count_last_epoch",
        "priors",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("label-prior state fields are invalid")
    if int(value["schema_version"]) != LABEL_PRIOR_STATE_SCHEMA_VERSION:
        raise ValueError("label-prior state schema_version is unsupported")
    if value["policy"] != LABEL_PRIOR_STATE_POLICY:
        raise ValueError("label-prior state policy is unsupported")
    if float(value["alpha"]) != LABEL_PRIOR_ALPHA:
        raise ValueError("label-prior alpha is unsupported")
    epoch_count = int(value["epoch_count"])
    frame_count = int(value["frame_count_last_epoch"])
    if epoch_count < 0 or frame_count < 0:
        raise ValueError("label-prior state counters are invalid")
    raw_priors = value["priors"]
    if not isinstance(raw_priors, list) or len(raw_priors) != vocab_size:
        raise ValueError("label-prior state vector length is invalid")
    priors = validate_label_priors(
        torch.tensor(raw_priors, dtype=torch.float64),
        vocab_size=vocab_size,
    )
    return label_prior_state(
        priors,
        epoch_count=epoch_count,
        frame_count_last_epoch=frame_count,
    )
