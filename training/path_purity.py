from __future__ import annotations

import math

import torch


PATH_PURITY_MARGIN_DEFAULT = 0.10
PATH_PURITY_MARGIN_MAX = 2.0


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
            raise ValueError("target length must be non-negative")
        rows.append(tuple(int(value) for value in flat[offset : offset + length]))
        offset += length
    if offset != len(flat):
        raise ValueError("flattened targets do not match target lengths")
    return rows


def ordered_path_purity_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    keyword_sequences: list[list[int]],
    blank: int = 0,
    margin: float = PATH_PURITY_MARGIN_DEFAULT,
) -> torch.Tensor:
    """Penalize out-of-order wake-token dominance on exact wake positives.

    Each exact wake target is split into the same coarse chronological regions
    used by ordered-token loss. Within a region, blank, the current target token,
    and the next target token are allowed. Other tokens that occur in any
    configured wake sequence are competitors. The maximum competitor-over-
    allowed log-probability hinge in each region is averaged across target
    occurrences.

    The loss is zero for non-wake/near-miss targets. It is deliberately a narrow
    positive-path purity objective: hard-negative suppression remains owned by
    CTC, sequence-margin, completion loss and replay.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must be [T,B,V]")
    batch = int(log_probs.shape[1])
    vocab = int(log_probs.shape[2])
    if input_lengths.ndim != 1 or int(input_lengths.numel()) != batch:
        raise ValueError("input_lengths must contain one value per sample")
    if target_lengths.ndim != 1 or int(target_lengths.numel()) != batch:
        raise ValueError("target_lengths must contain one value per sample")
    if not math.isfinite(margin) or not 0.0 <= margin <= PATH_PURITY_MARGIN_MAX:
        raise ValueError(
            f"path-purity margin must be finite and in [0,{PATH_PURITY_MARGIN_MAX}]"
        )
    if blank < 0 or blank >= vocab:
        raise ValueError("path-purity blank id is outside vocabulary")

    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    active_tokens: set[int] = set()
    for raw in keyword_sequences:
        sequence = tuple(int(value) for value in raw)
        if not sequence or any(
            value == blank or value < 0 or value >= vocab for value in sequence
        ):
            raise ValueError("path-purity keyword sequence contains invalid token ids")
        if sequence in seen:
            raise ValueError("path-purity keyword sequences must be unique")
        seen.add(sequence)
        normalized.append(sequence)
        active_tokens.update(sequence)

    rows = _target_rows(targets, target_lengths)
    losses: list[torch.Tensor] = []
    for batch_index, row in enumerate(rows):
        if row not in seen:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)
            continue
        steps = int(input_lengths[batch_index])
        if steps <= 0 or steps > int(log_probs.shape[0]):
            raise ValueError("input length is outside model output")

        occurrence_losses: list[torch.Tensor] = []
        count = len(row)
        for occurrence, current in enumerate(row):
            start = (occurrence * steps) // count
            stop = max(start + 1, ((occurrence + 1) * steps) // count)
            stop = min(stop, steps)

            allowed = {blank, current}
            if occurrence + 1 < count:
                allowed.add(row[occurrence + 1])
            competitors = sorted(active_tokens - allowed)
            if not competitors:
                continue

            allowed_ids = sorted(allowed)
            allowed_log = torch.logsumexp(
                log_probs[start:stop, batch_index, allowed_ids],
                dim=1,
            )
            competitor_log = torch.logsumexp(
                log_probs[start:stop, batch_index, competitors],
                dim=1,
            )
            occurrence_losses.append(
                torch.relu(competitor_log - allowed_log + margin).amax()
            )

        if occurrence_losses:
            losses.append(torch.stack(occurrence_losses).mean())
        else:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)

    return torch.stack(losses)
