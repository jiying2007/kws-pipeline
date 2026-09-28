from __future__ import annotations

import math

import torch


SUFFIX_ROOT_BLANK_MARGIN = 0.5
SUFFIX_ROOT_POLICY = "vad-inactive-one-token-prefix-omission-hinge-v1"


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


def suffix_root_suppression_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    vad_mask: torch.Tensor,
    keyword_sequences: list[list[int]],
    blank: int = 0,
    margin: float = SUFFIX_ROOT_BLANK_MARGIN,
) -> torch.Tensor:
    """Suppress a hallucinated missing root on one-token-short suffix negatives.

    The observed kw2 false accepts use a target such as (4,3,4) while the
    configured wake is (3,4,3,4). The C runtime can complete the wake only if
    the missing root token 3 is hallucinated before the real suffix. Development
    evidence localized that hallucination to a VAD-inactive frame.

    This narrow hinge applies only when a non-wake target is exactly one leading
    token shorter than a configured wake. On VAD-inactive frames it requires
    blank to exceed the missing root by margin in log-probability. Other
    negatives, exact wakes, and all VAD-active frames receive zero loss.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must be [T,B,V]")
    batch = int(log_probs.shape[1])
    vocab = int(log_probs.shape[2])
    if input_lengths.shape != (batch,):
        raise ValueError("input_lengths must contain one value per sample")
    if target_lengths.shape != (batch,):
        raise ValueError("target_lengths must contain one value per sample")
    if vad_mask.shape != (batch, int(log_probs.shape[0])) or vad_mask.dtype != torch.bool:
        raise ValueError("suffix-root VAD mask must be bool [B,T]")
    if blank < 0 or blank >= vocab:
        raise ValueError("suffix-root blank id is outside vocabulary")
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("suffix-root blank margin must be finite and >= 0")

    normalized: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw in keyword_sequences:
        sequence = tuple(int(value) for value in raw)
        if len(sequence) < 2 or any(
            value == blank or value < 0 or value >= vocab for value in sequence
        ):
            raise ValueError("suffix-root keyword sequence is invalid")
        if sequence in seen:
            raise ValueError("suffix-root keyword sequences must be unique")
        seen.add(sequence)
        normalized.append(sequence)

    rows = _target_rows(targets, target_lengths)
    losses: list[torch.Tensor] = []
    for batch_index, row in enumerate(rows):
        steps = int(input_lengths[batch_index])
        if steps <= 0 or steps > int(log_probs.shape[0]):
            raise ValueError("input length is outside model output")

        candidates = [
            keyword[0]
            for keyword in normalized
            if len(keyword) == len(row) + 1 and keyword[1:] == row
        ]
        if not row or not candidates:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)
            continue

        inactive = ~vad_mask[batch_index, :steps]
        if not bool(inactive.any()):
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)
            continue

        sample = log_probs[:steps, batch_index, :]
        hinges: list[torch.Tensor] = []
        for missing_root in sorted(set(candidates)):
            root = sample[inactive, missing_root]
            blank_log = sample[inactive, blank]
            hinges.append(torch.relu(root - blank_log + margin).amax())
        losses.append(torch.stack(hinges).amax())

    return torch.stack(losses)
