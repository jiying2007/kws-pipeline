from __future__ import annotations

import math

import torch


PREFIX_COMPLETION_TAIL_STEPS = 14


def _target_rows(
    targets: torch.Tensor, target_lengths: torch.Tensor
) -> list[tuple[int, ...]]:
    flat = targets.detach().cpu().tolist()
    rows: list[tuple[int, ...]] = []
    offset = 0
    for raw_length in target_lengths.detach().cpu().tolist():
        length = int(raw_length)
        rows.append(tuple(int(value) for value in flat[offset : offset + length]))
        offset += length
    if offset != len(flat):
        raise ValueError("flattened targets do not match target lengths")
    return rows


def strict_prefix_completion_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    keyword_sequences: list[list[int]],
    keyword_operating_points: list[dict],
    tail_steps: int = PREFIX_COMPLETION_TAIL_STEPS,
) -> torch.Tensor:
    """Penalize hallucinated terminal tokens after a genuine strict wake prefix.

    Sequence-margin already suppresses a completed wake path on every negative.
    This auxiliary hinge concentrates gradient on the missing next token for
    strict-prefix negatives such as ``ni3 hao3 xiao3``. The window covers the
    terminal 280 ms at the 20-ms training hop, matching the current synthetic
    200-ms tail plus up-to-70-ms echo without changing the runtime threshold.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must be [T,B,V]")
    batch = int(log_probs.shape[1])
    if input_lengths.ndim != 1 or int(input_lengths.numel()) != batch:
        raise ValueError("input_lengths must contain one value per sample")
    if target_lengths.ndim != 1 or int(target_lengths.numel()) != batch:
        raise ValueError("target_lengths must contain one value per sample")
    if tail_steps <= 0:
        raise ValueError("prefix completion tail must be positive")
    if len(keyword_sequences) != len(keyword_operating_points):
        raise ValueError("keyword operating points must align with keyword sequences")

    normalized = [tuple(int(value) for value in sequence) for sequence in keyword_sequences]
    if any(not sequence for sequence in normalized):
        raise ValueError("keyword sequence may not be empty")
    ceilings: list[torch.Tensor] = []
    for item in keyword_operating_points:
        threshold = float(item["threshold"])
        negative_margin = float(item["negative_margin"])
        ceiling = threshold - negative_margin
        if not math.isfinite(ceiling) or not 0.0 < ceiling < 1.0:
            raise ValueError("keyword negative operating ceiling is invalid")
        ceilings.append(log_probs.new_tensor(math.log(ceiling)))

    rows = _target_rows(targets, target_lengths)
    losses: list[torch.Tensor] = []
    for batch_index, true_row in enumerate(rows):
        steps = int(input_lengths[batch_index])
        if steps <= 0 or steps > int(log_probs.shape[0]):
            raise ValueError("input length is outside model output")
        if not true_row:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)
            continue
        hinges: list[torch.Tensor] = []
        start = max(0, steps - tail_steps)
        sample = log_probs[start:steps, batch_index, :]
        for index, keyword in enumerate(normalized):
            if len(true_row) >= len(keyword) or keyword[: len(true_row)] != true_row:
                continue
            next_token = int(keyword[len(true_row)])
            terminal_peak = sample[:, next_token].amax()
            hinges.append(torch.relu(terminal_peak - ceilings[index]))
        if hinges:
            losses.append(torch.stack(hinges).amax())
        else:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)
    return torch.stack(losses)
