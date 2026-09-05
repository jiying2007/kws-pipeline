from __future__ import annotations

import torch
import torch.nn.functional as F


def _target_rows(targets: torch.Tensor, target_lengths: torch.Tensor) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    offset = 0
    flat = targets.detach().cpu().tolist()
    for raw_length in target_lengths.detach().cpu().tolist():
        length = int(raw_length)
        rows.append(tuple(int(value) for value in flat[offset : offset + length]))
        offset += length
    if offset != len(flat):
        raise ValueError("flattened CTC targets do not match target lengths")
    return rows


def _keyword_completion_margin(
    *,
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    true_rows: list[tuple[int, ...]],
    keywords: set[tuple[int, ...]],
    blank: int,
    margin: float,
) -> torch.Tensor:
    """Return a temporal token-vs-blank margin for exact keyword positives.

    Whole-utterance CTC ranking can still accept a wake path whose weakest token
    is locally dominated by blank at the streaming completion frame. Reuse the
    ordered-token chronological regions, select the frame where each correct
    token is strongest, and require that token to beat blank at that same frame.
    Comparing at the same frame is deliberate: blank is allowed to dominate the
    gaps between CTC token emissions.

    The weakest target occurrence wins by max hinge, so a weak terminal token is
    never diluted by the other, already-strong keyword tokens.
    """
    per_sample: list[torch.Tensor] = []
    for batch_index, row in enumerate(true_rows):
        if row not in keywords:
            per_sample.append(log_probs[:, batch_index].sum() * 0.0)
            continue
        steps = int(input_lengths[batch_index])
        if steps <= 0:
            raise ValueError("input length must be positive")
        count = len(row)
        token_hinges: list[torch.Tensor] = []
        for occurrence, token in enumerate(row):
            start = (occurrence * steps) // count
            stop = max(start + 1, ((occurrence + 1) * steps) // count)
            stop = min(stop, steps)
            token_region = log_probs[start:stop, batch_index, token]
            best_frame = start + int(token_region.detach().argmax())
            token_hinges.append(
                torch.relu(
                    float(margin)
                    + log_probs[best_frame, batch_index, blank]
                    - log_probs[best_frame, batch_index, token]
                )
            )
        per_sample.append(torch.stack(token_hinges).amax())
    return torch.stack(per_sample)


def keyword_sequence_margin_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    true_ctc_nll: torch.Tensor,
    keyword_sequences: list[list[int]],
    blank: int = 0,
    margin: float = 0.05,
) -> torch.Tensor:
    """Return a per-sample wake/non-wake sequence-discriminative hinge.

    For a real wake example, its true keyword path must beat both the blank-only
    path and every other configured wake phrase. Exact keyword positives also
    require every chronological target occurrence to beat blank at its strongest
    token frame, aligning the recall objective with streaming completion timing.
    For a non-wake example, its true target path (including the blank-only target
    for background) must beat every configured wake phrase. All CTC NLLs are
    normalized by true acoustic length.

    The loss uses the worst competing path / weakest completion token, not an
    average. Product semantics are wake-on-any-keyword, so averaging would dilute
    one dangerous path as more custom keywords are configured. The sequence and
    completion terms are combined with max rather than sum so this objective keeps
    the qualified global 0.10 loss scale instead of silently increasing it.
    """
    if log_probs.ndim != 3:
        raise ValueError("log_probs must be [T,B,V]")
    batch = int(log_probs.shape[1])
    if true_ctc_nll.ndim != 1 or int(true_ctc_nll.numel()) != batch:
        raise ValueError("true_ctc_nll must contain one value per batch sample")
    if input_lengths.ndim != 1 or int(input_lengths.numel()) != batch:
        raise ValueError("input_lengths must contain one value per batch sample")
    if target_lengths.ndim != 1 or int(target_lengths.numel()) != batch:
        raise ValueError("target_lengths must contain one value per batch sample")
    if not keyword_sequences:
        return torch.zeros_like(true_ctc_nll)
    if not 0.0 <= float(margin) <= 10.0:
        raise ValueError("keyword sequence margin is invalid")

    vocab = int(log_probs.shape[2])
    normalized_keywords: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw in keyword_sequences:
        sequence = tuple(int(value) for value in raw)
        if not sequence or any(value <= blank or value >= vocab for value in sequence):
            raise ValueError("keyword sequence contains invalid token ids")
        if sequence in seen:
            raise ValueError("keyword sequences must be unique")
        seen.add(sequence)
        normalized_keywords.append(sequence)

    true_rows = _target_rows(targets, target_lengths)
    length_norm = input_lengths.to(dtype=log_probs.dtype).clamp_min(1.0)
    true_norm = true_ctc_nll / length_norm
    competing_hinges: list[torch.Tensor] = []

    # Positive wakes must not collapse toward the all-blank explanation. This is
    # the recall side of the same discriminative boundary that suppresses keyword
    # paths on hard negatives below.
    blank_targets = torch.empty(0, dtype=targets.dtype, device=targets.device)
    blank_lengths = torch.zeros_like(target_lengths)
    blank_nll = F.ctc_loss(
        log_probs,
        blank_targets,
        input_lengths,
        blank_lengths,
        blank=blank,
        reduction="none",
        zero_infinity=False,
    )
    wake_example = torch.tensor(
        [row in seen for row in true_rows],
        dtype=log_probs.dtype,
        device=log_probs.device,
    )
    competing_hinges.append(
        torch.relu(float(margin) + true_norm - blank_nll / length_norm)
        * wake_example
    )

    for sequence in normalized_keywords:
        keyword_targets = torch.tensor(
            sequence, dtype=targets.dtype, device=targets.device
        ).repeat(batch, 1)
        keyword_lengths = torch.full(
            (batch,),
            len(sequence),
            dtype=target_lengths.dtype,
            device=target_lengths.device,
        )
        keyword_nll = F.ctc_loss(
            log_probs,
            keyword_targets,
            input_lengths,
            keyword_lengths,
            blank=blank,
            reduction="none",
            # An impossible wake path should be safely separated, not converted
            # to zero loss (which would make it look like the best possible path).
            zero_infinity=False,
        )
        keyword_norm = keyword_nll / length_norm
        # A genuine wake does not compete against its own identical true path;
        # non-wake targets compete against every configured wake path.
        include = torch.tensor(
            [row != sequence for row in true_rows],
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        competing_hinges.append(
            torch.relu(float(margin) + true_norm - keyword_norm) * include
        )

    sequence_hinge = torch.stack(competing_hinges, dim=0).amax(dim=0)
    completion_hinge = _keyword_completion_margin(
        log_probs=log_probs,
        input_lengths=input_lengths,
        true_rows=true_rows,
        keywords=seen,
        blank=blank,
        margin=float(margin),
    )
    return torch.maximum(sequence_hinge, completion_hinge)
