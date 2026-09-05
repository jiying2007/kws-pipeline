from __future__ import annotations

import math

import torch
import torch.nn.functional as F

QUALIFIED_DECODER_CONFIDENCE_TARGET = 0.55


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


def _positive_decoder_confidence_hinge(
    *,
    log_probs: torch.Tensor,
    true_rows: list[tuple[int, ...]],
    input_lengths: torch.Tensor,
    keyword_sequences: set[tuple[int, ...]],
) -> torch.Tensor:
    """Mirror the shipping decoder confidence threshold on genuine wakes.

    Runtime confidence is ``exp(terminal_acoustic / keyword_depth)``: the
    geometric mean of token acoustic probabilities along the accepted trie path.
    This differentiable surrogate uses the same chronological token regions as
    ``ordered_token_loss`` and the strongest target posterior in each region.
    It activates only below the formally qualified 0.55 runtime threshold and
    adds no extra confidence margin or any loss to non-wake examples.
    """
    losses: list[torch.Tensor] = []
    for batch_index, row in enumerate(true_rows):
        if row not in keyword_sequences:
            losses.append(log_probs[:, batch_index].sum() * 0.0)
            continue
        steps = int(input_lengths[batch_index])
        count = len(row)
        if steps <= 0 or count <= 0:
            raise ValueError("positive decoder surrogate requires non-empty input/target")
        occurrence_peaks: list[torch.Tensor] = []
        for occurrence, token in enumerate(row):
            start = (occurrence * steps) // count
            stop = max(start + 1, ((occurrence + 1) * steps) // count)
            stop = min(stop, steps)
            occurrence_peaks.append(log_probs[start:stop, batch_index, token].max())
        surrogate_log_confidence = torch.stack(occurrence_peaks).mean()
        losses.append(
            torch.relu(
                math.log(QUALIFIED_DECODER_CONFIDENCE_TARGET)
                - surrogate_log_confidence
            )
        )
    if not losses:
        return log_probs.sum(dim=(0, 2)) * 0.0
    return torch.stack(losses)


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
    path and every other configured wake phrase. For a non-wake example, its true
    target path (including the blank-only target for background) must beat every
    configured wake phrase. All CTC NLLs are normalized by true acoustic length.

    The loss uses the worst competing path, not an average across keywords.
    Product semantics are wake-on-any-keyword, so averaging would dilute one
    dangerous path as more custom keywords are configured.
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

    # #121 eliminated formal false accepts but still missed two kw1 positives at
    # 5 m + critical SNR. Full-sequence CTC margin can prefer the correct wake
    # globally while the streaming decoder's geometric-mean token confidence
    # remains below its fixed threshold. Align recall training with that exact
    # product gate before evaluating the broader blank/keyword competitors.
    competing_hinges.append(
        _positive_decoder_confidence_hinge(
            log_probs=log_probs,
            true_rows=true_rows,
            input_lengths=input_lengths,
            keyword_sequences=seen,
        )
    )

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

    return torch.stack(competing_hinges, dim=0).amax(dim=0)
