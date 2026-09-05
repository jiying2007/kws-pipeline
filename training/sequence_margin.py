from __future__ import annotations

import math

import torch


DECODER_CONFIDENCE_THRESHOLD = 0.55


def _target_rows(
    targets: torch.Tensor, target_lengths: torch.Tensor
) -> list[tuple[int, ...]]:
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


def _decoder_sequence_log_confidence(
    sample_log_probs: torch.Tensor,
    sequence: tuple[int, ...],
) -> torch.Tensor:
    """Best chronological token-transition confidence in shipping-decoder units.

    The C decoder reports exp(acoustic_score / trie_depth), where acoustic_score
    contains only the log probability collected when a keyword token advances.
    This max-plus dynamic program is a differentiable surrogate for that acoustic
    path: choose one chronological frame per keyword token, then normalize the
    best accumulated log probability by keyword depth.

    Adjacent repeated tokens require one separating frame, matching the runtime
    decoder's repeated-token transition rule. Retention/dominance remain runtime
    constraints; this objective deliberately aligns the confidence numerator and
    threshold without cloning the entire discrete decoder state machine.
    """
    if sample_log_probs.ndim != 2:
        raise ValueError("sample_log_probs must be [T,V]")
    steps = int(sample_log_probs.shape[0])
    if not sequence:
        raise ValueError("decoder keyword sequence may not be empty")
    if steps <= 0:
        raise ValueError("decoder confidence requires positive input length")

    # Each state means: best acoustic sum whose current token was consumed at
    # exactly this frame. Strict chronological advance uses a shifted prefix max.
    score = sample_log_probs[:, sequence[0]]
    previous = sequence[0]
    for token in sequence[1:]:
        gap = 2 if token == previous else 1
        if steps <= gap:
            return sample_log_probs.new_tensor(float("-inf"))
        prefix_best = torch.cummax(score, dim=0).values
        shifted = sample_log_probs.new_full((steps,), float("-inf"))
        shifted[gap:] = prefix_best[:-gap]
        score = shifted + sample_log_probs[:, token]
        previous = token
    return score.amax() / float(len(sequence))


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
    confidence_threshold: float = DECODER_CONFIDENCE_THRESHOLD,
) -> torch.Tensor:
    """Return a per-sample decoder-confidence operating-band hinge.

    The previous relative CTC-likelihood margin improved development FR/FA but
    still optimized a score different from the shipping decoder. Runtime wakes
    on the geometric mean of token-transition acoustic probabilities crossing a
    keyword threshold. Train that operating band directly:

    * a genuine wake must reach at least ``threshold + margin``;
    * every competing/non-wake keyword path must stay at or below
      ``threshold - margin``;
    * the worst competing path wins, never an average, because product semantics
      are wake-on-any-keyword.

    Once a sample is safely inside the operating band this auxiliary loss is
    exactly zero, avoiding the late-round over-driving seen with a purely
    relative objective. CTC/ordered-token/recurrent-release remain the primary
    sequence, alignment and recurrent-state objectives.
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
    if not 0.0 < float(confidence_threshold) < 1.0:
        raise ValueError("decoder confidence threshold must be in (0,1)")
    if not 0.0 <= float(margin) < 0.5:
        raise ValueError("keyword sequence margin is invalid")
    lower = float(confidence_threshold) - float(margin)
    upper = float(confidence_threshold) + float(margin)
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("decoder confidence operating band must stay inside (0,1)")

    vocab = int(log_probs.shape[2])
    normalized_keywords: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw in keyword_sequences:
        sequence = tuple(int(value) for value in raw)
        if not sequence or any(
            value <= blank or value >= vocab for value in sequence
        ):
            raise ValueError("keyword sequence contains invalid token ids")
        if sequence in seen:
            raise ValueError("keyword sequences must be unique")
        seen.add(sequence)
        normalized_keywords.append(sequence)

    true_rows = _target_rows(targets, target_lengths)
    positive_floor = log_probs.new_tensor(math.log(upper))
    negative_ceiling = log_probs.new_tensor(math.log(lower))
    losses: list[torch.Tensor] = []

    for batch_index, true_row in enumerate(true_rows):
        steps = int(input_lengths[batch_index])
        if steps <= 0 or steps > int(log_probs.shape[0]):
            raise ValueError("input length is outside model output")
        sample = log_probs[:steps, batch_index, :]
        scores = [
            _decoder_sequence_log_confidence(sample, sequence)
            for sequence in normalized_keywords
        ]
        wake_index = next(
            (
                index
                for index, sequence in enumerate(normalized_keywords)
                if sequence == true_row
            ),
            None,
        )
        hinges: list[torch.Tensor] = []
        if wake_index is not None:
            hinges.append(torch.relu(positive_floor - scores[wake_index]))
        for index, score in enumerate(scores):
            if index == wake_index:
                continue
            hinges.append(torch.relu(score - negative_ceiling))
        if hinges:
            losses.append(torch.stack(hinges).amax())
        else:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)

    return torch.stack(losses)
