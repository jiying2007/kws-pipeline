from __future__ import annotations

import math

import torch

from objective_contract import (
    SEQUENCE_MARGIN_NEGATIVE_POLICIES,
    SEQUENCE_MARGIN_NEGATIVE_POLICY_DEFAULT,
    SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
)


DECODER_CONFIDENCE_THRESHOLD = 0.55
RUNTIME_STATE_RETENTION = 0.94
RUNTIME_BLANK_RETENTION = 0.70
RUNTIME_MIN_PATH_RETENTION_LOG = -16.0
RUNTIME_ROOT_START_LOGIT_MARGIN = 0.5
RUNTIME_FUZZY_CHILD_COST_LOG = -8.25


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


def _runtime_executable_sequence_log_confidence(
    sample_log_probs: torch.Tensor,
    sequence: tuple[int, ...],
    *,
    keyword_root_tokens: frozenset[int],
    blank: int,
) -> torch.Tensor:
    """Viterbi-style acoustic confidence for a runtime-executable keyword path.

    Path selection follows the shipping decoder's discrete search semantics on
    detached log-probabilities: root admissibility, blank/state retention,
    repeated-token separation, fuzzy-child cost, and the cumulative abandonment
    budget. The selected path's acoustic score remains attached so negative
    margin gradients flow only through paths the current runtime can execute.

    This scorer is intentionally used only on the negative side of the optional
    runtime-executable policy. Exact wake positives keep the historical sparse
    chronological scorer so a temporarily non-executable positive never loses
    all margin gradient. It receives acoustic posteriors only; speech/VAD gating,
    the inactivity boundary reset and refractory suppression still require
    evaluation with the C runtime before any candidate decision.
    """
    if sample_log_probs.ndim != 2:
        raise ValueError("sample_log_probs must be [T,V]")
    steps, vocab = (int(sample_log_probs.shape[0]), int(sample_log_probs.shape[1]))
    if steps <= 0 or not sequence:
        raise ValueError("runtime-executable score requires non-empty inputs")
    if blank < 0 or blank >= vocab:
        raise ValueError("runtime-executable blank id is outside vocabulary")

    detached = sample_log_probs.detach()
    top_tokens = detached.argmax(dim=1)
    state_log = math.log(RUNTIME_STATE_RETENTION)
    blank_log = math.log(RUNTIME_BLANK_RETENTION)
    neg_inf = float("-inf")
    depth_count = len(sequence)

    # Each prefix state stores detached search score plus the attached acoustic
    # numerator along the selected Viterbi path. Index zero is the trie root.
    nonblank_search = [neg_inf] * (depth_count + 1)
    separated_search = [neg_inf] * (depth_count + 1)
    nonblank_acoustic: list[torch.Tensor | None] = [None] * (depth_count + 1)
    separated_acoustic: list[torch.Tensor | None] = [None] * (depth_count + 1)
    nonblank_search[0] = 0.0
    nonblank_acoustic[0] = sample_log_probs.new_tensor(0.0)

    best_search = neg_inf
    best_acoustic: torch.Tensor | None = None

    def assign(
        dst_search: list[float],
        dst_acoustic: list[torch.Tensor | None],
        index: int,
        search_score: float,
        acoustic_score: torch.Tensor,
    ) -> None:
        if search_score > dst_search[index]:
            dst_search[index] = search_score
            dst_acoustic[index] = acoustic_score

    for frame in range(steps):
        top = int(top_tokens[frame])
        blank_dominant = top == blank
        top_is_keyword_root = top in keyword_root_tokens
        next_nonblank_search = [neg_inf] * (depth_count + 1)
        next_separated_search = [neg_inf] * (depth_count + 1)
        next_nonblank_acoustic: list[torch.Tensor | None] = [None] * (
            depth_count + 1
        )
        next_separated_acoustic: list[torch.Tensor | None] = [None] * (
            depth_count + 1
        )
        next_nonblank_search[0] = 0.0
        next_nonblank_acoustic[0] = sample_log_probs.new_tensor(0.0)

        for depth in range(depth_count + 1):
            nonblank = 0.0 if depth == 0 else nonblank_search[depth]
            separated = neg_inf if depth == 0 else separated_search[depth]
            nonblank_a = (
                sample_log_probs.new_tensor(0.0)
                if depth == 0
                else nonblank_acoustic[depth]
            )
            separated_a = None if depth == 0 else separated_acoustic[depth]

            if depth != 0 and nonblank_a is not None and math.isfinite(nonblank):
                if blank_dominant:
                    assign(
                        next_separated_search,
                        next_separated_acoustic,
                        depth,
                        nonblank + blank_log,
                        nonblank_a,
                    )
                elif top == sequence[depth - 1]:
                    assign(
                        next_nonblank_search,
                        next_nonblank_acoustic,
                        depth,
                        nonblank + state_log,
                        nonblank_a,
                    )
            if (
                depth != 0
                and separated_a is not None
                and math.isfinite(separated)
                and blank_dominant
            ):
                assign(
                    next_separated_search,
                    next_separated_acoustic,
                    depth,
                    separated + blank_log,
                    separated_a,
                )

            if depth >= depth_count:
                continue
            token = sequence[depth]
            repeated = depth != 0 and token == sequence[depth - 1]
            if repeated:
                base_search = separated
                base_acoustic = separated_a
            elif nonblank >= separated:
                base_search = nonblank
                base_acoustic = nonblank_a
            else:
                base_search = separated
                base_acoustic = separated_a
            if base_acoustic is None or not math.isfinite(base_search):
                continue

            if depth == 0:
                gap = float(detached[frame, top] - detached[frame, token])
                root_allowed = (
                    top == token
                    or (
                        (blank_dominant or top_is_keyword_root)
                        and gap <= RUNTIME_ROOT_START_LOGIT_MARGIN
                    )
                )
                if not root_allowed:
                    continue

            acoustic_log_probability = sample_log_probs[frame, token]
            detached_acoustic = float(detached[frame, token])
            fuzzy_cost = (
                RUNTIME_FUZZY_CHILD_COST_LOG
                if depth != 0 and top != token
                else 0.0
            )
            assign(
                next_nonblank_search,
                next_nonblank_acoustic,
                depth + 1,
                base_search + detached_acoustic + fuzzy_cost,
                base_acoustic + acoustic_log_probability,
            )

        nonblank_search = next_nonblank_search
        separated_search = next_separated_search
        nonblank_acoustic = next_nonblank_acoustic
        separated_acoustic = next_separated_acoustic

        terminal_depth = depth_count
        if nonblank_search[terminal_depth] >= separated_search[terminal_depth]:
            terminal_search = nonblank_search[terminal_depth]
            terminal_acoustic = nonblank_acoustic[terminal_depth]
        else:
            terminal_search = separated_search[terminal_depth]
            terminal_acoustic = separated_acoustic[terminal_depth]
        if terminal_acoustic is None or not math.isfinite(terminal_search):
            continue
        retention_log = terminal_search - float(terminal_acoustic.detach())
        if retention_log < RUNTIME_MIN_PATH_RETENTION_LOG:
            continue
        acoustic_value = float(terminal_acoustic.detach())
        if best_acoustic is None or acoustic_value > best_search:
            best_search = acoustic_value
            best_acoustic = terminal_acoustic

    if best_acoustic is None:
        return sample_log_probs.new_tensor(float("-inf"))
    return best_acoustic / float(depth_count)


def _operating_points(
    count: int,
    *,
    margin: float,
    confidence_threshold: float,
    keyword_operating_points: list[dict] | None,
) -> list[tuple[float, float, float]]:
    if not 0.0 < float(confidence_threshold) < 1.0:
        raise ValueError("decoder confidence threshold must be in (0,1)")
    if not 0.0 <= float(margin) < 0.5:
        raise ValueError("keyword sequence margin is invalid")
    if keyword_operating_points is None:
        raw = [
            {
                "threshold": confidence_threshold,
                "positive_margin": margin,
                "negative_margin": margin,
            }
            for _ in range(count)
        ]
    else:
        if len(keyword_operating_points) != count:
            raise ValueError("keyword operating points must align with keyword sequences")
        raw = keyword_operating_points

    result: list[tuple[float, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"keyword operating point {index} must be an object")
        threshold = float(item.get("threshold", confidence_threshold))
        positive_margin = float(item.get("positive_margin", margin))
        negative_margin = float(item.get("negative_margin", margin))
        values = (threshold, positive_margin, negative_margin)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("keyword operating point values must be finite")
        if not 0.0 < threshold < 1.0:
            raise ValueError("keyword threshold must be in (0,1)")
        if positive_margin < 0.0 or negative_margin < 0.0:
            raise ValueError("keyword margins must be non-negative")
        lower = threshold - negative_margin
        upper = threshold + positive_margin
        if not 0.0 < lower < threshold <= upper < 1.0:
            raise ValueError("keyword operating band must stay inside (0,1)")
        result.append((threshold, positive_margin, negative_margin))
    return result


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
    keyword_operating_points: list[dict] | None = None,
    negative_path_policy: str = SEQUENCE_MARGIN_NEGATIVE_POLICY_DEFAULT,
) -> torch.Tensor:
    """Return a per-sample decoder-confidence operating-band hinge.

    Each wake path may now own an explicit runtime-aligned operating point. This
    lets a multi-keyword product spend more negative margin on a confusion-prone
    wake word without forcing the same recall/false-accept tradeoff onto every
    other wake word.

    For each keyword:

    * a genuine wake must reach ``threshold + positive_margin``;
    * that path, when competing or non-wake, must stay at or below
      ``threshold - negative_margin``;
    * the worst hinge wins because product semantics are wake-on-any-keyword.

    Passing no per-keyword operating points preserves the historical single
    threshold/symmetric-margin behavior exactly.
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
    if negative_path_policy not in SEQUENCE_MARGIN_NEGATIVE_POLICIES:
        raise ValueError("sequence-margin negative path policy is invalid")

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

    points = _operating_points(
        len(normalized_keywords),
        margin=margin,
        confidence_threshold=confidence_threshold,
        keyword_operating_points=keyword_operating_points,
    )
    positive_floors = [
        log_probs.new_tensor(math.log(threshold + positive_margin))
        for threshold, positive_margin, _ in points
    ]
    negative_ceilings = [
        log_probs.new_tensor(math.log(threshold - negative_margin))
        for threshold, _, negative_margin in points
    ]

    true_rows = _target_rows(targets, target_lengths)
    losses: list[torch.Tensor] = []
    for batch_index, true_row in enumerate(true_rows):
        steps = int(input_lengths[batch_index])
        if steps <= 0 or steps > int(log_probs.shape[0]):
            raise ValueError("input length is outside model output")
        sample = log_probs[:steps, batch_index, :]
        sparse_scores = [
            _decoder_sequence_log_confidence(sample, sequence)
            for sequence in normalized_keywords
        ]
        if negative_path_policy == SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE:
            root_tokens = frozenset(sequence[0] for sequence in normalized_keywords)
            negative_scores = [
                _runtime_executable_sequence_log_confidence(
                    sample,
                    sequence,
                    keyword_root_tokens=root_tokens,
                    blank=blank,
                )
                for sequence in normalized_keywords
            ]
        else:
            negative_scores = sparse_scores
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
            hinges.append(
                torch.relu(positive_floors[wake_index] - sparse_scores[wake_index])
            )
        for index, score in enumerate(negative_scores):
            if index == wake_index:
                continue
            hinges.append(torch.relu(score - negative_ceilings[index]))
        if hinges:
            losses.append(torch.stack(hinges).amax())
        else:
            losses.append(log_probs[:, batch_index, :].sum() * 0.0)

    return torch.stack(losses)
