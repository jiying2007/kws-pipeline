#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn.functional as F

from completion_loss import strict_prefix_completion_loss
from sequence_margin import keyword_sequence_margin_loss
from train_ctc import (
    normalized_weighted_mean,
    ordered_token_loss,
    sample_weight_statistics,
    wake_example_weights,
)


def make_logits(tokens: list[int], *, steps: int = 12, vocab: int = 5) -> torch.Tensor:
    logits = torch.full((steps, 1, vocab), -6.0, dtype=torch.float32)
    logits[:, :, 0] = 4.0
    positions = [1 + 2 * index for index in range(len(tokens))]
    for position, token in zip(positions, tokens):
        logits[position, 0, 0] = -6.0
        logits[position, 0, token] = 8.0
    return logits.requires_grad_().log_softmax(dim=2)


def true_nll(log_probs: torch.Tensor, target: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.tensor(target, dtype=torch.long)
    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long)
    target_lengths = torch.tensor([len(target)], dtype=torch.long)
    raw = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    return targets, input_lengths, target_lengths, raw


def operating_points() -> list[dict]:
    return [
        {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.06},
        {"threshold": 0.55, "positive_margin": 0.055, "negative_margin": 0.06},
    ]


def margin(
    log_probs: torch.Tensor,
    target: list[int],
    keyword_sequences: list[list[int]] | None = None,
    keyword_operating_points: list[dict] | None = None,
) -> torch.Tensor:
    targets, input_lengths, target_lengths, raw = true_nll(log_probs, target)
    return keyword_sequence_margin_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        true_ctc_nll=raw,
        keyword_sequences=keyword_sequences
        or [[1, 2, 3, 4], [3, 4, 3, 4]],
        blank=0,
        margin=0.05,
        keyword_operating_points=keyword_operating_points,
    )


def completion(log_probs: torch.Tensor, target: list[int]) -> torch.Tensor:
    targets, input_lengths, target_lengths, _ = true_nll(log_probs, target)
    return strict_prefix_completion_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
        keyword_operating_points=operating_points(),
        tail_steps=6,
    )


def main() -> int:
    unsafe = make_logits([3, 4, 3, 4])
    unsafe_loss = margin(unsafe, [3, 4, 3])
    assert float(unsafe_loss.item()) > 0.05
    unsafe_loss.mean().backward()
    assert unsafe.grad_fn is not None

    unsafe_more_keywords = make_logits([3, 4, 3, 4])
    expanded_loss = margin(
        unsafe_more_keywords,
        [3, 4, 3],
        [[1, 2, 3, 4], [3, 4, 3, 4], [2, 3, 2, 3]],
    )
    assert abs(float(expanded_loss.item()) - float(unsafe_loss.item())) < 1.0e-5

    safe = make_logits([3, 4, 3])
    safe_loss = margin(safe, [3, 4, 3])
    assert float(safe_loss.item()) < 1.0e-6

    positive = make_logits([1, 2, 3, 4])
    positive_loss = margin(positive, [1, 2, 3, 4])
    assert float(positive_loss.item()) < 1.0e-6

    weak_logits = torch.full((12, 1, 5), -6.0, dtype=torch.float32)
    weak_logits[:, :, 0] = 6.0
    for position, token in zip([1, 3, 5, 7], [1, 2, 3, 4]):
        weak_logits[position, 0, token] = 4.0
    weak_log_probs = weak_logits.requires_grad_().log_softmax(dim=2)
    weak_loss = margin(weak_log_probs, [1, 2, 3, 4])
    assert float(weak_loss.item()) > 0.50
    weak_loss.mean().backward()
    assert weak_logits.grad is not None

    # The #132/#135 failure mode is a strict "你好小" prefix with a spurious
    # terminal wo1 posterior in the acoustic tail. The completion hinge must
    # concentrate gradient on that missing terminal token without changing the
    # shipping 0.55 threshold or global sequence margin.
    prefix_logits = torch.full((12, 1, 5), -6.0, dtype=torch.float32)
    prefix_logits[:, :, 0] = 5.0
    for position, token in zip([1, 3, 5], [1, 2, 3]):
        prefix_logits[position, 0, 0] = -6.0
        prefix_logits[position, 0, token] = 8.0
    prefix_logits[9, 0, 0] = -1.0
    prefix_logits[9, 0, 4] = 7.0
    prefix_log_probs = prefix_logits.requires_grad_().log_softmax(dim=2)
    prefix_completion = completion(prefix_log_probs, [1, 2, 3])
    assert float(prefix_completion.item()) > 0.05
    prefix_completion.mean().backward()
    assert prefix_logits.grad is not None
    assert abs(float(prefix_logits.grad[9, 0, 4])) > 0.0

    # A strict prefix with a blank-dominant terminal window is already safe.
    safe_prefix = make_logits([1, 2, 3])
    safe_prefix_completion = completion(safe_prefix, [1, 2, 3])
    assert float(safe_prefix_completion.item()) < 1.0e-6

    # Full wake positives and unrelated negatives must not receive the strict-
    # prefix terminal penalty; their discrimination remains owned by CTC,
    # ordered-token, sequence-margin and recurrent-release objectives.
    complete_wake = make_logits([1, 2, 3, 4])
    assert float(completion(complete_wake, [1, 2, 3, 4]).item()) == 0.0
    unrelated = make_logits([2, 1, 3, 4])
    assert float(completion(unrelated, [2, 1, 3]).item()) == 0.0

    # A stricter negative margin on keyword 2 must increase the penalty for a
    # partial "小窝小" target whose acoustic continuation realizes "小窝小窝".
    default_partial = make_logits([3, 4, 3, 4])
    default_partial_loss = margin(default_partial, [3, 4, 3])
    strict_partial = make_logits([3, 4, 3, 4])
    strict_partial_loss = margin(
        strict_partial,
        [3, 4, 3],
        keyword_operating_points=[
            {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.05},
            {"threshold": 0.55, "positive_margin": 0.055, "negative_margin": 0.06},
        ],
    )
    assert float(strict_partial_loss.item()) > float(default_partial_loss.item())

    # Per-keyword thresholds are independent: tightening keyword 1 must not move
    # the keyword-2 negative ceiling when keyword 2 keeps the same operating point.
    same_kw2 = make_logits([3, 4, 3, 4])
    same_kw2_loss = margin(
        same_kw2,
        [3, 4, 3],
        keyword_operating_points=[
            {"threshold": 0.60, "positive_margin": 0.03, "negative_margin": 0.03},
            {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.05},
        ],
    )
    assert abs(float(same_kw2_loss.item()) - float(default_partial_loss.item())) < 1.0e-5

    short_logits = torch.full((2, 1, 5), -6.0, dtype=torch.float32)
    short_logits[:, :, 0] = 6.0
    short_log_probs = short_logits.log_softmax(dim=2)
    empty_targets = torch.empty(0, dtype=torch.long)
    short_lengths = torch.tensor([2], dtype=torch.long)
    empty_lengths = torch.tensor([0], dtype=torch.long)
    blank_nll = F.ctc_loss(
        short_log_probs,
        empty_targets,
        short_lengths,
        empty_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    impossible_loss = keyword_sequence_margin_loss(
        log_probs=short_log_probs,
        targets=empty_targets,
        input_lengths=short_lengths,
        target_lengths=empty_lengths,
        true_ctc_nll=blank_nll,
        keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
        blank=0,
        margin=0.05,
    )
    assert torch.isfinite(impossible_loss).all()
    assert float(impossible_loss.item()) == 0.0

    # Per-keyword refinement weighting must distinguish the two exact wake
    # targets while leaving tokenized near-misses and empty targets at 1.0.
    wake_targets = torch.tensor(
        [1, 2, 3, 4, 3, 4, 3, 4, 1, 2, 3],
        dtype=torch.long,
    )
    wake_lengths = torch.tensor([4, 4, 3, 0], dtype=torch.long)
    per_keyword_weights = wake_example_weights(
        wake_targets,
        wake_lengths,
        [[1, 2, 3, 4], [3, 4, 3, 4]],
        [1, 2],
        default_weight=1.0,
        keyword_weights={1: 4.25, 2: 2.75},
    )
    assert torch.allclose(
        per_keyword_weights,
        torch.tensor([4.25, 2.75, 1.0, 1.0], dtype=torch.float32),
    )

    stats = sample_weight_statistics(
        [
            (pathlib.Path("wake1.wav"), [1, 2, 3, 4]),
            (pathlib.Path("wake2.wav"), [3, 4, 3, 4]),
            (pathlib.Path("near.wav"), [1, 2, 3]),
            (pathlib.Path("negative.wav"), []),
        ],
        [[1, 2, 3, 4], [3, 4, 3, 4]],
        [1, 2],
        positive_example_weight=2.0,
        default_wake_weight=1.0,
        keyword_weights={1: 4.0, 2: 2.0},
    )
    assert stats["policy"] == "dataset-mean-sample-weight-v1"
    assert stats["rows"] == 4
    assert stats["nonempty_rows"] == 3
    assert stats["exact_wake_rows"] == 2
    assert abs(float(stats["all_weight_sum"]) - 15.0) < 1.0e-12
    assert abs(float(stats["all_mean_weight"]) - 3.75) < 1.0e-12
    assert abs(float(stats["nonempty_weight_sum"]) - 14.0) < 1.0e-12
    assert abs(float(stats["nonempty_mean_weight"]) - (14.0 / 3.0)) < 1.0e-12

    # Dataset-mean normalization is invariant to batch partitioning when batch
    # losses are aggregated by sample count. A homogeneous high-weight batch
    # therefore keeps its intended larger contribution instead of cancelling
    # its own multiplier through a batch-local denominator.
    values = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float32)
    weights = torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float32)
    full = normalized_weighted_mean(values, weights, 3.75)
    first = normalized_weighted_mean(values[:1], weights[:1], 3.75)
    rest = normalized_weighted_mean(values[1:], weights[1:], 3.75)
    partitioned = (first * 1.0 + rest * 3.0) / 4.0
    assert abs(float(full.item()) - float(partitioned.item())) < 1.0e-7

    # Refinement wake balancing must also affect ordered-token loss. Equal
    # sample weights preserve the legacy mean exactly; only a non-uniform wake
    # weight may move this auxiliary objective.
    ordered_logits = torch.zeros((4, 2, 3), dtype=torch.float32)
    ordered_logits[:, 0, 0] = 4.0
    ordered_logits[:, 0, 1] = 0.0
    ordered_logits[:, 1, 0] = 0.0
    ordered_logits[:, 1, 1] = 4.0
    ordered_log_probs = ordered_logits.log_softmax(dim=2)
    ordered_targets = torch.tensor([1, 1], dtype=torch.long)
    ordered_input_lengths = torch.tensor([4, 4], dtype=torch.long)
    ordered_target_lengths = torch.tensor([1, 1], dtype=torch.long)
    unweighted_ordered, _, _ = ordered_token_loss(
        ordered_log_probs,
        ordered_targets,
        ordered_input_lengths,
        ordered_target_lengths,
    )
    equal_weight_ordered, _, _ = ordered_token_loss(
        ordered_log_probs,
        ordered_targets,
        ordered_input_lengths,
        ordered_target_lengths,
        torch.tensor([2.0, 2.0], dtype=torch.float32),
    )
    wake_weighted_ordered, _, _ = ordered_token_loss(
        ordered_log_probs,
        ordered_targets,
        ordered_input_lengths,
        ordered_target_lengths,
        torch.tensor([4.0, 1.0], dtype=torch.float32),
    )
    normalized_ordered, _, _ = ordered_token_loss(
        ordered_log_probs,
        ordered_targets,
        ordered_input_lengths,
        ordered_target_lengths,
        torch.tensor([4.0, 1.0], dtype=torch.float32),
        normalization_mean_weight=2.5,
    )
    ordered_first, _, _ = ordered_token_loss(
        ordered_log_probs[:, :1, :],
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4.0], dtype=torch.float32),
        normalization_mean_weight=2.5,
    )
    ordered_second, _, _ = ordered_token_loss(
        ordered_log_probs[:, 1:, :],
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1.0], dtype=torch.float32),
        normalization_mean_weight=2.5,
    )
    assert abs(float(equal_weight_ordered.item()) - float(unweighted_ordered.item())) < 1.0e-7
    assert float(wake_weighted_ordered.item()) > float(unweighted_ordered.item())
    assert abs(float(normalized_ordered.item()) - float(wake_weighted_ordered.item())) < 1.0e-7
    assert abs(
        float(normalized_ordered.item())
        - float(((ordered_first + ordered_second) / 2.0).item())
    ) < 1.0e-7
    try:
        ordered_token_loss(
            ordered_log_probs,
            ordered_targets,
            ordered_input_lengths,
            ordered_target_lengths,
            torch.tensor([1.0, 0.0], dtype=torch.float32),
        )
    except ValueError as exc:
        assert "finite and > 0" in str(exc)
    else:
        raise AssertionError("non-positive ordered-token sample weight was accepted")

    for bad in (
        [{"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.05}],
        [
            {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.05},
            {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.60},
        ],
    ):
        try:
            margin(make_logits([1, 2, 3, 4]), [1, 2, 3, 4], keyword_operating_points=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid keyword operating point must fail closed")

    print("test_sequence_margin: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
