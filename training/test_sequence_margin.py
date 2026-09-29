#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import pathlib

import torch
import torch.nn.functional as F

from completion_loss import strict_prefix_completion_loss
from objective_config import (
    optional_objective_cli_args,
    ordered_token_scope_setting,
    path_purity_settings,
    sequence_margin_negative_policy_setting,
    sequence_margin_positive_policy_setting,
)
from path_purity import ordered_path_purity_loss
from sequence_margin import (
    RUNTIME_BLANK_RETENTION,
    RUNTIME_FUZZY_CHILD_COST_LOG,
    RUNTIME_MIN_PATH_RETENTION_LOG,
    RUNTIME_ROOT_START_LOGIT_MARGIN,
    RUNTIME_STATE_RETENTION,
    keyword_sequence_margin_loss,
)
from train_ctc import (
    exact_keyword_sample_mask,
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
    negative_path_policy: str = "sparse-chronological-v1",
    positive_path_policy: str = "sparse-chronological-v1",
    ctc_log_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    effective_ctc = log_probs if ctc_log_probs is None else ctc_log_probs
    targets, input_lengths, target_lengths, raw = true_nll(
        effective_ctc, target
    )
    return keyword_sequence_margin_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        true_ctc_nll=raw,
        keyword_sequences=keyword_sequences
        or [[1, 2, 3, 4], [3, 4, 3, 4]],
        ctc_log_probs=effective_ctc,
        blank=0,
        margin=0.05,
        keyword_operating_points=keyword_operating_points,
        negative_path_policy=negative_path_policy,
        positive_path_policy=positive_path_policy,
    )


def path_purity(log_probs: torch.Tensor, target: list[int]) -> torch.Tensor:
    targets, input_lengths, target_lengths, _ = true_nll(log_probs, target)
    return ordered_path_purity_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
        blank=0,
        margin=0.10,
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
    contract = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "configs/parameter-contract.json")
        .read_text(encoding="utf-8")
    )
    runtime = contract["runtime"]
    constants = contract["algorithm_constants"]
    assert math.isclose(
        RUNTIME_STATE_RETENTION,
        float(runtime["state_retention"]["default"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        math.log(RUNTIME_BLANK_RETENTION),
        float(constants["KWS_SILENCE_RETENTION_LOG"]["default"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        RUNTIME_MIN_PATH_RETENTION_LOG,
        float(constants["KWS_MIN_PATH_RETENTION_LOG"]["default"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        RUNTIME_ROOT_START_LOGIT_MARGIN,
        float(constants["KWS_ROOT_START_LOGIT_MARGIN"]["default"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        RUNTIME_FUZZY_CHILD_COST_LOG,
        float(constants["KWS_FUZZY_CHILD_RETENTION_COST_LOG"]["default"]),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )

    assert optional_objective_cli_args({}) == []
    scope, scope_configured = ordered_token_scope_setting({})
    assert scope == "all-nonempty-targets-v1"
    assert scope_configured is False
    assert optional_objective_cli_args(
        {"ordered_token_scope": "exact-configured-wake-targets-v1"}
    ) == [
        "--ordered-token-scope",
        "exact-configured-wake-targets-v1",
    ]
    try:
        ordered_token_scope_setting({"ordered_token_scope": "unsupported"})
    except ValueError as exc:
        assert "ordered_token_scope" in str(exc)
    else:
        raise AssertionError("unsupported ordered-token scope was accepted")
    weight, purity_margin, configured = path_purity_settings({})
    assert weight == 0.0 and purity_margin == 0.10 and configured is False
    assert optional_objective_cli_args(
        {"path_purity_loss_weight": 0.10, "path_purity_margin": 0.10}
    ) == [
        "--path-purity-loss-weight",
        "0.1",
        "--path-purity-margin",
        "0.1",
    ]
    try:
        path_purity_settings({"path_purity_loss_weight": -0.1})
    except ValueError as exc:
        assert "path_purity_loss_weight" in str(exc)
    else:
        raise AssertionError("negative path-purity objective weight was accepted")

    negative_policy, negative_policy_configured = (
        sequence_margin_negative_policy_setting({})
    )
    assert negative_policy == "sparse-chronological-v1"
    assert negative_policy_configured is False
    assert optional_objective_cli_args(
        {"sequence_margin_negative_policy": "runtime-executable-v1"}
    ) == [
        "--sequence-margin-negative-policy",
        "runtime-executable-v1",
    ]
    try:
        sequence_margin_negative_policy_setting(
            {"sequence_margin_negative_policy": "unsupported"}
        )
    except ValueError as exc:
        assert "sequence_margin_negative_policy" in str(exc)
    else:
        raise AssertionError("unsupported sequence-margin negative policy was accepted")

    positive_policy, positive_policy_configured = (
        sequence_margin_positive_policy_setting({})
    )
    assert positive_policy == "sparse-chronological-v1"
    assert positive_policy_configured is False
    assert optional_objective_cli_args(
        {"sequence_margin_positive_policy": "ctc-token-state-target-blank-v1"}
    ) == [
        "--sequence-margin-positive-policy",
        "ctc-token-state-target-blank-v1",
    ]
    for retired_or_invalid in (
        "runtime-search-aligned-v1",
        "ctc-keyword-competition-v1",
        "unsupported",
    ):
        try:
            optional_objective_cli_args(
                {"sequence_margin_positive_policy": retired_or_invalid}
            )
        except ValueError as exc:
            assert "sequence_margin_positive_policy" in str(exc)
        else:
            raise AssertionError(
                "retired/unsupported positive policy was accepted: "
                + retired_or_invalid
            )

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

    # The current sparse chronological surrogate can assemble a high-confidence
    # wake from token frames that the runtime cannot execute because multiple
    # child advances are non-dominant and exhaust the fuzzy path budget.
    loose_logits = torch.full((12, 1, 6), -6.0, dtype=torch.float32)
    loose_logits[:, :, 0] = 4.0
    loose_logits[1, 0, 0] = -6.0
    loose_logits[1, 0, 1] = 8.0
    for position, token in zip([3, 5, 7], [2, 3, 4]):
        loose_logits[position, 0, 0] = -6.0
        loose_logits[position, 0, token] = 5.0
        loose_logits[position, 0, 5] = 5.1
    loose_log_probs = loose_logits.requires_grad_().log_softmax(dim=2)
    sparse_loose = margin(
        loose_log_probs,
        [],
        keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
    )
    runtime_loose = margin(
        loose_log_probs,
        [],
        keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
        negative_path_policy="runtime-executable-v1",
    )
    assert float(sparse_loose.item()) > 0.01
    assert float(runtime_loose.item()) < 1.0e-6

    # A genuinely executable negative path must remain penalized; the new policy
    # narrows negative pressure rather than deleting it.
    executable_negative = make_logits([3, 4, 3, 4])
    sparse_executable = margin(executable_negative, [3, 4, 3])
    runtime_executable = margin(
        executable_negative,
        [3, 4, 3],
        negative_path_policy="runtime-executable-v1",
    )
    assert float(sparse_executable.item()) > 0.05
    assert float(runtime_executable.item()) > 0.05

    # Negative runtime alignment remains independent from positive policy.
    weak_runtime_positive = margin(
        weak_log_probs,
        [1, 2, 3, 4],
        negative_path_policy="runtime-executable-v1",
    )
    assert abs(
        float(weak_runtime_positive.item()) - float(weak_loss.item())
    ) < 1.0e-6

    # The optional positive runtime-search policy must remain finite and
    # differentiable even when the selected terminal path is below the runtime
    # retention gate. This is the retained #400 failure mode: fuzzy target
    # advances reach the terminal but are not executable as a detection.
    fuzzy_positive_logits = torch.full((12, 1, 6), -6.0, dtype=torch.float32)
    fuzzy_positive_logits[:, :, 0] = 4.0
    fuzzy_positive_logits[1, 0, 0] = -6.0
    fuzzy_positive_logits[1, 0, 1] = 8.0
    for position, token in zip([3, 5, 7], [2, 3, 4]):
        fuzzy_positive_logits[position, 0, 0] = -6.0
        fuzzy_positive_logits[position, 0, token] = 5.0
        fuzzy_positive_logits[position, 0, 5] = 5.1
    fuzzy_positive = fuzzy_positive_logits.requires_grad_().log_softmax(dim=2)
    aligned_positive = margin(
        fuzzy_positive,
        [1, 2, 3, 4],
        positive_path_policy="runtime-search-aligned-v1",
    )
    assert torch.isfinite(aligned_positive).all()
    assert float(aligned_positive.item()) > 0.0
    aligned_positive.mean().backward()
    assert fuzzy_positive_logits.grad is not None

    clean_runtime_positive = margin(
        make_logits([1, 2, 3, 4]),
        [1, 2, 3, 4],
        positive_path_policy="runtime-search-aligned-v1",
    )
    assert float(clean_runtime_positive.item()) < 1.0e-6

    # The CTC keyword-competition policy directly ranks configured wake
    # sequences.  Unlike the threshold hinge it remains differentiable when the
    # true wake already clears its runtime confidence floor.
    clean_competition_logits = torch.full(
        (14, 1, 5), -6.0, dtype=torch.float32
    )
    clean_competition_logits[:, :, 0] = 5.0
    for position, token in zip([1, 4, 7, 10], [1, 2, 3, 4]):
        clean_competition_logits[position, 0, 0] = -6.0
        clean_competition_logits[position, 0, token] = 8.0
    clean_competition = clean_competition_logits.requires_grad_().log_softmax(
        dim=2
    )
    clean_competition_loss = margin(
        clean_competition,
        [1, 2, 3, 4],
        positive_path_policy="ctc-keyword-competition-v1",
        negative_path_policy="runtime-executable-v1",
    )
    assert torch.isfinite(clean_competition_loss).all()
    assert float(clean_competition_loss.item()) > 0.0
    clean_competition_loss.mean().backward()
    assert clean_competition_logits.grad is not None

    ambiguous_logits = torch.full((14, 1, 5), -6.0, dtype=torch.float32)
    ambiguous_logits[:, :, 0] = 5.0
    for position, token in zip([1, 4, 7, 10], [1, 2, 3, 4]):
        ambiguous_logits[position, 0, 0] = -6.0
        ambiguous_logits[position, 0, token] = 8.0
    for position, token in zip([2, 5, 8, 11], [3, 4, 3, 4]):
        ambiguous_logits[position, 0, 0] = -6.0
        ambiguous_logits[position, 0, token] = 8.0
    ambiguous = ambiguous_logits.requires_grad_().log_softmax(dim=2)
    ambiguous_competition = margin(
        ambiguous,
        [1, 2, 3, 4],
        positive_path_policy="ctc-keyword-competition-v1",
        negative_path_policy="runtime-executable-v1",
    )
    assert float(ambiguous_competition.item()) > float(
        clean_competition_loss.item()
    )
    ambiguous_competition.mean().backward()
    assert ambiguous_logits.grad is not None

    # CTC competition must consume the same VAD-aligned probabilities as the
    # trainer's primary CTC objective, rather than silently falling back to raw
    # decoder log-probabilities.
    masked_logits = ambiguous.detach().clone()
    masked_logits[:6, 0, :] = -20.0
    masked_logits[:6, 0, 0] = 0.0
    masked_competition = margin(
        ambiguous.detach(),
        [1, 2, 3, 4],
        positive_path_policy="ctc-keyword-competition-v1",
        ctc_log_probs=masked_logits,
    )
    raw_competition = margin(
        ambiguous.detach(),
        [1, 2, 3, 4],
        positive_path_policy="ctc-keyword-competition-v1",
    )
    assert abs(
        float(masked_competition.item()) - float(raw_competition.item())
    ) > 1.0e-4

    # The fixed-logit-qualified token-state policy must strengthen every
    # true wake token against blank on detached exact CTC token-state support.
    token_state_logits = torch.full((14, 1, 5), -8.0, dtype=torch.float32)
    token_state_logits[:, :, 0] = 3.0
    for position, token in zip([1, 4, 7, 10], [1, 2, 3, 4]):
        token_state_logits[position, 0, token] = 4.0
    token_state_logits.requires_grad_()
    token_state_log_probs = token_state_logits.log_softmax(dim=2)
    token_state_loss = margin(
        token_state_log_probs,
        [1, 2, 3, 4],
        positive_path_policy="ctc-token-state-target-blank-v1",
        negative_path_policy="runtime-executable-v1",
    )
    assert torch.isfinite(token_state_loss).all()
    assert float(token_state_loss.item()) > 0.0
    token_state_loss.mean().backward()
    assert token_state_logits.grad is not None
    assert float(token_state_logits.grad[:, 0, 0].sum()) > 0.0
    assert (
        sum(
            float(token_state_logits.grad[:, 0, token].sum())
            for token in (1, 2, 3, 4)
        )
        < 0.0
    )

    # The policy must consume the same CTC/VAD-aligned probabilities as the
    # primary CTC objective rather than silently using decoder probabilities.
    detached_token_state = token_state_log_probs.detach()
    aligned_token_state = detached_token_state.clone()
    aligned_token_state[:3, 0, :] = -30.0
    aligned_token_state[:3, 0, 0] = 0.0
    aligned_token_state_loss = margin(
        detached_token_state,
        [1, 2, 3, 4],
        positive_path_policy="ctc-token-state-target-blank-v1",
        negative_path_policy="runtime-executable-v1",
        ctc_log_probs=aligned_token_state,
    )
    raw_token_state_loss = margin(
        detached_token_state,
        [1, 2, 3, 4],
        positive_path_policy="ctc-token-state-target-blank-v1",
        negative_path_policy="runtime-executable-v1",
    )
    assert abs(
        float(aligned_token_state_loss.item()) - float(raw_token_state_loss.item())
    ) > 1.0e-6

    # Path-purity targets the #240 failure mode: the target sequence may be
    # present as a loose subsequence while an unrelated wake token dominates
    # inside an ordered region. Clean wake evidence should be free of this
    # penalty, while an inserted out-of-order token must receive gradient.
    clean_path = make_logits([1, 2, 3, 4])
    clean_purity = path_purity(clean_path, [1, 2, 3, 4])
    assert float(clean_purity.item()) < 1.0e-6

    polluted_logits = torch.full((12, 1, 5), -6.0, dtype=torch.float32)
    polluted_logits[:, :, 0] = 4.0
    for position, token in zip([1, 3, 6, 9], [1, 2, 3, 4]):
        polluted_logits[position, 0, 0] = -6.0
        polluted_logits[position, 0, token] = 8.0
    # wo1 is unrelated to the allowed {blank, hao3, xiao3} set in region 2.
    polluted_logits[4, 0, 0] = -6.0
    polluted_logits[4, 0, 4] = 10.0
    polluted_log_probs = polluted_logits.requires_grad_().log_softmax(dim=2)
    polluted_purity = path_purity(polluted_log_probs, [1, 2, 3, 4])
    assert float(polluted_purity.item()) > 0.05
    polluted_purity.mean().backward()
    assert polluted_logits.grad is not None
    assert abs(float(polluted_logits.grad[4, 0, 4])) > 0.0

    # Tokenized near-misses remain outside this positive-only objective.
    near_miss = make_logits([1, 4, 2])
    near_miss_purity = path_purity(near_miss, [1, 4, 2])
    assert float(near_miss_purity.item()) == 0.0

    repeat_wake = make_logits([3, 4, 3, 4])
    repeat_purity = path_purity(repeat_wake, [3, 4, 3, 4])
    assert float(repeat_purity.item()) < 1.0e-6

    try:
        targets, input_lengths, target_lengths, _ = true_nll(clean_path, [1, 2, 3, 4])
        ordered_path_purity_loss(
            log_probs=clean_path,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
            margin=2.1,
        )
    except ValueError as exc:
        assert "path-purity margin" in str(exc)
    else:
        raise AssertionError("out-of-range path-purity margin was accepted")

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
    exact_mask = exact_keyword_sample_mask(
        wake_targets,
        wake_lengths,
        [[1, 2, 3, 4], [3, 4, 3, 4]],
    )
    assert torch.equal(
        exact_mask,
        torch.tensor([True, True, False, False], dtype=torch.bool),
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
    assert abs(float(stats["exact_wake_weight_sum"]) - 12.0) < 1.0e-12
    assert abs(float(stats["exact_wake_mean_weight"]) - 6.0) < 1.0e-12
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
    wake_only_ordered, wake_only_correct, wake_only_total = ordered_token_loss(
        ordered_log_probs,
        ordered_targets,
        ordered_input_lengths,
        ordered_target_lengths,
        torch.tensor([4.0, 1.0], dtype=torch.float32),
        normalization_mean_weight=4.0,
        sample_mask=torch.tensor([True, False], dtype=torch.bool),
    )
    ordered_first, _, _ = ordered_token_loss(
        ordered_log_probs[:, :1, :],
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4.0], dtype=torch.float32),
        normalization_mean_weight=2.5,
    )
    ordered_first_exact_scope, _, _ = ordered_token_loss(
        ordered_log_probs[:, :1, :],
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([4.0], dtype=torch.float32),
        normalization_mean_weight=4.0,
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
    assert wake_only_total == 1
    assert wake_only_correct == 0
    assert (
        abs(
            float(wake_only_ordered.item())
            - float(ordered_first_exact_scope.item())
        )
        < 1.0e-7
    )
    assert abs(
        float(normalized_ordered.item())
        - float(((ordered_first + ordered_second) / 2.0).item())
    ) < 1.0e-7
    # Empty targets have no ordered-token term. The dataset participation rate
    # keeps the epoch contribution invariant when the same rows are regrouped.
    mixed_lengths = torch.tensor([1, 0], dtype=torch.long)
    mixed_weights = torch.tensor([2.0, 1.0], dtype=torch.float32)
    mixed, _, _ = ordered_token_loss(
        ordered_log_probs,
        torch.tensor([1], dtype=torch.long),
        ordered_input_lengths,
        mixed_lengths,
        mixed_weights,
        normalization_mean_weight=2.0,
        normalization_participation_rate=0.5,
    )
    positive_only, _, _ = ordered_token_loss(
        ordered_log_probs[:, :1, :],
        torch.tensor([1], dtype=torch.long),
        ordered_input_lengths[:1],
        mixed_lengths[:1],
        mixed_weights[:1],
        normalization_mean_weight=2.0,
        normalization_participation_rate=0.5,
    )
    empty_only, _, _ = ordered_token_loss(
        ordered_log_probs[:, 1:, :],
        torch.empty(0, dtype=torch.long),
        ordered_input_lengths[1:],
        mixed_lengths[1:],
        mixed_weights[1:],
        normalization_mean_weight=2.0,
        normalization_participation_rate=0.5,
    )
    assert (
        abs(float(mixed.item()) - float(((positive_only + empty_only) / 2).item()))
        < 1.0e-7
    )
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
