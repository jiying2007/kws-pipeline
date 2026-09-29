#!/usr/bin/env python3
"""Audit a local CTC token-state target-vs-blank objective on retained logits."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from diagnose_decoder_search_path_decomposition import load_positive_sample  # noqa: E402
from diagnose_keyword_ctc_competition_gradient import (  # noqa: E402
    ctc_true_nll,
    read_margin_weight,
    scalar_stats,
)
from diagnose_sequence_margin_runtime_gap import (  # noqa: E402
    load_keywords,
    load_tokens,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "ctc-token-state-blank-gradient-audit-development-v1"
POLICY = "detached-ctc-token-state-target-vs-blank-v1"
CANDIDATE = "ctc-token-state-target-blank-v1"
PRIMARY_GRADIENT_RATIO_MAX = 1.0


def logsumexp(values: list[float]) -> float:
    maximum = max(values)
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def extended_target(sequence: tuple[int, ...], blank: int = 0) -> tuple[int, ...]:
    result = [blank]
    for token in sequence:
        result.extend((int(token), blank))
    return tuple(result)


def allowed_predecessors(
    states: tuple[int, ...], index: int, blank: int = 0
) -> tuple[int, ...]:
    result = [index]
    if index > 0:
        result.append(index - 1)
    if (
        index > 1
        and states[index] != blank
        and states[index] != states[index - 2]
    ):
        result.append(index - 2)
    return tuple(result)


def allowed_successors(
    states: tuple[int, ...], index: int, blank: int = 0
) -> tuple[int, ...]:
    result = [index]
    if index + 1 < len(states):
        result.append(index + 1)
    if (
        index + 2 < len(states)
        and states[index + 2] != blank
        and states[index + 2] != states[index]
    ):
        result.append(index + 2)
    return tuple(result)


def token_state_temporal_weights(
    log_probs: list[list[float]],
    sequence: tuple[int, ...],
    *,
    blank: int = 0,
) -> list[list[float]]:
    """Return normalized frame posterior for each target-token CTC state."""
    if not log_probs or not sequence:
        raise ValueError("token-state alignment requires non-empty trace/sequence")
    vocab = len(log_probs[0])
    if any(len(row) != vocab for row in log_probs):
        raise ValueError("posterior rows must share one vocabulary")
    if blank < 0 or blank >= vocab:
        raise ValueError("blank token is outside vocabulary")
    if any(token <= blank or token >= vocab for token in sequence):
        raise ValueError("target token is outside vocabulary")

    states = extended_target(sequence, blank)
    steps = len(log_probs)
    count = len(states)
    alpha = [[float("-inf")] * count for _ in range(steps)]
    alpha[0][0] = log_probs[0][blank]
    if count > 1:
        alpha[0][1] = log_probs[0][states[1]]
    for frame in range(1, steps):
        for state, token in enumerate(states):
            alpha[frame][state] = (
                logsumexp(
                    [
                        alpha[frame - 1][prior]
                        for prior in allowed_predecessors(states, state, blank)
                    ]
                )
                + log_probs[frame][token]
            )

    log_probability = logsumexp(alpha[-1][-2:])
    if not math.isfinite(log_probability):
        raise ValueError("CTC sequence probability is non-finite")

    beta = [[float("-inf")] * count for _ in range(steps)]
    beta[-1][-1] = 0.0
    if count > 1:
        beta[-1][-2] = 0.0
    for frame in range(steps - 2, -1, -1):
        for state in range(count):
            beta[frame][state] = logsumexp(
                [
                    beta[frame + 1][successor]
                    + log_probs[frame + 1][states[successor]]
                    for successor in allowed_successors(states, state, blank)
                ]
            )

    result: list[list[float]] = []
    for occurrence in range(len(sequence)):
        state = occurrence * 2 + 1
        posterior = [
            math.exp(alpha[frame][state] + beta[frame][state] - log_probability)
            for frame in range(steps)
        ]
        occupancy = sum(posterior)
        if occupancy <= 0.0 or not math.isfinite(occupancy):
            raise ValueError("CTC token-state occupancy is invalid")
        normalized = [value / occupancy for value in posterior]
        if abs(sum(normalized) - 1.0) > 1.0e-6:
            raise ValueError("CTC token-state posterior did not normalize")
        result.append(normalized)
    return result


def common_suffix_length(
    left: tuple[int, ...], right: tuple[int, ...]
) -> int:
    count = 0
    for a, b in zip(reversed(left), reversed(right)):
        if a != b:
            break
        count += 1
    return count


def discriminative_prefix_length(
    sequence: tuple[int, ...],
    keywords: list[tuple[int, ...]],
) -> int:
    shared = max(
        (
            common_suffix_length(sequence, other)
            for other in keywords
            if other != sequence
        ),
        default=0,
    )
    if shared >= len(sequence):
        raise ValueError("configured keyword sequences must be unique")
    return len(sequence) - shared


def candidate_gradient(
    raw_logits: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    temporal: list[list[float]],
) -> tuple[float, torch.Tensor]:
    logits = raw_logits.detach().clone().requires_grad_(True)
    log_probs = logits.log_softmax(dim=1)
    losses: list[torch.Tensor] = []
    for token, weights in zip(sequence, temporal):
        weight = torch.tensor(
            weights,
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        losses.append(
            (
                weight
                * F.softplus(log_probs[:, 0] - log_probs[:, int(token)])
            ).sum()
        )
    loss = torch.stack(losses).mean()
    grad = torch.autograd.grad(loss, logits)[0]
    return float(loss.detach()), grad


def occurrence_updates(
    weighted_update: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    temporal: list[list[float]],
) -> list[dict]:
    rows: list[dict] = []
    for occurrence, (token, weights) in enumerate(zip(sequence, temporal), 1):
        weight = torch.tensor(
            weights,
            dtype=weighted_update.dtype,
            device=weighted_update.device,
        )
        target = float((weight * weighted_update[:, int(token)]).sum())
        blank = float((weight * weighted_update[:, 0]).sum())
        rows.append(
            {
                "occurrence": occurrence,
                "token": int(token),
                "target_update": target,
                "blank_update": blank,
                "target_minus_blank_update": target - blank,
            }
        )
    return rows


def aggregate_occurrences(rows: list[dict]) -> dict:
    return {
        "target_update": sum(float(row["target_update"]) for row in rows),
        "blank_update": sum(float(row["blank_update"]) for row in rows),
        "target_minus_blank_update": sum(
            float(row["target_minus_blank_update"]) for row in rows
        ),
    }


def summarize(records: list[dict]) -> dict:
    scalar_names = (
        "candidate_loss",
        "weighted_candidate_grad_l2",
        "primary_ctc_grad_l2",
        "candidate_to_primary_ctc_grad_l2_ratio",
        "global_blank_update",
        "prefix_target_update",
        "prefix_blank_update",
        "prefix_target_minus_blank_update",
        "suffix_target_update",
        "suffix_blank_update",
        "suffix_target_minus_blank_update",
    )
    result = {
        "recordings": len(records),
        **{
            name: scalar_stats([float(row[name]) for row in records])
            for name in scalar_names
        },
        "prefix_target_minus_blank_increased": sum(
            float(row["prefix_target_minus_blank_update"]) > 0.0
            for row in records
        ),
        "suffix_target_minus_blank_increased": sum(
            int(row["shared_suffix_length"]) == 0
            or float(row["suffix_target_minus_blank_update"]) > 0.0
            for row in records
        ),
        "suffix_blank_nonincreasing": sum(
            int(row["shared_suffix_length"]) == 0
            or float(row["suffix_blank_update"]) <= 0.0
            for row in records
        ),
        "candidate_not_stronger_than_primary_ctc": sum(
            float(row["candidate_to_primary_ctc_grad_l2_ratio"])
            <= PRIMARY_GRADIENT_RATIO_MAX + 1.0e-9
            for row in records
        ),
    }
    return result


def build_gate(records: list[dict]) -> dict:
    failures: list[str] = []
    for row in records:
        audio_sha = str(row["audio_sha256"])
        if float(row["prefix_target_minus_blank_update"]) <= 0.0:
            failures.append(f"{audio_sha}: prefix target-vs-blank did not improve")
        if int(row["shared_suffix_length"]) > 0:
            if float(row["suffix_target_minus_blank_update"]) <= 0.0:
                failures.append(
                    f"{audio_sha}: shared suffix target-vs-blank did not improve"
                )
            if float(row["suffix_blank_update"]) > 0.0:
                failures.append(f"{audio_sha}: shared suffix blank increased")
        if (
            float(row["candidate_to_primary_ctc_grad_l2_ratio"])
            > PRIMARY_GRADIENT_RATIO_MAX + 1.0e-9
        ):
            failures.append(
                f"{audio_sha}: candidate gradient exceeds primary CTC"
            )
    return {
        "policy": "complete-path-local-gradient-safety-v1",
        "primary_ctc_gradient_ratio_max": PRIMARY_GRADIENT_RATIO_MAX,
        "pass": not failures,
        "failures": failures,
    }


def self_test() -> None:
    raw = torch.tensor(
        [
            [2.0, 5.0, -2.0],
            [5.0, 0.0, -2.0],
            [2.0, -2.0, 5.0],
            [5.0, -2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    sequence = (1, 2)
    log_probs = raw.log_softmax(dim=1).tolist()
    temporal = token_state_temporal_weights(log_probs, sequence)
    assert len(temporal) == 2
    assert all(abs(sum(row) - 1.0) < 1.0e-6 for row in temporal)
    loss, grad = candidate_gradient(raw, sequence=sequence, temporal=temporal)
    assert math.isfinite(loss) and loss > 0.0
    update = -grad
    rows = occurrence_updates(update, sequence=sequence, temporal=temporal)
    assert all(float(row["target_update"]) > 0.0 for row in rows)
    assert all(float(row["blank_update"]) < 0.0 for row in rows)
    assert all(float(row["target_minus_blank_update"]) > 0.0 for row in rows)
    print("CTC token-state blank gradient self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--training-readback", type=pathlib.Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    required = {
        "tokens": args.tokens,
        "keywords_tsv": args.keywords_tsv,
        "posterior_cache": args.posterior_cache,
        "acoustic_alignment": args.acoustic_alignment,
        "training_readback": args.training_readback,
        "model_sha256": args.model_sha256,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))

    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    training_readback = args.training_readback.resolve()
    output = args.output.resolve()
    model_sha256 = str(args.model_sha256)

    token_map = load_tokens(tokens)
    keyword_map = load_keywords(keywords_tsv, token_map)
    keyword_ids = sorted(keyword_map)
    keywords = [
        tuple(int(value) for value in keyword_map[keyword_id]["tokens"])
        for keyword_id in keyword_ids
    ]
    if len(keywords) < 2:
        raise ValueError("candidate audit requires multiple configured keywords")

    positive = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    traces = trace_paths(posterior_cache, model_sha256)
    by_sha = {path.stem: path for path in traces}
    weight = read_margin_weight(training_readback, model_sha256=model_sha256)

    records: list[dict] = []
    for audio_sha, sample in sorted(positive.items()):
        trace = by_sha.get(audio_sha)
        if trace is None:
            raise ValueError("retained positive posterior trace is missing")
        keyword_id = int(sample["keyword_id"])
        try:
            keyword_index = keyword_ids.index(keyword_id)
        except ValueError as exc:
            raise ValueError("positive sample references unknown keyword") from exc
        sequence = keywords[keyword_index]

        raw = torch.tensor(read_trace_logits(trace), dtype=torch.float32)
        log_probs = raw.log_softmax(dim=1)
        temporal = token_state_temporal_weights(
            log_probs.detach().tolist(),
            sequence,
        )

        candidate_loss, candidate_grad = candidate_gradient(
            raw,
            sequence=sequence,
            temporal=temporal,
        )

        primary_logits = raw.detach().clone().requires_grad_(True)
        primary_log_probs = primary_logits.log_softmax(dim=1)
        primary_loss = ctc_true_nll(
            primary_log_probs,
            sequence,
        ) / float(primary_log_probs.shape[0])
        primary_grad = torch.autograd.grad(primary_loss, primary_logits)[0]
        primary_grad_l2 = float(primary_grad.norm())
        if primary_grad_l2 <= 0.0:
            raise ValueError("primary CTC gradient norm must be positive")

        weighted_update = -float(weight) * candidate_grad
        occurrence = occurrence_updates(
            weighted_update,
            sequence=sequence,
            temporal=temporal,
        )
        prefix_length = discriminative_prefix_length(sequence, keywords)
        prefix = aggregate_occurrences(occurrence[:prefix_length])
        suffix = aggregate_occurrences(occurrence[prefix_length:])
        shared_suffix_length = len(sequence) - prefix_length

        records.append(
            {
                "audio_sha256": audio_sha,
                "split": sample["split"],
                "keyword_id": keyword_id,
                "runtime_hit": bool(sample["runtime_detected_expected"]),
                "runtime_gap": bool(
                    sample["surrogate_above_runtime_threshold"]
                    and not sample["runtime_detected_expected"]
                ),
                "candidate_loss": candidate_loss,
                "candidate_weight": weight,
                "weighted_candidate_grad_l2": (
                    float(weight) * float(candidate_grad.norm())
                ),
                "primary_ctc_grad_l2": primary_grad_l2,
                "candidate_to_primary_ctc_grad_l2_ratio": (
                    float(weight) * float(candidate_grad.norm())
                    / primary_grad_l2
                ),
                "global_blank_update": float(weighted_update[:, 0].sum()),
                "discriminative_prefix_length": prefix_length,
                "shared_suffix_length": shared_suffix_length,
                "prefix_target_update": prefix["target_update"],
                "prefix_blank_update": prefix["blank_update"],
                "prefix_target_minus_blank_update": (
                    prefix["target_minus_blank_update"]
                ),
                "suffix_target_update": suffix["target_update"],
                "suffix_blank_update": suffix["blank_update"],
                "suffix_target_minus_blank_update": (
                    suffix["target_minus_blank_update"]
                ),
                "occurrences": occurrence,
            }
        )

    cohorts: dict[str, list[dict]] = {"all": records}
    for keyword_id in keyword_ids:
        rows = [row for row in records if int(row["keyword_id"]) == keyword_id]
        cohorts[f"kw{keyword_id}_all"] = rows
        cohorts[f"kw{keyword_id}_runtime_hit"] = [
            row for row in rows if row["runtime_hit"]
        ]
        cohorts[f"kw{keyword_id}_runtime_gap"] = [
            row for row in rows if row["runtime_gap"]
        ]

    gate = build_gate(records)
    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "candidate": CANDIDATE,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "candidate_weight_source": (
            "retained keyword_sequence_margin_loss_weight"
        ),
        "candidate_weight": weight,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "training_readback_sha256": sha256_file(training_readback),
        "pretraining_candidate_gate": gate,
        "cohorts": {
            name: summarize(rows)
            for name, rows in cohorts.items()
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "weight": weight,
                "gate": gate,
                "kw1_runtime_gap": result["cohorts"].get("kw1_runtime_gap"),
                "kw2_runtime_gap": result["cohorts"].get("kw2_runtime_gap"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
