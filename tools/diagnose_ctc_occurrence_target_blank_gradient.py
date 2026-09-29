#!/usr/bin/env python3
"""Audit a local CTC occurrence target-vs-blank positive candidate on retained logits."""
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
from diagnose_keyword_ctc_sequence_competition import (  # noqa: E402
    allowed_predecessors,
    allowed_successors,
    common_suffix_length,
    extended_target,
    logsumexp,
)
from diagnose_sequence_margin_runtime_gap import (  # noqa: E402
    load_keywords,
    load_tokens,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "ctc-occurrence-target-blank-gradient-audit-development-v1"
POLICY = "ctc-token-state-posterior-weighted-target-vs-blank-v1"


def token_temporal_weights(
    raw_logits: torch.Tensor,
    sequence: tuple[int, ...],
    *,
    blank: int = 0,
) -> list[torch.Tensor]:
    if raw_logits.ndim != 2:
        raise ValueError("raw logits must be [T,V]")
    log_probs = raw_logits.detach().log_softmax(dim=1).cpu().tolist()
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

    result: list[torch.Tensor] = []
    for occurrence, _ in enumerate(sequence):
        state = occurrence * 2 + 1
        posterior = [
            math.exp(alpha[frame][state] + beta[frame][state] - log_probability)
            for frame in range(steps)
        ]
        occupancy = sum(posterior)
        if not math.isfinite(occupancy) or occupancy <= 0.0:
            raise ValueError("CTC token-state occupancy is invalid")
        temporal = torch.tensor(
            [value / occupancy for value in posterior],
            dtype=raw_logits.dtype,
            device=raw_logits.device,
        )
        if not torch.isfinite(temporal).all():
            raise ValueError("CTC token temporal weights are non-finite")
        if abs(float(temporal.sum()) - 1.0) > 1.0e-5:
            raise ValueError("CTC token temporal weights must sum to one")
        result.append(temporal)
    return result


def candidate_loss(
    logits: torch.Tensor,
    sequence: tuple[int, ...],
    temporal: list[torch.Tensor],
    *,
    blank: int = 0,
) -> torch.Tensor:
    if len(sequence) != len(temporal):
        raise ValueError("candidate temporal weights must align with target occurrences")
    terms: list[torch.Tensor] = []
    for token, weights in zip(sequence, temporal):
        if int(weights.numel()) != int(logits.shape[0]):
            raise ValueError("candidate temporal weight length mismatch")
        terms.append(
            (
                weights.detach()
                * F.softplus(logits[:, blank] - logits[:, token])
            ).sum()
        )
    if not terms:
        raise ValueError("candidate requires non-empty sequence")
    return torch.stack(terms).mean()


def primary_ctc_gradient(
    raw_logits: torch.Tensor,
    sequence: tuple[int, ...],
) -> tuple[float, torch.Tensor]:
    logits = raw_logits.detach().clone().requires_grad_(True)
    log_probs = logits.log_softmax(dim=1)
    loss = ctc_true_nll(log_probs, sequence) / float(log_probs.shape[0])
    grad = torch.autograd.grad(loss, logits)[0]
    return float(loss.detach()), grad


def candidate_gradient(
    raw_logits: torch.Tensor,
    sequence: tuple[int, ...],
) -> tuple[float, torch.Tensor, list[torch.Tensor]]:
    temporal = token_temporal_weights(raw_logits, sequence)
    logits = raw_logits.detach().clone().requires_grad_(True)
    loss = candidate_loss(logits, sequence, temporal)
    grad = torch.autograd.grad(loss, logits)[0]
    return float(loss.detach()), grad, temporal


def occurrence_region_update(
    update: torch.Tensor,
    sequence: tuple[int, ...],
    temporal: list[torch.Tensor],
    *,
    start: int,
    stop: int,
    blank: int = 0,
) -> dict:
    target_total = 0.0
    blank_total = 0.0
    for index in range(start, stop):
        token = sequence[index]
        weights = temporal[index].to(dtype=update.dtype, device=update.device)
        target_total += float((weights * update[:, token]).sum())
        blank_total += float((weights * update[:, blank]).sum())
    return {
        "target": target_total,
        "blank": blank_total,
        "target_minus_blank": target_total - blank_total,
    }


def max_shared_suffix(
    sequence: tuple[int, ...],
    keywords: list[tuple[int, ...]],
) -> int:
    values = [
        common_suffix_length(sequence, other)
        for other in keywords
        if other != sequence
    ]
    return max(values) if values else 0


def summarize(records: list[dict]) -> dict:
    return {
        "recordings": len(records),
        "candidate_loss": scalar_stats(
            [float(row["candidate_loss"]) for row in records]
        ),
        "candidate_to_primary_grad_l2_ratio": scalar_stats(
            [float(row["candidate_to_primary_grad_l2_ratio"]) for row in records]
        ),
        "candidate_primary_gradient_cosine": scalar_stats(
            [float(row["candidate_primary_gradient_cosine"]) for row in records]
        ),
        "candidate_prefix_target_minus_blank_update": scalar_stats(
            [
                float(row["candidate_prefix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "candidate_suffix_target_minus_blank_update": scalar_stats(
            [
                float(row["candidate_suffix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "primary_prefix_target_minus_blank_update": scalar_stats(
            [
                float(row["primary_prefix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "primary_suffix_target_minus_blank_update": scalar_stats(
            [
                float(row["primary_suffix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "combined_prefix_target_minus_blank_update": scalar_stats(
            [
                float(row["combined_prefix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "combined_suffix_target_minus_blank_update": scalar_stats(
            [
                float(row["combined_suffix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "combined_prefix_improves_primary": sum(
            float(row["combined_prefix"]["target_minus_blank"])
            > float(row["primary_prefix"]["target_minus_blank"])
            for row in records
        ),
        "combined_suffix_improves_primary": sum(
            float(row["combined_suffix"]["target_minus_blank"])
            > float(row["primary_suffix"]["target_minus_blank"])
            for row in records
        ),
        "combined_prefix_positive": sum(
            float(row["combined_prefix"]["target_minus_blank"]) > 0.0
            for row in records
        ),
        "combined_suffix_positive": sum(
            float(row["combined_suffix"]["target_minus_blank"]) > 0.0
            for row in records
        ),
        "candidate_global_blank_update": scalar_stats(
            [float(row["candidate_global_blank_update"]) for row in records]
        ),
    }


def self_test() -> None:
    raw = torch.tensor(
        [
            [4.0, -4.0, -4.0],
            [-4.0, 6.0, -4.0],
            [4.0, -4.0, -4.0],
            [-4.0, -4.0, 6.0],
            [4.0, -4.0, -4.0],
        ],
        dtype=torch.float32,
    )
    sequence = (1, 2)
    weights = token_temporal_weights(raw, sequence)
    assert len(weights) == 2
    assert all(abs(float(item.sum()) - 1.0) < 1.0e-6 for item in weights)
    loss, grad, _ = candidate_gradient(raw, sequence)
    assert loss > 0.0
    update = -grad
    first = occurrence_region_update(
        update, sequence, weights, start=0, stop=1
    )
    second = occurrence_region_update(
        update, sequence, weights, start=1, stop=2
    )
    assert first["target_minus_blank"] > 0.0
    assert second["target_minus_blank"] > 0.0
    assert float(update[:, 0].sum()) < 0.0
    print("ctc occurrence target-blank gradient self-test: PASS")


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
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))

    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    training_readback = args.training_readback.resolve()
    output = args.output.resolve()
    model_sha256 = str(args.model_sha256)

    for path, label in (
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (acoustic_alignment, "acoustic alignment"),
        (training_readback, "training readback"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError("posterior cache is missing")

    token_map = load_tokens(tokens)
    keyword_map = load_keywords(keywords_tsv, token_map)
    keyword_ids = sorted(keyword_map)
    keywords = [
        tuple(int(value) for value in keyword_map[keyword_id]["tokens"])
        for keyword_id in keyword_ids
    ]
    if len(keywords) < 2:
        raise ValueError("candidate audit requires at least two keywords")

    positive = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    traces = trace_paths(posterior_cache, model_sha256)
    by_sha = {path.stem: path for path in traces}
    weight = read_margin_weight(
        training_readback,
        model_sha256=model_sha256,
    )

    records: list[dict] = []
    for audio_sha, sample in sorted(positive.items()):
        trace = by_sha.get(audio_sha)
        if trace is None:
            raise ValueError("retained positive posterior trace is missing")
        keyword_id = int(sample["keyword_id"])
        try:
            wake_index = keyword_ids.index(keyword_id)
        except ValueError as exc:
            raise ValueError("positive sample references unknown keyword") from exc
        sequence = keywords[wake_index]
        raw = torch.tensor(read_trace_logits(trace), dtype=torch.float32)

        candidate_value, candidate_grad, temporal = candidate_gradient(
            raw, sequence
        )
        primary_value, primary_grad = primary_ctc_gradient(raw, sequence)
        candidate_norm = float(candidate_grad.norm())
        primary_norm = float(primary_grad.norm())
        if candidate_norm <= 0.0 or primary_norm <= 0.0:
            raise ValueError("candidate/primary gradient norm must be positive")

        weighted_candidate_grad = candidate_grad * weight
        candidate_update = -candidate_grad
        primary_update = -primary_grad
        combined_update = -(primary_grad + weighted_candidate_grad)

        shared_suffix = max_shared_suffix(sequence, keywords)
        prefix_count = len(sequence) - shared_suffix
        if prefix_count <= 0:
            raise ValueError("candidate audit requires a discriminative prefix")

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
                "shared_suffix_length": shared_suffix,
                "discriminative_prefix_length": prefix_count,
                "candidate_loss": candidate_value,
                "primary_ctc_loss": primary_value,
                "candidate_to_primary_grad_l2_ratio": (
                    weight * candidate_norm / primary_norm
                ),
                "candidate_primary_gradient_cosine": float(
                    F.cosine_similarity(
                        candidate_grad.flatten(),
                        primary_grad.flatten(),
                        dim=0,
                    )
                ),
                "candidate_prefix": occurrence_region_update(
                    candidate_update,
                    sequence,
                    temporal,
                    start=0,
                    stop=prefix_count,
                ),
                "candidate_suffix": occurrence_region_update(
                    candidate_update,
                    sequence,
                    temporal,
                    start=prefix_count,
                    stop=len(sequence),
                ),
                "primary_prefix": occurrence_region_update(
                    primary_update,
                    sequence,
                    temporal,
                    start=0,
                    stop=prefix_count,
                ),
                "primary_suffix": occurrence_region_update(
                    primary_update,
                    sequence,
                    temporal,
                    start=prefix_count,
                    stop=len(sequence),
                ),
                "combined_prefix": occurrence_region_update(
                    combined_update,
                    sequence,
                    temporal,
                    start=0,
                    stop=prefix_count,
                ),
                "combined_suffix": occurrence_region_update(
                    combined_update,
                    sequence,
                    temporal,
                    start=prefix_count,
                    stop=len(sequence),
                ),
                "candidate_global_blank_update": float(
                    candidate_update[:, 0].sum()
                ),
            }
        )

    cohorts: dict[str, list[dict]] = {"all": records}
    for keyword_id in keyword_ids:
        rows = [row for row in records if row["keyword_id"] == keyword_id]
        cohorts[f"kw{keyword_id}_all"] = rows
        cohorts[f"kw{keyword_id}_runtime_hit"] = [
            row for row in rows if row["runtime_hit"]
        ]
        cohorts[f"kw{keyword_id}_runtime_gap"] = [
            row for row in rows if row["runtime_gap"]
        ]

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "candidate_applied_to_exact_wake_positives_only": True,
        "candidate_weight": weight,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "training_readback_sha256": sha256_file(training_readback),
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
                "candidate_weight": weight,
                "kw1_runtime_gap": result["cohorts"].get("kw1_runtime_gap", {}),
                "kw2_runtime_gap": result["cohorts"].get("kw2_runtime_gap", {}),
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
