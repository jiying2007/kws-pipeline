#!/usr/bin/env python3
"""Audit label-prior CTC gradients on retained KWS posteriors.

The candidate follows the label-prior CTC formulation from ICASSP 2024:
path scores divide frame posteriors by unigram label priors raised to alpha.
This audit fixes alpha=0.3 (the paper's reported setting), estimates priors
from the retained development posterior cache, and compares the resulting
primary-loss gradient to standard CTC without changing trainer behavior.
The workflow is fixed-evidence only and never changes training configuration.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

import torch
import torch.nn.functional as F

from diagnose_decoder_search_path_decomposition import load_positive_sample
from diagnose_keyword_ctc_competition_gradient import scalar_stats
from diagnose_keyword_ctc_sequence_competition import (
    allowed_predecessors,
    allowed_successors,
    common_suffix_length,
    extended_target,
)
from diagnose_sequence_margin_runtime_gap import (
    load_keywords,
    load_tokens,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "ctc-label-prior-gradient-audit-development-v1"
POLICY = "posterior-marginal-label-prior-ctc-v1"
LABEL_PRIOR_ALPHA = 0.3
NEGATIVE_SENTINEL = -1.0e4


def standard_ctc_loss(
    raw_logits: torch.Tensor,
    sequence: tuple[int, ...],
) -> torch.Tensor:
    log_probs = raw_logits.log_softmax(dim=1)
    target = torch.tensor(sequence, dtype=torch.long, device=raw_logits.device)
    steps = int(raw_logits.shape[0])
    return F.ctc_loss(
        log_probs.unsqueeze(1),
        target,
        torch.tensor([steps], dtype=torch.long, device=raw_logits.device),
        torch.tensor([len(sequence)], dtype=torch.long, device=raw_logits.device),
        blank=0,
        reduction="sum",
        zero_infinity=True,
    ) / float(steps)


def label_prior_ctc_loss(
    raw_logits: torch.Tensor,
    sequence: tuple[int, ...],
    priors: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if raw_logits.ndim != 2:
        raise ValueError("raw logits must be [T,V]")
    if priors.ndim != 1 or int(priors.numel()) != int(raw_logits.shape[1]):
        raise ValueError("label priors must match vocabulary")
    if not torch.isfinite(priors).all() or bool((priors <= 0).any()):
        raise ValueError("label priors must be finite and positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("label-prior alpha must be finite and >= 0")

    adjusted = (
        raw_logits.log_softmax(dim=1)
        - alpha * priors.to(raw_logits.device).log().unsqueeze(0)
    )
    states = extended_target(sequence, 0)
    count = len(states)
    steps = int(adjusted.shape[0])
    sentinel = adjusted.new_tensor(NEGATIVE_SENTINEL)

    previous = [sentinel for _ in range(count)]
    previous[0] = adjusted[0, 0]
    if count > 1:
        previous[1] = adjusted[0, states[1]]

    for frame in range(1, steps):
        current: list[torch.Tensor] = []
        for state, token in enumerate(states):
            values = torch.stack(
                [
                    previous[prior]
                    for prior in allowed_predecessors(states, state, 0)
                ]
            )
            current.append(torch.logsumexp(values, dim=0) + adjusted[frame, token])
        previous = current

    log_probability = torch.logsumexp(torch.stack(previous[-2:]), dim=0)
    return -log_probability / float(steps)


def token_temporal_weights(
    raw_logits: torch.Tensor,
    sequence: tuple[int, ...],
) -> list[torch.Tensor]:
    log_probs = raw_logits.detach().log_softmax(dim=1)
    states = extended_target(sequence, 0)
    steps = int(log_probs.shape[0])
    count = len(states)

    with torch.no_grad():
        alpha = torch.full(
            (steps, count),
            float("-inf"),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        alpha[0, 0] = log_probs[0, 0]
        if count > 1:
            alpha[0, 1] = log_probs[0, states[1]]
        for frame in range(1, steps):
            for state, token in enumerate(states):
                alpha[frame, state] = (
                    torch.logsumexp(
                        alpha[
                            frame - 1,
                            allowed_predecessors(states, state, 0),
                        ],
                        dim=0,
                    )
                    + log_probs[frame, token]
                )

        log_probability = torch.logsumexp(alpha[-1, -2:], dim=0)
        beta = torch.full(
            (steps, count),
            float("-inf"),
            dtype=log_probs.dtype,
            device=log_probs.device,
        )
        beta[-1, -1] = 0.0
        if count > 1:
            beta[-1, -2] = 0.0
        for frame in range(steps - 2, -1, -1):
            for state in range(count):
                beta[frame, state] = torch.logsumexp(
                    torch.stack(
                        [
                            beta[frame + 1, successor]
                            + log_probs[frame + 1, states[successor]]
                            for successor in allowed_successors(states, state, 0)
                        ]
                    ),
                    dim=0,
                )

        result: list[torch.Tensor] = []
        for occurrence, _ in enumerate(sequence):
            state = occurrence * 2 + 1
            posterior = torch.exp(
                alpha[:, state] + beta[:, state] - log_probability
            )
            occupancy = posterior.sum()
            if not torch.isfinite(occupancy) or float(occupancy) <= 0.0:
                raise ValueError("CTC token-state occupancy is invalid")
            temporal = posterior / occupancy
            if abs(float(temporal.sum()) - 1.0) > 1.0e-5:
                raise ValueError("CTC temporal weights must sum to one")
            result.append(temporal)
        return result


def estimate_label_priors(
    traces: list[pathlib.Path],
) -> tuple[torch.Tensor, int]:
    total: torch.Tensor | None = None
    frames = 0
    for trace in traces:
        raw = torch.tensor(read_trace_logits(trace), dtype=torch.float32)
        posterior_sum = raw.softmax(dim=1).sum(dim=0)
        total = posterior_sum if total is None else total + posterior_sum
        frames += int(raw.shape[0])
    if total is None or frames <= 0:
        raise ValueError("label-prior estimation requires retained traces")
    priors = total / float(frames)
    if not torch.isfinite(priors).all() or bool((priors <= 0).any()):
        raise ValueError("estimated label priors are invalid")
    if abs(float(priors.sum()) - 1.0) > 1.0e-5:
        raise ValueError("estimated label priors must sum to one")
    return priors, frames


def local_update(
    update: torch.Tensor,
    sequence: tuple[int, ...],
    temporal: list[torch.Tensor],
    *,
    start: int,
    stop: int,
) -> dict:
    target_total = 0.0
    blank_total = 0.0
    for index in range(start, stop):
        weights = temporal[index].to(update.device)
        token = sequence[index]
        target_total += float((weights * update[:, token]).sum())
        blank_total += float((weights * update[:, 0]).sum())
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
        "prior_to_standard_grad_l2_ratio": scalar_stats(
            [float(row["prior_to_standard_grad_l2_ratio"]) for row in records]
        ),
        "prior_standard_gradient_cosine": scalar_stats(
            [float(row["prior_standard_gradient_cosine"]) for row in records]
        ),
        "standard_prefix_target_minus_blank": scalar_stats(
            [
                float(row["standard_prefix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "prior_prefix_target_minus_blank": scalar_stats(
            [
                float(row["prior_prefix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "standard_suffix_target_minus_blank": scalar_stats(
            [
                float(row["standard_suffix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "prior_suffix_target_minus_blank": scalar_stats(
            [
                float(row["prior_suffix"]["target_minus_blank"])
                for row in records
            ]
        ),
        "prefix_improves_standard": sum(
            float(row["prior_prefix"]["target_minus_blank"])
            > float(row["standard_prefix"]["target_minus_blank"])
            for row in records
        ),
        "suffix_improves_standard": sum(
            float(row["prior_suffix"]["target_minus_blank"])
            > float(row["standard_suffix"]["target_minus_blank"])
            for row in records
        ),
        "prior_prefix_positive": sum(
            float(row["prior_prefix"]["target_minus_blank"]) > 0.0
            for row in records
        ),
        "prior_suffix_positive": sum(
            float(row["prior_suffix"]["target_minus_blank"]) > 0.0
            for row in records
        ),
        "standard_global_blank_update": scalar_stats(
            [float(row["standard_global_blank_update"]) for row in records]
        ),
        "prior_global_blank_update": scalar_stats(
            [float(row["prior_global_blank_update"]) for row in records]
        ),
        "global_blank_update_reduced": sum(
            float(row["prior_global_blank_update"])
            < float(row["standard_global_blank_update"])
            for row in records
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
    priors = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float32)

    standard_logits = raw.detach().clone().requires_grad_(True)
    standard = standard_ctc_loss(standard_logits, sequence)
    standard_grad = torch.autograd.grad(standard, standard_logits)[0]

    zero_logits = raw.detach().clone().requires_grad_(True)
    zero = label_prior_ctc_loss(
        zero_logits, sequence, priors, alpha=0.0
    )
    zero_grad = torch.autograd.grad(zero, zero_logits)[0]
    assert abs(float(standard - zero)) < 1.0e-5
    assert torch.allclose(standard_grad, zero_grad, atol=1.0e-5, rtol=1.0e-5)

    prior_logits = raw.detach().clone().requires_grad_(True)
    prior = label_prior_ctc_loss(
        prior_logits, sequence, priors, alpha=LABEL_PRIOR_ALPHA
    )
    prior_grad = torch.autograd.grad(prior, prior_logits)[0]
    assert torch.isfinite(prior)
    assert torch.isfinite(prior_grad).all()

    temporal = token_temporal_weights(raw, sequence)
    assert len(temporal) == len(sequence)
    print("ctc label-prior gradient self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    values = {
        "tokens": args.tokens,
        "keywords_tsv": args.keywords_tsv,
        "posterior_cache": args.posterior_cache,
        "acoustic_alignment": args.acoustic_alignment,
        "model_sha256": args.model_sha256,
        "output": args.output,
    }
    missing = sorted(key for key, value in values.items() if value is None)
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))

    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
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
        raise ValueError("label-prior audit requires multiple keywords")

    positive = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    traces = trace_paths(posterior_cache, model_sha256)
    priors, prior_frames = estimate_label_priors(traces)
    by_sha = {path.stem: path for path in traces}

    records: list[dict] = []
    for audio_sha, sample in sorted(positive.items()):
        trace = by_sha.get(audio_sha)
        if trace is None:
            raise ValueError("retained positive posterior trace is missing")
        keyword_id = int(sample["keyword_id"])
        wake_index = keyword_ids.index(keyword_id)
        sequence = keywords[wake_index]
        raw = torch.tensor(read_trace_logits(trace), dtype=torch.float32)

        temporal = token_temporal_weights(raw, sequence)

        standard_logits = raw.detach().clone().requires_grad_(True)
        standard_loss = standard_ctc_loss(standard_logits, sequence)
        standard_grad = torch.autograd.grad(
            standard_loss, standard_logits
        )[0]

        zero_logits = raw.detach().clone().requires_grad_(True)
        zero_loss = label_prior_ctc_loss(
            zero_logits, sequence, priors, alpha=0.0
        )
        zero_grad = torch.autograd.grad(zero_loss, zero_logits)[0]
        if abs(float(standard_loss - zero_loss)) > 1.0e-5:
            raise ValueError("alpha=0 label-prior loss differs from standard CTC")
        if not torch.allclose(
            standard_grad, zero_grad, atol=1.0e-5, rtol=1.0e-5
        ):
            raise ValueError("alpha=0 label-prior gradient differs from standard CTC")

        prior_logits = raw.detach().clone().requires_grad_(True)
        prior_loss = label_prior_ctc_loss(
            prior_logits,
            sequence,
            priors,
            alpha=LABEL_PRIOR_ALPHA,
        )
        prior_grad = torch.autograd.grad(prior_loss, prior_logits)[0]
        if not torch.isfinite(prior_grad).all():
            raise ValueError("label-prior CTC gradient is non-finite")

        standard_norm = float(standard_grad.norm())
        prior_norm = float(prior_grad.norm())
        if standard_norm <= 0.0 or prior_norm <= 0.0:
            raise ValueError("CTC gradient norm must be positive")

        shared_suffix = max_shared_suffix(sequence, keywords)
        prefix_count = len(sequence) - shared_suffix
        if prefix_count <= 0:
            raise ValueError("label-prior audit requires a discriminative prefix")

        standard_update = -standard_grad
        prior_update = -prior_grad
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
                "standard_ctc_loss": float(standard_loss.detach()),
                "prior_ctc_loss": float(prior_loss.detach()),
                "prior_to_standard_grad_l2_ratio": prior_norm / standard_norm,
                "prior_standard_gradient_cosine": float(
                    F.cosine_similarity(
                        prior_grad.flatten(),
                        standard_grad.flatten(),
                        dim=0,
                    )
                ),
                "standard_prefix": local_update(
                    standard_update,
                    sequence,
                    temporal,
                    start=0,
                    stop=prefix_count,
                ),
                "prior_prefix": local_update(
                    prior_update,
                    sequence,
                    temporal,
                    start=0,
                    stop=prefix_count,
                ),
                "standard_suffix": local_update(
                    standard_update,
                    sequence,
                    temporal,
                    start=prefix_count,
                    stop=len(sequence),
                ),
                "prior_suffix": local_update(
                    prior_update,
                    sequence,
                    temporal,
                    start=prefix_count,
                    stop=len(sequence),
                ),
                "standard_global_blank_update": float(
                    standard_update[:, 0].sum()
                ),
                "prior_global_blank_update": float(
                    prior_update[:, 0].sum()
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
        "label_prior_alpha": LABEL_PRIOR_ALPHA,
        "label_prior_source": "retained-posterior-marginal-v1",
        "label_prior_trace_count": len(traces),
        "label_prior_frame_count": prior_frames,
        "label_priors": [float(value) for value in priors],
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
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
                "alpha": LABEL_PRIOR_ALPHA,
                "label_priors": result["label_priors"],
                "kw1_runtime_gap": result["cohorts"].get("kw1_runtime_gap", {}),
                "kw1_runtime_hit": result["cohorts"].get("kw1_runtime_hit", {}),
                "kw2_runtime_gap": result["cohorts"].get("kw2_runtime_gap", {}),
                "kw2_runtime_hit": result["cohorts"].get("kw2_runtime_hit", {}),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
