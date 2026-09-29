#!/usr/bin/env python3
"""Compare retained token-region training gradients for baseline and candidate objectives."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from completion_loss import strict_prefix_completion_loss  # noqa: E402
from diagnose_ctc_label_prior_gradient import (  # noqa: E402
    LABEL_PRIOR_ALPHA,
    estimate_label_priors,
    label_prior_ctc_loss,
)
from diagnose_ctc_token_state_blank_gradient import (  # noqa: E402
    aggregate_occurrences,
    discriminative_prefix_length,
    occurrence_updates,
    token_state_temporal_weights,
)
from diagnose_decoder_search_path_decomposition import load_positive_sample  # noqa: E402
from diagnose_keyword_ctc_competition_gradient import (  # noqa: E402
    ctc_true_nll,
    scalar_stats,
)
from diagnose_sequence_margin_runtime_gap import (  # noqa: E402
    TRACE_HEADER_BYTES,
    TRACE_MAGIC,
    load_keywords,
    load_tokens,
    read_trace_logits,
    sha256_file,
    trace_paths,
)
from objective_contract import (  # noqa: E402
    SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
    SEQUENCE_MARGIN_POSITIVE_POLICY_CTC_TOKEN_STATE_TARGET_BLANK,
    SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE,
)
from sequence_margin import keyword_sequence_margin_loss  # noqa: E402
from train_ctc import ordered_token_loss, vad_aligned_ctc_log_probs  # noqa: E402

EVIDENCE_CLASS = "full-speech-objective-gradient-audit-development-v2"
POLICY = "exact-wake-token-region-training-objective-v2"
VARIANT_BASELINE = "standard-ctc+sparse-positive"
VARIANT_TOKEN_STATE = "standard-ctc+token-state-positive"
VARIANT_LABEL_PRIOR = "label-prior-ctc+sparse-positive"

EXPECTED_ORDERED_WEIGHT = 0.35
EXPECTED_MARGIN_WEIGHT = 0.10
EXPECTED_COMPLETION_WEIGHT = 0.10
EXPECTED_RELEASE_WEIGHT = 0.05
EXPECTED_SUFFIX_ROOT_WEIGHT = 0.0
EXPECTED_PATH_PURITY_WEIGHT = 0.0


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_checkpoint(path: pathlib.Path) -> dict:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("checkpoint must be a dict")
    return value


def read_trace_speech_active(path: pathlib.Path) -> list[bool]:
    data = path.read_bytes()
    if len(data) < TRACE_HEADER_BYTES or data[:8] != TRACE_MAGIC:
        raise ValueError(f"invalid posterior trace header: {path}")
    vocab_size = struct.unpack_from("<H", data, 12)[0]
    frame_count = struct.unpack_from("<Q", data, 40)[0]
    if vocab_size < 2 or frame_count <= 0:
        raise ValueError(f"invalid posterior trace dimensions: {path}")
    record_bytes = 16 + 4 * vocab_size
    expected = TRACE_HEADER_BYTES + frame_count * record_bytes
    if len(data) != expected:
        raise ValueError(f"posterior trace size mismatch: {path}")
    result: list[bool] = []
    offset = TRACE_HEADER_BYTES
    for _ in range(frame_count):
        flags = data[offset + 8 : offset + 16]
        if flags[0] not in (0, 1) or any(flags[1:]):
            raise ValueError(f"posterior trace flags are invalid: {path}")
        result.append(bool(flags[0]))
        offset += record_bytes
    return result


def effective_ctc_log_probs(
    log_probs: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    speech_active: list[bool] | None,
) -> torch.Tensor:
    if speech_active is None:
        return log_probs
    if len(speech_active) != int(log_probs.shape[0]):
        raise ValueError("trace speech-active flags do not match posterior frames")
    mask = torch.tensor(
        speech_active,
        dtype=torch.bool,
        device=log_probs.device,
    ).unsqueeze(0)
    target_lengths = torch.tensor(
        [len(sequence)],
        dtype=torch.long,
        device=log_probs.device,
    )
    return vad_aligned_ctc_log_probs(
        log_probs.unsqueeze(1),
        mask,
        target_lengths,
    ).squeeze(1)


def label_prior_ctc_log_probs_loss(
    ctc_log_probs: torch.Tensor,
    sequence: tuple[int, ...],
    priors: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if ctc_log_probs.ndim != 2:
        raise ValueError("CTC log probabilities must be [T,V]")
    if priors.ndim != 1 or int(priors.numel()) != int(ctc_log_probs.shape[1]):
        raise ValueError("label priors must match vocabulary")
    if not torch.isfinite(priors).all() or bool((priors <= 0).any()):
        raise ValueError("label priors must be finite and positive")
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError("label-prior alpha must be finite and >= 0")

    from diagnose_keyword_ctc_sequence_competition import (
        allowed_predecessors,
        extended_target,
    )

    adjusted = (
        ctc_log_probs
        - alpha * priors.to(ctc_log_probs.device).log().unsqueeze(0)
    )
    states = extended_target(sequence, 0)
    count = len(states)
    steps = int(adjusted.shape[0])
    sentinel = adjusted.new_tensor(-1.0e4)
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
            current.append(
                torch.logsumexp(values, dim=0) + adjusted[frame, token]
            )
        previous = current
    log_probability = torch.logsumexp(torch.stack(previous[-2:]), dim=0)
    return -log_probability / float(steps)


def checkpoint_contract(checkpoint: dict) -> dict:
    weights = checkpoint.get("auxiliary_loss_weights")
    if not isinstance(weights, dict):
        raise ValueError("checkpoint auxiliary loss weights are missing")
    expected_weights = {
        "ordered_token_loss_weight": EXPECTED_ORDERED_WEIGHT,
        "keyword_sequence_margin_loss_weight": EXPECTED_MARGIN_WEIGHT,
        "prefix_completion_loss_weight": EXPECTED_COMPLETION_WEIGHT,
        "recurrent_release_loss_weight": EXPECTED_RELEASE_WEIGHT,
        "suffix_root_suppression_loss_weight": EXPECTED_SUFFIX_ROOT_WEIGHT,
    }
    for key, expected in expected_weights.items():
        actual = float(weights.get(key, float("nan")))
        if not math.isfinite(actual) or abs(actual - expected) > 1.0e-12:
            raise ValueError(f"checkpoint {key} drifted: {actual} != {expected}")

    ordered_scope = checkpoint.get("ordered_token_scope")
    if ordered_scope != "all-nonempty-targets-v1":
        raise ValueError("checkpoint ordered-token scope is unsupported")
    negative_policy = checkpoint.get("sequence_margin_negative_policy")
    if negative_policy != SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE:
        raise ValueError("checkpoint negative sequence-margin policy drifted")
    positive_policy = checkpoint.get(
        "sequence_margin_positive_policy",
        SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE,
    )
    if positive_policy is None:
        positive_policy = SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE
    if positive_policy != SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE:
        raise ValueError("checkpoint positive sequence-margin policy is not sparse baseline")
    if float(checkpoint.get("path_purity_loss_weight", 0.0)) != EXPECTED_PATH_PURITY_WEIGHT:
        raise ValueError("checkpoint path-purity weight drifted")

    ctc_alignment = checkpoint.get("ctc_vad_alignment")
    if not isinstance(ctc_alignment, dict):
        raise ValueError("checkpoint CTC VAD alignment contract is missing")
    if checkpoint.get("development_recipe") != "development-pcm-dbfs-gated-ctc-v1":
        raise ValueError("checkpoint development CTC recipe drifted")
    threshold = float(ctc_alignment.get("threshold_dbfs", float("nan")))
    if not math.isfinite(threshold) or abs(threshold - (-55.0)) > 1.0e-12:
        raise ValueError("checkpoint CTC VAD threshold drifted")

    stats = checkpoint.get("sample_weight_normalization")
    if not isinstance(stats, dict):
        raise ValueError("checkpoint sample-weight normalization is missing")
    all_mean = float(stats.get("all_mean_weight", float("nan")))
    nonempty_mean = float(stats.get("nonempty_mean_weight", float("nan")))
    nonempty_rows = int(stats.get("nonempty_rows", -1))
    rows = int(stats.get("rows", -1))
    if (
        not math.isfinite(all_mean)
        or not math.isfinite(nonempty_mean)
        or all_mean <= 0.0
        or abs(all_mean - nonempty_mean) > 1.0e-12
        or nonempty_rows != rows
        or rows <= 0
    ):
        raise ValueError(
            "checkpoint does not preserve common exact-wake token-region sample scaling"
        )

    operating_points = checkpoint.get("keyword_operating_points")
    sequences = checkpoint.get("keyword_sequences")
    if not isinstance(operating_points, list) or not operating_points:
        raise ValueError("checkpoint keyword operating points are missing")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("checkpoint keyword sequences are missing")

    return {
        "auxiliary_loss_weights": {
            key: float(value) for key, value in sorted(weights.items())
        },
        "ordered_token_scope": ordered_scope,
        "sequence_margin_negative_policy": negative_policy,
        "sequence_margin_positive_policy": positive_policy,
        "path_purity_loss_weight": EXPECTED_PATH_PURITY_WEIGHT,
        "common_sample_scale_omitted": True,
        "sample_weight_all_mean": all_mean,
        "sample_weight_nonempty_mean": nonempty_mean,
        "keyword_operating_points": operating_points,
        "keyword_sequences": sequences,
        "recurrent_release_scope": "post-input-tail-only-disjoint-from-token-occurrence-metrics",
        "prefix_completion_exact_wake": "zero-by-definition",
        "suffix_root_weight": EXPECTED_SUFFIX_ROOT_WEIGHT,
        "ctc_vad_alignment": ctc_alignment,
    }


def primary_loss(
    ctc_log_probs: torch.Tensor,
    sequence: tuple[int, ...],
    *,
    variant: str,
    priors: torch.Tensor,
) -> torch.Tensor:
    steps = int(ctc_log_probs.shape[0])
    if variant in (VARIANT_BASELINE, VARIANT_TOKEN_STATE):
        return ctc_true_nll(ctc_log_probs, sequence) / float(steps)
    if variant == VARIANT_LABEL_PRIOR:
        return label_prior_ctc_log_probs_loss(
            ctc_log_probs,
            sequence,
            priors,
            alpha=LABEL_PRIOR_ALPHA,
        )
    raise ValueError(f"unsupported objective variant: {variant}")


def speech_objective_gradient(
    raw_logits: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    keywords: list[tuple[int, ...]],
    operating_points: list[dict],
    variant: str,
    priors: torch.Tensor,
    speech_active: list[bool] | None = None,
) -> tuple[dict, torch.Tensor]:
    logits = raw_logits.detach().clone().requires_grad_(True)
    log_probs = logits.log_softmax(dim=1)
    steps = int(log_probs.shape[0])
    target = torch.tensor(sequence, dtype=torch.long, device=log_probs.device)
    input_lengths = torch.tensor([steps], dtype=torch.long, device=log_probs.device)
    target_lengths = torch.tensor(
        [len(sequence)],
        dtype=torch.long,
        device=log_probs.device,
    )
    ctc_probs = effective_ctc_log_probs(
        log_probs,
        sequence=sequence,
        speech_active=speech_active,
    )

    primary = primary_loss(
        ctc_probs,
        sequence,
        variant=variant,
        priors=priors,
    )
    ordered, _, _ = ordered_token_loss(
        log_probs.unsqueeze(1),
        target,
        input_lengths,
        target_lengths,
    )

    positive_policy = (
        SEQUENCE_MARGIN_POSITIVE_POLICY_CTC_TOKEN_STATE_TARGET_BLANK
        if variant == VARIANT_TOKEN_STATE
        else SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE
    )
    standard_sum = ctc_true_nll(ctc_probs, sequence).reshape(1)
    margin = keyword_sequence_margin_loss(
        log_probs=log_probs.unsqueeze(1),
        ctc_log_probs=ctc_probs.unsqueeze(1),
        targets=target,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        true_ctc_nll=standard_sum,
        keyword_sequences=[list(item) for item in keywords],
        blank=0,
        margin=0.05,
        keyword_operating_points=operating_points,
        negative_path_policy=SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
        positive_path_policy=positive_policy,
    )[0]

    completion = strict_prefix_completion_loss(
        log_probs=log_probs.unsqueeze(1),
        targets=target,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        keyword_sequences=[list(item) for item in keywords],
        keyword_operating_points=operating_points,
    )[0]
    if abs(float(completion.detach())) > 1.0e-8:
        raise ValueError("exact-wake prefix-completion loss must be zero")

    total = (
        primary
        + EXPECTED_ORDERED_WEIGHT * ordered
        + EXPECTED_MARGIN_WEIGHT * margin
        + EXPECTED_COMPLETION_WEIGHT * completion
    )
    grad = torch.autograd.grad(total, logits)[0]
    return {
        "total": float(total.detach()),
        "primary": float(primary.detach()),
        "ordered": float(ordered.detach()),
        "margin": float(margin.detach()),
        "completion": float(completion.detach()),
        "positive_policy": positive_policy,
    }, grad


def local_summary(
    update: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    temporal: torch.Tensor,
    prefix_length: int,
) -> tuple[dict, dict, float]:
    occurrences = occurrence_updates(
        update,
        sequence=sequence,
        temporal=temporal,
    )
    def normalize(rows: list[dict]) -> dict:
        value = aggregate_occurrences(rows)
        return {
            "target": float(value["target_update"]),
            "blank": float(value["blank_update"]),
            "target_minus_blank": float(
                value["target_minus_blank_update"]
            ),
        }
    prefix = normalize(occurrences[:prefix_length])
    suffix = normalize(occurrences[prefix_length:])
    return prefix, suffix, float(update[:, 0].sum())


def compare_variant(
    baseline: dict,
    candidate: dict,
) -> dict:
    return {
        "prefix_target_minus_blank_delta": (
            float(candidate["prefix"]["target_minus_blank"])
            - float(baseline["prefix"]["target_minus_blank"])
        ),
        "suffix_target_minus_blank_delta": (
            float(candidate["suffix"]["target_minus_blank"])
            - float(baseline["suffix"]["target_minus_blank"])
        ),
        "global_blank_update_delta": (
            float(candidate["global_blank_update"])
            - float(baseline["global_blank_update"])
        ),
        "prefix_improves_baseline": (
            float(candidate["prefix"]["target_minus_blank"])
            > float(baseline["prefix"]["target_minus_blank"])
        ),
        "suffix_improves_baseline": (
            float(candidate["suffix"]["target_minus_blank"])
            > float(baseline["suffix"]["target_minus_blank"])
        ),
        "global_blank_reduced": (
            float(candidate["global_blank_update"])
            < float(baseline["global_blank_update"])
        ),
        "candidate_prefix_positive": (
            float(candidate["prefix"]["target_minus_blank"]) > 0.0
        ),
        "candidate_suffix_positive": (
            float(candidate["suffix"]["target_minus_blank"]) > 0.0
        ),
    }


def summarize_variant(records: list[dict], key: str) -> dict:
    return {
        "recordings": len(records),
        "grad_l2_ratio_to_baseline": scalar_stats(
            [float(row[key]["grad_l2_ratio_to_baseline"]) for row in records]
        ),
        "gradient_cosine_to_baseline": scalar_stats(
            [float(row[key]["gradient_cosine_to_baseline"]) for row in records]
        ),
        "prefix_target_minus_blank": scalar_stats(
            [float(row[key]["prefix"]["target_minus_blank"]) for row in records]
        ),
        "suffix_target_minus_blank": scalar_stats(
            [float(row[key]["suffix"]["target_minus_blank"]) for row in records]
        ),
        "global_blank_update": scalar_stats(
            [float(row[key]["global_blank_update"]) for row in records]
        ),
        "prefix_improves_baseline": sum(
            bool(row[key]["comparison"]["prefix_improves_baseline"])
            for row in records
        ),
        "suffix_improves_baseline": sum(
            bool(row[key]["comparison"]["suffix_improves_baseline"])
            for row in records
        ),
        "global_blank_reduced": sum(
            bool(row[key]["comparison"]["global_blank_reduced"])
            for row in records
        ),
        "prefix_positive": sum(
            bool(row[key]["comparison"]["candidate_prefix_positive"])
            for row in records
        ),
        "suffix_positive": sum(
            bool(row[key]["comparison"]["candidate_suffix_positive"])
            for row in records
        ),
    }


def build_gate(records: list[dict], key: str) -> dict:
    failures: list[str] = []
    for row in records:
        audio_sha = str(row["audio_sha256"])
        comparison = row[key]["comparison"]
        if not bool(comparison["prefix_improves_baseline"]):
            failures.append(f"{audio_sha}: prefix did not improve baseline")
        if int(row["shared_suffix_length"]) > 0 and not bool(
            comparison["suffix_improves_baseline"]
        ):
            failures.append(f"{audio_sha}: shared suffix did not improve baseline")
        if not bool(comparison["global_blank_reduced"]):
            failures.append(f"{audio_sha}: global blank update was not reduced")
        cosine = float(row[key]["gradient_cosine_to_baseline"])
        if cosine <= 0.0:
            failures.append(f"{audio_sha}: candidate full-objective gradient opposes baseline")
    return {
        "policy": "full-token-region-relative-safety-v1",
        "absolute_positive_is_diagnostic_not_gate": True,
        "pass": not failures,
        "failures": failures,
    }


def self_test() -> None:
    raw = torch.tensor(
        [
            [3.0, -1.0, -2.0],
            [-1.0, 4.0, -2.0],
            [3.0, -1.0, -2.0],
            [-1.0, -2.0, 4.0],
            [3.0, -1.0, -2.0],
        ],
        dtype=torch.float32,
    )
    sequence = (1, 2)
    keywords = [sequence, (2, 1)]
    operating_points = [
        {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.05},
        {"threshold": 0.55, "positive_margin": 0.05, "negative_margin": 0.05},
    ]
    priors = torch.tensor([0.8, 0.1, 0.1], dtype=torch.float32)
    baseline_metrics, baseline_grad = speech_objective_gradient(
        raw,
        sequence=sequence,
        keywords=keywords,
        operating_points=operating_points,
        variant=VARIANT_BASELINE,
        priors=priors,
    )
    prior_metrics, prior_grad = speech_objective_gradient(
        raw,
        sequence=sequence,
        keywords=keywords,
        operating_points=operating_points,
        variant=VARIANT_LABEL_PRIOR,
        priors=priors,
    )
    assert baseline_metrics["completion"] == 0.0
    assert prior_metrics["completion"] == 0.0
    assert torch.isfinite(baseline_grad).all()
    assert torch.isfinite(prior_grad).all()
    print("full speech-objective gradient self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--provenance", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--trace-vad-align", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0

    required = {
        "checkpoint": args.checkpoint,
        "provenance": args.provenance,
        "tokens": args.tokens,
        "keywords_tsv": args.keywords_tsv,
        "posterior_cache": args.posterior_cache,
        "acoustic_alignment": args.acoustic_alignment,
        "model_sha256": args.model_sha256,
        "output": args.output,
    }
    missing = sorted(key for key, value in required.items() if value is None)
    if missing:
        raise ValueError("missing required arguments: " + ", ".join(missing))

    checkpoint_path = args.checkpoint.resolve()
    provenance_path = args.provenance.resolve()
    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    output = args.output.resolve()
    model_sha256 = str(args.model_sha256)

    for path, label in (
        (checkpoint_path, "checkpoint"),
        (provenance_path, "provenance"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (acoustic_alignment, "acoustic alignment"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError("posterior cache is missing")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected_checkpoint_sha = str(provenance["checkpoint"]["sha256"])
    if sha256_file(checkpoint_path) != expected_checkpoint_sha:
        raise ValueError("checkpoint SHA does not match provenance")
    if str(provenance["model"]["sha256"]) != model_sha256:
        raise ValueError("model SHA does not match provenance")

    checkpoint = load_checkpoint(checkpoint_path)
    contract = checkpoint_contract(checkpoint)

    token_map = load_tokens(tokens)
    keyword_map = load_keywords(keywords_tsv, token_map)
    keyword_ids = sorted(keyword_map)
    keywords = [
        tuple(int(value) for value in keyword_map[keyword_id]["tokens"])
        for keyword_id in keyword_ids
    ]
    checkpoint_sequences = [
        tuple(int(value) for value in row)
        for row in contract["keyword_sequences"]
    ]
    if keywords != checkpoint_sequences:
        raise ValueError("calibrated keyword sequences drifted from checkpoint")
    operating_points = contract["keyword_operating_points"]
    if len(operating_points) != len(keywords):
        raise ValueError("keyword operating points do not align with keywords")

    positive = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    traces = trace_paths(posterior_cache, model_sha256)
    by_sha = {path.stem: path for path in traces}
    priors, prior_frames = estimate_label_priors(traces)

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
        speech_active = (
            read_trace_speech_active(trace)
            if args.trace_vad_align
            else None
        )
        metric_ctc_probs = effective_ctc_log_probs(
            raw.log_softmax(dim=1),
            sequence=sequence,
            speech_active=speech_active,
        )
        temporal = token_state_temporal_weights(
            metric_ctc_probs.detach().tolist(),
            sequence,
        )
        prefix_length = discriminative_prefix_length(sequence, keywords)
        shared_suffix = len(sequence) - prefix_length

        variant_rows: dict[str, dict] = {}
        gradients: dict[str, torch.Tensor] = {}
        for variant in (
            VARIANT_BASELINE,
            VARIANT_TOKEN_STATE,
            VARIANT_LABEL_PRIOR,
        ):
            metrics, grad = speech_objective_gradient(
                raw,
                sequence=sequence,
                keywords=keywords,
                operating_points=operating_points,
                variant=variant,
                priors=priors,
                speech_active=speech_active,
            )
            update = -grad
            prefix, suffix, global_blank = local_summary(
                update,
                sequence=sequence,
                temporal=temporal,
                prefix_length=prefix_length,
            )
            gradients[variant] = grad
            variant_rows[variant] = {
                "metrics": metrics,
                "grad_l2": float(grad.norm()),
                "prefix": prefix,
                "suffix": suffix,
                "global_blank_update": global_blank,
            }

        baseline = variant_rows[VARIANT_BASELINE]
        baseline_grad = gradients[VARIANT_BASELINE]
        baseline_norm = float(baseline_grad.norm())
        if baseline_norm <= 0.0:
            raise ValueError("baseline speech-objective gradient must be nonzero")

        row: dict = {
            "audio_sha256": audio_sha,
            "split": sample["split"],
            "keyword_id": keyword_id,
            "runtime_hit": bool(sample["runtime_detected_expected"]),
            "runtime_gap": bool(
                sample["surrogate_above_runtime_threshold"]
                and not sample["runtime_detected_expected"]
            ),
            "discriminative_prefix_length": prefix_length,
            "shared_suffix_length": shared_suffix,
            "baseline": baseline,
        }
        for key, variant in (
            ("token_state", VARIANT_TOKEN_STATE),
            ("label_prior", VARIANT_LABEL_PRIOR),
        ):
            candidate = variant_rows[variant]
            grad = gradients[variant]
            candidate["grad_l2_ratio_to_baseline"] = (
                float(grad.norm()) / baseline_norm
            )
            candidate["gradient_cosine_to_baseline"] = float(
                F.cosine_similarity(
                    grad.flatten(),
                    baseline_grad.flatten(),
                    dim=0,
                )
            )
            candidate["comparison"] = compare_variant(
                baseline,
                candidate,
            )
            row[key] = candidate
        records.append(row)

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

    token_gate = build_gate(records, "token_state")
    label_prior_gate = build_gate(records, "label_prior")
    result = {
        "schema_version": 2,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "ctc_alignment_mode": (
            "trace-runtime-speech-active-v1"
            if args.trace_vad_align
            else "raw-posterior-v1"
        ),
        "model_sha256": model_sha256,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "provenance_sha256": sha256_file(provenance_path),
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "source_checkpoint_contract": contract,
        "label_prior": {
            "alpha": LABEL_PRIOR_ALPHA,
            "estimated_frames": prior_frames,
            "values": [float(value) for value in priors.tolist()],
            "diagnostic_estimation_source": "retained-development-posteriors-only-not-training-authority",
        },
        "gates": {
            "token_state": token_gate,
            "label_prior": label_prior_gate,
        },
        "cohorts": {
            name: {
                "recordings": len(rows),
                "token_state": summarize_variant(rows, "token_state"),
                "label_prior": summarize_variant(rows, "label_prior"),
            }
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
                "gates": result["gates"],
                "kw1_runtime_gap": result["cohorts"].get("kw1_runtime_gap", {}),
                "kw1_runtime_hit": result["cohorts"].get("kw1_runtime_hit", {}),
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
