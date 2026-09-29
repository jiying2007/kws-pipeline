#!/usr/bin/env python3
"""Audit CTC keyword-competition policy gradients on retained exact-wake logits."""
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

from objective_contract import (  # noqa: E402
    SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
    SEQUENCE_MARGIN_POSITIVE_POLICY_CTC_KEYWORD_COMPETITION,
    SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE,
)
from sequence_margin import keyword_sequence_margin_loss  # noqa: E402
from diagnose_decoder_search_path_decomposition import load_positive_sample  # noqa: E402
from diagnose_sequence_margin_runtime_gap import (  # noqa: E402
    load_keywords,
    load_tokens,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "keyword-ctc-competition-gradient-audit-development-v1"
POLICY = "fixed-logit-actual-training-objective-gradient-v1"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def scalar_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def read_margin_weight(path: pathlib.Path, *, model_sha256: str) -> float:
    value = json.loads(path.read_text(encoding="utf-8"))
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("training readback candidates are missing")
    matches = [
        item
        for item in candidates
        if isinstance(item, dict) and item.get("model_sha256") == model_sha256
    ]
    if len(matches) != 1:
        raise ValueError("training readback does not uniquely bind retained model")
    weights = matches[0].get("auxiliary_loss_weights")
    if not isinstance(weights, dict):
        raise ValueError("training readback auxiliary weights are missing")
    weight = float(weights.get("keyword_sequence_margin_loss_weight", float("nan")))
    if not math.isfinite(weight) or weight <= 0.0:
        raise ValueError("retained keyword sequence-margin weight is invalid")
    return weight


def ctc_true_nll(
    log_probs: torch.Tensor,
    sequence: tuple[int, ...],
) -> torch.Tensor:
    steps = int(log_probs.shape[0])
    target = torch.tensor(sequence, dtype=torch.long, device=log_probs.device)
    return F.ctc_loss(
        log_probs.unsqueeze(1),
        target,
        torch.tensor([steps], dtype=torch.long, device=log_probs.device),
        torch.tensor([len(sequence)], dtype=torch.long, device=log_probs.device),
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )


def margin_loss(
    log_probs: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    keywords: list[tuple[int, ...]],
    positive_policy: str,
) -> torch.Tensor:
    target = torch.tensor(sequence, dtype=torch.long, device=log_probs.device)
    input_lengths = torch.tensor(
        [int(log_probs.shape[0])],
        dtype=torch.long,
        device=log_probs.device,
    )
    target_lengths = torch.tensor(
        [len(sequence)],
        dtype=torch.long,
        device=log_probs.device,
    )
    true_nll = ctc_true_nll(log_probs, sequence).reshape(1)
    return keyword_sequence_margin_loss(
        log_probs=log_probs.unsqueeze(1),
        ctc_log_probs=log_probs.unsqueeze(1),
        targets=target,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        true_ctc_nll=true_nll,
        keyword_sequences=[list(item) for item in keywords],
        blank=0,
        margin=0.05,
        keyword_operating_points=[
            {
                "threshold": 0.60,
                "positive_margin": 0.05,
                "negative_margin": 0.05,
            }
            for _ in keywords
        ],
        negative_path_policy=SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE,
        positive_path_policy=positive_policy,
    )[0]


def gradient(
    raw_logits: torch.Tensor,
    *,
    sequence: tuple[int, ...],
    keywords: list[tuple[int, ...]],
    positive_policy: str,
) -> tuple[float, torch.Tensor]:
    logits = raw_logits.detach().clone().requires_grad_(True)
    log_probs = logits.log_softmax(dim=1)
    loss = margin_loss(
        log_probs,
        sequence=sequence,
        keywords=keywords,
        positive_policy=positive_policy,
    )
    grad = torch.autograd.grad(loss, logits)[0]
    return float(loss.detach()), grad


def prefix_update(
    update: torch.Tensor,
    positions: list[dict],
    count: int,
) -> float:
    total = 0.0
    for item in positions[:count]:
        token = int(item["token"])
        start = int(item["q10_frame"])
        stop = int(item["q90_frame"]) + 1
        if start < 0 or stop > int(update.shape[0]) or start >= stop:
            raise ValueError("sequence competition position window is invalid")
        total += float(update[start:stop, token].sum())
    return total


def summarize(records: list[dict]) -> dict:
    return {
        "recordings": len(records),
        "old_loss": scalar_stats([float(row["old_loss"]) for row in records]),
        "new_loss": scalar_stats([float(row["new_loss"]) for row in records]),
        "old_weighted_grad_l2": scalar_stats(
            [float(row["old_weighted_grad_l2"]) for row in records]
        ),
        "new_weighted_grad_l2": scalar_stats(
            [float(row["new_weighted_grad_l2"]) for row in records]
        ),
        "correct_prefix_update": scalar_stats(
            [float(row["correct_prefix_update"]) for row in records]
        ),
        "competitor_prefix_update": scalar_stats(
            [float(row["competitor_prefix_update"]) for row in records]
        ),
        "correct_prefix_increased": sum(
            float(row["correct_prefix_update"]) > 0.0 for row in records
        ),
        "competitor_prefix_decreased": sum(
            float(row["competitor_prefix_update"]) < 0.0 for row in records
        ),
        "blank_global_update": scalar_stats(
            [float(row["blank_global_update"]) for row in records]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords-tsv", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", required=True, type=pathlib.Path)
    parser.add_argument("--sequence-competition", required=True, type=pathlib.Path)
    parser.add_argument("--training-readback", required=True, type=pathlib.Path)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    sequence_competition = args.sequence_competition.resolve()
    training_readback = args.training_readback.resolve()
    output = args.output.resolve()
    model_sha256 = str(args.model_sha256)

    for path, label in (
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (acoustic_alignment, "acoustic alignment"),
        (sequence_competition, "sequence competition"),
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
        raise ValueError("competition gradient audit requires multiple keywords")

    positive = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    traces = trace_paths(posterior_cache, model_sha256)
    by_sha = {path.stem: path for path in traces}
    competition = json.loads(sequence_competition.read_text(encoding="utf-8"))
    if competition.get("model_sha256") != model_sha256:
        raise ValueError("sequence competition model SHA mismatch")
    competition_rows = {
        str(row["audio_sha256"]): row
        for row in competition.get("records", [])
        if isinstance(row, dict)
    }
    if set(positive) - set(competition_rows):
        raise ValueError("sequence competition misses retained positive rows")

    alignment_value = json.loads(acoustic_alignment.read_text(encoding="utf-8"))
    alignment_rows = {
        str(row["wav_sha256"]): row
        for row in alignment_value["records"]
        if isinstance(row, dict)
    }

    weight = read_margin_weight(training_readback, model_sha256=model_sha256)
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

        raw = torch.tensor(
            read_trace_logits(trace),
            dtype=torch.float32,
        )
        retained_nll = float(alignment_rows[audio_sha]["ctc_nll_per_token"])
        recomputed_nll = float(
            ctc_true_nll(raw.log_softmax(dim=1), sequence).detach()
        ) / float(len(sequence))
        if abs(recomputed_nll - retained_nll) > 1.0e-4:
            raise ValueError(
                f"retained CTC NLL drifted for {audio_sha}: "
                f"{recomputed_nll} != {retained_nll}"
            )

        old_loss, old_grad = gradient(
            raw,
            sequence=sequence,
            keywords=keywords,
            positive_policy=SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE,
        )
        new_loss, new_grad = gradient(
            raw,
            sequence=sequence,
            keywords=keywords,
            positive_policy=SEQUENCE_MARGIN_POSITIVE_POLICY_CTC_KEYWORD_COMPETITION,
        )
        update = -new_grad
        row = competition_rows[audio_sha]
        correct_prefix = int(row["correct_discriminative_prefix_length"])
        competitor_prefix = int(row["competitor_discriminative_prefix_length"])

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
                "old_loss": old_loss,
                "new_loss": new_loss,
                "old_weighted_grad_l2": weight * float(old_grad.norm()),
                "new_weighted_grad_l2": weight * float(new_grad.norm()),
                "correct_prefix_update": prefix_update(
                    update,
                    row["correct_positions"],
                    correct_prefix,
                ),
                "competitor_prefix_update": prefix_update(
                    update,
                    row["closest_competitor_positions"],
                    competitor_prefix,
                ),
                "blank_global_update": float(update[:, 0].sum()),
            }
        )

    cohorts: dict[str, list[dict]] = {
        "all": records,
    }
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
        "candidate_positive_policy": (
            SEQUENCE_MARGIN_POSITIVE_POLICY_CTC_KEYWORD_COMPETITION
        ),
        "baseline_positive_policy": SEQUENCE_MARGIN_POSITIVE_POLICY_SPARSE,
        "negative_policy_fixed": (
            SEQUENCE_MARGIN_NEGATIVE_POLICY_RUNTIME_EXECUTABLE
        ),
        "keyword_sequence_margin_loss_weight": weight,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "sequence_competition_sha256": sha256_file(sequence_competition),
        "training_readback_sha256": sha256_file(training_readback),
        "cohorts": {
            name: summarize(rows)
            for name, rows in cohorts.items()
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "weight": weight,
        "cohorts": result["cohorts"],
        "selection_feedback_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
