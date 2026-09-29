#!/usr/bin/env python3
"""Decompose exact CTC keyword competition and token-position ambiguity."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

from diagnose_decoder_search_path_decomposition import load_positive_sample
from diagnose_sequence_margin_runtime_gap import (
    load_keywords,
    load_tokens,
    log_softmax,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "keyword-ctc-sequence-competition-development-v1"
POLICY = "exact-ctc-forward-backward-position-competition-v1"


def logsumexp(values: list[float]) -> float:
    finite = [value for value in values if value != float("-inf")]
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in finite)
    )


def extended_target(sequence: tuple[int, ...], blank: int = 0) -> list[int]:
    result = [blank]
    for token in sequence:
        result.extend((token, blank))
    return result


def allowed_predecessors(states: list[int], index: int, blank: int = 0) -> list[int]:
    result = [index]
    if index > 0:
        result.append(index - 1)
    if (
        index > 1
        and states[index] != blank
        and states[index] != states[index - 2]
    ):
        result.append(index - 2)
    return result


def allowed_successors(states: list[int], index: int, blank: int = 0) -> list[int]:
    result = [index]
    if index + 1 < len(states):
        result.append(index + 1)
    if (
        index + 2 < len(states)
        and states[index + 2] != blank
        and states[index + 2] != states[index]
    ):
        result.append(index + 2)
    return result


def percentile_from_distribution(probabilities: list[float], fraction: float) -> int:
    cumulative = 0.0
    for index, value in enumerate(probabilities):
        cumulative += value
        if cumulative >= fraction:
            return index
    return len(probabilities) - 1


def ctc_forward_backward(
    log_probs: list[list[float]],
    sequence: tuple[int, ...],
    *,
    blank: int = 0,
) -> tuple[float, list[dict], list[dict]]:
    if not log_probs or not sequence:
        raise ValueError("CTC competition requires non-empty trace and sequence")
    vocab = len(log_probs[0])
    if blank < 0 or blank >= vocab:
        raise ValueError("blank token is outside vocabulary")
    if any(len(row) != vocab for row in log_probs):
        raise ValueError("posterior rows must share one vocabulary")
    if any(token <= blank or token >= vocab for token in sequence):
        raise ValueError("keyword token is outside vocabulary")

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

    positions: list[dict] = []
    for occurrence, token in enumerate(sequence):
        state = occurrence * 2 + 1
        posterior = [
            math.exp(alpha[frame][state] + beta[frame][state] - log_probability)
            for frame in range(steps)
        ]
        occupancy = sum(posterior)
        if occupancy <= 0.0 or not math.isfinite(occupancy):
            raise ValueError("CTC token-state occupancy is invalid")
        temporal = [value / occupancy for value in posterior]
        expected = sum(
            float(frame) * value for frame, value in enumerate(temporal)
        )
        peak_frame = max(range(steps), key=temporal.__getitem__)
        q10 = percentile_from_distribution(temporal, 0.10)
        q50 = percentile_from_distribution(temporal, 0.50)
        q90 = percentile_from_distribution(temporal, 0.90)
        entropy = -sum(
            value * math.log(value)
            for value in temporal
            if value > 0.0
        )
        positions.append(
            {
                "occurrence": occurrence + 1,
                "token": int(token),
                "expected_frame": expected,
                "q10_frame": q10,
                "q50_frame": q50,
                "q90_frame": q90,
                "width80_frames": q90 - q10,
                "state_occupancy_frames": occupancy,
                "temporal_peak_frame": peak_frame,
                "temporal_peak_fraction": temporal[peak_frame],
                "temporal_entropy_nats": entropy,
            }
        )

    viterbi = [[float("-inf")] * count for _ in range(steps)]
    backpointer = [[-1] * count for _ in range(steps)]
    viterbi[0][0] = log_probs[0][blank]
    if count > 1:
        viterbi[0][1] = log_probs[0][states[1]]
    for frame in range(1, steps):
        for state, token in enumerate(states):
            candidates = [
                (viterbi[frame - 1][prior], prior)
                for prior in allowed_predecessors(states, state, blank)
            ]
            best_score, best_state = max(candidates, key=lambda item: item[0])
            viterbi[frame][state] = best_score + log_probs[frame][token]
            backpointer[frame][state] = best_state

    terminal = max(
        (count - 2, count - 1),
        key=lambda state: viterbi[-1][state],
    )
    state_path = [terminal]
    for frame in range(steps - 1, 0, -1):
        terminal = backpointer[frame][terminal]
        if terminal < 0:
            raise ValueError("CTC Viterbi backpointer is invalid")
        state_path.append(terminal)
    state_path.reverse()

    viterbi_positions: list[dict] = []
    for occurrence, token in enumerate(sequence):
        state = occurrence * 2 + 1
        frames = [
            frame
            for frame, current_state in enumerate(state_path)
            if current_state == state
        ]
        if not frames:
            raise ValueError("CTC Viterbi path skipped target occurrence")
        viterbi_positions.append(
            {
                "occurrence": occurrence + 1,
                "token": int(token),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "frames": len(frames),
            }
        )

    return log_probability, positions, viterbi_positions


def common_suffix_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    count = 0
    while (
        count < len(left)
        and count < len(right)
        and left[-1 - count] == right[-1 - count]
    ):
        count += 1
    return count


def ordered_token_region_coverage(
    log_probs: list[list[float]],
    sequence: tuple[int, ...],
) -> dict:
    """Replay the trainer's ordered_token_loss region semantics exactly."""
    steps = len(log_probs)
    positions: list[dict] = []
    correct = 0
    losses: list[float] = []
    for occurrence, token in enumerate(sequence):
        start = (occurrence * steps) // len(sequence)
        stop = max(start + 1, ((occurrence + 1) * steps) // len(sequence))
        stop = min(stop, steps)
        best_frame = max(
            range(start, stop),
            key=lambda frame: log_probs[frame][token],
        )
        predicted = max(
            range(len(log_probs[best_frame])),
            key=log_probs[best_frame].__getitem__,
        )
        token_log_probability = float(log_probs[best_frame][token])
        is_correct = predicted == token
        correct += int(is_correct)
        losses.append(-token_log_probability)
        positions.append(
            {
                "occurrence": occurrence + 1,
                "token": int(token),
                "region_start_frame": start,
                "region_stop_frame_exclusive": stop,
                "best_frame": best_frame,
                "predicted_token": int(predicted),
                "top1_correct": bool(is_correct),
                "best_token_log_probability": token_log_probability,
            }
        )
    return {
        "positions": positions,
        "top1_correct_positions": correct,
        "positions_total": len(sequence),
        "all_positions_top1_correct": correct == len(sequence),
        "mean_negative_log_probability": statistics.fmean(losses),
    }


def top1_token_summary(
    logits: list[list[float]],
    sequence: tuple[int, ...],
) -> list[dict]:
    result: list[dict] = []
    for occurrence, token in enumerate(sequence):
        top1_frames: list[int] = []
        best_rank = len(logits[0]) + 1
        best_margin = float("-inf")
        best_margin_frame = -1
        for frame, row in enumerate(logits):
            order = sorted(range(len(row)), key=row.__getitem__, reverse=True)
            rank = order.index(token) + 1
            best_rank = min(best_rank, rank)
            if order[0] == token:
                top1_frames.append(frame)
            competitor = max(
                value
                for index, value in enumerate(row)
                if index != token
            )
            margin = float(row[token] - competitor)
            if margin > best_margin:
                best_margin = margin
                best_margin_frame = frame
        result.append(
            {
                "occurrence": occurrence + 1,
                "token": int(token),
                "top1_frames": len(top1_frames),
                "first_top1_frame": top1_frames[0] if top1_frames else None,
                "last_top1_frame": top1_frames[-1] if top1_frames else None,
                "best_rank": best_rank,
                "best_logit_margin": best_margin,
                "best_margin_frame": best_margin_frame,
            }
        )
    return result


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
    }


def cohort_summary(records: list[dict]) -> dict:
    margins = [
        float(row["closest_competitor_minus_correct_nll_per_token"])
        for row in records
    ]
    correct_ranks = [float(row["correct_keyword_rank"]) for row in records]
    ordered_counts = [
        int(row["ordered_token_region_coverage"]["top1_correct_positions"])
        for row in records
    ]
    ordered_losses = [
        float(row["ordered_token_region_coverage"]["mean_negative_log_probability"])
        for row in records
    ]
    result = {
        "recordings": len(records),
        "ordered_token_all_positions_top1_correct": sum(
            bool(row["ordered_token_region_coverage"]["all_positions_top1_correct"])
            for row in records
        ),
        "ordered_token_top1_correct_positions": scalar_stats(
            [float(value) for value in ordered_counts]
        ),
        "ordered_token_mean_negative_log_probability": scalar_stats(
            ordered_losses
        ),
        "correct_keyword_wins": sum(
            float(row["closest_competitor_minus_correct_nll_per_token"]) > 0.0
            for row in records
        ),
        "correct_keyword_ties_or_loses": sum(
            float(row["closest_competitor_minus_correct_nll_per_token"]) <= 0.0
            for row in records
        ),
        "closest_competitor_minus_correct_nll_per_token": scalar_stats(margins),
        "correct_keyword_rank": scalar_stats(correct_ranks),
    }
    if records:
        prefix_widths: list[float] = []
        prefix_peaks: list[float] = []
        for row in records:
            for position in row["correct_positions"][
                : int(row["correct_discriminative_prefix_length"])
            ]:
                prefix_widths.append(float(position["width80_frames"]))
                prefix_peaks.append(float(position["temporal_peak_fraction"]))
        result["correct_discriminative_prefix_width80_frames"] = scalar_stats(
            prefix_widths
        )
        result["correct_discriminative_prefix_temporal_peak_fraction"] = scalar_stats(
            prefix_peaks
        )
    return result


def self_test() -> None:
    rows = log_softmax(
        [
            [0.0, 5.0, -5.0],
            [5.0, -5.0, -5.0],
            [0.0, -5.0, 5.0],
            [5.0, -5.0, -5.0],
        ]
    )
    value, positions, viterbi = ctc_forward_backward(rows, (1, 2))
    assert math.isfinite(value)
    assert len(positions) == 2
    assert positions[0]["expected_frame"] < positions[1]["expected_frame"]
    assert viterbi[0]["first_frame"] < viterbi[1]["first_frame"]
    assert common_suffix_length((1, 2, 3, 4), (3, 4, 3, 4)) == 2
    assert common_suffix_length((1, 2), (1, 2)) == 2
    ordered = ordered_token_region_coverage(rows, (1, 2))
    assert ordered["positions_total"] == 2
    assert ordered["top1_correct_positions"] == 2
    assert ordered["all_positions_top1_correct"] is True
    print("keyword CTC sequence competition self-test: PASS")


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

    required = {
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "posterior-cache": args.posterior_cache,
        "acoustic-alignment": args.acoustic_alignment,
        "model-sha256": args.model_sha256,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    output = args.output.resolve()
    model_sha256 = str(args.model_sha256)

    for path, label in (
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (acoustic_alignment, "acoustic alignment"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError("posterior cache is missing")
    if (
        len(model_sha256) != 64
        or any(char not in "0123456789abcdef" for char in model_sha256)
    ):
        raise ValueError("model SHA256 is invalid")

    token_map = load_tokens(tokens)
    keywords = load_keywords(keywords_tsv, token_map)
    if len(keywords) < 2:
        raise ValueError("sequence competition requires at least two keywords")
    positive_sample = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    alignment = json.loads(acoustic_alignment.read_text(encoding="utf-8"))
    alignment_by_sha = {
        str(row["wav_sha256"]): row
        for row in alignment["records"]
    }
    traces = trace_paths(posterior_cache, model_sha256)
    by_sha = {path.stem: path for path in traces}

    missing_traces = sorted(set(positive_sample) - set(by_sha))
    if missing_traces:
        raise ValueError(
            "positive sample lacks posterior traces: "
            + ",".join(missing_traces[:8])
        )

    records: list[dict] = []
    for audio_sha, sample in sorted(positive_sample.items()):
        true_keyword = int(sample["keyword_id"])
        if true_keyword not in keywords:
            raise ValueError("positive sample references unknown keyword")
        logits = read_trace_logits(by_sha[audio_sha])
        log_probs = log_softmax(logits)

        hypotheses: list[dict] = []
        for keyword_id, item in sorted(keywords.items()):
            sequence = tuple(int(value) for value in item["tokens"])
            log_probability, positions, viterbi = ctc_forward_backward(
                log_probs,
                sequence,
            )
            hypotheses.append(
                {
                    "keyword_id": int(keyword_id),
                    "sequence": list(sequence),
                    "ctc_log_probability": log_probability,
                    "ctc_nll_per_token": -log_probability / float(len(sequence)),
                    "positions": positions,
                    "viterbi_positions": viterbi,
                    "top1_token_summary": top1_token_summary(logits, sequence),
                }
            )

        ranked = sorted(
            hypotheses,
            key=lambda row: (
                float(row["ctc_nll_per_token"]),
                int(row["keyword_id"]),
            ),
        )
        correct = next(
            row for row in hypotheses if int(row["keyword_id"]) == true_keyword
        )
        competitors = [
            row for row in ranked if int(row["keyword_id"]) != true_keyword
        ]
        closest = competitors[0]
        correct_rank = (
            next(
                index
                for index, row in enumerate(ranked, 1)
                if int(row["keyword_id"]) == true_keyword
            )
        )

        alignment_row = alignment_by_sha.get(audio_sha)
        if not isinstance(alignment_row, dict):
            raise ValueError("acoustic alignment row is missing")
        retained_nll = float(alignment_row["ctc_nll_per_token"])
        actual_nll = float(correct["ctc_nll_per_token"])
        if abs(actual_nll - retained_nll) > 1.0e-5:
            raise ValueError(
                f"exact CTC recomputation drifted for {audio_sha}: "
                f"{actual_nll} != {retained_nll}"
            )

        correct_sequence = tuple(int(value) for value in correct["sequence"])
        competitor_sequence = tuple(int(value) for value in closest["sequence"])
        shared_suffix = common_suffix_length(
            correct_sequence,
            competitor_sequence,
        )
        correct_prefix = len(correct_sequence) - shared_suffix
        competitor_prefix = len(competitor_sequence) - shared_suffix

        records.append(
            {
                "audio_sha256": audio_sha,
                "split": sample["split"],
                "true_keyword_id": true_keyword,
                "runtime_hit": bool(sample["runtime_detected_expected"]),
                "surrogate_above_threshold": bool(
                    sample["surrogate_above_runtime_threshold"]
                ),
                "runtime_gap": bool(
                    sample["surrogate_above_runtime_threshold"]
                    and not sample["runtime_detected_expected"]
                ),
                "correct_keyword_rank": correct_rank,
                "correct_ctc_nll_per_token": actual_nll,
                "closest_competitor_keyword_id": int(closest["keyword_id"]),
                "closest_competitor_ctc_nll_per_token": float(
                    closest["ctc_nll_per_token"]
                ),
                "closest_competitor_minus_correct_nll_per_token": (
                    float(closest["ctc_nll_per_token"]) - actual_nll
                ),
                "shared_suffix_length": shared_suffix,
                "correct_discriminative_prefix_length": correct_prefix,
                "competitor_discriminative_prefix_length": competitor_prefix,
                "correct_positions": correct["positions"],
                "correct_viterbi_positions": correct["viterbi_positions"],
                "correct_top1_token_summary": correct["top1_token_summary"],
                "ordered_token_region_coverage": ordered_token_region_coverage(
                    log_probs,
                    correct_sequence,
                ),
                "closest_competitor_positions": closest["positions"],
                "closest_competitor_viterbi_positions": closest[
                    "viterbi_positions"
                ],
                "hypotheses": [
                    {
                        "keyword_id": int(row["keyword_id"]),
                        "ctc_nll_per_token": float(row["ctc_nll_per_token"]),
                    }
                    for row in ranked
                ],
            }
        )

    by_keyword: dict[str, dict] = {}
    for keyword_id in sorted(keywords):
        keyword_records = [
            row
            for row in records
            if int(row["true_keyword_id"]) == int(keyword_id)
        ]
        by_keyword[str(keyword_id)] = {
            "all": cohort_summary(keyword_records),
            "runtime_hit": cohort_summary(
                [row for row in keyword_records if row["runtime_hit"]]
            ),
            "runtime_gap": cohort_summary(
                [row for row in keyword_records if row["runtime_gap"]]
            ),
            "surrogate_below_runtime_miss": cohort_summary(
                [
                    row
                    for row in keyword_records
                    if not row["surrogate_above_threshold"]
                    and not row["runtime_hit"]
                ]
            ),
        }

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "decoder_math_changed": False,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "by_keyword": by_keyword,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "by_keyword": by_keyword,
                "records": len(records),
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
