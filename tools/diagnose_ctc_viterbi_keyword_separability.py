#!/usr/bin/env python3
"""Evaluate a token-synchronous CTC Viterbi keyword scorer on retained evidence."""
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

EVIDENCE_CLASS = "ctc-viterbi-keyword-separability-development-v1"
POLICY = "max-over-start-token-synchronous-ctc-viterbi-per-token-v1"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def ctc_viterbi_segment_log_score(
    log_probs: list[list[float]],
    target: tuple[int, ...],
    *,
    blank: int = 0,
) -> float:
    """Best finite CTC path for target, with an arbitrary segment start.

    The path starts when the first target token is emitted. After that every
    frame is consumed by either a target-token state or a blank state, so long
    gaps and unrelated-token regions cannot be crossed for free.

    This is Viterbi (max path), not a CTC prefix beam / log-sum over paths.
    Adjacent unequal labels may transition directly; adjacent equal labels must
    pass through blank, matching standard CTC collapse semantics.
    """
    if not log_probs or not target:
        raise ValueError("CTC Viterbi score requires non-empty trace/target")
    vocab = len(log_probs[0])
    if blank < 0 or blank >= vocab:
        raise ValueError("blank token is outside vocabulary")
    if any(
        len(row) != vocab or any(not math.isfinite(value) for value in row)
        for row in log_probs
    ):
        raise ValueError("posterior rows are ragged/non-finite")
    if any(token == blank or token < 0 or token >= vocab for token in target):
        raise ValueError("target contains an invalid token")

    neg_inf = float("-inf")
    length = len(target)
    token_state = [neg_inf] * length
    blank_state = [neg_inf] * length
    best_terminal = neg_inf

    for row in log_probs:
        next_token = [neg_inf] * length
        next_blank = [neg_inf] * length

        # Local restart: pre-keyword frames carry no score. This keeps the
        # metric focused on the best keyword-shaped segment in the utterance.
        next_token[0] = row[target[0]]

        for index, token in enumerate(target):
            if math.isfinite(token_state[index]):
                next_token[index] = max(
                    next_token[index],
                    token_state[index] + row[token],
                )

            if index > 0:
                if math.isfinite(blank_state[index - 1]):
                    next_token[index] = max(
                        next_token[index],
                        blank_state[index - 1] + row[token],
                    )
                if (
                    target[index] != target[index - 1]
                    and math.isfinite(token_state[index - 1])
                ):
                    next_token[index] = max(
                        next_token[index],
                        token_state[index - 1] + row[token],
                    )

            prior = max(token_state[index], blank_state[index])
            if math.isfinite(prior):
                next_blank[index] = prior + row[blank]

        token_state = next_token
        blank_state = next_blank
        best_terminal = max(
            best_terminal,
            token_state[-1],
            blank_state[-1],
        )

    return best_terminal


def normalized_score(
    log_probs: list[list[float]],
    target: tuple[int, ...],
) -> float:
    raw = ctc_viterbi_segment_log_score(log_probs, target)
    return raw / float(len(target))


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def exact_positive_summary(
    *,
    traces: list[pathlib.Path],
    model_sha256: str,
    keywords: dict[int, dict],
    positive_sample: dict[str, dict],
) -> dict:
    by_sha = {path.stem: path for path in traces}
    missing = sorted(set(positive_sample) - set(by_sha))
    if missing:
        raise ValueError(
            "retained positive sample lacks posterior traces: "
            + ",".join(missing[:8])
        )

    records: list[dict] = []
    by_split_keyword: dict[str, list[dict]] = {}
    for audio_sha, sample in sorted(positive_sample.items()):
        keyword_id = int(sample["keyword_id"])
        if keyword_id not in keywords:
            raise ValueError("positive sample references unknown keyword")
        logits = read_trace_logits(by_sha[audio_sha])
        log_probs = log_softmax(logits)
        scores = {
            int(other_id): normalized_score(
                log_probs,
                tuple(int(value) for value in row["tokens"]),
            )
            for other_id, row in keywords.items()
        }
        target_score = scores[keyword_id]
        wrong_score = max(
            value for other_id, value in scores.items()
            if other_id != keyword_id
        )
        target_wins = target_score > wrong_score
        row = {
            "audio_sha256": audio_sha,
            "split": sample["split"],
            "keyword_id": keyword_id,
            "runtime_detected_expected": bool(
                sample["runtime_detected_expected"]
            ),
            "target_score": target_score,
            "best_wrong_score": wrong_score,
            "target_minus_wrong": target_score - wrong_score,
            "target_wins": target_wins,
            "scores": {str(key): value for key, value in scores.items()},
        }
        records.append(row)
        key = f"{sample['split']}:kw{keyword_id}"
        by_split_keyword.setdefault(key, []).append(row)

    def summarize(rows: list[dict]) -> dict:
        margins = [float(row["target_minus_wrong"]) for row in rows]
        targets = [float(row["target_score"]) for row in rows]
        return {
            "recordings": len(rows),
            "target_wins": sum(bool(row["target_wins"]) for row in rows),
            "runtime_hits": sum(
                bool(row["runtime_detected_expected"]) for row in rows
            ),
            "target_score": stats(targets),
            "target_minus_wrong": stats(margins),
        }

    return {
        "recordings": len(records),
        "summary": summarize(records),
        "by_split_keyword": {
            key: summarize(rows)
            for key, rows in sorted(by_split_keyword.items())
        },
        "records": records,
    }


def boundary_summary(
    *,
    boundary: dict,
    boundary_trace_dir: pathlib.Path,
    keywords: dict[int, dict],
) -> dict:
    records = boundary.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("boundary summary has no records")

    paired: list[dict] = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("boundary record must be an object")
        keyword_id = int(item["keyword_id"])
        keyword = keywords.get(keyword_id)
        if keyword is None:
            raise ValueError("boundary record references unknown keyword")
        target = tuple(int(value) for value in keyword["tokens"])

        pos_sha = str(item["within_word_wav_sha256"])
        neg_sha = str(item["cross_boundary_wav_sha256"])
        pos_trace = boundary_trace_dir / f"{pos_sha}.kwtr"
        neg_trace = boundary_trace_dir / f"{neg_sha}.kwtr"
        if not pos_trace.is_file() or not neg_trace.is_file():
            raise ValueError("boundary posterior trace is missing")

        pos_lp = log_softmax(read_trace_logits(pos_trace))
        neg_lp = log_softmax(read_trace_logits(neg_trace))
        pos_scores = {
            int(other_id): normalized_score(
                pos_lp,
                tuple(int(value) for value in row["tokens"]),
            )
            for other_id, row in keywords.items()
        }
        neg_scores = {
            int(other_id): normalized_score(
                neg_lp,
                tuple(int(value) for value in row["tokens"]),
            )
            for other_id, row in keywords.items()
        }
        pos_target = pos_scores[keyword_id]
        neg_target = neg_scores[keyword_id]
        paired.append(
            {
                "voice_id": str(item["voice_id"]),
                "keyword_id": keyword_id,
                "within_word_wav_sha256": pos_sha,
                "cross_boundary_wav_sha256": neg_sha,
                "within_word_target_score": pos_target,
                "cross_boundary_target_score": neg_target,
                "paired_delta": pos_target - neg_target,
                "within_word_target_minus_wrong": (
                    pos_target
                    - max(
                        value for other_id, value in pos_scores.items()
                        if other_id != keyword_id
                    )
                ),
                "cross_boundary_target_minus_wrong": (
                    neg_target
                    - max(
                        value for other_id, value in neg_scores.items()
                        if other_id != keyword_id
                    )
                ),
                "within_word_scores": {
                    str(key): value for key, value in pos_scores.items()
                },
                "cross_boundary_scores": {
                    str(key): value for key, value in neg_scores.items()
                },
            }
        )

    def summarize(rows: list[dict]) -> dict:
        pos = [float(row["within_word_target_score"]) for row in rows]
        neg = [float(row["cross_boundary_target_score"]) for row in rows]
        delta = [float(row["paired_delta"]) for row in rows]
        return {
            "pairs": len(rows),
            "paired_positive_above_negative": sum(value > 0.0 for value in delta),
            "within_word_target_score": stats(pos),
            "cross_boundary_target_score": stats(neg),
            "paired_delta": stats(delta),
            "strict_pooled_separation": bool(pos and neg and min(pos) > max(neg)),
            "pooled_separation_margin": (
                min(pos) - max(neg) if pos and neg else None
            ),
        }

    by_keyword: dict[str, list[dict]] = {}
    for row in paired:
        by_keyword.setdefault(str(row["keyword_id"]), []).append(row)

    return {
        "summary": summarize(paired),
        "by_keyword": {
            key: summarize(rows)
            for key, rows in sorted(by_keyword.items())
        },
        "records": paired,
    }


def self_test() -> None:
    # Strong token1 -> blank -> token2 path should beat a wrong [2,1] path.
    rows = log_softmax(
        [
            [0.0, 5.0, -5.0],
            [5.0, -5.0, -5.0],
            [0.0, -5.0, 5.0],
        ]
    )
    good = normalized_score(rows, (1, 2))
    wrong = normalized_score(rows, (2, 1))
    assert good > wrong

    # Waiting through a frame where blank is very unlikely must cost score.
    clean = log_softmax(
        [
            [0.0, 5.0, -5.0, -5.0],
            [5.0, -5.0, -5.0, -5.0],
            [0.0, -5.0, 5.0, -5.0],
        ]
    )
    interfered = log_softmax(
        [
            [0.0, 5.0, -5.0, -5.0],
            [-5.0, -5.0, -5.0, 5.0],
            [0.0, -5.0, 5.0, -5.0],
        ]
    )
    assert normalized_score(clean, (1, 2)) > normalized_score(
        interfered, (1, 2)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--boundary-summary", type=pathlib.Path)
    parser.add_argument("--boundary-trace-dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("CTC Viterbi keyword scorer self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "posterior-cache": args.posterior_cache,
        "acoustic-alignment": args.acoustic_alignment,
        "boundary-summary": args.boundary_summary,
        "boundary-trace-dir": args.boundary_trace_dir,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing arguments: " + ", ".join(missing))

    model = args.model.resolve()
    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    boundary_path = args.boundary_summary.resolve()
    boundary_trace_dir = args.boundary_trace_dir.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (acoustic_alignment, "acoustic alignment"),
        (boundary_path, "boundary summary"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir() or not boundary_trace_dir.is_dir():
        raise ValueError("posterior trace directory is missing")

    model_sha = sha256_file(model)
    token_map = load_tokens(tokens)
    keywords = load_keywords(keywords_tsv, token_map)
    traces = trace_paths(posterior_cache, model_sha)
    positive_sample = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha,
    )
    boundary = load_object(boundary_path, "boundary summary")
    if boundary.get("model_sha256") != model_sha:
        raise ValueError("boundary summary model SHA mismatch")

    exact = exact_positive_summary(
        traces=traces,
        model_sha256=model_sha,
        keywords=keywords,
        positive_sample=positive_sample,
    )
    boundary_result = boundary_summary(
        boundary=boundary,
        boundary_trace_dir=boundary_trace_dir,
        keywords=keywords,
    )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed_for_exact_positive": True,
        "training_changed": False,
        "decoder_changed": False,
        "score_definition": {
            "path_aggregation": "viterbi-max",
            "start_policy": "restart-on-first-target-token-any-frame",
            "frame_accounting": "consume-every-frame-after-start",
            "wait_state": "actual-ctc-blank-log-probability",
            "adjacent_repeat_policy": "blank-separation-required",
            "normalization": "divide-by-target-token-count",
            "not_prefix_beam": True,
        },
        "model_sha256": model_sha,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "exact_positive": exact,
        "boundary": boundary_result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "exact_positive": exact["summary"],
                "boundary": boundary_result["summary"],
                "boundary_by_keyword": boundary_result["by_keyword"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
