#!/usr/bin/env python3
"""Decompose sequence-margin/runtime misses on retained posterior traces."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import subprocess

from diagnose_sequence_margin_runtime_gap import (
    decoder_sequence_log_confidence,
    load_keywords,
    load_tokens,
    log_softmax,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "sequence-margin-runtime-gap-decomposition-development-v1"
POLICY = "fixed-posterior-independent-decoder-probes-v1"

PROBES = {
    "refractory_zero": ("--refractory-ms", "0"),
    "state_retention_097": ("--state-retention", "0.97"),
    "blank_retention_085": ("--blank-retention", "0.85"),
    "blank_retention_095": ("--blank-retention", "0.95"),
    "fuzzy_child_cost_m4": ("--fuzzy-child-cost-log", "-4.0"),
}


def runtime_hits(
    *,
    decoder_replay: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    trace: pathlib.Path,
    recording: str,
    option: tuple[str, str] | None = None,
) -> set[int]:
    command = [
        str(decoder_replay),
        str(model),
        str(pack),
        str(trace),
        recording,
    ]
    if option is not None:
        command.extend(option)
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    result: set[int] = set()
    for line_no, raw in enumerate(completed.stdout.splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("recording") != recording:
            raise ValueError(
                f"decoder replay emitted invalid event at {trace}:{line_no}"
            )
        result.add(int(value["keyword_id"]))
    return result


def greedy_ctc_tokens(log_probs: list[list[float]]) -> tuple[int, ...]:
    result: list[int] = []
    previous: int | None = None
    for row in log_probs:
        token = max(range(len(row)), key=row.__getitem__)
        if token != previous and token != 0:
            result.append(token)
        previous = token
    return tuple(result)


def contains_subsequence(sequence: tuple[int, ...], target: tuple[int, ...]) -> bool:
    if not target:
        return True
    index = 0
    for token in sequence:
        if token == target[index]:
            index += 1
            if index == len(target):
                return True
    return False


def longest_prefix_depth(sequence: tuple[int, ...], target: tuple[int, ...]) -> int:
    if not target:
        return 0
    depth = 0
    for token in sequence:
        if token == target[depth]:
            depth += 1
            if depth == len(target):
                return depth
    return depth


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def summarize(
    *,
    traces: list[pathlib.Path],
    model: pathlib.Path,
    pack: pathlib.Path,
    keywords: dict[int, dict],
    decoder_replay: pathlib.Path,
) -> tuple[dict[str, dict], dict]:
    rows = {
        keyword_id: {
            "traces": 0,
            "surrogate_above_threshold": 0,
            "runtime_hit": 0,
            "surrogate_above_threshold_runtime_miss": 0,
            "miss_greedy_contains_target": 0,
            "miss_greedy_incomplete_target_order": 0,
            "miss_wrong_keyword_hit": 0,
            "miss_no_runtime_hit": 0,
            "miss_unrecovered_all_probes": 0,
            "recoveries": {name: 0 for name in PROBES},
            "miss_prefix_depth_histogram": {},
            "_miss_confidences": [],
            "_hit_confidences": [],
        }
        for keyword_id in keywords
    }
    probe_trace_count = 0
    baseline_trace_count = 0

    for trace_index, trace in enumerate(traces):
        logits = read_trace_logits(trace)
        log_probs = log_softmax(logits)
        recording = f"posterior-trace-{trace_index:06d}"
        baseline = runtime_hits(
            decoder_replay=decoder_replay,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording,
        )
        baseline_trace_count += 1
        greedy = greedy_ctc_tokens(log_probs)

        trace_misses: list[tuple[int, float]] = []
        for keyword_id, item in keywords.items():
            target = tuple(item["tokens"])
            score = decoder_sequence_log_confidence(log_probs, target)
            confidence = (
                0.0 if not math.isfinite(score) else min(1.0, math.exp(score))
            )
            surrogate = confidence >= float(item["threshold"])
            runtime = keyword_id in baseline
            row = rows[keyword_id]
            row["traces"] += 1
            row["surrogate_above_threshold"] += int(surrogate)
            row["runtime_hit"] += int(runtime)
            if surrogate and runtime:
                row["_hit_confidences"].append(confidence)
            if not surrogate or runtime:
                continue

            row["surrogate_above_threshold_runtime_miss"] += 1
            row["_miss_confidences"].append(confidence)
            complete = contains_subsequence(greedy, target)
            row["miss_greedy_contains_target"] += int(complete)
            row["miss_greedy_incomplete_target_order"] += int(not complete)
            wrong = bool(baseline - {keyword_id})
            row["miss_wrong_keyword_hit"] += int(wrong)
            row["miss_no_runtime_hit"] += int(not baseline)
            depth = longest_prefix_depth(greedy, target)
            histogram = row["miss_prefix_depth_histogram"]
            histogram[str(depth)] = int(histogram.get(str(depth), 0)) + 1
            trace_misses.append((keyword_id, confidence))

        if not trace_misses:
            continue

        probe_trace_count += 1
        probe_hits = {
            name: runtime_hits(
                decoder_replay=decoder_replay,
                model=model,
                pack=pack,
                trace=trace,
                recording=f"{recording}-{name}",
                option=option,
            )
            for name, option in PROBES.items()
        }
        for keyword_id, _ in trace_misses:
            recovered_any = False
            row = rows[keyword_id]
            for name, hits in probe_hits.items():
                recovered = keyword_id in hits
                row["recoveries"][name] += int(recovered)
                recovered_any = recovered_any or recovered
            row["miss_unrecovered_all_probes"] += int(not recovered_any)

    aggregate = {}
    totals = {
        "traces": len(traces),
        "baseline_decoder_replays": baseline_trace_count,
        "probe_traces": probe_trace_count,
        "probe_decoder_replays": probe_trace_count * len(PROBES),
        "surrogate_above_threshold": 0,
        "runtime_hit": 0,
        "surrogate_above_threshold_runtime_miss": 0,
    }
    for keyword_id, row in sorted(rows.items()):
        miss_confidences = row.pop("_miss_confidences")
        hit_confidences = row.pop("_hit_confidences")
        miss_count = int(row["surrogate_above_threshold_runtime_miss"])
        row["miss_greedy_target_order_complete_fraction"] = (
            row["miss_greedy_contains_target"] / miss_count if miss_count else 0.0
        )
        row["miss_wrong_keyword_hit_fraction"] = (
            row["miss_wrong_keyword_hit"] / miss_count if miss_count else 0.0
        )
        row["miss_confidence"] = {
            "mean": statistics.fmean(miss_confidences) if miss_confidences else None,
            "p50": percentile(miss_confidences, 0.50),
            "p90": percentile(miss_confidences, 0.90),
        }
        row["runtime_hit_confidence"] = {
            "mean": statistics.fmean(hit_confidences) if hit_confidences else None,
            "p50": percentile(hit_confidences, 0.50),
            "p90": percentile(hit_confidences, 0.90),
        }
        aggregate[str(keyword_id)] = row
        totals["surrogate_above_threshold"] += int(row["surrogate_above_threshold"])
        totals["runtime_hit"] += int(row["runtime_hit"])
        totals["surrogate_above_threshold_runtime_miss"] += miss_count

    return aggregate, totals


def verify_baseline(
    *,
    baseline: dict,
    model_sha256: str,
    per_keyword: dict[str, dict],
    totals: dict,
) -> None:
    if baseline.get("evidence_class") != "sequence-margin-runtime-gap-development-v1":
        raise ValueError("baseline summary evidence_class mismatch")
    if baseline.get("model_sha256") != model_sha256:
        raise ValueError("baseline summary model SHA mismatch")
    expected = baseline.get("per_keyword")
    if not isinstance(expected, dict):
        raise ValueError("baseline summary per_keyword is missing")
    for keyword_id, row in per_keyword.items():
        other = expected.get(keyword_id)
        if not isinstance(other, dict):
            raise ValueError(f"baseline summary missing keyword {keyword_id}")
        for key in (
            "traces",
            "surrogate_above_threshold",
            "runtime_hit",
            "surrogate_above_threshold_runtime_miss",
        ):
            if int(row[key]) != int(other.get(key, -1)):
                raise ValueError(
                    f"baseline summary mismatch for keyword {keyword_id}.{key}"
                )
    expected_totals = baseline.get("totals")
    if not isinstance(expected_totals, dict):
        raise ValueError("baseline summary totals are missing")
    for key in (
        "surrogate_above_threshold",
        "runtime_hit",
        "surrogate_above_threshold_runtime_miss",
    ):
        if int(totals[key]) != int(expected_totals.get(key, -1)):
            raise ValueError(f"baseline summary mismatch for totals.{key}")


def self_test() -> None:
    rows = log_softmax(
        [
            [0.0, 5.0, -5.0],
            [5.0, -5.0, -5.0],
            [0.0, -5.0, 5.0],
        ]
    )
    assert greedy_ctc_tokens(rows) == (1, 2)
    assert contains_subsequence((4, 1, 3, 2), (1, 2))
    assert not contains_subsequence((1, 3), (1, 2))
    assert longest_prefix_depth((4, 1, 3, 2), (1, 2)) == 2
    assert longest_prefix_depth((4, 1, 3), (1, 2)) == 1
    assert percentile([0.1, 0.2, 0.3], 0.5) == 0.2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--pack", type=pathlib.Path)
    parser.add_argument("--decoder-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--baseline-summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("sequence-margin/runtime-gap decomposition self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "pack": args.pack,
        "decoder-replay": args.decoder_replay,
        "posterior-cache": args.posterior_cache,
        "baseline-summary": args.baseline_summary,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    model = args.model.resolve()
    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    pack = args.pack.resolve()
    decoder_replay = args.decoder_replay.resolve()
    posterior_cache = args.posterior_cache.resolve()
    baseline_summary = args.baseline_summary.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (pack, "keyword pack"),
        (decoder_replay, "decoder replay"),
        (baseline_summary, "baseline summary"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError(f"posterior cache is missing: {posterior_cache}")

    model_sha256 = sha256_file(model)
    token_map = load_tokens(tokens)
    keywords = load_keywords(keywords_tsv, token_map)
    traces = trace_paths(posterior_cache, model_sha256)
    per_keyword, totals = summarize(
        traces=traces,
        model=model,
        pack=pack,
        keywords=keywords,
        decoder_replay=decoder_replay,
    )
    baseline = load_object(baseline_summary, "baseline summary")
    verify_baseline(
        baseline=baseline,
        model_sha256=model_sha256,
        per_keyword=per_keyword,
        totals=totals,
    )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "keyword_pack_sha256": sha256_file(pack),
        "decoder_replay_sha256": sha256_file(decoder_replay),
        "baseline_summary_sha256": sha256_file(baseline_summary),
        "probe_semantics": {
            "overlapping_recovery_signals": True,
            "thresholds_not_recalibrated": True,
            "posterior_fixed": True,
            "training_changed": False,
            "probes": {
                name: {"argv": list(option)}
                for name, option in PROBES.items()
            },
        },
        "per_keyword": per_keyword,
        "totals": totals,
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
                "traces": totals["traces"],
                "misses": totals["surrogate_above_threshold_runtime_miss"],
                "probe_traces": totals["probe_traces"],
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
