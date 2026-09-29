#!/usr/bin/env python3
"""Classify fixed-posterior runtime misses by exact decoder trie-path state."""
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

EVIDENCE_CLASS = "decoder-search-path-decomposition-development-v1"
POLICY = "exact-decoder-trie-path-observability-v1"

CATEGORIES = (
    "wrong_keyword_competition",
    "root_never_admitted",
    "prefix_dropped_before_terminal",
    "terminal_never_alive_on_speech",
    "terminal_retention_gate",
    "terminal_confidence_below_threshold",
    "terminal_pending_not_emitted",
    "terminal_eligible_no_emit",
)


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def replay_summary(
    *,
    tool: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    trace: pathlib.Path,
    recording: str,
) -> dict:
    completed = subprocess.run(
        [str(tool), str(model), str(pack), str(trace), recording],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"path replay must emit exactly one JSON row: {trace}")
    value = json.loads(lines[0])
    if not isinstance(value, dict) or value.get("recording") != recording:
        raise ValueError(f"path replay emitted invalid summary: {trace}")
    keywords = value.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        raise ValueError(f"path replay summary lacks keyword rows: {trace}")
    return value


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def classify(row: dict, *, other_detected: bool) -> str:
    if other_detected:
        return "wrong_keyword_competition"
    target_depth = int(row["target_depth"])
    max_depth = int(row["max_depth_reached"])
    if max_depth <= 0:
        return "root_never_admitted"
    if max_depth < target_depth:
        return "prefix_dropped_before_terminal"
    if int(row["terminal_speech_alive_frames"]) <= 0:
        return "terminal_never_alive_on_speech"
    if int(row["terminal_retention_pass_frames"]) <= 0:
        return "terminal_retention_gate"
    if int(row["terminal_threshold_pass_frames"]) <= 0:
        return "terminal_confidence_below_threshold"
    if int(row["pending_frames"]) > 0:
        return "terminal_pending_not_emitted"
    return "terminal_eligible_no_emit"


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


def summarize(
    *,
    traces: list[pathlib.Path],
    model: pathlib.Path,
    pack: pathlib.Path,
    keywords: dict[int, dict],
    path_replay: pathlib.Path,
) -> tuple[dict[str, dict], dict]:
    rows = {
        keyword_id: {
            "traces": 0,
            "surrogate_above_threshold": 0,
            "runtime_hit": 0,
            "surrogate_above_threshold_runtime_miss": 0,
            "categories": {name: 0 for name in CATEGORIES},
            "max_depth_histogram": {},
            "_root_alive_frames": [],
            "_terminal_confidence": [],
            "_terminal_retention_log": [],
        }
        for keyword_id in keywords
    }

    for trace_index, trace in enumerate(traces):
        logits = read_trace_logits(trace)
        log_probs = log_softmax(logits)
        recording = f"posterior-trace-{trace_index:06d}"
        replay = replay_summary(
            tool=path_replay,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording,
        )
        by_id = {
            int(item["keyword_id"]): item
            for item in replay["keywords"]
            if isinstance(item, dict) and "keyword_id" in item
        }
        if set(by_id) != set(keywords):
            raise ValueError(f"path replay keyword set mismatch: {trace}")
        detected_ids = {
            keyword_id
            for keyword_id, item in by_id.items()
            if int(item.get("detections", 0)) > 0
        }

        for keyword_id, item in keywords.items():
            target = tuple(item["tokens"])
            score = decoder_sequence_log_confidence(log_probs, target)
            confidence = (
                0.0 if not math.isfinite(score) else min(1.0, math.exp(score))
            )
            surrogate = confidence >= float(item["threshold"])
            runtime = keyword_id in detected_ids
            row = rows[keyword_id]
            row["traces"] += 1
            row["surrogate_above_threshold"] += int(surrogate)
            row["runtime_hit"] += int(runtime)
            if not surrogate or runtime:
                continue

            row["surrogate_above_threshold_runtime_miss"] += 1
            path = by_id[keyword_id]
            category = classify(
                path,
                other_detected=bool(detected_ids - {keyword_id}),
            )
            row["categories"][category] += 1
            depth = str(int(path["max_depth_reached"]))
            histogram = row["max_depth_histogram"]
            histogram[depth] = int(histogram.get(depth, 0)) + 1
            row["_root_alive_frames"].append(float(path["root_alive_frames"]))
            terminal_confidence = path.get("max_terminal_confidence")
            if isinstance(terminal_confidence, (int, float)):
                row["_terminal_confidence"].append(float(terminal_confidence))
            terminal_retention = path.get("max_terminal_retention_log")
            if isinstance(terminal_retention, (int, float)):
                row["_terminal_retention_log"].append(float(terminal_retention))

    aggregate: dict[str, dict] = {}
    totals = {
        "traces": len(traces),
        "surrogate_above_threshold": 0,
        "runtime_hit": 0,
        "surrogate_above_threshold_runtime_miss": 0,
        "categories": {name: 0 for name in CATEGORIES},
    }
    for keyword_id, row in sorted(rows.items()):
        misses = int(row["surrogate_above_threshold_runtime_miss"])
        if sum(int(value) for value in row["categories"].values()) != misses:
            raise ValueError(f"miss categories do not partition keyword {keyword_id}")
        roots = row.pop("_root_alive_frames")
        confidences = row.pop("_terminal_confidence")
        retentions = row.pop("_terminal_retention_log")
        row["miss_root_alive_frames"] = {
            "mean": statistics.fmean(roots) if roots else None,
            "p50": percentile(roots, 0.50),
            "p90": percentile(roots, 0.90),
        }
        row["miss_max_terminal_confidence"] = {
            "mean": statistics.fmean(confidences) if confidences else None,
            "p50": percentile(confidences, 0.50),
            "p90": percentile(confidences, 0.90),
        }
        row["miss_max_terminal_retention_log"] = {
            "mean": statistics.fmean(retentions) if retentions else None,
            "p50": percentile(retentions, 0.50),
            "p90": percentile(retentions, 0.90),
        }
        aggregate[str(keyword_id)] = row
        totals["surrogate_above_threshold"] += int(row["surrogate_above_threshold"])
        totals["runtime_hit"] += int(row["runtime_hit"])
        totals["surrogate_above_threshold_runtime_miss"] += misses
        for name in CATEGORIES:
            totals["categories"][name] += int(row["categories"][name])

    if sum(totals["categories"].values()) != totals["surrogate_above_threshold_runtime_miss"]:
        raise ValueError("total miss categories do not form an exact partition")
    return aggregate, totals


def self_test() -> None:
    fixtures = [
        (
            {
                "target_depth": 4,
                "max_depth_reached": 0,
                "terminal_speech_alive_frames": 0,
                "terminal_retention_pass_frames": 0,
                "terminal_threshold_pass_frames": 0,
                "pending_frames": 0,
            },
            False,
            "root_never_admitted",
        ),
        (
            {
                "target_depth": 4,
                "max_depth_reached": 2,
                "terminal_speech_alive_frames": 0,
                "terminal_retention_pass_frames": 0,
                "terminal_threshold_pass_frames": 0,
                "pending_frames": 0,
            },
            False,
            "prefix_dropped_before_terminal",
        ),
        (
            {
                "target_depth": 4,
                "max_depth_reached": 4,
                "terminal_speech_alive_frames": 2,
                "terminal_retention_pass_frames": 1,
                "terminal_threshold_pass_frames": 0,
                "pending_frames": 0,
            },
            False,
            "terminal_confidence_below_threshold",
        ),
    ]
    for row, other, expected in fixtures:
        assert classify(row, other_detected=other) == expected
    assert classify(fixtures[0][0], other_detected=True) == "wrong_keyword_competition"
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--pack", type=pathlib.Path)
    parser.add_argument("--decoder-path-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--baseline-summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("decoder search-path decomposition self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "pack": args.pack,
        "decoder-path-replay": args.decoder_path_replay,
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
    path_replay = args.decoder_path_replay.resolve()
    posterior_cache = args.posterior_cache.resolve()
    baseline_summary = args.baseline_summary.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (pack, "keyword pack"),
        (path_replay, "decoder path replay"),
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
        path_replay=path_replay,
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
        "decoder_path_replay_sha256": sha256_file(path_replay),
        "baseline_summary_sha256": sha256_file(baseline_summary),
        "posterior_fixed": True,
        "training_changed": False,
        "decoder_math_changed": False,
        "categories_are_exclusive": True,
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
                "categories": totals["categories"],
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
