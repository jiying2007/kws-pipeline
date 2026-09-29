#!/usr/bin/env python3
"""Compare exact selected decoder-path provenance on retained positive samples."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics

from diagnose_decoder_search_path_decomposition import (
    classify,
    load_positive_sample,
    replay_summary,
)
from diagnose_sequence_margin_runtime_gap import (
    decoder_sequence_log_confidence,
    load_keywords,
    load_tokens,
    log_softmax,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "decoder-selected-path-provenance-development-v4"
POLICY = "exact-shadow-backpointer-temporal-neighborhood-v4"


def empty_bucket() -> dict:
    return {
        "recordings": 0,
        "any_non_top": 0,
        "any_fuzzy": 0,
        "root_ambiguous": 0,
        "fuzzy_advances": [],
        "non_top_advances": [],
        "blank_retentions": [],
        "same_token_retentions": [],
        "fuzzy_logit_gap_mean": [],
        "fuzzy_logit_gap_max": [],
        "fuzzy_target_rank_mean": [],
        "fuzzy_target_rank_max": [],
        "fuzzy_events": [],
        "retention_log": [],
        "confidence": [],
    }


def add_snapshot(bucket: dict, snapshot: dict) -> None:
    provenance = snapshot.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("selected path snapshot lacks provenance")
    fuzzy = int(provenance.get("fuzzy_advances", -1))
    root_ambiguous = int(provenance.get("root_ambiguous_starts", -1))
    blank = int(provenance.get("blank_retentions", -1))
    same = int(provenance.get("same_token_retentions", -1))
    token_advances = int(provenance.get("token_advances", -1))
    exact_top = int(provenance.get("exact_top_advances", -1))
    root_exact = int(provenance.get("root_exact_starts", -1))
    fuzzy_gap_sum = float(provenance.get("fuzzy_logit_gap_sum", float("nan")))
    fuzzy_gap_max = float(provenance.get("fuzzy_logit_gap_max", float("nan")))
    fuzzy_rank_sum = int(provenance.get("fuzzy_target_rank_sum", -1))
    fuzzy_rank_max = int(provenance.get("fuzzy_target_rank_max", -1))
    fuzzy_events = provenance.get("fuzzy_events")
    if not isinstance(fuzzy_events, list) or len(fuzzy_events) != fuzzy:
        raise ValueError("selected path fuzzy event list does not match count")
    normalized_events: list[dict] = []
    for event in fuzzy_events:
        if not isinstance(event, dict) or set(event) != {
            "frame_index", "depth", "target_token", "top_token",
            "logit_gap", "target_rank"
        }:
            raise ValueError("selected path fuzzy event schema is invalid")
        frame_index = int(event["frame_index"])
        depth = int(event["depth"])
        target_token = int(event["target_token"])
        top_token = int(event["top_token"])
        gap = float(event["logit_gap"])
        rank = int(event["target_rank"])
        if (
            frame_index < 0
            or depth <= 1
            or target_token <= 0
            or top_token == target_token
            or top_token < 0
            or not math.isfinite(gap)
            or gap < 0.0
            or rank < 2
        ):
            raise ValueError("selected path fuzzy event value is invalid")
        normalized_events.append({
            "frame_index": frame_index,
            "depth": depth,
            "target_token": target_token,
            "top_token": top_token,
            "logit_gap": gap,
            "target_rank": rank,
        })
    values = (fuzzy, root_ambiguous, blank, same, token_advances, exact_top, root_exact)
    if any(value < 0 for value in values):
        raise ValueError("selected path provenance contains invalid counters")
    if (
        not math.isfinite(fuzzy_gap_sum)
        or not math.isfinite(fuzzy_gap_max)
        or fuzzy_rank_sum < 0
        or fuzzy_rank_max < 0
    ):
        raise ValueError("selected path fuzzy provenance is invalid")
    if fuzzy == 0:
        if (
            abs(fuzzy_gap_sum) > 1.0e-9
            or abs(fuzzy_gap_max) > 1.0e-9
            or fuzzy_rank_sum != 0
            or fuzzy_rank_max != 0
        ):
            raise ValueError("zero-fuzzy path carries fuzzy provenance")
    elif (
        fuzzy_gap_sum < 0.0
        or fuzzy_gap_max < 0.0
        or fuzzy_rank_sum < fuzzy
        or fuzzy_rank_max < 1
    ):
        raise ValueError("fuzzy path carries inconsistent gap/rank provenance")
    if root_exact + root_ambiguous != 1:
        raise ValueError("selected path must have exactly one root start")
    if token_advances <= 0:
        raise ValueError("selected path token advance count must be positive")
    non_top = fuzzy + root_ambiguous
    retention = float(snapshot["retention_log"])
    confidence = float(snapshot["confidence"])
    if not math.isfinite(retention) or not math.isfinite(confidence):
        raise ValueError("selected path snapshot contains non-finite values")

    bucket["recordings"] += 1
    bucket["any_non_top"] += int(non_top > 0)
    bucket["any_fuzzy"] += int(fuzzy > 0)
    bucket["root_ambiguous"] += int(root_ambiguous > 0)
    bucket["fuzzy_advances"].append(fuzzy)
    bucket["non_top_advances"].append(non_top)
    bucket["blank_retentions"].append(blank)
    bucket["same_token_retentions"].append(same)
    if fuzzy > 0:
        bucket["fuzzy_logit_gap_mean"].append(fuzzy_gap_sum / float(fuzzy))
        bucket["fuzzy_logit_gap_max"].append(fuzzy_gap_max)
        bucket["fuzzy_target_rank_mean"].append(fuzzy_rank_sum / float(fuzzy))
        bucket["fuzzy_target_rank_max"].append(fuzzy_rank_max)
        bucket["fuzzy_events"].extend(normalized_events)
    bucket["retention_log"].append(retention)
    bucket["confidence"].append(confidence)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def histogram(values: list[int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: int(item[0])))


def summarize_fuzzy_events(events: list[dict]) -> dict:
    def stats(rows: list[dict]) -> dict:
        gaps = [float(row["logit_gap"]) for row in rows]
        ranks = [float(row["target_rank"]) for row in rows]
        return {
            "events": len(rows),
            "logit_gap": {
                "mean": statistics.fmean(gaps) if gaps else None,
                "p50": percentile(gaps, 0.50),
                "p90": percentile(gaps, 0.90),
            },
            "target_rank": {
                "mean": statistics.fmean(ranks) if ranks else None,
                "p50": percentile(ranks, 0.50),
                "p90": percentile(ranks, 0.90),
            },
        }

    grouped: dict[str, dict[str, list[dict]]] = {
        "by_depth": {},
        "by_target_token": {},
        "by_top_token": {},
        "by_pair": {},
    }
    for event in events:
        keys = {
            "by_depth": str(int(event["depth"])),
            "by_target_token": str(int(event["target_token"])),
            "by_top_token": str(int(event["top_token"])),
            "by_pair": (
                f'{int(event["target_token"])}->{int(event["top_token"])}'
            ),
        }
        for group, key in keys.items():
            grouped[group].setdefault(key, []).append(event)

    result = {"events": len(events)}
    for group, rows in grouped.items():
        result[group] = {
            key: stats(value)
            for key, value in sorted(rows.items())
        }
    return result



TEMPORAL_RADII_FRAMES = (1, 2, 3, 5)


def dominant_token(row: list[float]) -> int:
    if not row:
        raise ValueError("posterior row may not be empty")
    return max(range(len(row)), key=row.__getitem__)


def target_margin(row: list[float], target: int) -> float:
    if target < 0 or target >= len(row):
        raise ValueError("target token is outside posterior vocabulary")
    competitor = max(
        value for index, value in enumerate(row)
        if index != target
    )
    return float(row[target] - competitor)


def temporal_neighborhood_event(
    logits: list[list[float]],
    event: dict,
) -> dict:
    frame = int(event["frame_index"])
    target = int(event["target_token"])
    if frame < 0 or frame >= len(logits):
        raise ValueError("fuzzy event frame is outside retained posterior trace")
    result = {
        "frame_index": frame,
        "depth": int(event["depth"]),
        "target_token": target,
        "top_token": int(event["top_token"]),
        "logit_gap": float(event["logit_gap"]),
        "target_rank": int(event["target_rank"]),
        "top1_within": {},
        "best_target_margin": {},
    }
    for radius in TEMPORAL_RADII_FRAMES:
        start = max(0, frame - radius)
        stop = min(len(logits), frame + radius + 1)
        rows = logits[start:stop]
        result["top1_within"][str(radius)] = any(
            dominant_token(row) == target for row in rows
        )
        result["best_target_margin"][str(radius)] = max(
            target_margin(row, target) for row in rows
        )
    recovered = [
        radius
        for radius in TEMPORAL_RADII_FRAMES
        if result["top1_within"][str(radius)]
    ]
    result["nearest_top1_radius_frames"] = min(recovered) if recovered else None
    return result


def summarize_temporal_events(events: list[dict]) -> dict:
    def stats(rows: list[dict]) -> dict:
        count = len(rows)
        recovered = {}
        margins = {}
        for radius in TEMPORAL_RADII_FRAMES:
            key = str(radius)
            hits = sum(bool(row["top1_within"][key]) for row in rows)
            recovered[key] = {
                "events": hits,
                "fraction": float(hits) / count if count else None,
            }
            values = [float(row["best_target_margin"][key]) for row in rows]
            margins[key] = {
                "mean": statistics.fmean(values) if values else None,
                "p50": percentile(values, 0.50),
                "p90": percentile(values, 0.90),
            }
        nearest = [
            int(row["nearest_top1_radius_frames"])
            for row in rows
            if row["nearest_top1_radius_frames"] is not None
        ]
        return {
            "events": count,
            "top1_recovered_within_frames": recovered,
            "best_target_margin_by_radius": margins,
            "nearest_top1_radius_histogram": histogram(nearest),
        }

    grouped: dict[str, dict[str, list[dict]]] = {
        "by_keyword": {},
        "by_depth": {},
        "by_pair": {},
        "by_category": {},
    }
    for event in events:
        keys = {
            "by_keyword": str(int(event["keyword_id"])),
            "by_depth": str(int(event["depth"])),
            "by_pair": (
                f'{int(event["target_token"])}->{int(event["top_token"])}'
            ),
            "by_category": str(event["miss_category"]),
        }
        for group, key in keys.items():
            grouped[group].setdefault(key, []).append(event)

    result = {"overall": stats(events)}
    for group, rows in grouped.items():
        result[group] = {
            key: stats(value)
            for key, value in sorted(rows.items())
        }
    return result


def finish_bucket(bucket: dict) -> dict:
    count = int(bucket["recordings"])
    result = {
        "recordings": count,
        "any_non_top": int(bucket["any_non_top"]),
        "any_fuzzy": int(bucket["any_fuzzy"]),
        "root_ambiguous": int(bucket["root_ambiguous"]),
        "any_non_top_fraction": (
            float(bucket["any_non_top"]) / count if count else None
        ),
        "any_fuzzy_fraction": (
            float(bucket["any_fuzzy"]) / count if count else None
        ),
        "root_ambiguous_fraction": (
            float(bucket["root_ambiguous"]) / count if count else None
        ),
        "fuzzy_advances_histogram": histogram(bucket["fuzzy_advances"]),
        "non_top_advances_histogram": histogram(bucket["non_top_advances"]),
        "blank_retentions_histogram": histogram(bucket["blank_retentions"]),
        "same_token_retentions_histogram": histogram(
            bucket["same_token_retentions"]
        ),
        "fuzzy_competition": summarize_fuzzy_events(bucket["fuzzy_events"]),
    }
    for key in (
        "retention_log",
        "confidence",
        "fuzzy_logit_gap_mean",
        "fuzzy_logit_gap_max",
        "fuzzy_target_rank_mean",
        "fuzzy_target_rank_max",
    ):
        values = [float(value) for value in bucket[key]]
        result[key] = {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
        }
    return result


def representative_snapshot(path: dict, *, runtime: bool, category: str | None) -> tuple[str, dict]:
    selected = path.get("selected_path")
    if not isinstance(selected, dict):
        raise ValueError("path replay lacks selected_path evidence")

    if runtime:
        value = selected.get("best_eligible")
        if not isinstance(value, dict):
            raise ValueError("runtime hit lacks eligible selected path")
        return "runtime_best_eligible", value

    order: tuple[tuple[str, str], ...]
    if category == "terminal_retention_gate":
        order = (
            ("miss_best_retention", "best_retention"),
            ("miss_best_confidence_retention_pass", "best_confidence_retention_pass"),
            ("miss_best_eligible", "best_eligible"),
        )
    elif category == "terminal_confidence_below_threshold":
        order = (
            ("miss_best_confidence_retention_pass", "best_confidence_retention_pass"),
            ("miss_best_retention", "best_retention"),
            ("miss_best_eligible", "best_eligible"),
        )
    elif category == "wrong_keyword_competition":
        order = (
            ("miss_best_eligible", "best_eligible"),
            ("miss_best_confidence_retention_pass", "best_confidence_retention_pass"),
            ("miss_best_retention", "best_retention"),
        )
    else:
        order = (
            ("miss_best_eligible", "best_eligible"),
            ("miss_best_confidence_retention_pass", "best_confidence_retention_pass"),
            ("miss_best_retention", "best_retention"),
        )
    for label, key in order:
        value = selected.get(key)
        if isinstance(value, dict):
            return label, value
    raise ValueError(f"miss category {category!r} has no selected terminal path")


def summarize(
    *,
    traces: list[pathlib.Path],
    model: pathlib.Path,
    pack: pathlib.Path,
    keywords: dict[int, dict],
    path_replay: pathlib.Path,
    positive_sample: dict[str, dict],
) -> dict:
    by_sha = {path.stem: path for path in traces}
    missing = sorted(set(positive_sample) - set(by_sha))
    if missing:
        raise ValueError(
            "positive sample is missing retained posterior traces: "
            + ",".join(missing[:8])
        )

    buckets: dict[str, dict] = {
        "runtime_hit": empty_bucket(),
        "surrogate_runtime_hit": empty_bucket(),
        "surrogate_runtime_miss": empty_bucket(),
        "terminal_retention_gate": empty_bucket(),
        "terminal_confidence_below_threshold": empty_bucket(),
        "wrong_keyword_competition": empty_bucket(),
    }
    by_keyword: dict[str, dict[str, dict]] = {
        str(keyword_id): {
            name: empty_bucket()
            for name in (
                "runtime_hit",
                "surrogate_runtime_hit",
                "surrogate_runtime_miss",
                "terminal_retention_gate",
                "terminal_confidence_below_threshold",
                "wrong_keyword_competition",
            )
        }
        for keyword_id in sorted(keywords)
    }
    records: list[dict] = []
    temporal_events: list[dict] = []
    excluded_non_surrogate_runtime_miss = 0

    for audio_sha, sample in sorted(positive_sample.items()):
        trace = by_sha[audio_sha]
        keyword_id = int(sample["keyword_id"])
        keyword = keywords.get(keyword_id)
        if not isinstance(keyword, dict):
            raise ValueError("positive sample references unknown keyword")
        logits = read_trace_logits(trace)
        log_probs = log_softmax(logits)
        score = decoder_sequence_log_confidence(log_probs, tuple(keyword["tokens"]))
        confidence = 0.0 if not math.isfinite(score) else min(1.0, math.exp(score))
        surrogate = confidence >= float(keyword["threshold"])

        replay = replay_summary(
            tool=path_replay,
            model=model,
            pack=pack,
            trace=trace,
            recording=f"positive-{audio_sha[:16]}",
        )
        by_id = {
            int(item["keyword_id"]): item
            for item in replay["keywords"]
            if isinstance(item, dict) and "keyword_id" in item
        }
        if set(by_id) != set(keywords):
            raise ValueError("path replay keyword set mismatch")
        detected_ids = {
            key
            for key, row in by_id.items()
            if int(row.get("detections", 0)) > 0
        }
        runtime = keyword_id in detected_ids
        if surrogate != bool(sample["surrogate_above_runtime_threshold"]):
            raise ValueError(f"surrogate flag drifted for positive {audio_sha}")
        if runtime != bool(sample["runtime_detected_expected"]):
            raise ValueError(f"runtime flag drifted for positive {audio_sha}")

        target_path = by_id[keyword_id]
        category: str | None = None
        if surrogate and not runtime:
            category = classify(
                target_path,
                other_detected=bool(detected_ids - {keyword_id}),
            )

        if not runtime and not surrogate:
            excluded_non_surrogate_runtime_miss += 1
            records.append(
                {
                    "audio_sha256": audio_sha,
                    "split": sample["split"],
                    "keyword_id": keyword_id,
                    "surrogate_above_threshold": False,
                    "runtime_hit": False,
                    "miss_category": None,
                    "representative_policy": None,
                    "representative": None,
                    "selected_path_cohort": False,
                    "exclusion_reason": "surrogate_below_threshold_runtime_miss",
                }
            )
            continue

        label, snapshot = representative_snapshot(
            target_path,
            runtime=runtime,
            category=category,
        )

        if runtime:
            add_snapshot(buckets["runtime_hit"], snapshot)
            add_snapshot(by_keyword[str(keyword_id)]["runtime_hit"], snapshot)
            if surrogate:
                add_snapshot(buckets["surrogate_runtime_hit"], snapshot)
                add_snapshot(
                    by_keyword[str(keyword_id)]["surrogate_runtime_hit"],
                    snapshot,
                )
        if surrogate and not runtime:
            add_snapshot(buckets["surrogate_runtime_miss"], snapshot)
            add_snapshot(
                by_keyword[str(keyword_id)]["surrogate_runtime_miss"],
                snapshot,
            )
            if category in (
                "terminal_retention_gate",
                "terminal_confidence_below_threshold",
                "wrong_keyword_competition",
            ):
                add_snapshot(buckets[category], snapshot)
                add_snapshot(by_keyword[str(keyword_id)][category], snapshot)

        provenance = snapshot["provenance"]
        for event in provenance["fuzzy_events"]:
            temporal_events.append(
                {
                    "split": sample["split"],
                    "keyword_id": keyword_id,
                    "miss_category": category,
                    **temporal_neighborhood_event(logits, event),
                }
            )
        records.append(
            {
                "audio_sha256": audio_sha,
                "split": sample["split"],
                "keyword_id": keyword_id,
                "surrogate_above_threshold": surrogate,
                "runtime_hit": runtime,
                "miss_category": category,
                "representative_policy": label,
                "selected_path_cohort": True,
                "exclusion_reason": None,
                "representative": {
                    "retention_log": float(snapshot["retention_log"]),
                    "confidence": float(snapshot["confidence"]),
                    "provenance": {
                        "token_advances": int(provenance["token_advances"]),
                        "exact_top_advances": int(provenance["exact_top_advances"]),
                        "fuzzy_advances": int(provenance["fuzzy_advances"]),
                        "fuzzy_logit_gap_sum": float(
                            provenance["fuzzy_logit_gap_sum"]
                        ),
                        "fuzzy_logit_gap_max": float(
                            provenance["fuzzy_logit_gap_max"]
                        ),
                        "fuzzy_target_rank_sum": int(
                            provenance["fuzzy_target_rank_sum"]
                        ),
                        "fuzzy_target_rank_max": int(
                            provenance["fuzzy_target_rank_max"]
                        ),
                        "fuzzy_events": [
                            {
                                "frame_index": int(event["frame_index"]),
                                "depth": int(event["depth"]),
                                "target_token": int(event["target_token"]),
                                "top_token": int(event["top_token"]),
                                "logit_gap": float(event["logit_gap"]),
                                "target_rank": int(event["target_rank"]),
                            }
                            for event in provenance["fuzzy_events"]
                        ],
                        "root_exact_starts": int(provenance["root_exact_starts"]),
                        "root_ambiguous_starts": int(
                            provenance["root_ambiguous_starts"]
                        ),
                        "same_token_retentions": int(
                            provenance["same_token_retentions"]
                        ),
                        "blank_retentions": int(provenance["blank_retentions"]),
                    },
                },
            }
        )

    return {
        "recordings": len(records),
        "selected_path_cohort_recordings": (
            len(records) - excluded_non_surrogate_runtime_miss
        ),
        "excluded_non_surrogate_runtime_miss": (
            excluded_non_surrogate_runtime_miss
        ),
        "cohorts": {name: finish_bucket(value) for name, value in buckets.items()},
        "by_keyword": {
            keyword_id: {
                name: finish_bucket(value)
                for name, value in groups.items()
            }
            for keyword_id, groups in by_keyword.items()
        },
        "records": records,
        "fuzzy_temporal_neighborhood": summarize_temporal_events(
            temporal_events
        ),
    }


def self_test() -> None:
    hit = {
        "selected_path": {
            "best_eligible": {
                "retention_log": -1.0,
                "confidence": 0.9,
                "provenance": {
                    "token_advances": 4,
                    "exact_top_advances": 3,
                    "fuzzy_advances": 0,
                    "fuzzy_logit_gap_sum": 0.0,
                    "fuzzy_logit_gap_max": 0.0,
                    "fuzzy_target_rank_sum": 0,
                    "fuzzy_target_rank_max": 0,
                    "fuzzy_events": [],
                    "root_exact_starts": 1,
                    "root_ambiguous_starts": 0,
                    "same_token_retentions": 2,
                    "blank_retentions": 1,
                },
            }
        }
    }
    label, snapshot = representative_snapshot(hit, runtime=True, category=None)
    assert label == "runtime_best_eligible"
    bucket = empty_bucket()
    add_snapshot(bucket, snapshot)
    finished = finish_bucket(bucket)
    assert finished["recordings"] == 1
    assert finished["any_fuzzy"] == 0
    assert finished["any_non_top"] == 0

    miss = {
        "selected_path": {
            "best_retention": {
                "retention_log": -17.0,
                "confidence": 0.8,
                "provenance": {
                    "token_advances": 4,
                    "exact_top_advances": 2,
                    "fuzzy_advances": 1,
                    "fuzzy_logit_gap_sum": 0.4,
                    "fuzzy_logit_gap_max": 0.4,
                    "fuzzy_target_rank_sum": 2,
                    "fuzzy_target_rank_max": 2,
                    "fuzzy_events": [
                        {
                            "frame_index": 1,
                            "depth": 3,
                            "target_token": 3,
                            "top_token": 0,
                            "logit_gap": 0.4,
                            "target_rank": 2,
                        }
                    ],
                    "root_exact_starts": 1,
                    "root_ambiguous_starts": 0,
                    "same_token_retentions": 0,
                    "blank_retentions": 1,
                },
            },
            "best_confidence_retention_pass": None,
            "best_eligible": None,
        }
    }
    label, snapshot = representative_snapshot(
        miss,
        runtime=False,
        category="terminal_retention_gate",
    )
    assert label == "miss_best_retention"
    bucket = empty_bucket()
    add_snapshot(bucket, snapshot)
    finished = finish_bucket(bucket)
    assert finished["any_fuzzy"] == 1
    assert finished["fuzzy_advances_histogram"] == {"1": 1}
    assert finished["fuzzy_logit_gap_mean"]["mean"] == 0.4
    assert finished["fuzzy_target_rank_mean"]["mean"] == 2.0
    assert finished["fuzzy_competition"]["events"] == 1
    assert finished["fuzzy_competition"]["by_pair"]["3->0"]["events"] == 1
    temporal = temporal_neighborhood_event(
        [
            [3.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        {
            "frame_index": 1,
            "depth": 3,
            "target_token": 3,
            "top_token": 0,
            "logit_gap": 1.0,
            "target_rank": 2,
        },
    )
    assert temporal["top1_within"]["1"] is True
    assert temporal["nearest_top1_radius_frames"] == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--pack", type=pathlib.Path)
    parser.add_argument("--decoder-path-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("decoder selected-path provenance self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "pack": args.pack,
        "decoder-path-replay": args.decoder_path_replay,
        "posterior-cache": args.posterior_cache,
        "acoustic-alignment": args.acoustic_alignment,
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
    acoustic_alignment = args.acoustic_alignment.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (pack, "keyword pack"),
        (path_replay, "decoder path replay"),
        (acoustic_alignment, "acoustic alignment"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError(f"posterior cache is missing: {posterior_cache}")

    model_sha256 = sha256_file(model)
    token_map = load_tokens(tokens)
    keywords = load_keywords(keywords_tsv, token_map)
    traces = trace_paths(posterior_cache, model_sha256)
    positive_sample = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    positive = summarize(
        traces=traces,
        model=model,
        pack=pack,
        keywords=keywords,
        path_replay=path_replay,
        positive_sample=positive_sample,
    )

    result = {
        "schema_version": 4,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "decoder_math_changed": False,
        "shadow_runtime_state_verified": True,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "keyword_pack_sha256": sha256_file(pack),
        "decoder_path_replay_sha256": sha256_file(path_replay),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "positive_sample": positive,
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
                "recordings": positive["recordings"],
                "selected_path_cohort_recordings": positive[
                    "selected_path_cohort_recordings"
                ],
                "excluded_non_surrogate_runtime_miss": positive[
                    "excluded_non_surrogate_runtime_miss"
                ],
                "cohorts": {
                    name: {
                        "recordings": row["recordings"],
                        "any_fuzzy": row["any_fuzzy"],
                        "any_non_top": row["any_non_top"],
                    }
                    for name, row in positive["cohorts"].items()
                },
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
