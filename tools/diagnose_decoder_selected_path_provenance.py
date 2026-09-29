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

EVIDENCE_CLASS = "decoder-selected-path-provenance-development-v1"
POLICY = "exact-shadow-backpointer-positive-cohort-v1"


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
    values = (fuzzy, root_ambiguous, blank, same, token_advances, exact_top, root_exact)
    if any(value < 0 for value in values):
        raise ValueError("selected path provenance contains invalid counters")
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
    }
    for key in ("retention_log", "confidence"):
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
        records.append(
            {
                "audio_sha256": audio_sha,
                "split": sample["split"],
                "keyword_id": keyword_id,
                "surrogate_above_threshold": surrogate,
                "runtime_hit": runtime,
                "miss_category": category,
                "representative_policy": label,
                "representative": {
                    "retention_log": float(snapshot["retention_log"]),
                    "confidence": float(snapshot["confidence"]),
                    "provenance": {
                        key: int(provenance[key])
                        for key in (
                            "token_advances",
                            "exact_top_advances",
                            "fuzzy_advances",
                            "root_exact_starts",
                            "root_ambiguous_starts",
                            "same_token_retentions",
                            "blank_retentions",
                        )
                    },
                },
            }
        )

    return {
        "recordings": len(records),
        "cohorts": {name: finish_bucket(value) for name, value in buckets.items()},
        "by_keyword": {
            keyword_id: {
                name: finish_bucket(value)
                for name, value in groups.items()
            }
            for keyword_id, groups in by_keyword.items()
        },
        "records": records,
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
        "schema_version": 1,
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
