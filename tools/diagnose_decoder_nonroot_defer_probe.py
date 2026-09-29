#!/usr/bin/env python3
"""Counterfactual retained-posterior probe for deferred non-root mismatch admission."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess

from diagnose_decoder_search_path_decomposition import load_positive_sample
from diagnose_sequence_margin_runtime_gap import (
    decoder_sequence_log_confidence,
    load_keywords,
    load_tokens,
    log_softmax,
    read_trace_logits,
    sha256_file,
    trace_paths,
)

EVIDENCE_CLASS = "decoder-nonroot-defer-counterfactual-development-v1"


def replay_events(
    tool: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    trace: pathlib.Path,
    recording: str,
    *,
    defer: bool,
) -> list[dict]:
    command = [str(tool), str(model), str(pack), str(trace), recording]
    if defer:
        command += ["--require-nonroot-defer", "1"]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    events: list[dict] = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("recording") != recording:
            raise ValueError("decoder replay emitted invalid event")
        events.append(
            {
                "keyword_id": int(value["keyword_id"]),
                "time_s": float(value["time_s"]),
                "confidence": float(value["confidence"]),
            }
        )
    return events


def empty_positive_bucket() -> dict:
    return {
        "recordings": 0,
        "baseline_target_hits": 0,
        "defer_target_hits": 0,
        "target_gains": 0,
        "target_losses": 0,
        "baseline_wrong_keyword_hits": 0,
        "defer_wrong_keyword_hits": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords-tsv", required=True, type=pathlib.Path)
    parser.add_argument("--pack", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-decoder-replay", required=True, type=pathlib.Path)
    parser.add_argument("--probe-decoder-replay", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    model = args.model.resolve()
    tokens = args.tokens.resolve()
    keywords_tsv = args.keywords_tsv.resolve()
    pack = args.pack.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    baseline_replay = args.baseline_decoder_replay.resolve()
    probe_replay = args.probe_decoder_replay.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (pack, "keyword pack"),
        (acoustic_alignment, "acoustic alignment"),
        (baseline_replay, "baseline decoder replay"),
        (probe_replay, "probe decoder replay"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError("posterior cache is missing")

    model_sha = sha256_file(model)
    token_map = load_tokens(tokens)
    keywords = load_keywords(keywords_tsv, token_map)
    positive_sample = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha,
    )
    traces = trace_paths(posterior_cache, model_sha)
    positive_seen: set[str] = set()

    positive = empty_positive_bucket()
    positive["by_split_keyword"] = {
        f"{split}:kw{keyword_id}": empty_positive_bucket()
        for split in ("calibration", "test")
        for keyword_id in sorted(keywords)
    }
    runtime_gap = {
        str(keyword_id): {
            "surrogate_above_baseline_miss": 0,
            "defer_recovered": 0,
            "defer_still_miss": 0,
        }
        for keyword_id in sorted(keywords)
    }
    totals = {
        "traces": len(traces),
        "baseline_events": 0,
        "defer_events": 0,
        "traces_changed": 0,
        "traces_with_defer_only_keyword": 0,
        "traces_with_baseline_only_keyword": 0,
        "defer_only_keyword_memberships": 0,
        "baseline_only_keyword_memberships": 0,
    }

    for index, trace in enumerate(traces):
        recording = f"counterfactual-{index:06d}"
        baseline = replay_events(
            baseline_replay, model, pack, trace, recording, defer=False
        )
        probe_default = replay_events(
            probe_replay, model, pack, trace, recording, defer=False
        )
        if baseline != probe_default:
            raise ValueError(
                f"probe-off replay drifted from exact retained baseline: {trace}"
            )
        defer = replay_events(
            probe_replay, model, pack, trace, recording, defer=True
        )

        baseline_ids = {int(row["keyword_id"]) for row in baseline}
        defer_ids = {int(row["keyword_id"]) for row in defer}
        defer_only = defer_ids - baseline_ids
        baseline_only = baseline_ids - defer_ids

        totals["baseline_events"] += len(baseline)
        totals["defer_events"] += len(defer)
        totals["traces_changed"] += int(baseline != defer)
        totals["traces_with_defer_only_keyword"] += int(bool(defer_only))
        totals["traces_with_baseline_only_keyword"] += int(bool(baseline_only))
        totals["defer_only_keyword_memberships"] += len(defer_only)
        totals["baseline_only_keyword_memberships"] += len(baseline_only)

        logits = read_trace_logits(trace)
        log_probs = log_softmax(logits)
        for keyword_id, item in keywords.items():
            target = tuple(item["tokens"])
            score = decoder_sequence_log_confidence(log_probs, target)
            confidence = (
                0.0 if not math.isfinite(score) else min(1.0, math.exp(score))
            )
            surrogate = confidence >= float(item["threshold"])
            baseline_hit = keyword_id in baseline_ids
            defer_hit = keyword_id in defer_ids
            if surrogate and not baseline_hit:
                row = runtime_gap[str(keyword_id)]
                row["surrogate_above_baseline_miss"] += 1
                row["defer_recovered"] += int(defer_hit)
                row["defer_still_miss"] += int(not defer_hit)

        sample = positive_sample.get(trace.stem)
        if sample is None:
            continue
        positive_seen.add(trace.stem)
        keyword_id = int(sample["keyword_id"])
        baseline_target = keyword_id in baseline_ids
        defer_target = keyword_id in defer_ids
        if baseline_target != bool(sample["runtime_detected_expected"]):
            raise ValueError(
                f"retained positive runtime flag drifted: {trace.stem}"
            )
        buckets = [
            positive,
            positive["by_split_keyword"][
                f"{sample['split']}:kw{keyword_id}"
            ],
        ]
        for bucket in buckets:
            bucket["recordings"] += 1
            bucket["baseline_target_hits"] += int(baseline_target)
            bucket["defer_target_hits"] += int(defer_target)
            bucket["target_gains"] += int(defer_target and not baseline_target)
            bucket["target_losses"] += int(baseline_target and not defer_target)
            bucket["baseline_wrong_keyword_hits"] += int(
                bool(baseline_ids - {keyword_id})
            )
            bucket["defer_wrong_keyword_hits"] += int(
                bool(defer_ids - {keyword_id})
            )

    missing = sorted(set(positive_sample) - positive_seen)
    if missing:
        raise ValueError(
            "positive retained sample missing posterior trace: "
            + ",".join(missing[:8])
        )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "thresholds_changed": False,
        "root_admission_changed": False,
        "counterfactual": "top1-child-with-parent-mismatch-defer-v1",
        "probe_off_matches_exact_retained_baseline": True,
        "model_sha256": model_sha,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "keyword_pack_sha256": sha256_file(pack),
        "positive": positive,
        "runtime_gap": runtime_gap,
        "totals": totals,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "positive": positive,
        "runtime_gap": runtime_gap,
        "totals": totals,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
