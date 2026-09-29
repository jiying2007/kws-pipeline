#!/usr/bin/env python3
"""Screen strict non-root exact-top child admission on retained posteriors."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

from diagnose_decoder_search_path_decomposition import load_positive_sample
from diagnose_sequence_margin_runtime_gap import sha256_file, trace_paths

EVIDENCE_CLASS = "nonroot-exact-top-admission-screen-development-v1"
POLICY = "fixed-posterior-source-default-equivalence-then-strict-nonroot-v1"


def replay_events(
    *,
    tool: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    trace: pathlib.Path,
    recording: str,
    strict: bool = False,
) -> list[dict]:
    command = [str(tool), str(model), str(pack), str(trace), recording]
    if strict:
        command.extend(["--nonroot-exact-top-only", "1"])
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    events: list[dict] = []
    for line_no, raw in enumerate(completed.stdout.splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if (
            not isinstance(value, dict)
            or value.get("recording") != recording
            or isinstance(value.get("keyword_id"), bool)
            or not isinstance(value.get("keyword_id"), int)
        ):
            raise ValueError(
                f"decoder replay emitted invalid event at {trace}:{line_no}"
            )
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
        "baseline_hits": 0,
        "strict_hits": 0,
        "recovered": 0,
        "lost": 0,
        "surrogate_runtime_gap_recordings": 0,
        "surrogate_runtime_gap_recovered": 0,
        "baseline_wrong_keyword_recordings": 0,
        "strict_wrong_keyword_recordings": 0,
    }


def add_positive(
    bucket: dict,
    *,
    keyword_id: int,
    sample: dict,
    baseline_ids: set[int],
    strict_ids: set[int],
) -> None:
    baseline_hit = keyword_id in baseline_ids
    strict_hit = keyword_id in strict_ids
    expected_runtime = bool(sample["runtime_detected_expected"])
    if baseline_hit != expected_runtime:
        raise ValueError("retained positive runtime flag drifted from exact source replay")
    gap = bool(sample["surrogate_above_runtime_threshold"]) and not baseline_hit

    bucket["recordings"] += 1
    bucket["baseline_hits"] += int(baseline_hit)
    bucket["strict_hits"] += int(strict_hit)
    bucket["recovered"] += int(not baseline_hit and strict_hit)
    bucket["lost"] += int(baseline_hit and not strict_hit)
    bucket["surrogate_runtime_gap_recordings"] += int(gap)
    bucket["surrogate_runtime_gap_recovered"] += int(gap and strict_hit)
    bucket["baseline_wrong_keyword_recordings"] += int(bool(baseline_ids - {keyword_id}))
    bucket["strict_wrong_keyword_recordings"] += int(bool(strict_ids - {keyword_id}))


def summarize(
    *,
    traces: list[pathlib.Path],
    model: pathlib.Path,
    pack: pathlib.Path,
    baseline_decoder: pathlib.Path,
    probe_decoder: pathlib.Path,
    positive_sample: dict[str, dict],
) -> dict:
    positive = empty_positive_bucket()
    by_split_keyword: dict[str, dict] = {}
    positive_seen: set[str] = set()

    baseline_pairs: set[tuple[str, int]] = set()
    strict_pairs: set[tuple[str, int]] = set()
    baseline_event_count = 0
    strict_event_count = 0
    changed_traces = 0

    for trace_index, trace in enumerate(traces):
        recording = f"posterior-trace-{trace_index:06d}"
        source_events = replay_events(
            tool=baseline_decoder,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording,
        )
        current_default_events = replay_events(
            tool=probe_decoder,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording,
        )
        if current_default_events != source_events:
            raise ValueError(
                "current default replay differs from exact retained source: "
                + trace.stem
            )
        strict_events = replay_events(
            tool=probe_decoder,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording,
            strict=True,
        )

        source_ids = {int(item["keyword_id"]) for item in source_events}
        strict_ids = {int(item["keyword_id"]) for item in strict_events}
        baseline_event_count += len(source_events)
        strict_event_count += len(strict_events)
        changed_traces += int(source_events != strict_events)
        baseline_pairs.update((trace.stem, keyword_id) for keyword_id in source_ids)
        strict_pairs.update((trace.stem, keyword_id) for keyword_id in strict_ids)

        sample = positive_sample.get(trace.stem)
        if sample is None:
            continue
        positive_seen.add(trace.stem)
        keyword_id = int(sample["keyword_id"])
        key = f'{sample["split"]}:kw{keyword_id}'
        bucket = by_split_keyword.setdefault(key, empty_positive_bucket())
        add_positive(
            positive,
            keyword_id=keyword_id,
            sample=sample,
            baseline_ids=source_ids,
            strict_ids=strict_ids,
        )
        add_positive(
            bucket,
            keyword_id=keyword_id,
            sample=sample,
            baseline_ids=source_ids,
            strict_ids=strict_ids,
        )

    missing = sorted(set(positive_sample) - positive_seen)
    if missing:
        raise ValueError(
            "retained positive sample is missing posterior traces: "
            + ",".join(missing[:8])
        )

    new_pairs = strict_pairs - baseline_pairs
    removed_pairs = baseline_pairs - strict_pairs
    promising = (
        positive["surrogate_runtime_gap_recovered"] > 0
        and positive["lost"] == 0
    )
    return {
        "default_equivalence": {
            "traces": len(traces),
            "exact_source_matches_current_default": True,
        },
        "all_traces": {
            "traces": len(traces),
            "changed_traces": changed_traces,
            "baseline_detection_events": baseline_event_count,
            "strict_detection_events": strict_event_count,
            "baseline_trace_keyword_pairs": len(baseline_pairs),
            "strict_trace_keyword_pairs": len(strict_pairs),
            "new_trace_keyword_pairs": len(new_pairs),
            "removed_trace_keyword_pairs": len(removed_pairs),
            "new_pairs_by_keyword": {
                str(keyword_id): sum(1 for _, kid in new_pairs if kid == keyword_id)
                for keyword_id in sorted({kid for _, kid in new_pairs})
            },
            "removed_pairs_by_keyword": {
                str(keyword_id): sum(1 for _, kid in removed_pairs if kid == keyword_id)
                for keyword_id in sorted({kid for _, kid in removed_pairs})
            },
        },
        "positive_sample": {
            **positive,
            "by_split_keyword": dict(sorted(by_split_keyword.items())),
        },
        "screen_result": (
            "promising-requires-reference-scored-eval"
            if promising
            else "negative-close-no-shipping-change"
        ),
        "full_reference_score_required": promising,
    }


def self_test() -> None:
    bucket = empty_positive_bucket()
    sample = {
        "surrogate_above_runtime_threshold": True,
        "runtime_detected_expected": False,
    }
    add_positive(
        bucket,
        keyword_id=1,
        sample=sample,
        baseline_ids=set(),
        strict_ids={1},
    )
    assert bucket["recovered"] == 1
    assert bucket["surrogate_runtime_gap_recovered"] == 1
    assert bucket["lost"] == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--pack", type=pathlib.Path)
    parser.add_argument("--baseline-decoder", type=pathlib.Path)
    parser.add_argument("--probe-decoder", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("non-root exact-top admission probe self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "pack": args.pack,
        "baseline-decoder": args.baseline_decoder,
        "probe-decoder": args.probe_decoder,
        "posterior-cache": args.posterior_cache,
        "acoustic-alignment": args.acoustic_alignment,
        "output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))

    model = args.model.resolve()
    pack = args.pack.resolve()
    baseline_decoder = args.baseline_decoder.resolve()
    probe_decoder = args.probe_decoder.resolve()
    posterior_cache = args.posterior_cache.resolve()
    acoustic_alignment = args.acoustic_alignment.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (pack, "keyword pack"),
        (baseline_decoder, "baseline decoder"),
        (probe_decoder, "probe decoder"),
        (acoustic_alignment, "acoustic alignment"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")
    if not posterior_cache.is_dir():
        raise ValueError(f"posterior cache is missing: {posterior_cache}")

    model_sha256 = sha256_file(model)
    traces = trace_paths(posterior_cache, model_sha256)
    positive_sample = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    summary = summarize(
        traces=traces,
        model=model,
        pack=pack,
        baseline_decoder=baseline_decoder,
        probe_decoder=probe_decoder,
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
        "shipping_decoder_changed": False,
        "probe_change": "non-root child requires target token to be frame top-1",
        "root_admission_changed": False,
        "thresholds_changed": False,
        "model_sha256": model_sha256,
        "keyword_pack_sha256": sha256_file(pack),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        **summary,
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
                "screen_result": result["screen_result"],
                "all_traces": result["all_traces"],
                "positive_sample": result["positive_sample"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
