#!/usr/bin/env python3
"""Compare default vs strict fuzzy-child cost on exact retained posteriors."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
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

EVIDENCE_CLASS = "strict-fuzzy-child-cost-fixed-posterior-development-v1"
POLICY = "default-vs-min-budget-fuzzy-child-cost-v1"
TREATMENT_FUZZY_COST_LOG = -16.0


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def runtime_hits(
    *,
    decoder_replay: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    trace: pathlib.Path,
    recording: str,
    fuzzy_child_cost_log: float | None = None,
) -> set[int]:
    command = [
        str(decoder_replay),
        str(model),
        str(pack),
        str(trace),
        recording,
    ]
    if fuzzy_child_cost_log is not None:
        command.extend(
            ["--fuzzy-child-cost-log", format(fuzzy_child_cost_log, ".9g")]
        )
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


def empty_transition_bucket() -> dict:
    return {
        "traces": 0,
        "surrogate_above_threshold": 0,
        "control_hit": 0,
        "treatment_hit": 0,
        "gained_hit": 0,
        "lost_hit": 0,
        "stable_hit": 0,
        "stable_miss": 0,
        "surrogate_control_miss": 0,
        "surrogate_miss_recovered": 0,
        "surrogate_hit_lost": 0,
    }


def add_transition(
    bucket: dict,
    *,
    surrogate: bool,
    control: bool,
    treatment: bool,
) -> None:
    bucket["traces"] += 1
    bucket["surrogate_above_threshold"] += int(surrogate)
    bucket["control_hit"] += int(control)
    bucket["treatment_hit"] += int(treatment)
    bucket["gained_hit"] += int(treatment and not control)
    bucket["lost_hit"] += int(control and not treatment)
    bucket["stable_hit"] += int(control and treatment)
    bucket["stable_miss"] += int(not control and not treatment)
    bucket["surrogate_control_miss"] += int(surrogate and not control)
    bucket["surrogate_miss_recovered"] += int(
        surrogate and not control and treatment
    )
    bucket["surrogate_hit_lost"] += int(surrogate and control and not treatment)


def empty_positive_bucket() -> dict:
    return {
        "recordings": 0,
        "control_target_hit": 0,
        "treatment_target_hit": 0,
        "gained_target_hit": 0,
        "lost_target_hit": 0,
        "stable_target_hit": 0,
        "stable_target_miss": 0,
        "control_wrong_keyword": 0,
        "treatment_wrong_keyword": 0,
        "wrong_keyword_increase": 0,
        "wrong_keyword_decrease": 0,
    }


def add_positive(
    bucket: dict,
    *,
    control_target: bool,
    treatment_target: bool,
    control_wrong: bool,
    treatment_wrong: bool,
) -> None:
    bucket["recordings"] += 1
    bucket["control_target_hit"] += int(control_target)
    bucket["treatment_target_hit"] += int(treatment_target)
    bucket["gained_target_hit"] += int(treatment_target and not control_target)
    bucket["lost_target_hit"] += int(control_target and not treatment_target)
    bucket["stable_target_hit"] += int(control_target and treatment_target)
    bucket["stable_target_miss"] += int(
        not control_target and not treatment_target
    )
    bucket["control_wrong_keyword"] += int(control_wrong)
    bucket["treatment_wrong_keyword"] += int(treatment_wrong)
    bucket["wrong_keyword_increase"] += int(
        treatment_wrong and not control_wrong
    )
    bucket["wrong_keyword_decrease"] += int(
        control_wrong and not treatment_wrong
    )


def verify_baseline(
    *,
    baseline: dict,
    model_sha256: str,
    per_keyword: dict[str, dict],
) -> None:
    if baseline.get("evidence_class") != "sequence-margin-runtime-gap-development-v1":
        raise ValueError("baseline evidence_class mismatch")
    if baseline.get("model_sha256") != model_sha256:
        raise ValueError("baseline model SHA mismatch")
    expected = baseline.get("per_keyword")
    if not isinstance(expected, dict):
        raise ValueError("baseline per_keyword is missing")
    for keyword_id, row in per_keyword.items():
        other = expected.get(keyword_id)
        if not isinstance(other, dict):
            raise ValueError(f"baseline missing keyword {keyword_id}")
        expected_values = {
            "traces": int(other.get("traces", -1)),
            "surrogate_above_threshold": int(
                other.get("surrogate_above_threshold", -1)
            ),
            "control_hit": int(other.get("runtime_hit", -1)),
            "surrogate_control_miss": int(
                other.get("surrogate_above_threshold_runtime_miss", -1)
            ),
        }
        for key, expected_value in expected_values.items():
            if int(row[key]) != expected_value:
                raise ValueError(
                    f"baseline mismatch for keyword {keyword_id}.{key}: "
                    f"{row[key]} != {expected_value}"
                )


def summarize(
    *,
    traces: list[pathlib.Path],
    model: pathlib.Path,
    pack: pathlib.Path,
    keywords: dict[int, dict],
    decoder_replay: pathlib.Path,
    positive_sample: dict[str, dict],
) -> tuple[dict[str, dict], dict, dict]:
    per_keyword = {
        str(keyword_id): empty_transition_bucket()
        for keyword_id in sorted(keywords)
    }
    positive = empty_positive_bucket()
    positive["by_split_keyword"] = {
        f"{split}:kw{keyword_id}": empty_positive_bucket()
        for split in ("calibration", "test")
        for keyword_id in sorted(keywords)
    }
    positive_seen: set[str] = set()

    aggregate = {
        "traces": len(traces),
        "control_detection_pairs": 0,
        "treatment_detection_pairs": 0,
        "detection_pairs_added": 0,
        "detection_pairs_removed": 0,
        "traces_detection_set_changed": 0,
    }

    for trace_index, trace in enumerate(traces):
        logits = read_trace_logits(trace)
        log_probs = log_softmax(logits)
        recording = f"strict-fuzzy-probe-{trace_index:06d}"
        control = runtime_hits(
            decoder_replay=decoder_replay,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording + "-control",
        )
        treatment = runtime_hits(
            decoder_replay=decoder_replay,
            model=model,
            pack=pack,
            trace=trace,
            recording=recording + "-treatment",
            fuzzy_child_cost_log=TREATMENT_FUZZY_COST_LOG,
        )

        aggregate["control_detection_pairs"] += len(control)
        aggregate["treatment_detection_pairs"] += len(treatment)
        aggregate["detection_pairs_added"] += len(treatment - control)
        aggregate["detection_pairs_removed"] += len(control - treatment)
        aggregate["traces_detection_set_changed"] += int(control != treatment)

        for keyword_id, item in keywords.items():
            target = tuple(item["tokens"])
            score = decoder_sequence_log_confidence(log_probs, target)
            confidence = (
                0.0 if not math.isfinite(score) else min(1.0, math.exp(score))
            )
            surrogate = confidence >= float(item["threshold"])
            add_transition(
                per_keyword[str(keyword_id)],
                surrogate=surrogate,
                control=keyword_id in control,
                treatment=keyword_id in treatment,
            )

        sample = positive_sample.get(trace.stem)
        if sample is None:
            continue
        keyword_id = int(sample["keyword_id"])
        if keyword_id not in keywords:
            raise ValueError("positive sample references unknown keyword")
        control_target = keyword_id in control
        treatment_target = keyword_id in treatment
        if control_target != bool(sample["runtime_detected_expected"]):
            raise ValueError(
                f"control runtime drifted for positive {trace.stem}"
            )
        control_wrong = bool(control - {keyword_id})
        treatment_wrong = bool(treatment - {keyword_id})
        positive_seen.add(trace.stem)
        for bucket in (
            positive,
            positive["by_split_keyword"][
                f"{sample['split']}:kw{keyword_id}"
            ],
        ):
            add_positive(
                bucket,
                control_target=control_target,
                treatment_target=treatment_target,
                control_wrong=control_wrong,
                treatment_wrong=treatment_wrong,
            )

    missing = sorted(set(positive_sample) - positive_seen)
    if missing:
        raise ValueError(
            "positive sample traces were not found: " + ",".join(missing[:8])
        )
    return per_keyword, positive, aggregate


def classify_result(positive: dict, aggregate: dict) -> tuple[str, str]:
    gains = int(positive["gained_target_hit"])
    losses = int(positive["lost_target_hit"])
    wrong_increase = int(positive["wrong_keyword_increase"])
    detection_growth = (
        int(aggregate["treatment_detection_pairs"])
        > int(aggregate["control_detection_pairs"])
    )
    if gains > 0 and losses == 0 and wrong_increase == 0 and not detection_growth:
        return (
            "promising-fixed-posterior",
            "run-broader-labeled-runtime-evaluation-before-shipping-change",
        )
    if gains == 0 and losses == 0 and wrong_increase == 0:
        return ("no-positive-effect", "close-no-sweep")
    return ("negative", "close-no-sweep")


def self_test() -> None:
    bucket = empty_transition_bucket()
    add_transition(bucket, surrogate=True, control=False, treatment=True)
    assert bucket["gained_hit"] == 1
    assert bucket["surrogate_miss_recovered"] == 1
    add_transition(bucket, surrogate=True, control=True, treatment=False)
    assert bucket["lost_hit"] == 1
    assert bucket["surrogate_hit_lost"] == 1

    positive = empty_positive_bucket()
    add_positive(
        positive,
        control_target=False,
        treatment_target=True,
        control_wrong=False,
        treatment_wrong=False,
    )
    result, action = classify_result(
        positive,
        {
            "control_detection_pairs": 10,
            "treatment_detection_pairs": 9,
        },
    )
    assert result == "promising-fixed-posterior"
    assert action.startswith("run-broader")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", type=pathlib.Path)
    parser.add_argument("--tokens", type=pathlib.Path)
    parser.add_argument("--keywords-tsv", type=pathlib.Path)
    parser.add_argument("--pack", type=pathlib.Path)
    parser.add_argument("--decoder-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument("--acoustic-alignment", type=pathlib.Path)
    parser.add_argument("--baseline-summary", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("strict fuzzy-child cost probe self-test: PASS")
        return 0

    required = {
        "model": args.model,
        "tokens": args.tokens,
        "keywords-tsv": args.keywords_tsv,
        "pack": args.pack,
        "decoder-replay": args.decoder_replay,
        "posterior-cache": args.posterior_cache,
        "acoustic-alignment": args.acoustic_alignment,
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
    acoustic_alignment = args.acoustic_alignment.resolve()
    baseline_summary = args.baseline_summary.resolve()
    output = args.output.resolve()

    for path, label in (
        (model, "model"),
        (tokens, "tokens"),
        (keywords_tsv, "keywords TSV"),
        (pack, "keyword pack"),
        (decoder_replay, "decoder replay"),
        (acoustic_alignment, "acoustic alignment"),
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
    positive_sample = load_positive_sample(
        acoustic_alignment,
        model_sha256=model_sha256,
    )
    per_keyword, positive, aggregate = summarize(
        traces=traces,
        model=model,
        pack=pack,
        keywords=keywords,
        decoder_replay=decoder_replay,
        positive_sample=positive_sample,
    )
    baseline = load_object(baseline_summary, "baseline summary")
    verify_baseline(
        baseline=baseline,
        model_sha256=model_sha256,
        per_keyword=per_keyword,
    )
    result_class, next_action = classify_result(positive, aggregate)

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_fixed": True,
        "training_changed": False,
        "thresholds_changed": False,
        "single_variable": "fuzzy_child_cost_log",
        "control": {
            "policy": "exact-source-default",
            "fuzzy_child_cost_log": -8.25,
        },
        "treatment": {
            "policy": "debug-search-policy-override",
            "fuzzy_child_cost_log": TREATMENT_FUZZY_COST_LOG,
        },
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(tokens),
        "keywords_sha256": sha256_file(keywords_tsv),
        "keyword_pack_sha256": sha256_file(pack),
        "decoder_replay_sha256": sha256_file(decoder_replay),
        "acoustic_alignment_sha256": sha256_file(acoustic_alignment),
        "baseline_summary_sha256": sha256_file(baseline_summary),
        "per_keyword": per_keyword,
        "positive_sample": positive,
        "aggregate": aggregate,
        "result_class": result_class,
        "next_action": next_action,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result_class": result_class,
                "next_action": next_action,
                "positive_sample": {
                    key: value
                    for key, value in positive.items()
                    if key != "by_split_keyword"
                },
                "aggregate": aggregate,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
