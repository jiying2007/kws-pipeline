#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path) -> tuple[dict | None, str | None]:
    if not path.is_file() or path.stat().st_size == 0:
        return None, "missing"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid:{type(exc).__name__}"
    if not isinstance(value, dict):
        return None, "not-object"
    return value, None


def metric_slice(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in (
        "expected",
        "matched",
        "false_rejects",
        "false_accepts",
        "frr",
        "far_per_hour",
        "p95_post_end_latency_ms",
    ):
        if key in value:
            result[key] = value[key]
    per_keyword = value.get("per_keyword")
    if isinstance(per_keyword, dict):
        result["per_keyword"] = {
            str(key): metric_slice(row) or {}
            for key, row in sorted(per_keyword.items(), key=lambda item: str(item[0]))
            if isinstance(row, dict)
        }
    return result


def threshold_slice(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    selected = value.get("calibrated_thresholds")
    grid = value.get("calibration_threshold_grid")
    if not isinstance(selected, dict) or not selected or not isinstance(grid, list) or not grid:
        return None
    thresholds = sorted(float(item) for item in grid)
    lower = thresholds[0]
    upper = thresholds[-1]
    edge_keywords: dict[str, str] = {}
    normalized: dict[str, float] = {}
    for raw_key, raw_value in sorted(selected.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        threshold = float(raw_value)
        normalized[key] = threshold
        if math.isclose(threshold, lower, rel_tol=0.0, abs_tol=1.0e-12):
            edge_keywords[key] = "min"
        elif math.isclose(threshold, upper, rel_tol=0.0, abs_tol=1.0e-12):
            edge_keywords[key] = "max"
    result = {
        "selected": normalized,
        "grid": thresholds,
        "grid_min": lower,
        "grid_max": upper,
        "edge_keywords": edge_keywords,
        "grid_saturated": bool(edge_keywords),
        "coordinate_rounds": value.get("calibration_coordinate_rounds"),
        "coordinate_rounds_executed": value.get(
            "calibration_coordinate_rounds_executed"
        ),
        "parallel_trials": value.get("calibration_parallel_trials"),
    }
    curve_summary = value.get("calibration_operating_curve_summary")
    if isinstance(curve_summary, dict):
        result["operating_curve_summary"] = curve_summary
    curve_path = value.get("calibration_operating_curve_path")
    if isinstance(curve_path, str) and curve_path:
        result["operating_curve_path"] = curve_path
    curve_sha = value.get("calibration_operating_curve_sha256")
    if isinstance(curve_sha, str) and curve_sha:
        result["operating_curve_sha256"] = curve_sha
    return result


def domain_slice(value: Any) -> dict | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key in ("worst_domain", "worst_domain_score"):
        if key in value:
            result[key] = value[key]
    confusion = value.get("keyword_confusion")
    if isinstance(confusion, dict):
        result["keyword_confusion"] = {
            key: confusion[key]
            for key in (
                "assignment",
                "expected_events",
                "correct_keyword",
                "wrong_keyword",
                "missed",
                "matrix",
            )
            if key in confusion
        }
    return result or None


def round_slice(row: dict) -> dict:
    return {
        key: row[key]
        for key in (
            "round",
            "stage",
            "frontend",
            "score",
            "model_sha256",
            "calibration_gate",
            "test_gate",
            "hard_negative_replay_examples",
            "hard_negative_replay_manifest_sha256",
            "adversarial_policy",
            "adversarial_selection_policy",
            "adversarial_top_k",
            "adversarial_replay_examples",
            "adversarial_manifest_sha256",
            "development_failure_replay_examples",
            "development_failure_manifest_sha256",
        )
        if key in row
    } | {
        "calibration": metric_slice(row.get("calibration")),
        "test": metric_slice(row.get("test")),
        "calibration_operating_point": threshold_slice(row.get("calibration")),
        "calibration_domain_summary": domain_slice(row.get("calibration_domains")),
        "test_domain_summary": domain_slice(row.get("test_domains")),
    }


def acoustic_slice(summary: dict | None) -> dict | None:
    if not isinstance(summary, dict):
        return None
    return {
        key: summary[key]
        for key in (
            "schema_version",
            "evidence_class",
            "development_only",
            "source_round",
            "source_frontend",
            "source_selection_policy",
            "source_was_strict",
            "model_sha256",
            "tokens_sha256",
            "keywords_sha256",
            "domain_index_sha256",
            "feature_dump_sha256",
            "parameter_contract_sha256",
            "root_start_logit_margin",
            "max_recordings_per_keyword_split",
            "model",
            "aggregates",
        )
        if key in summary
    }


def evidence_signals(
    rounds: list[dict],
    refinement_eligibility: dict | None,
    acoustic_alignment: dict | None,
) -> dict:
    threshold_saturation: list[dict] = []
    keyword_confusion: list[dict] = []
    round_by_index: dict[int, dict] = {}

    for row in rounds:
        if not isinstance(row, dict):
            continue
        round_index = int(row.get("round", -1))
        round_by_index[round_index] = row
        operating = row.get("calibration_operating_point")
        if isinstance(operating, dict) and operating.get("grid_saturated") is True:
            threshold_saturation.append(
                {
                    "round": round_index,
                    "edge_keywords": dict(operating.get("edge_keywords", {})),
                    "grid_min": operating.get("grid_min"),
                    "grid_max": operating.get("grid_max"),
                }
            )
        for split, key in (
            ("calibration", "calibration_domain_summary"),
            ("test", "test_domain_summary"),
        ):
            domains = row.get(key)
            confusion = domains.get("keyword_confusion") if isinstance(domains, dict) else None
            if isinstance(confusion, dict) and int(confusion.get("wrong_keyword", 0)) > 0:
                keyword_confusion.append(
                    {
                        "round": round_index,
                        "split": split,
                        "wrong_keyword": int(confusion["wrong_keyword"]),
                        "matrix": confusion.get("matrix", {}),
                    }
                )

    collapsed_keyword_ids: list[str] = []
    if isinstance(refinement_eligibility, dict) and refinement_eligibility.get("eligible") is False:
        raw = refinement_eligibility.get("collapsed_keyword_ids", [])
        if isinstance(raw, list):
            collapsed_keyword_ids = [str(value) for value in raw]

    acoustic_runtime_gaps: list[dict] = []
    sampled_acoustic_sequence_absent: list[dict] = []
    if isinstance(acoustic_alignment, dict):
        source_round = int(acoustic_alignment.get("source_round", -1))
        source = round_by_index.get(source_round)
        aggregates = acoustic_alignment.get("aggregates")
        if isinstance(source, dict) and isinstance(aggregates, dict):
            keyword_ids: set[str] = set()
            for split in ("calibration", "test"):
                split_aggregates = aggregates.get(split)
                if isinstance(split_aggregates, dict):
                    keyword_ids.update(str(key) for key in split_aggregates)

            for keyword_id in sorted(keyword_ids):
                subsequence_total = 0
                sampled_total = 0
                for split in ("calibration", "test"):
                    split_aggregates = aggregates.get(split)
                    aggregate = (
                        split_aggregates.get(keyword_id)
                        if isinstance(split_aggregates, dict)
                        else None
                    )
                    if not isinstance(aggregate, dict):
                        continue
                    recordings = int(aggregate.get("recordings", 0))
                    subsequences = int(aggregate.get("greedy_subsequence_recordings", 0))
                    sampled_total += recordings
                    subsequence_total += subsequences

                    metrics = source.get(split)
                    per_keyword = (
                        metrics.get("per_keyword") if isinstance(metrics, dict) else None
                    )
                    runtime = (
                        per_keyword.get(keyword_id)
                        if isinstance(per_keyword, dict)
                        else None
                    )
                    if (
                        subsequences > 0
                        and isinstance(runtime, dict)
                        and int(runtime.get("matched", 0)) == 0
                    ):
                        acoustic_runtime_gaps.append(
                            {
                                "round": source_round,
                                "split": split,
                                "keyword_id": keyword_id,
                                "sampled_recordings": recordings,
                                "acoustic_greedy_subsequence_recordings": subsequences,
                                "runtime_matched": 0,
                            }
                        )
                if sampled_total > 0 and subsequence_total == 0:
                    sampled_acoustic_sequence_absent.append(
                        {
                            "round": source_round,
                            "keyword_id": keyword_id,
                            "sampled_recordings": sampled_total,
                        }
                    )

    return {
        "cross_split_keyword_collapse": {
            "observed": bool(collapsed_keyword_ids),
            "keyword_ids": collapsed_keyword_ids,
        },
        "threshold_grid_saturation": {
            "observed": bool(threshold_saturation),
            "occurrences": threshold_saturation,
        },
        "keyword_confusion": {
            "observed": bool(keyword_confusion),
            "occurrences": keyword_confusion,
        },
        "acoustic_sequence_observed_but_runtime_missed": {
            "observed": bool(acoustic_runtime_gaps),
            "occurrences": acoustic_runtime_gaps,
        },
        "sampled_acoustic_sequence_absent": {
            "observed": bool(sampled_acoustic_sequence_absent),
            "occurrences": sampled_acoustic_sequence_absent,
        },
    }


def shadow_slice(summary: dict | None) -> dict | None:
    if not isinstance(summary, dict):
        return None
    results = summary.get("results")
    compact_results: list[dict] = []
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            item = {
                key: row[key]
                for key in (
                    "seed",
                    "qualified",
                    "runtime_qualified",
                    "surrogate_separation_qualified",
                )
                if key in row
            }
            item["qualification"] = metric_slice(row.get("qualification"))
            surrogate = row.get("surrogate")
            if isinstance(surrogate, dict):
                item["surrogate"] = {
                    key: surrogate[key]
                    for key in (
                        "minimum_separation",
                        "minimum_positive_confidence",
                        "maximum_negative_confidence",
                    )
                    if key in surrogate
                }
            compact_results.append(item)
    return {
        key: summary[key]
        for key in (
            "schema_version",
            "evidence_class",
            "qualified",
            "seeds",
            "expected_wakes_per_seed",
            "min_surrogate_separation",
            "formal_qualification_seed_consumed",
            "development_manifest_sha256",
            "development_selected_round",
            "development_selected_frontend",
        )
        if key in summary
    } | {"results": compact_results}


def file_entry(root: pathlib.Path, relative: str) -> dict:
    path = root / relative
    value, error = load_json(path)
    entry: dict[str, Any] = {
        "path": relative,
        "present": value is not None,
    }
    if path.is_file():
        entry["bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    if error is not None:
        entry["error"] = error
    return entry


def build(config_path: pathlib.Path, root: pathlib.Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest, manifest_error = load_json(root / "domain-loop-manifest.json")
    refinement, refinement_error = load_json(root / "adversarial-refinement/summary.json")
    adversarial, adversarial_error = load_json(root / "best/adversarial-lexicon.json")
    failure_replay, failure_error = load_json(root / "development-failure-replay/evidence.json")
    shadow, shadow_error = load_json(root / "shadow-qualification/summary.json")
    preflight, preflight_error = load_json(root / "formal-preflight.json")
    cohort, cohort_error = load_json(root / "qualification-dataset/qualification-cohort.json")
    training, training_error = load_json(root / "training-run-summary.json")
    robustness, robustness_error = load_json(root / "robustness-summary.json")
    continuous_far, continuous_far_error = load_json(root / "hard-negative-stream/summary.json")
    refinement_eligibility, refinement_eligibility_error = load_json(
        root / "base-refinement-eligibility.json"
    )
    acoustic_alignment, acoustic_alignment_error = load_json(
        root / "acoustic-alignment.json"
    )

    diagnostics_errors = {
        name: error
        for name, error in (
            ("domain_loop_manifest", manifest_error),
            ("adversarial_refinement", refinement_error),
            ("adversarial_lexicon", adversarial_error),
            ("development_failure_replay", failure_error),
            ("shadow_qualification", shadow_error),
            ("formal_preflight", preflight_error),
            ("qualification_cohort", cohort_error),
            ("training_summary", training_error),
            ("robustness", robustness_error),
            ("continuous_far", continuous_far_error),
            ("base_refinement_eligibility", refinement_eligibility_error),
            ("acoustic_alignment", acoustic_alignment_error),
        )
        if error not in (None, "missing")
    }

    records = manifest.get("records", []) if isinstance(manifest, dict) else []
    compact_rounds = [round_slice(row) for row in records if isinstance(row, dict)]
    selection = manifest.get("candidate_selection") if isinstance(manifest, dict) else None

    active_seed = int(config.get("qualification_holdout_seed", -1))
    retired = [int(value) for value in config.get("retired_qualification_holdout_seeds", [])]
    formal_cohort_generated = isinstance(cohort, dict) and (
        int(cohort.get("qualification_seed", -2)) == active_seed
        and str(cohort.get("formal_preflight_policy", "")) == "shadow-adversarial-formal-preflight-v1"
    )

    return {
        "schema_version": 1,
        "evidence_class": "compact-model-training-diagnostics",
        "config_sha256": sha256_file(config_path),
        "formal_seed": {
            "active": active_seed,
            "retired": retired,
            "cohort_generated": formal_cohort_generated,
            "consumed_by_this_run": formal_cohort_generated,
        },
        "development": {
            "qualified": manifest.get("development_qualified") if isinstance(manifest, dict) else None,
            "qualification_qualified": manifest.get("qualification_qualified") if isinstance(manifest, dict) else None,
            "candidate_selection": selection if isinstance(selection, dict) else None,
            "refinement_eligibility": refinement_eligibility,
            "acoustic_alignment": acoustic_slice(acoustic_alignment),
            "rounds": compact_rounds,
        },
        "evidence_signals": evidence_signals(
            compact_rounds,
            refinement_eligibility,
            acoustic_alignment,
        ),
        "adversarial_refinement": refinement,
        "adversarial_lexicon": ({
            key: adversarial[key]
            for key in (
                "evidence_class",
                "selection_policy",
                "data_policy",
                "enumerated_sequences",
                "top_k",
                "probes_per_sequence",
                "replay_examples_per_sequence",
                "replay_examples",
                "per_keyword_selected",
                "strict_prefix_anchor_count",
                "manifest_sha256",
                "formal_qualification_used",
            )
            if key in adversarial
        } if isinstance(adversarial, dict) else None),
        "development_failure_replay": failure_replay,
        "shadow_qualification": shadow_slice(shadow),
        "formal_preflight": preflight,
        "qualification_cohort": cohort,
        "training_summary": ({
            key: training[key]
            for key in (
                "qualified",
                "iteration_exit_code",
                "training_exit_code",
                "best_round",
                "best_frontend",
                "best_score",
                "candidate_selection",
                "qualification",
            )
            if key in training
        } if isinstance(training, dict) else None),
        "robustness": robustness,
        "continuous_far": continuous_far,
        "files": [
            file_entry(root, path)
            for path in (
                "domain-loop-manifest.json",
                "adversarial-refinement/summary.json",
                "best/adversarial-lexicon.json",
                "development-failure-replay/evidence.json",
                "shadow-qualification/summary.json",
                "formal-preflight.json",
                "qualification-dataset/qualification-cohort.json",
                "training-run-summary.json",
                "robustness-summary.json",
                "hard-negative-stream/summary.json",
                "base-refinement-eligibility.json",
                "acoustic-alignment.json",
            )
        ],
        "diagnostic_errors": diagnostics_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact model-training diagnostic sidecar.")
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = build(args.config.resolve(), args.work_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "formal_seed": result["formal_seed"],
        "rounds": len(result["development"]["rounds"]),
        "shadow_present": result["shadow_qualification"] is not None,
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
