#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
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
            "rounds": compact_rounds,
        },
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
