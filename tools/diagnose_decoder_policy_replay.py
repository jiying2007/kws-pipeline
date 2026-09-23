#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from iterate_domain import (  # noqa: E402
    base_gate,
    calibrate,
    domain_gate,
    evaluate,
    gate_values,
    resolve_posterior_replay,
    sha256_file,
)
from diagnose_kws_threshold_operating_curve import (  # noqa: E402
    development_round_evidence,
)

EVIDENCE_CLASS = "decoder-policy-posterior-replay-grid-development-v1"
POLICY = "blank-retention-fuzzy-child-grid-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def compact_metrics(base: dict, domains: dict) -> dict:
    far_domain = domains.get("domains", {}).get("distance:far", {})
    return {
        "frr": float(base.get("frr", 1.0)),
        "far_per_hour": float(base.get("far_per_hour", math.inf)),
        "negative_recording_far_per_hour": base.get(
            "negative_recording_far_per_hour"
        ),
        "negative_recording_far_upper_95_per_hour": base.get(
            "negative_recording_far_upper_95_per_hour"
        ),
        "p95_post_end_latency_ms": float(
            base.get("p95_post_end_latency_ms", math.inf)
        ),
        "per_keyword": base.get("per_keyword", {}),
        "far_field_frr": (
            float(far_domain["frr"])
            if isinstance(far_domain, dict) and "frr" in far_domain
            else None
        ),
        "worst_domain_score": domains.get("worst_domain_score"),
    }


def posterior_counts(root: pathlib.Path) -> tuple[int, int, int]:
    hits = 0
    misses = 0
    files = 0
    for path in sorted(root.rglob("detections.provenance.json")):
        value = load_object(path)
        if value.get("evaluation_mode") != "posterior-replay-cache-v1":
            continue
        hits += int(value.get("posterior_cache_hits", 0))
        misses += int(value.get("posterior_cache_misses", 0))
        files += 1
    return hits, misses, files


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only decoder search-policy sweep over exact posterior "
            "traces. Recalibrates keyword thresholds for every policy point and "
            "never changes shipping settings."
        )
    )
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-references", required=True, type=pathlib.Path)
    parser.add_argument("--test-references", required=True, type=pathlib.Path)
    parser.add_argument("--development-manifest", type=pathlib.Path)
    parser.add_argument("--development-authority-receipt", type=pathlib.Path)
    parser.add_argument("--round-index", type=int)
    parser.add_argument(
        "--diagnostic-round-selection-policy",
        default="caller-supplied-round-v1",
    )
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument(
        "--blank-retentions", required=True, nargs="+", type=float
    )
    parser.add_argument(
        "--fuzzy-child-cost-logs", required=True, nargs="+", type=float
    )
    parser.add_argument(
        "--coordinate-rounds",
        type=int,
        default=1,
        help="bounded diagnostic calibration rounds per decoder policy point",
    )
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    for path, label in (
        (args.runner, "runner"),
        (args.model, "model"),
        (args.tokens, "tokens"),
        (args.keywords, "keywords"),
        (args.config, "config"),
        (args.calibration_references, "calibration references"),
        (args.test_references, "test references"),
        (args.posterior_dump, "posterior dump"),
        (args.decoder_replay, "decoder replay"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    blank_retentions = sorted(set(float(v) for v in args.blank_retentions))
    fuzzy_costs = sorted(set(float(v) for v in args.fuzzy_child_cost_logs))
    if (
        not blank_retentions
        or any(not math.isfinite(v) or not 0.0 < v < 1.0 for v in blank_retentions)
    ):
        raise ValueError("blank retentions must be unique finite values in (0,1)")
    if (
        not fuzzy_costs
        or any(not math.isfinite(v) or not -16.0 <= v <= 0.0 for v in fuzzy_costs)
    ):
        raise ValueError("fuzzy child cost logs must be unique finite values in [-16,0]")
    if not 1 <= args.coordinate_rounds <= 2:
        raise ValueError("coordinate rounds must be in [1,2]")

    cfg = load_object(args.config)
    calibration_cfg = cfg.get("calibration", {})
    thresholds = [float(v) for v in calibration_cfg.get("thresholds", [])]
    parallel_trials = int(calibration_cfg.get("max_parallel_trials", 1))
    if (
        not thresholds
        or any(not math.isfinite(v) or not 0.0 < v < 1.0 for v in thresholds)
        or not 1 <= parallel_trials <= 4
    ):
        raise ValueError("config calibration contract is invalid")
    gates = gate_values(cfg.get("domain_gates", {}))

    if (
        args.development_authority_receipt is not None
        and args.development_manifest is None
    ):
        raise ValueError(
            "--development-authority-receipt requires --development-manifest"
        )
    development_evidence = development_round_evidence(
        args.development_manifest,
        args.round_index,
        gates,
        args.development_authority_receipt,
    )
    model_sha256 = sha256_file(args.model)
    config_sha256 = sha256_file(args.config)
    actual_calibration = sha256_file(args.calibration_references)
    actual_test = sha256_file(args.test_references)
    if development_evidence is not None:
        if development_evidence["development_record_model_sha256"] != model_sha256:
            raise ValueError("diagnostic model does not match development round model")
        if development_evidence["development_config_sha256"] != config_sha256:
            raise ValueError("diagnostic config does not match development manifest config")
        if development_evidence["calibration_references_sha256"] != actual_calibration:
            raise ValueError("calibration references do not match development manifest")
        if development_evidence["test_references_sha256"] != actual_test:
            raise ValueError("test references do not match development manifest")

    work = args.work_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    posterior_replay = resolve_posterior_replay(
        args.posterior_dump,
        args.decoder_replay,
        args.posterior_cache,
    )
    assert posterior_replay is not None

    rows: list[dict] = []
    for blank_retention in blank_retentions:
        for fuzzy_cost in fuzzy_costs:
            trial = work / (
                f"blank-{blank_retention:.3f}-fuzzy-{abs(fuzzy_cost):.3f}"
            )
            calibrated_tsv, calibrated_pack, cal_base, cal_domains = calibrate(
                runner=args.runner.resolve(),
                model=args.model.resolve(),
                tokens=args.tokens.resolve(),
                source_keywords=args.keywords.resolve(),
                references=args.calibration_references.resolve(),
                output=trial / "calibration",
                thresholds=thresholds,
                rounds=args.coordinate_rounds,
                gates=gates,
                parallel_trials=parallel_trials,
                posterior_replay=posterior_replay,
                decoder_blank_retention=blank_retention,
                decoder_fuzzy_child_cost_log=fuzzy_cost,
            )
            test_base, test_domains = evaluate(
                runner=args.runner.resolve(),
                model=args.model.resolve(),
                pack=calibrated_pack,
                references=args.test_references.resolve(),
                output=trial / "test",
                posterior_replay=posterior_replay,
                decoder_blank_retention=blank_retention,
                decoder_fuzzy_child_cost_log=fuzzy_cost,
            )
            hits, misses, provenance_files = posterior_counts(trial)
            rows.append(
                {
                    "blank_retention": blank_retention,
                    "fuzzy_child_cost_log": fuzzy_cost,
                    "calibrated_thresholds": dict(cal_base["calibrated_thresholds"]),
                    "calibrated_keywords_sha256": sha256_file(calibrated_tsv),
                    "calibrated_pack_sha256": sha256_file(calibrated_pack),
                    "calibration": compact_metrics(cal_base, cal_domains),
                    "test": compact_metrics(test_base, test_domains),
                    "strict": (
                        base_gate(cal_base, gates)
                        and domain_gate(cal_domains, gates)
                        and base_gate(test_base, gates)
                        and domain_gate(test_domains, gates)
                    ),
                    "posterior_cache": {
                        "hits": hits,
                        "misses": misses,
                        "provenance_files": provenance_files,
                    },
                }
            )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "thresholds_recalibrated_per_policy_point": True,
        "model_sha256": model_sha256,
        "source_keyword_pack_sha256": sha256_file(args.keywords),
        "config_sha256": config_sha256,
        "calibration_references_sha256": actual_calibration,
        "test_references_sha256": actual_test,
        "diagnostic_round_selection_policy": str(
            args.diagnostic_round_selection_policy
        ),
        "development_round_evidence": development_evidence,
        "formal_threshold_grid": thresholds,
        "coordinate_rounds": args.coordinate_rounds,
        "parallel_trials": parallel_trials,
        "blank_retentions": blank_retentions,
        "fuzzy_child_cost_logs": fuzzy_costs,
        "operating_grid": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "policy_points": len(rows),
                "thresholds_recalibrated": True,
                "selection_feedback_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
