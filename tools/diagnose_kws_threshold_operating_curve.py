#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    calibration_behavior_key,
    compile_pack,
    domain_gate,
    evaluate,
    gate_values,
    keyword_rows,
    resolve_posterior_replay,
    select_calibration_threshold,
    write_keywords,
)

EVIDENCE_CLASS = "kws-v2-threshold-operating-curve-diagnostic-v2"
MODE = "common-threshold-sweep-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def far_frr(domains: dict) -> float:
    row = domains.get("domains", {}).get("distance:far", {})
    if not isinstance(row, dict):
        return 1.0
    return finite(row.get("frr", 1.0), "distance:far.frr")


def compact_metrics(base: dict, domains: dict, gates: dict) -> dict:
    return {
        "frr": finite(base["frr"], "frr"),
        "far_per_hour": finite(base["far_per_hour"], "far_per_hour"),
        "p95_post_end_latency_ms": finite(
            base["p95_post_end_latency_ms"], "p95_post_end_latency_ms"
        ),
        "far_distance_frr": far_frr(domains),
        "base_gate": base_gate(base, gates),
        "domain_gate": domain_gate(domains, gates),
    }


def pareto_indices(rows: list[dict]) -> list[int]:
    result: list[int] = []
    for index, row in enumerate(rows):
        frr = float(row["calibration"]["frr"])
        far = float(row["calibration"]["far_per_hour"])
        dominated = False
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            other_frr = float(other["calibration"]["frr"])
            other_far = float(other["calibration"]["far_per_hour"])
            if (
                other_frr <= frr
                and other_far <= far
                and (other_frr < frr or other_far < far)
            ):
                dominated = True
                break
        if not dominated:
            result.append(index)
    return result


def development_round_evidence(
    manifest_path: pathlib.Path | None,
    round_index: int | None,
    gates: dict,
    authority_receipt_path: pathlib.Path | None = None,
) -> dict | None:
    if manifest_path is None:
        if round_index is not None:
            raise ValueError("--round-index requires --development-manifest")
        return None
    value = load_object(manifest_path)
    manifest_config_sha256 = str(value.get("config_sha256", ""))
    authority_policy: str
    authority_receipt_sha256: str | None = None

    if value.get("evidence_scope") == "development-only":
        for key in ("qualification_used", "shadow_used", "formal_qualification_used"):
            if value.get(key) is not False:
                raise ValueError(f"development manifest used protected evidence: {key}")
        if authority_receipt_path is not None:
            raise ValueError(
                "development authority receipt is only valid for PR-head experiment manifests"
            )
        authority_policy = "development-manifest-self-declared-v1"
    else:
        if authority_receipt_path is None:
            raise ValueError(
                "development manifest must be development-only or bound to "
                "an exact PR-head development receipt"
            )
        receipt = load_object(authority_receipt_path)
        if receipt.get("evidence_class") != "product-development-pr-head-experiment-v1":
            raise ValueError("development authority receipt evidence_class mismatch")
        if receipt.get("development_only") is not True:
            raise ValueError("development authority receipt is not development-only")
        if receipt.get("source_policy") != "exact-pr-head":
            raise ValueError("development authority receipt source_policy mismatch")
        if receipt.get("protected_evidence_used") is not False:
            raise ValueError("development authority receipt used protected evidence")
        receipt_config_sha256 = str(receipt.get("experiment_config_sha256", ""))
        if not manifest_config_sha256 or receipt_config_sha256 != manifest_config_sha256:
            raise ValueError(
                "development authority receipt config does not match manifest"
            )
        for key in ("pr_base_sha", "pr_head_sha"):
            raw = receipt.get(key)
            if (
                not isinstance(raw, str)
                or len(raw) != 40
                or any(ch not in "0123456789abcdef" for ch in raw)
            ):
                raise ValueError(f"development authority receipt {key} is invalid")
        authority_policy = "exact-pr-head-development-receipt-v1"
        authority_receipt_sha256 = sha256_file(authority_receipt_path)
    records = value.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("development manifest has no records")
    if round_index is None:
        raise ValueError("--round-index is required with --development-manifest")
    matches = [
        row
        for row in records
        if isinstance(row, dict) and int(row.get("round", -1)) == round_index
    ]
    if len(matches) != 1:
        raise ValueError(f"development round must match exactly once: {round_index}")
    row = matches[0]
    calibration = row.get("calibration")
    calibration_domains = row.get("calibration_domains")
    test = row.get("test")
    test_domains = row.get("test_domains")
    if not isinstance(calibration, dict) or not isinstance(calibration_domains, dict):
        raise ValueError("selected development round lacks calibration evidence")
    if not isinstance(test, dict) or not isinstance(test_domains, dict):
        raise ValueError("selected development round lacks test evidence")
    selected = calibration.get("calibrated_thresholds")
    grid = calibration.get("calibration_threshold_grid")
    coordinate_rounds = calibration.get("calibration_coordinate_rounds")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("selected development round lacks calibrated thresholds")
    if not isinstance(grid, list) or not grid:
        raise ValueError("selected development round lacks threshold grid")
    manifest_selected_round = value.get("selected_round")
    if manifest_selected_round is not None:
        manifest_selected_round = int(manifest_selected_round)
    selection_policy = value.get("selection_policy")
    return {
        "round": round_index,
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_authority_policy": authority_policy,
        "development_authority_receipt_sha256": authority_receipt_sha256,
        "development_config_sha256": manifest_config_sha256,
        "development_record_model_sha256": str(row.get("model_sha256", "")),
        "development_record_provenance_sha256": str(row.get("provenance_sha256", "")),
        "development_record_frontend": str(row.get("frontend", "")),
        "development_record_score": finite(row.get("score"), "development record score"),
        "development_qualified": bool(value.get("development_qualified", False)),
        "development_manifest_selection_policy": (
            str(selection_policy) if selection_policy is not None else None
        ),
        "development_manifest_selected_round": manifest_selected_round,
        "round_matches_development_selected_round": (
            manifest_selected_round == round_index
            if manifest_selected_round is not None
            else False
        ),
        "calibration_references_sha256": str(calibration.get("references_sha256", "")),
        "test_references_sha256": str(test.get("references_sha256", "")),
        "calibrated_thresholds": {
            str(key): finite(value, f"development threshold {key}")
            for key, value in selected.items()
        },
        "threshold_grid": [
            finite(value, "development threshold grid") for value in grid
        ],
        "coordinate_rounds": int(coordinate_rounds),
        "calibration_metrics": compact_metrics(
            calibration, calibration_domains, gates
        ),
        "test_metrics": compact_metrics(test, test_domains, gates),
        "calibration_behavior_key": list(
            calibration_behavior_key(calibration, calibration_domains, gates)
        ),
        "test_behavior_key": list(
            calibration_behavior_key(test, test_domains, gates)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only common-threshold operating-curve diagnostic. "
            "This tool never edits the product config or selects a shipping threshold."
        )
    )
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-references", required=True, type=pathlib.Path)
    parser.add_argument("--test-references", required=True, type=pathlib.Path)
    parser.add_argument("--thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--development-manifest", type=pathlib.Path)
    parser.add_argument("--development-authority-receipt", type=pathlib.Path)
    parser.add_argument("--development-domain-summary", type=pathlib.Path)
    parser.add_argument("--round-index", type=int)
    parser.add_argument(
        "--diagnostic-round-selection-policy",
        default="caller-supplied-round-v1",
    )
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", type=pathlib.Path)
    parser.add_argument("--decoder-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    args = parser.parse_args()

    posterior_replay = resolve_posterior_replay(
        args.posterior_dump,
        args.decoder_replay,
        args.posterior_cache,
    )

    for path, label in (
        (args.runner, "runner"),
        (args.model, "model"),
        (args.tokens, "tokens"),
        (args.keywords, "keywords"),
        (args.config, "config"),
        (args.calibration_references, "calibration references"),
        (args.test_references, "test references"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    thresholds = sorted(set(float(value) for value in args.thresholds))
    if (
        not thresholds
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in thresholds)
    ):
        raise ValueError("diagnostic thresholds must be unique finite values in (0,1)")

    cfg = load_object(args.config)
    gates = gate_values(cfg.get("domain_gates", {}))
    if args.development_authority_receipt is not None and args.development_manifest is None:
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
        if not development_evidence["development_record_model_sha256"]:
            raise ValueError("development round is missing model SHA256")
        if development_evidence["development_record_model_sha256"] != model_sha256:
            raise ValueError("diagnostic model does not match development round model")
        if not development_evidence["development_config_sha256"]:
            raise ValueError("development manifest is missing config SHA256")
        if development_evidence["development_config_sha256"] != config_sha256:
            raise ValueError("diagnostic config does not match development manifest config")
        expected_calibration = development_evidence["calibration_references_sha256"]
        expected_test = development_evidence["test_references_sha256"]
        if not expected_calibration or expected_calibration != actual_calibration:
            raise ValueError("calibration references do not match development manifest")
        if not expected_test or expected_test != actual_test:
            raise ValueError("test references do not match development manifest")

    domain_summary_evidence = None
    if args.development_domain_summary is not None:
        if not args.development_domain_summary.is_file():
            raise ValueError(
                f"development domain summary is missing: {args.development_domain_summary}"
            )
        summary = load_object(args.development_domain_summary)
        splits = summary.get("splits")
        if not isinstance(splits, dict):
            raise ValueError("development domain summary has no split evidence")
        calibration_split = splits.get("calibration")
        test_split = splits.get("test")
        if not isinstance(calibration_split, dict) or not isinstance(test_split, dict):
            raise ValueError("development domain summary lacks calibration/test splits")
        expected_calibration = str(calibration_split.get("references_sha256", ""))
        expected_test = str(test_split.get("references_sha256", ""))
        if not expected_calibration or expected_calibration != actual_calibration:
            raise ValueError("calibration references do not match development domain summary")
        if not expected_test or expected_test != actual_test:
            raise ValueError("test references do not match development domain summary")
        domain_summary_evidence = {
            "sha256": sha256_file(args.development_domain_summary),
            "calibration_references_sha256": actual_calibration,
            "test_references_sha256": actual_test,
        }

    work = args.work_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    source_rows = keyword_rows(args.keywords)
    rows: list[dict] = []
    for threshold in thresholds:
        trial = [dict(row) for row in source_rows]
        for row in trial:
            row["threshold"] = threshold
        trial_root = work / f"threshold-{threshold:.3f}"
        tsv = trial_root / "keywords.tsv"
        pack = trial_root / "keywords.kwk"
        write_keywords(trial, tsv)
        compile_pack(args.tokens, tsv, pack)
        cal_base, cal_domains = evaluate(
            runner=args.runner,
            model=args.model,
            pack=pack,
            references=args.calibration_references,
            output=trial_root / "calibration",
            posterior_replay=posterior_replay,
        )
        test_base, test_domains = evaluate(
            runner=args.runner,
            model=args.model,
            pack=pack,
            references=args.test_references,
            output=trial_root / "test",
            posterior_replay=posterior_replay,
        )
        rows.append(
            {
                "threshold": threshold,
                "keyword_thresholds": {
                    str(row["id"]): threshold for row in source_rows
                },
                "calibration": compact_metrics(cal_base, cal_domains, gates),
                "test": compact_metrics(test_base, test_domains, gates),
                "official_calibration_behavior_key": list(
                    calibration_behavior_key(cal_base, cal_domains, gates)
                ),
            }
        )

    pareto = pareto_indices(rows)
    min_frr_value = min(float(row["calibration"]["frr"]) for row in rows)
    min_far_value = min(float(row["calibration"]["far_per_hour"]) for row in rows)
    min_frr_plateau = [
        index
        for index, row in enumerate(rows)
        if math.isclose(
            float(row["calibration"]["frr"]),
            min_frr_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    min_far_plateau = [
        index
        for index, row in enumerate(rows)
        if math.isclose(
            float(row["calibration"]["far_per_hour"]),
            min_far_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ]
    min_calibration_frr = min(
        min_frr_plateau,
        key=lambda index: (
            float(rows[index]["calibration"]["far_per_hour"]),
            rows[index]["threshold"],
        ),
    )
    min_calibration_far = min(
        min_far_plateau,
        key=lambda index: (
            float(rows[index]["calibration"]["frr"]),
            -rows[index]["threshold"],
        ),
    )
    official_candidates = [
        (
            float(row["threshold"]),
            tuple(float(value) for value in row["official_calibration_behavior_key"]),
        )
        for row in rows
    ]
    official_best_key = min(key for _, key in official_candidates)
    official_order_plateau = [
        index
        for index, row in enumerate(rows)
        if tuple(float(value) for value in row["official_calibration_behavior_key"])
        == official_best_key
    ]
    official_order_threshold = select_calibration_threshold(official_candidates)

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "development-only",
        "mode": MODE,
        "diagnostic_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "posterior_replay_enabled": posterior_replay is not None,
        "model_sha256": model_sha256,
        "tokens_sha256": sha256_file(args.tokens),
        "keywords_sha256": sha256_file(args.keywords),
        "config_sha256": config_sha256,
        "calibration_references_sha256": sha256_file(args.calibration_references),
        "test_references_sha256": sha256_file(args.test_references),
        "thresholds": thresholds,
        "diagnostic_round_selection_policy": str(args.diagnostic_round_selection_policy),
        "common_threshold_sweep_only": True,
        "development_round_evidence": development_evidence,
        "development_domain_summary_evidence": domain_summary_evidence,
        "operating_curve": rows,
        "calibration_frr_far_pareto_thresholds": [
            rows[index]["threshold"] for index in pareto
        ],
        "pareto_thresholds": [rows[index]["threshold"] for index in pareto],
        "min_calibration_frr_thresholds": [
            rows[index]["threshold"] for index in min_frr_plateau
        ],
        "min_calibration_far_thresholds": [
            rows[index]["threshold"] for index in min_far_plateau
        ],
        "min_calibration_frr_threshold": rows[min_calibration_frr]["threshold"],
        "min_calibration_far_threshold": rows[min_calibration_far]["threshold"],
        "official_behavior_order_key": list(official_best_key),
        "official_behavior_order_thresholds": [
            rows[index]["threshold"] for index in official_order_plateau
        ],
        "official_behavior_order_threshold": official_order_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "threshold diagnostic complete: "
        f"points={len(rows)} pareto={result['pareto_thresholds']} "
        f"min-frr={result['min_calibration_frr_threshold']} "
        f"official-order={result['official_behavior_order_threshold']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
