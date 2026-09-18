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
    write_keywords,
)

EVIDENCE_CLASS = "kws-v2-threshold-operating-curve-diagnostic-v1"
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
    manifest_path: pathlib.Path | None, round_index: int | None
) -> dict | None:
    if manifest_path is None:
        if round_index is not None:
            raise ValueError("--round-index requires --development-manifest")
        return None
    value = load_object(manifest_path)
    if value.get("evidence_scope") != "development-only":
        raise ValueError("development manifest must be development-only")
    for key in ("qualification_used", "shadow_used", "formal_qualification_used"):
        if value.get(key) is not False:
            raise ValueError(f"development manifest used protected evidence: {key}")
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
    if not isinstance(calibration, dict):
        raise ValueError("selected development round lacks calibration evidence")
    selected = calibration.get("calibrated_thresholds")
    grid = calibration.get("calibration_threshold_grid")
    coordinate_rounds = calibration.get("calibration_coordinate_rounds")
    if not isinstance(selected, dict) or not selected:
        raise ValueError("selected development round lacks calibrated thresholds")
    if not isinstance(grid, list) or not grid:
        raise ValueError("selected development round lacks threshold grid")
    return {
        "round": round_index,
        "development_manifest_sha256": sha256_file(manifest_path),
        "formal_selected_thresholds": {
            str(key): finite(value, f"formal threshold {key}")
            for key, value in selected.items()
        },
        "formal_threshold_grid": [
            finite(value, "formal threshold grid") for value in grid
        ],
        "formal_coordinate_rounds": int(coordinate_rounds),
        "formal_calibration_metrics": {
            "frr": finite(calibration["frr"], "formal calibration frr"),
            "far_per_hour": finite(
                calibration["far_per_hour"], "formal calibration far_per_hour"
            ),
        },
        "formal_test_metrics": {
            "frr": finite(row["test"]["frr"], "formal test frr"),
            "far_per_hour": finite(
                row["test"]["far_per_hour"], "formal test far_per_hour"
            ),
        },
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
    parser.add_argument("--round-index", type=int)
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
    formal = development_round_evidence(
        args.development_manifest, args.round_index
    )

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
        )
        test_base, test_domains = evaluate(
            runner=args.runner,
            model=args.model,
            pack=pack,
            references=args.test_references,
            output=trial_root / "test",
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
    min_calibration_frr = min(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["calibration"]["frr"]),
            float(rows[index]["calibration"]["far_per_hour"]),
            rows[index]["threshold"],
        ),
    )
    min_calibration_far = min(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["calibration"]["far_per_hour"]),
            float(rows[index]["calibration"]["frr"]),
            -rows[index]["threshold"],
        ),
    )
    official_order = min(
        range(len(rows)),
        key=lambda index: tuple(rows[index]["official_calibration_behavior_key"]),
    )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "development-only",
        "mode": MODE,
        "diagnostic_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "model_sha256": sha256_file(args.model),
        "tokens_sha256": sha256_file(args.tokens),
        "keywords_sha256": sha256_file(args.keywords),
        "config_sha256": sha256_file(args.config),
        "calibration_references_sha256": sha256_file(args.calibration_references),
        "test_references_sha256": sha256_file(args.test_references),
        "thresholds": thresholds,
        "formal_development_evidence": formal,
        "operating_curve": rows,
        "pareto_thresholds": [rows[index]["threshold"] for index in pareto],
        "min_calibration_frr_threshold": rows[min_calibration_frr]["threshold"],
        "min_calibration_far_threshold": rows[min_calibration_far]["threshold"],
        "official_behavior_order_threshold": rows[official_order]["threshold"],
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
