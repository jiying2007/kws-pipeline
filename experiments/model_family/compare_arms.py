#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def shadow_stats(shadow: dict | None) -> dict:
    if not isinstance(shadow, dict):
        return {
            "qualified": False,
            "failure_count": 999,
            "false_rejects": 999,
            "false_accepts": 999,
            "separation_failures": 999,
        }
    results = [row for row in shadow.get("results", []) if isinstance(row, dict)]
    failures = [row for row in results if not bool(row.get("qualified"))]
    fr = 0
    fa = 0
    sep = 0
    for row in results:
        runtime = row.get("runtime", {})
        if isinstance(runtime, dict):
            fr += int(runtime.get("false_rejects", 0))
            fa += int(runtime.get("false_accepts", 0))
        separation = row.get("surrogate_separation", {})
        if isinstance(separation, dict) and not bool(separation.get("qualified", True)):
            sep += 1
    return {
        "qualified": bool(shadow.get("qualified")),
        "failure_count": len(failures),
        "false_rejects": fr,
        "false_accepts": fa,
        "separation_failures": sep,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-data", required=True, type=pathlib.Path)
    parser.add_argument("--arm-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    shared = load(args.shared_data)
    rows: list[dict] = []

    baseline_path = pathlib.Path(shared.get("baseline", {}).get("shadow_summary", ""))
    if baseline_path.is_file():
        baseline_shadow = load(baseline_path)
        stats = shadow_stats(baseline_shadow)
        rows.append(
            {
                "arm": "rnn-v1-product-baseline",
                "source": "model-training-140",
                **stats,
                "model_bytes": None,
                "parameter_count": None,
                "estimated_macs_per_frame": None,
            }
        )

    for summary_path in sorted(args.arm_root.rglob("arm-summary.json")):
        summary = load(summary_path)
        stats = shadow_stats(summary.get("shadow"))
        rows.append(
            {
                "arm": str(summary["arm"]),
                "source": "model-family-experiment",
                **stats,
                "calibration_gate": bool(summary.get("calibration_gate")),
                "test_gate": bool(summary.get("test_gate")),
                "development_qualification_gate": bool(summary.get("development_qualification_gate")),
                "model_bytes": int(summary.get("model_bytes", 0)),
                "parameter_count": int(summary.get("parameter_count", 0)),
                "estimated_macs_per_frame": int(summary.get("estimated_macs_per_frame", 0)),
                "training_manifest_count": int(summary.get("training_manifest_count", 0)),
                "train_only_seed_count": int(summary.get("train_only_seed_count", 0)),
            }
        )

    def rank(row: dict) -> tuple:
        huge = 10**18
        return (
            0 if bool(row.get("qualified")) else 1,
            int(row.get("failure_count", 999)),
            int(row.get("false_rejects", 999)) + int(row.get("false_accepts", 999)),
            int(row.get("separation_failures", 999)),
            huge if row.get("model_bytes") is None else int(row["model_bytes"]),
            huge if row.get("estimated_macs_per_frame") is None else int(row["estimated_macs_per_frame"]),
            str(row["arm"]),
        )

    ranked = sorted(rows, key=rank)
    result = {
        "schema_version": 1,
        "policy": "model-family-comparison-v1",
        "formal_qualification_used": False,
        "ranking_priority": [
            "shadow_qualified",
            "shadow_failure_count",
            "runtime_false_rejects_plus_false_accepts",
            "surrogate_separation_failures",
            "model_bytes",
            "estimated_macs_per_frame",
        ],
        "rows": ranked,
        "leader": ranked[0]["arm"] if ranked else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
