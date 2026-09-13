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
    fr = fa = sep = 0
    for row in results:
        qualification = row.get("qualification", {})
        if isinstance(qualification, dict):
            fr += int(qualification.get("false_rejects", 0))
            fa += int(qualification.get("false_accepts", 0))
        if not bool(row.get("surrogate_separation_qualified", True)):
            sep += 1
    return {
        "qualified": bool(shadow.get("qualified")),
        "failure_count": len(failures),
        "false_rejects": fr,
        "false_accepts": fa,
        "separation_failures": sep,
    }


def rank(row: dict) -> tuple:
    return (
        0 if bool(row.get("qualified")) else 1,
        int(row.get("failure_count", 999)),
        int(row.get("false_rejects", 999)) + int(row.get("false_accepts", 999)),
        int(row.get("separation_failures", 999)),
        int(row.get("model_bytes", 10**18)),
        str(row.get("candidate", row.get("arm", ""))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True, type=pathlib.Path)
    parser.add_argument("--reference-comparison", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    rows: list[dict] = []
    reference = load(args.reference_comparison)
    for arm in ("gru-64", "rnn-multiseed-terminal"):
        matched = [row for row in reference.get("rows", []) if row.get("arm") == arm]
        if len(matched) != 1:
            raise ValueError(f"exactly one {arm} reference row is required")
        row = dict(matched[0])
        row["candidate"] = f"{arm}-reference"
        row["source"] = "model-family-ab-6"
        rows.append(row)

    for path in sorted(args.candidate_root.rglob("tuning-summary.json")):
        summary = load(path)
        stats = shadow_stats(summary.get("shadow"))
        rows.append(
            {
                "candidate": str(summary["candidate"]),
                "source": "gru-tuning-ab",
                **stats,
                "calibration_gate": bool(summary.get("calibration_gate")),
                "test_gate": bool(summary.get("test_gate")),
                "development_qualification_gate": bool(summary.get("development_qualification_gate")),
                "parameters": dict(summary.get("parameters", {})),
                "model_bytes": int(summary.get("model_bytes", 0)),
                "parameter_count": int(summary.get("parameter_count", 0)),
                "estimated_macs_per_frame": int(summary.get("estimated_macs_per_frame", 0)),
                "training_manifest_count": int(summary.get("training_manifest_count", 0)),
            }
        )

    ranked = sorted(rows, key=rank)
    result = {
        "schema_version": 1,
        "policy": "focused-gru-tuning-comparison-v1",
        "formal_qualification_used": False,
        "ranking_priority": [
            "shadow_qualified",
            "shadow_failure_count",
            "runtime_false_rejects_plus_false_accepts",
            "surrogate_separation_failures",
            "model_bytes",
        ],
        "leader": ranked[0].get("candidate") if ranked else None,
        "rows": ranked,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
