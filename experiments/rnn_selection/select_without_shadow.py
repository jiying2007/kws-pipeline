#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

POLICY = "shadow-blind-rnn-seed-selection-v1"


def load(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select one RNN optimizer seed using development evidence only."
    )
    parser.add_argument("--probe-root", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    rows: list[dict] = []
    for summary_path in sorted(args.probe_root.rglob("tuning-summary.json")):
        summary = load(summary_path)
        manifest_path = summary_path.parent / "domain-loop-manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing domain-loop manifest beside {summary_path}")
        manifest = load(manifest_path)
        selection = manifest.get("candidate_selection")
        if not isinstance(selection, dict):
            raise ValueError("probe manifest lacks candidate_selection")
        if bool(selection.get("qualification_used_for_selection", True)):
            raise ValueError("RNN seed selection must not use qualification for ranking")
        if bool(summary.get("formal_qualification_used", True)):
            raise ValueError("RNN seed probe touched formal qualification")
        params = summary.get("parameters")
        if not isinstance(params, dict):
            raise ValueError("probe summary lacks parameters")
        if "training_seed_offset" not in params or "training_seed" not in params:
            raise ValueError("probe summary lacks explicit training seed identity")
        eligible = (
            bool(summary.get("calibration_gate"))
            and bool(summary.get("test_gate"))
            and bool(summary.get("development_qualification_gate"))
        )
        rows.append(
            {
                "candidate": str(summary["candidate"]),
                "training_seed_offset": int(params["training_seed_offset"]),
                "training_seed": int(params["training_seed"]),
                "eligible": eligible,
                "development_score": float(selection["selected_score"]),
                "calibration_gate": bool(summary.get("calibration_gate")),
                "test_gate": bool(summary.get("test_gate")),
                "development_qualification_gate": bool(
                    summary.get("development_qualification_gate")
                ),
                # Retain only a diagnostic flag proving shadow may exist; its
                # metrics are deliberately not copied into the selection table.
                "shadow_evidence_present": isinstance(summary.get("shadow"), dict),
            }
        )

    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("no optimizer seed passes all development gates")
    ranked = sorted(
        eligible,
        key=lambda row: (
            float(row["development_score"]),
            int(row["training_seed_offset"]),
            str(row["candidate"]),
        ),
    )
    winner = ranked[0]
    result = {
        "schema_version": 1,
        "policy": POLICY,
        "formal_qualification_used": False,
        "shadow_used_for_selection": False,
        "qualification_used_for_ranking": False,
        "qualification_used_as_gate": True,
        "ranking_metric": "calibration_plus_test_selected_score",
        "rows": sorted(rows, key=lambda row: int(row["training_seed_offset"])),
        "selected_candidate": winner["candidate"],
        "selected_training_seed_offset": winner["training_seed_offset"],
        "selected_training_seed": winner["training_seed"],
        "selected_development_score": winner["development_score"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
