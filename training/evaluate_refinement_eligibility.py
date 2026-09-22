#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib

from adversarial_refinement import select_refinement_source
from verify_product_development_preflight import (
    expected_keyword_ids_from_config,
    load_object,
)

POLICY = "selected-refinement-source-signal-v1"


def _keyword_counts(metrics: object, split: str, keyword_id: str) -> dict:
    if not isinstance(metrics, dict):
        raise ValueError(f"{split} metrics are missing")
    per_keyword = metrics.get("per_keyword")
    if not isinstance(per_keyword, dict):
        raise ValueError(f"{split} per-keyword metrics are missing")
    row = per_keyword.get(keyword_id)
    if not isinstance(row, dict):
        raise ValueError(f"{split} keyword {keyword_id} metrics are missing")
    expected = int(row.get("expected", -1))
    matched = int(row.get("matched", -1))
    false_rejects = int(row.get("false_rejects", -1))
    if (
        expected <= 0
        or matched < 0
        or false_rejects < 0
        or matched + false_rejects != expected
    ):
        raise ValueError(
            f"{split} keyword {keyword_id} counts are invalid: "
            f"expected={expected} matched={matched} false_rejects={false_rejects}"
        )
    return {
        "expected": expected,
        "matched": matched,
        "false_rejects": false_rejects,
    }


def evaluate_refinement_eligibility(
    manifest: dict,
    *,
    expected_keyword_ids: tuple[str, ...],
) -> dict:
    if not expected_keyword_ids or len(set(expected_keyword_ids)) != len(expected_keyword_ids):
        raise ValueError("expected keyword ids must be non-empty and unique")

    source, source_policy = select_refinement_source(manifest)
    keyword_signal: dict[str, dict] = {}
    collapsed: list[str] = []
    for keyword_id in expected_keyword_ids:
        calibration = _keyword_counts(source.get("calibration"), "calibration", keyword_id)
        test = _keyword_counts(source.get("test"), "test", keyword_id)
        matched = int(calibration["matched"]) + int(test["matched"])
        if matched == 0:
            collapsed.append(keyword_id)
        keyword_signal[keyword_id] = {
            "calibration": calibration,
            "test": test,
            "combined_matched": matched,
            "has_signal": matched > 0,
        }

    return {
        "schema_version": 1,
        "policy": POLICY,
        "eligible": not collapsed,
        "source_round": int(source["round"]),
        "source_frontend": str(source["frontend"]),
        "source_selection_policy": source_policy,
        "source_was_strict": (
            source.get("calibration_gate") is True and source.get("test_gate") is True
        ),
        "expected_keyword_ids": list(expected_keyword_ids),
        "collapsed_keyword_ids": collapsed,
        "keyword_signal": keyword_signal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decide whether the recall-first base checkpoint has enough observed "
            "development signal to justify bounded refinement."
        )
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    manifest = load_object(args.manifest.resolve(), "base manifest")
    result = evaluate_refinement_eligibility(
        manifest,
        expected_keyword_ids=expected_keyword_ids_from_config(args.config.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
