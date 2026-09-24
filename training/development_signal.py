"""Development-only non-collapse ordering; not an acoustic qualification gate."""
from __future__ import annotations

import math

POLICY = "per-keyword-per-split-observed-signal-v1"


def selection_enabled(config: dict) -> bool:
    iteration = config.get("domain_iteration", {})
    if not isinstance(iteration, dict):
        raise ValueError("domain_iteration must be an object")
    value = iteration.get("nondegenerate_selection_enabled", False)
    if not isinstance(value, bool):
        raise ValueError("nondegenerate_selection_enabled must be boolean")
    return value


def metric_signal(metrics: dict, keyword_ids: tuple[str, ...]) -> dict:
    if (not keyword_ids or any(not isinstance(k, str) or not k for k in keyword_ids)
            or len(set(keyword_ids)) != len(keyword_ids)):
        raise ValueError("required keyword ids must be non-empty and unique strings")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("per_keyword"), dict):
        raise ValueError("development signal requires per-keyword metrics")
    counts = {}
    for keyword_id in keyword_ids:
        row = metrics["per_keyword"].get(keyword_id)
        if not isinstance(row, dict):
            raise ValueError(f"development signal missing keyword {keyword_id}")
        values = {name: row.get(name) for name in ("expected", "matched", "false_rejects")}
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in values.values()):
            raise ValueError(f"keyword {keyword_id}: counts must be non-negative integers")
        if values["expected"] <= 0 or values["matched"] + values["false_rejects"] != values["expected"]:
            raise ValueError(f"keyword {keyword_id}: inconsistent or absent positive support")
        counts[keyword_id] = values
    collapsed = [key for key in keyword_ids if counts[key]["matched"] == 0]
    return {"nondegenerate": not collapsed, "collapsed_keyword_ids": collapsed, "counts": counts}


def record_signal(record: dict, keyword_ids: tuple[str, ...]) -> dict:
    splits = {split: metric_signal(record.get(split), keyword_ids) for split in ("calibration", "test")}
    collapsed = [f"{split}:{key}" for split, row in splits.items() for key in row["collapsed_keyword_ids"]]
    return {"policy": POLICY, "nondegenerate": not collapsed,
            "collapsed_keyword_splits": collapsed, "splits": splits}


def record_rank(record: dict, keyword_ids: tuple[str, ...] | None = None) -> tuple[int, float]:
    score = record.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError("development score must be finite numeric")
    collapsed = 0 if keyword_ids is None else len(record_signal(record, keyword_ids)["collapsed_keyword_splits"])
    return collapsed, float(score)


def selection_report(records: list[dict], selected: dict, keyword_ids: tuple[str, ...]) -> dict:
    signals = [record_signal(row, keyword_ids) for row in records]
    selected_signal = record_signal(selected, keyword_ids)
    count = sum(row["nondegenerate"] is True for row in signals)
    if count and selected_signal["nondegenerate"] is not True:
        raise ValueError("a collapsed diagnostic source displaced a nondegenerate candidate")
    identity = {key: selected[key] for key in ("round", "frontend", "model_sha256")}
    return {
        "policy": POLICY, "development_only": True, "release_authority": False,
        "minimum_observed_matches_per_keyword_per_split": 1,
        "required_keyword_ids": list(keyword_ids),
        "nondegenerate_candidate_count": count,
        "selected_candidate": identity if count else None,
        "diagnostic_source": identity,
        "best_artifact_role": "development-only-candidate" if count else "diagnostic-only-no-nondegenerate-candidate",
        "selected_signal": selected_signal,
    }
