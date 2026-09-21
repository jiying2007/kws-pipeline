#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib


def compact_metrics(metrics: dict) -> dict:
    if not isinstance(metrics, dict):
        raise TypeError("metrics must be an object")
    required = (
        "expected",
        "matched",
        "false_rejects",
        "false_accepts",
        "frr",
        "far_per_hour",
    )
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"metrics missing required fields: {', '.join(missing)}")
    raw_per_keyword = metrics.get("per_keyword", {})
    if not isinstance(raw_per_keyword, dict):
        raise ValueError("metrics.per_keyword must be an object")
    per_keyword = {}
    for keyword_id, values in sorted(raw_per_keyword.items(), key=lambda item: str(item[0])):
        if not isinstance(values, dict):
            raise ValueError(f"metrics.per_keyword[{keyword_id}] must be an object")
        per_keyword[str(keyword_id)] = {
            key: values[key]
            for key in (
                "expected",
                "matched",
                "false_rejects",
                "false_accepts",
                "frr",
            )
            if key in values
        }
    result = {key: metrics[key] for key in required}
    if "p95_post_end_latency_ms" in metrics:
        result["p95_post_end_latency_ms"] = metrics["p95_post_end_latency_ms"]
    result["per_keyword"] = per_keyword
    return result


def build_round_progress(record: dict, *, curriculum_sha256: str) -> dict:
    if not isinstance(record, dict):
        raise TypeError("round record must be an object")
    result = {
        "schema_version": 1,
        "round": record["round"],
        "frontend": record.get("frontend"),
        "candidate": record.get("candidate"),
        "score": record["score"],
        "calibration_gate": record.get("calibration_gate") is True,
        "test_gate": record.get("test_gate") is True,
        "calibration": compact_metrics(record["calibration"]),
        "test": compact_metrics(record["test"]),
        "curriculum_sha256": curriculum_sha256,
    }
    for key in (
        "training_epochs",
        "training_learning_rate",
        "training_seed",
        "warm_started",
        "warm_start_strategy",
        "hard_negative_replay_examples",
    ):
        if key in record:
            result[key] = record[key]
    return result


def append_round_progress(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
