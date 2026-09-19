#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from model_family_resource_estimate import (  # noqa: E402
    estimate_dense_macs_per_step,
    estimate_model_bytes,
)


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def parse_macro(path: pathlib.Path, name: str) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^#define\s+{re.escape(name)}\s+([0-9]+)u?\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{path}: missing {name}")
    return int(match.group(1))


def parse_python_constant(path: pathlib.Path, name: str) -> int:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*([0-9]+)\s*$",
        text,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{path}: missing {name}")
    return int(match.group(1))


def build_report(config: dict) -> dict:
    runtime = config["runtime_contract"]
    header = ROOT / str(runtime["header"])
    exporter = ROOT / str(runtime["exporter"])

    observed = {
        "runtime_max_feature_dim": parse_macro(header, "KWS_MAX_FEATURE_DIM"),
        "runtime_max_hidden_dim": parse_macro(header, "KWS_MAX_HIDDEN_DIM"),
        "runtime_max_vocab_size": parse_macro(header, "KWS_MAX_VOCAB_SIZE"),
        "exporter_max_feature_dim": parse_python_constant(exporter, "MAX_FEATURE_DIM"),
        "exporter_max_hidden_dim": parse_python_constant(exporter, "MAX_HIDDEN_DIM"),
        "exporter_max_vocab_size": parse_python_constant(exporter, "MAX_VOCAB_SIZE"),
    }
    expected = {
        "runtime_max_feature_dim": int(runtime["max_feature_dim"]),
        "runtime_max_hidden_dim": int(runtime["max_hidden_dim"]),
        "runtime_max_vocab_size": int(runtime["max_vocab_size"]),
        "exporter_max_feature_dim": int(runtime["max_feature_dim"]),
        "exporter_max_hidden_dim": int(runtime["max_hidden_dim"]),
        "exporter_max_vocab_size": int(runtime["max_vocab_size"]),
    }
    if observed != expected:
        raise ValueError(f"runtime/exporter contract drift: {observed} != {expected}")

    feature_dim = int(config["shipping_shape"]["feature_dim"])
    vocab_size = int(config["shipping_shape"]["vocab_size"])
    steps_per_second = float(runtime["steps_per_second"])
    target_board = config["target_board_evidence"]
    if bool(target_board["hosted_timing_is_shipping_evidence"]):
        raise ValueError("hosted timing may not be shipping evidence")

    rows = {}
    for name, spec in config["candidates"].items():
        family = str(spec["family"])
        hidden_dim = int(spec["hidden_dim"])
        model_bytes = estimate_model_bytes(
            family,
            feature_dim,
            hidden_dim,
            vocab_size,
        )
        dense_macs_per_step = estimate_dense_macs_per_step(
            family,
            feature_dim,
            hidden_dim,
            vocab_size,
        )
        static_runtime_fit = (
            feature_dim <= observed["runtime_max_feature_dim"]
            and feature_dim <= observed["exporter_max_feature_dim"]
            and hidden_dim <= observed["runtime_max_hidden_dim"]
            and hidden_dim <= observed["exporter_max_hidden_dim"]
            and vocab_size <= observed["runtime_max_vocab_size"]
            and vocab_size <= observed["exporter_max_vocab_size"]
        )
        shipping_evidence_ready = (
            bool(target_board["bound"])
            and bool(target_board["required_for_shipping"])
            and bool(target_board["approved_budget_required"])
            and bool(target_board["physical_measurement_required"])
        )
        rows[name] = {
            "role": spec["role"],
            "family": family,
            "frontend": spec["frontend"],
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab_size,
            "model_bytes_estimate": model_bytes,
            "dense_macs_per_step": dense_macs_per_step,
            "dense_mmac_per_s": dense_macs_per_step * steps_per_second / 1_000_000.0,
            "static_runtime_fit": static_runtime_fit,
            "shipping_evidence_ready": shipping_evidence_ready,
            "shipping_candidate_allowed": (
                static_runtime_fit and shipping_evidence_ready
            ),
        }

    target_name = str(config["decision_rules"]["research_target"])
    target = rows[target_name]
    if not target["static_runtime_fit"]:
        next_action = str(config["decision_rules"]["runtime_mismatch_next"])
    elif not target["shipping_evidence_ready"]:
        next_action = str(config["decision_rules"]["shipping_evidence_missing_next"])
    else:
        next_action = str(config["decision_rules"]["runtime_fit_next"])

    return {
        "schema_version": 1,
        "evidence_class": "kws-v2-research-architecture-fit-report-v1",
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "promotion_allowed": False,
        "shipping_approval_allowed": False,
        "hosted_timing_is_shipping_evidence": False,
        "runtime_contract_observed": observed,
        "candidates": rows,
        "research_target": target_name,
        "research_target_static_runtime_fit": bool(target["static_runtime_fit"]),
        "research_target_shipping_candidate_allowed": bool(
            target["shipping_candidate_allowed"]
        ),
        "next_action": next_action,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    config = load_json(args.config)
    report = build_report(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
