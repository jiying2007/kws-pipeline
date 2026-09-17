#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib

RNN_HEADER_BYTES = 72
GRU_HEADER_BYTES = 80
FLOAT_BYTES = 4


def align4(value: int) -> int:
    return (int(value) + 3) & ~3


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return result


def estimate_model_bytes(family: str, feature_dim: int, hidden_dim: int, vocab_size: int) -> int:
    feature_dim = _positive_int(feature_dim, "feature_dim")
    hidden_dim = _positive_int(hidden_dim, "hidden_dim")
    vocab_size = _positive_int(vocab_size, "vocab_size")
    if family == "rnn":
        blocks = (
            hidden_dim * feature_dim,
            hidden_dim * hidden_dim,
            hidden_dim * FLOAT_BYTES,
            vocab_size * hidden_dim,
            vocab_size * FLOAT_BYTES,
        )
        total = RNN_HEADER_BYTES
    elif family == "gru":
        blocks = (
            3 * hidden_dim * feature_dim,
            3 * hidden_dim * hidden_dim,
            3 * hidden_dim * FLOAT_BYTES,
            3 * hidden_dim * FLOAT_BYTES,
            vocab_size * hidden_dim,
            vocab_size * FLOAT_BYTES,
        )
        total = GRU_HEADER_BYTES
    else:
        raise ValueError(f"unsupported family: {family}")
    for block in blocks:
        total = align4(total)
        total += int(block)
    return total


def estimate_dense_macs_per_step(family: str, feature_dim: int, hidden_dim: int, vocab_size: int) -> int:
    feature_dim = _positive_int(feature_dim, "feature_dim")
    hidden_dim = _positive_int(hidden_dim, "hidden_dim")
    vocab_size = _positive_int(vocab_size, "vocab_size")
    output = vocab_size * hidden_dim
    if family == "rnn":
        return hidden_dim * feature_dim + hidden_dim * hidden_dim + output
    if family == "gru":
        return 3 * hidden_dim * feature_dim + 3 * hidden_dim * hidden_dim + output
    raise ValueError(f"unsupported family: {family}")


def estimate_non_linear_units_per_step(family: str, hidden_dim: int) -> int:
    hidden_dim = _positive_int(hidden_dim, "hidden_dim")
    if family == "rnn":
        return hidden_dim
    if family == "gru":
        return 3 * hidden_dim
    raise ValueError(f"unsupported family: {family}")


def load_contract(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("schema_version", 0)) != 1:
        raise ValueError("resource contract must be schema_version 1")
    if value.get("policy") != "model-family-resource-contract-v1":
        raise ValueError("resource contract policy identity mismatch")
    feature_dim = _positive_int(value.get("feature_dim"), "feature_dim")
    steps_per_second = float(value.get("steps_per_second", 0.0))
    if not math.isfinite(steps_per_second) or steps_per_second <= 0.0:
        raise ValueError("steps_per_second must be finite and positive")
    if feature_dim > 40:
        raise ValueError("feature_dim exceeds current runtime contract")
    vocabularies = value.get("vocabularies")
    candidates = value.get("candidates")
    if not isinstance(vocabularies, list) or not vocabularies:
        raise ValueError("vocabularies must be a non-empty list")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    return value


def build_report(contract: dict) -> dict:
    feature_dim = int(contract["feature_dim"])
    steps_per_second = float(contract["steps_per_second"])
    vocabularies = []
    seen_vocab_names: set[str] = set()
    for raw in contract["vocabularies"]:
        if not isinstance(raw, dict):
            raise ValueError("vocabulary entry must be an object")
        name = str(raw.get("name", ""))
        size = _positive_int(raw.get("size"), f"vocabulary {name}.size")
        if not name or name in seen_vocab_names:
            raise ValueError("vocabulary names must be unique and non-empty")
        seen_vocab_names.add(name)
        vocabularies.append((name, size))

    rows: list[dict] = []
    seen_candidates: set[str] = set()
    for candidate in contract["candidates"]:
        if not isinstance(candidate, dict):
            raise ValueError("candidate entry must be an object")
        name = str(candidate.get("name", ""))
        family = str(candidate.get("family", ""))
        hidden_dim = _positive_int(candidate.get("hidden_dim"), f"candidate {name}.hidden_dim")
        if not name or name in seen_candidates:
            raise ValueError("candidate names must be unique and non-empty")
        if family not in {"rnn", "gru"}:
            raise ValueError(f"candidate {name}: unsupported family {family}")
        if hidden_dim > 64:
            raise ValueError(f"candidate {name}: hidden_dim exceeds current runtime/export bound")
        seen_candidates.add(name)
        for vocab_name, vocab_size in vocabularies:
            macs = estimate_dense_macs_per_step(family, feature_dim, hidden_dim, vocab_size)
            rows.append(
                {
                    "candidate": name,
                    "family": family,
                    "feature_dim": feature_dim,
                    "hidden_dim": hidden_dim,
                    "vocabulary": vocab_name,
                    "vocab_size": vocab_size,
                    "model_bytes_estimate": estimate_model_bytes(family, feature_dim, hidden_dim, vocab_size),
                    "dense_macs_per_step": macs,
                    "dense_mmac_per_s": macs * steps_per_second / 1_000_000.0,
                    "nonlinear_units_per_step": estimate_non_linear_units_per_step(family, hidden_dim),
                    "steps_per_second": steps_per_second,
                }
            )
    return {
        "schema_version": 1,
        "policy": contract["policy"],
        "evidence_class": "static-model-family-resource-estimate",
        "hosted_timing_is_shipping_evidence": False,
        "requires_target_board_measurement_for_shipping": True,
        "rows": rows,
    }


def verify_contract(contract: dict, report: dict) -> None:
    rows = {(row["candidate"], row["vocabulary"]): row for row in report["rows"]}
    assertions = contract.get("reference_assertions", [])
    if not isinstance(assertions, list) or not assertions:
        raise ValueError("reference_assertions must be a non-empty list")
    for item in assertions:
        key = (str(item["candidate"]), str(item["vocabulary"]))
        row = rows.get(key)
        if row is None:
            raise ValueError(f"reference assertion row missing: {key}")
        if "model_bytes_estimate" in item and int(row["model_bytes_estimate"]) != int(item["model_bytes_estimate"]):
            raise ValueError(f"model byte estimate drifted for {key}")
        if "dense_mmac_per_s" in item:
            expected = float(item["dense_mmac_per_s"])
            if not math.isclose(float(row["dense_mmac_per_s"]), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"MMAC/s estimate drifted for {key}: {row['dense_mmac_per_s']} != {expected}")

    # Repository performance docs use this as the canonical arithmetic sizing point.
    baseline = rows.get(("rnn-h48", "full-pinyin-sizing"))
    if baseline is None or not math.isclose(float(baseline["dense_mmac_per_s"]), 1.2, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("RNN-H48 / 420-token baseline must remain 1.2 MMAC/s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Static RNN/GRU model-family resource estimator.")
    parser.add_argument("--contract", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    report = build_report(contract)
    if args.verify:
        verify_contract(contract, report)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
