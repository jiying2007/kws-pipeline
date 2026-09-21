#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib

PREFLIGHT_POLICY = "product-development-refinement-preflight-v1"
WAKE_BALANCE_POLICY = "per-keyword-provenance-pressure-balance-v3"
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPECTED_BASE_EPOCHS = (12, 6)
EXPECTED_REFINEMENT_EPOCHS = 6


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def expected_keyword_ids_from_config(config_path: pathlib.Path) -> tuple[str, ...]:
    config = load_object(config_path, "preflight config")
    raw_keywords = config.get("keywords")
    if not isinstance(raw_keywords, str) or not raw_keywords.strip():
        raise ValueError("preflight config keywords path is missing")
    keywords_path = pathlib.Path(raw_keywords)
    if not keywords_path.is_absolute():
        keywords_path = (ROOT / keywords_path).resolve()
    rows: list[str] = []
    for line_no, raw in enumerate(keywords_path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        parts = raw.split("\t")
        if not parts or not parts[0].strip():
            raise ValueError(f"keywords line {line_no} is missing an id")
        try:
            keyword_id = int(parts[0].strip())
        except ValueError as exc:
            raise ValueError(f"keywords line {line_no} id is invalid") from exc
        if keyword_id <= 0:
            raise ValueError(f"keywords line {line_no} id must be positive")
        rows.append(str(keyword_id))
    if not rows:
        raise ValueError("preflight keyword set is empty")
    if len(set(rows)) != len(rows):
        raise ValueError("preflight keyword ids must be unique")
    return tuple(rows)


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_metrics(
    metrics: object,
    label: str,
    *,
    expected_keywords: tuple[str, ...],
    require_noncollapse: bool = True,
) -> dict:
    if not isinstance(metrics, dict):
        raise ValueError(f"{label} metrics are missing")
    frr = finite(metrics.get("frr"), f"{label}.frr")
    far = finite(metrics.get("far_per_hour"), f"{label}.far_per_hour")
    if not 0.0 <= frr <= 1.0 or far < 0.0:
        raise ValueError(f"{label} aggregate metrics are invalid")
    per_keyword = metrics.get("per_keyword")
    if not isinstance(per_keyword, dict):
        raise ValueError(f"{label} per-keyword metrics are missing")
    result: dict[str, dict] = {}
    for keyword_id in expected_keywords:
        row = per_keyword.get(keyword_id)
        if not isinstance(row, dict):
            raise ValueError(f"{label} keyword {keyword_id} metrics are missing")
        expected = int(row.get("expected", -1))
        matched = int(row.get("matched", -1))
        false_rejects = int(row.get("false_rejects", -1))
        keyword_frr = finite(row.get("frr"), f"{label}.keyword.{keyword_id}.frr")
        invalid = (
            expected <= 0
            or matched < 0
            or false_rejects < 0
            or matched + false_rejects != expected
            or not 0.0 <= keyword_frr <= 1.0
        )
        collapsed = matched == 0 or math.isclose(
            keyword_frr, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        )
        if invalid or (require_noncollapse and collapsed):
            suffix = " collapsed" if collapsed else " invalid"
            raise ValueError(
                f"{label} keyword {keyword_id}{suffix}: "
                f"expected={expected} matched={matched} false_rejects={false_rejects} "
                f"frr={keyword_frr}"
            )
        result[keyword_id] = {
            "expected": expected,
            "matched": matched,
            "false_rejects": false_rejects,
            "frr": keyword_frr,
        }
    return {
        "frr": frr,
        "far_per_hour": far,
        "per_keyword": result,
    }


def compact_base_round_metrics(
    records: list[dict],
    *,
    expected_keywords: tuple[str, ...],
) -> list[dict]:
    result: list[dict] = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("preflight base record must be an object")
        item = {
            "round": int(row.get("round", -1)),
            "score": finite(row.get("score"), "base.score"),
            "calibration": validate_metrics(
                row.get("calibration"),
                "base.calibration",
                expected_keywords=expected_keywords,
                require_noncollapse=False,
            ),
            "test": validate_metrics(
                row.get("test"),
                "base.test",
                expected_keywords=expected_keywords,
                require_noncollapse=False,
            ),
            "calibration_gate": bool(row.get("calibration_gate")),
            "test_gate": bool(row.get("test_gate")),
        }
        result.append(item)
    return result


def verify(
    *,
    base_manifest_path: pathlib.Path,
    refinement_summary_path: pathlib.Path,
    work_dir: pathlib.Path,
    expected_keyword_ids: tuple[str, ...],
) -> dict:
    base = load_object(base_manifest_path, "base manifest")
    refinement = load_object(refinement_summary_path, "refinement preflight")

    if base.get("qualification_deferred") is not True:
        raise ValueError("preflight base must defer qualification")
    if base.get("qualification_qualified") is not None:
        raise ValueError("preflight base unexpectedly contains qualification result")
    records = base.get("records")
    if not isinstance(records, list) or len(records) != 2:
        raise ValueError("preflight base must contain exactly two development rounds")
    if not expected_keyword_ids or len(set(expected_keyword_ids)) != len(expected_keyword_ids):
        raise ValueError("expected keyword ids must be non-empty and unique")
    base_round_metrics = compact_base_round_metrics(
        records,
        expected_keywords=expected_keyword_ids,
    )
    epochs = tuple(int(row.get("training_epochs", -1)) for row in records)
    if epochs != EXPECTED_BASE_EPOCHS:
        raise ValueError(f"preflight base epoch budget drifted: {epochs}")

    if (
        int(refinement.get("schema_version", 0)) != 1
        or refinement.get("policy") != PREFLIGHT_POLICY
        or refinement.get("development_only") is not True
        or refinement.get("formal_qualification_used") is not False
        or refinement.get("qualification_used") is not False
        or refinement.get("shadow_used") is not False
    ):
        raise ValueError("refinement preflight evidence boundary drifted")
    if int(refinement.get("refinement_epochs", -1)) != EXPECTED_REFINEMENT_EPOCHS:
        raise ValueError("refinement preflight epoch budget drifted")

    wake_balance = refinement.get("wake_balance")
    if (
        not isinstance(wake_balance, dict)
        or int(wake_balance.get("schema_version", 0)) != 3
        or wake_balance.get("policy") != WAKE_BALANCE_POLICY
        or int(wake_balance.get("explicit_focus_nonwake_rows", 0)) <= 0
    ):
        raise ValueError("refinement preflight did not exercise provenance-first wake pressure")

    record = refinement.get("record")
    if not isinstance(record, dict):
        raise ValueError("refinement preflight record is missing")
    calibration = validate_metrics(
        record.get("calibration"),
        "calibration",
        expected_keywords=expected_keyword_ids,
    )
    test = validate_metrics(
        record.get("test"),
        "test",
        expected_keywords=expected_keyword_ids,
    )

    forbidden = (
        work_dir / "development-qualification-mining",
        work_dir / "qualification-dataset",
        work_dir / "shadow-qualification",
        work_dir / "formal-preflight.json",
    )
    leaked = [str(path) for path in forbidden if path.exists()]
    if leaked:
        raise ValueError(
            "development preflight crossed qualification boundary: " + ", ".join(leaked)
        )

    return {
        "schema_version": 1,
        "policy": "product-development-preflight-guard-v1",
        "passed": True,
        "expected_keyword_ids": list(expected_keyword_ids),
        "base_rounds": len(records),
        "base_epochs": list(epochs),
        "base_round_metrics": base_round_metrics,
        "refinement_source_round": int(refinement.get("source_round", -1)),
        "refinement_source_was_strict": bool(refinement.get("source_was_strict")),
        "refinement_epochs": int(refinement["refinement_epochs"]),
        "strict_dual_pass": bool(refinement.get("strict_dual_pass")),
        "calibration": calibration,
        "test": test,
        "formal_qualification_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--refinement-summary", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    result = verify(
        base_manifest_path=args.base_manifest.resolve(),
        refinement_summary_path=args.refinement_summary.resolve(),
        work_dir=args.work_dir.resolve(),
        expected_keyword_ids=expected_keyword_ids_from_config(args.config.resolve()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
