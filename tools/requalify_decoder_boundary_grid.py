#!/usr/bin/env python3
"""Requalify a retained decoder-policy grid against corrected boundary references.

This tool intentionally does not recalibrate thresholds or retrain a model.
It verifies that one retained development grid, model, and calibrated keyword
pack are hash-consistent, then replays only the corrected boundary corpora at
each retained decoder policy point. The retained row's original `strict`
verdict is immutable input; `joint_strict` requires both that source verdict
and the new boundary acceptance contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path[:0] = [str(TRAINING), str(TOOLS)]

from diagnose_decoder_policy_replay import (  # noqa: E402
    boundary_acceptance,
    boundary_reference_contract,
    compact_boundary_metrics,
)
from iterate_domain import evaluate, safe_reset  # noqa: E402

POLICY = "retained-decoder-grid-boundary-requalification-v1"
SOURCE_GRID_CLASS = "decoder-policy-posterior-replay-grid-development-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def load_json_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def validate_source_grid(
    grid: dict,
    *,
    model_sha256: str,
    pack_sha256: str,
) -> list[dict]:
    if int(grid.get("schema_version", 0)) != 1:
        raise ValueError("source decoder grid schema mismatch")
    if grid.get("evidence_class") != SOURCE_GRID_CLASS:
        raise ValueError("source decoder grid evidence class mismatch")
    if grid.get("development_only") is not True:
        raise ValueError("source decoder grid must remain development-only")
    if grid.get("protected_evidence_used") is not False:
        raise ValueError("source decoder grid must not use protected evidence")
    if grid.get("selection_feedback_allowed") is not False:
        raise ValueError("source decoder grid must forbid selection feedback")
    if str(grid.get("model_sha256") or "") != model_sha256:
        raise ValueError("retained model SHA differs from source decoder grid")

    rows = grid.get("operating_grid")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source decoder grid has no operating points")
    expected_packs = {
        str(row.get("calibrated_pack_sha256") or "")
        for row in rows
        if isinstance(row, dict)
    }
    if expected_packs != {pack_sha256}:
        raise ValueError(
            "retained calibrated pack must exactly match every source grid point"
        )

    seen: set[tuple[float, float]] = set()
    normalized: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"source grid row {index} must be an object")
        blank = finite_number(row.get("blank_retention"), f"row {index}.blank_retention")
        fuzzy = finite_number(
            row.get("fuzzy_child_cost_log"), f"row {index}.fuzzy_child_cost_log"
        )
        if not 0.0 < blank < 1.0:
            raise ValueError("source blank retention must be in (0,1)")
        if not -16.0 <= fuzzy <= 0.0:
            raise ValueError("source fuzzy child cost must be in [-16,0]")
        if type(row.get("strict")) is not bool:
            raise ValueError(f"source grid row {index}.strict must be boolean")
        coordinate = (blank, fuzzy)
        if coordinate in seen:
            raise ValueError(f"duplicate source decoder policy point: {coordinate}")
        seen.add(coordinate)
        normalized.append(row)
    return normalized


def compose_point(
    source_row: dict,
    *,
    within_word_metrics: dict,
    cross_boundary_metrics: dict,
) -> dict:
    acceptance = boundary_acceptance(
        within_word_metrics,
        cross_boundary_metrics,
    )
    boundary = {
        "within_word_pause": compact_boundary_metrics(within_word_metrics),
        "cross_boundary_negative": compact_boundary_metrics(cross_boundary_metrics),
        "acceptance": acceptance,
    }
    source_strict = source_row.get("strict") is True
    return {
        "blank_retention": float(source_row["blank_retention"]),
        "fuzzy_child_cost_log": float(source_row["fuzzy_child_cost_log"]),
        "source_strict": source_strict,
        "source_row_sha256": canonical_sha256(source_row),
        "calibrated_pack_sha256": str(source_row["calibrated_pack_sha256"]),
        "calibrated_thresholds": source_row.get("calibrated_thresholds"),
        "boundary": boundary,
        "joint_strict": source_strict and acceptance["qualified"] is True,
    }


def build(args: argparse.Namespace) -> dict:
    os.chdir(ROOT)

    source_grid_path = args.source_grid.resolve()
    model = args.model.resolve()
    pack = args.pack.resolve()
    within_refs = args.within_word_pause_references.resolve()
    cross_refs = args.cross_boundary_negative_references.resolve()
    runner = args.runner.resolve()
    posterior_dump = args.posterior_dump.resolve()
    decoder_replay = args.decoder_replay.resolve()
    posterior_cache = args.posterior_cache.resolve()
    output = args.output.resolve()

    for path, label in (
        (source_grid_path, "source grid"),
        (model, "model"),
        (pack, "calibrated pack"),
        (within_refs, "within-word references"),
        (cross_refs, "cross-boundary references"),
        (runner, "runner"),
        (posterior_dump, "posterior dump"),
        (decoder_replay, "decoder replay"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    grid = load_json_object(source_grid_path)
    model_sha = sha256_file(model)
    pack_sha = sha256_file(pack)
    source_rows = validate_source_grid(
        grid,
        model_sha256=model_sha,
        pack_sha256=pack_sha,
    )
    boundary_contract = {
        "policy": "corrected-boundary-labels-v1",
        "development_only": True,
        "selection_feedback_allowed": False,
        "within_word_pause": boundary_reference_contract(within_refs, positive=True),
        "cross_boundary_negative": boundary_reference_contract(cross_refs, positive=False),
    }

    work = safe_reset(args.work_dir)
    posterior_cache.mkdir(parents=True, exist_ok=True)
    replay = (posterior_dump, decoder_replay, posterior_cache)

    points: list[dict] = []
    for index, source_row in enumerate(source_rows):
        blank = float(source_row["blank_retention"])
        fuzzy = float(source_row["fuzzy_child_cost_log"])
        trial = work / f"point-{index:02d}-blank-{blank:.4f}-fuzzy-{fuzzy:.4f}"
        within_metrics, _ = evaluate(
            runner=runner,
            model=model,
            pack=pack,
            references=within_refs,
            output=trial / "within-word-pause",
            posterior_replay=replay,
            decoder_blank_retention=blank,
            decoder_fuzzy_child_cost_log=fuzzy,
        )
        cross_metrics, _ = evaluate(
            runner=runner,
            model=model,
            pack=pack,
            references=cross_refs,
            output=trial / "cross-boundary-negative",
            posterior_replay=replay,
            decoder_blank_retention=blank,
            decoder_fuzzy_child_cost_log=fuzzy,
        )
        points.append(
            compose_point(
                source_row,
                within_word_metrics=within_metrics,
                cross_boundary_metrics=cross_metrics,
            )
        )

    summary = {
        "schema_version": 1,
        "evidence_class": "decoder-boundary-retained-grid-requalification-v1",
        "policy": POLICY,
        "development_only": True,
        "protected_evidence_used": False,
        "selection_feedback_allowed": False,
        "source_grid_sha256": sha256_file(source_grid_path),
        "source_grid_evidence_class": grid["evidence_class"],
        "source_grid_policy": grid.get("policy"),
        "source_model_sha256": model_sha,
        "source_calibrated_pack_sha256": pack_sha,
        "boundary_reference_contract": boundary_contract,
        "source_strict_points": sum(1 for point in points if point["source_strict"]),
        "boundary_qualified_points": sum(
            1 for point in points if point["boundary"]["acceptance"]["qualified"] is True
        ),
        "joint_strict_points": sum(1 for point in points if point["joint_strict"]),
        "operating_grid": points,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Requalify one retained decoder grid against corrected boundary refs."
    )
    parser.add_argument("--source-grid", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--pack", required=True, type=pathlib.Path)
    parser.add_argument("--within-word-pause-references", required=True, type=pathlib.Path)
    parser.add_argument("--cross-boundary-negative-references", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    summary = build(args)
    print(
        json.dumps(
            {
                "source_strict_points": summary["source_strict_points"],
                "boundary_qualified_points": summary["boundary_qualified_points"],
                "joint_strict_points": summary["joint_strict_points"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
