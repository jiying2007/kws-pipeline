#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from domain_progress import append_round_progress, build_round_progress  # noqa: E402
from verify_product_development_preflight import verify  # noqa: E402


def metrics(matched1: int = 2, matched2: int = 2) -> dict:
    expected = 4
    rows = {}
    for keyword_id, matched in (("1", matched1), ("2", matched2)):
        false_rejects = expected - matched
        rows[keyword_id] = {
            "expected": expected,
            "matched": matched,
            "false_rejects": false_rejects,
            "false_accepts": 0,
            "frr": false_rejects / expected,
        }
    return {
        "expected": expected * 2,
        "matched": matched1 + matched2,
        "false_rejects": expected * 2 - matched1 - matched2,
        "false_accepts": 0,
        "frr": (expected * 2 - matched1 - matched2) / (expected * 2),
        "far_per_hour": 100.0,
        "per_keyword": rows,
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="product-preflight-test-") as tmp:
        root = pathlib.Path(tmp)
        work = root / "work"
        work.mkdir()
        base_path = work / "domain-loop-manifest.json"
        refinement_path = work / "adversarial-refinement" / "preflight-summary.json"
        refinement_path.parent.mkdir(parents=True)

        base = {
            "qualification_deferred": True,
            "qualification_qualified": None,
            "records": [
                {
                    "round": 0,
                    "training_epochs": 12,
                    "score": 10.0,
                    "calibration": metrics(0, 2),
                    "test": metrics(0, 1),
                    "calibration_gate": False,
                    "test_gate": False,
                },
                {
                    "round": 1,
                    "training_epochs": 6,
                    "score": 8.0,
                    "calibration": metrics(2, 2),
                    "test": metrics(1, 3),
                    "calibration_gate": False,
                    "test_gate": False,
                },
            ],
        }
        refinement = {
            "schema_version": 1,
            "policy": "product-development-refinement-preflight-v1",
            "development_only": True,
            "formal_qualification_used": False,
            "qualification_used": False,
            "shadow_used": False,
            "refinement_epochs": 6,
            "strict_dual_pass": False,
            "wake_balance": {
                "schema_version": 3,
                "policy": "per-keyword-provenance-pressure-balance-v3",
                "explicit_focus_nonwake_rows": 8,
            },
            "record": {
                "calibration": metrics(),
                "test": metrics(1, 3),
            },
        }
        base_path.write_text(json.dumps(base), encoding="utf-8")
        refinement_path.write_text(json.dumps(refinement), encoding="utf-8")

        result = verify(
            base_manifest_path=base_path,
            refinement_summary_path=refinement_path,
            work_dir=work,
        )
        assert result["passed"] is True
        assert result["base_epochs"] == [12, 6]
        assert len(result["base_round_metrics"]) == 2
        assert result["base_round_metrics"][0]["test"]["per_keyword"]["1"]["matched"] == 0
        assert result["refinement_source_round"] == -1
        assert result["formal_qualification_used"] is False

        progress = build_round_progress(
            {
                "round": 1,
                "frontend": "logmel",
                "candidate": 0,
                "score": 8.0,
                "calibration_gate": False,
                "test_gate": False,
                "training_epochs": 6,
                "training_learning_rate": 0.0005,
                "training_seed": 2026,
                "calibration": metrics(2, 2),
                "test": metrics(1, 3),
            },
            curriculum_sha256="a" * 64,
        )
        assert progress["training_epochs"] == 6
        assert progress["test"]["per_keyword"]["1"]["matched"] == 1
        progress_path = work / "domain-loop-progress.jsonl"
        append_round_progress(progress_path, progress)
        append_round_progress(progress_path, progress)
        progress_rows = [
            json.loads(line)
            for line in progress_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(progress_rows) == 2
        assert progress_rows[-1]["curriculum_sha256"] == "a" * 64

        collapsed = copy.deepcopy(refinement)
        collapsed["record"]["test"] = metrics(0, 3)
        refinement_path.write_text(json.dumps(collapsed), encoding="utf-8")
        try:
            verify(
                base_manifest_path=base_path,
                refinement_summary_path=refinement_path,
                work_dir=work,
            )
        except ValueError as exc:
            assert "keyword 1 collapsed" in str(exc)
        else:
            raise AssertionError("100% keyword collapse passed preflight")

        refinement_path.write_text(json.dumps(refinement), encoding="utf-8")
        leaked = work / "qualification-dataset"
        leaked.mkdir()
        try:
            verify(
                base_manifest_path=base_path,
                refinement_summary_path=refinement_path,
                work_dir=work,
            )
        except ValueError as exc:
            assert "crossed qualification boundary" in str(exc)
        else:
            raise AssertionError("qualification leakage passed preflight")

    workflow = (ROOT / ".github/workflows/model-training-preflight.yml").read_text(
        encoding="utf-8"
    )
    assert "--compact-log" in workflow
    assert "domain-loop-progress.jsonl" in workflow

    print("product development preflight guard: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
