#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_training_diagnostics import build  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="training-diagnostics-test-") as tmp:
        root = pathlib.Path(tmp)
        work = root / "work"
        work.mkdir()
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "qualification_holdout_seed": 42,
                    "retired_qualification_holdout_seeds": [40, 41],
                }
            ),
            encoding="utf-8",
        )
        metrics = {
            "expected": 8,
            "matched": 2,
            "false_rejects": 6,
            "false_accepts": 3,
            "frr": 0.75,
            "far_per_hour": 12.5,
            "p95_post_end_latency_ms": 180.0,
            "per_keyword": {
                "1": {
                    "expected": 4,
                    "matched": 0,
                    "false_rejects": 4,
                    "false_accepts": 1,
                    "frr": 1.0,
                },
                "2": {
                    "expected": 4,
                    "matched": 2,
                    "false_rejects": 2,
                    "false_accepts": 2,
                    "frr": 0.5,
                },
            },
            "calibrated_thresholds": {"1": 0.6, "2": 0.55},
            "calibration_threshold_grid": [0.5, 0.55, 0.6],
            "calibration_coordinate_rounds": 2,
            "calibration_coordinate_rounds_executed": 2,
            "calibration_parallel_trials": 2,
            "calibration_operating_curve_path": "build/calibration-operating-curve.json",
            "calibration_operating_curve_sha256": "9" * 64,
            "calibration_operating_curve_summary": {
                "policy": "coordinate-threshold-trials-v1",
                "trial_count": 28,
                "coordinates_executed": 2,
                "grid_saturated": True,
            },
        }
        domains = {
            "worst_domain": "distance:far",
            "worst_domain_score": 9.5,
            "keyword_confusion": {
                "assignment": "global-monotonic-one-to-one-v1",
                "expected_events": 8,
                "correct_keyword": 2,
                "wrong_keyword": 1,
                "missed": 5,
                "matrix": {"1": {"2": 1}},
            },
        }
        manifest = {
            "development_qualified": False,
            "qualification_qualified": None,
            "candidate_selection": {
                "objective_fallback_used": True,
                "objective_best_round": 0,
                "objective_best_frontend": "logmel",
            },
            "records": [
                {
                    "round": 0,
                    "frontend": "logmel",
                    "score": 123.0,
                    "model_sha256": "a" * 64,
                    "calibration_gate": False,
                    "test_gate": False,
                    "calibration": metrics,
                    "test": metrics,
                    "calibration_domains": domains,
                    "test_domains": domains,
                }
            ],
        }
        (work / "domain-loop-manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        eligibility = {
            "schema_version": 1,
            "policy": "selected-refinement-source-signal-v1",
            "eligible": True,
            "source_round": 0,
            "collapsed_keyword_ids": [],
        }
        (work / "base-refinement-eligibility.json").write_text(
            json.dumps(eligibility), encoding="utf-8"
        )
        acoustic = {
            "schema_version": 1,
            "evidence_class": "kws-acoustic-alignment-diagnostic-v1",
            "development_only": True,
            "source_round": 0,
            "source_frontend": "logmel",
            "source_selection_policy": "development-recall-first-refinement-source-v1",
            "source_was_strict": False,
            "model_sha256": "b" * 64,
            "tokens_sha256": "c" * 64,
            "keywords_sha256": "d" * 64,
            "domain_index_sha256": "e" * 64,
            "feature_dump_sha256": "f" * 64,
            "parameter_contract_sha256": "1" * 64,
            "root_start_logit_margin": 0.5,
            "max_recordings_per_keyword_split": 8,
            "model": {"feature_dim": 32, "hidden_dim": 64, "vocab_size": 5},
            "aggregates": {
                "calibration": {
                    "1": {"recordings": 8, "decoder_root_admissible_recordings": 1}
                },
                "test": {
                    "1": {"recordings": 8, "decoder_root_admissible_recordings": 2}
                },
            },
            "records": [{"wav_sha256": "2" * 64}],
        }
        (work / "acoustic-alignment.json").write_text(
            json.dumps(acoustic), encoding="utf-8"
        )

        result = build(config, work)
        assert result["development"]["refinement_eligibility"]["eligible"] is True
        compact_acoustic = result["development"]["acoustic_alignment"]
        assert compact_acoustic["source_round"] == 0
        assert compact_acoustic["aggregates"]["calibration"]["1"]["recordings"] == 8
        assert "records" not in compact_acoustic
        row = result["development"]["rounds"][0]
        operating = row["calibration_operating_point"]
        assert operating["grid_saturated"] is True
        assert operating["edge_keywords"] == {"1": "max"}
        assert operating["selected"] == {"1": 0.6, "2": 0.55}
        assert operating["coordinate_rounds_executed"] == 2
        assert operating["operating_curve_summary"]["trial_count"] == 28
        assert operating["operating_curve_summary"]["grid_saturated"] is True
        assert operating["operating_curve_sha256"] == "9" * 64
        confusion = row["test_domain_summary"]["keyword_confusion"]
        assert confusion["wrong_keyword"] == 1
        assert confusion["missed"] == 5
        assert confusion["matrix"] == {"1": {"2": 1}}
        file_rows = {item["path"]: item for item in result["files"]}
        assert file_rows["base-refinement-eligibility.json"]["present"] is True
        assert file_rows["acoustic-alignment.json"]["present"] is True
        assert result["diagnostic_errors"] == {}

    print("compact training diagnostics: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
