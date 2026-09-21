#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from domain_progress import append_round_progress, build_round_progress  # noqa: E402
from product_preflight_handoff import (  # noqa: E402
    pack_handoff,
    restore_handoff,
    sha256_file,
    verify_materialization,
)
from verify_product_development_preflight import (  # noqa: E402
    expected_keyword_ids_from_config,
    verify,
)


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



def validate_split_job_handoff() -> None:
    head_sha = "a" * 40
    base_sha = "b" * 40
    with tempfile.TemporaryDirectory(prefix="product-preflight-handoff-test-") as tmp:
        root = pathlib.Path(tmp)
        request = root / ".github/triggers/model-training-request.json"
        effective = root / ".generated/xiaowo.product-effective.json"
        config = root / ".generated/xiaowo.product-preflight.json"
        work = root / "build/model-training-preflight"
        for path, payload in (
            (request, '{"request_id":"fixture"}\n'),
            (effective, '{"effective":"fixture"}\n'),
            (config, '{"preflight":"fixture"}\n'),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")

        candidate = work / "candidates/r00-logmel-00"
        checkpoint = candidate / "model.pt"
        model = candidate / "model.kwm"
        provenance = candidate / "model.kwm.provenance.json"
        keywords = candidate / "calibration/calibrated-keywords.tsv"
        pack = candidate / "calibration/calibrated-keywords.kwk"
        cal_fp = candidate / "calibration/final-eval/false-positives.jsonl"
        cal_fr = candidate / "calibration/final-eval/false-rejects.jsonl"
        test_fp = candidate / "test/false-positives.jsonl"
        test_fr = candidate / "test/false-rejects.jsonl"
        domain_index = work / "datasets/round-00/domain-index.jsonl"
        audit = work / "datasets/round-00/audit.json"
        curriculum = work / "curriculum/round-00.json"
        clean_cache = (
            work
            / "hard-negative-replay/.clean-command-tts-cache/aa/cache.wav"
        )
        for path, payload in (
            (checkpoint, b"checkpoint"),
            (model, b"model"),
            (provenance, b'{"provider":"fixture"}\n'),
            (keywords, b"1\twake\t0.5\ta b\n"),
            (pack, b"pack"),
            (cal_fp, b""),
            (cal_fr, b""),
            (test_fp, b""),
            (test_fr, b""),
            (domain_index, b'{"split":"calibration"}\n'),
            (audit, b'{"ok":true}\n'),
            (curriculum, b'{"schema_version":1}\n'),
            (clean_cache, b"RIFFfixture"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

        record_metrics = {
            **metrics(),
            "false_positives_path": str(cal_fp),
            "false_rejects_path": str(cal_fr),
        }
        test_metrics = {
            **metrics(),
            "false_positives_path": str(test_fp),
            "false_rejects_path": str(test_fr),
        }
        manifest = {
            "qualification_deferred": True,
            "qualification_qualified": None,
            "records": [
                {
                    "round": 0,
                    "frontend": "logmel",
                    "candidate": 0,
                    "score": 1.0,
                    "model": str(model),
                    "model_sha256": sha256_file(model),
                    "checkpoint": str(checkpoint),
                    "provenance": str(provenance),
                    "provenance_sha256": sha256_file(provenance),
                    "keywords": str(keywords),
                    "pack": str(pack),
                    "calibration": record_metrics,
                    "test": test_metrics,
                    "calibration_gate": False,
                    "test_gate": False,
                }
            ],
        }
        manifest_path = work / "domain-loop-manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (work / "domain-loop-progress.jsonl").write_text(
            '{"round":0}\n', encoding="utf-8"
        )

        archive = root / "handoff/base-stage.tar"
        metadata = pack_handoff(
            work_dir=work,
            config_path=config,
            effective_config_path=effective,
            request_path=request,
            output_path=archive,
            head_sha=head_sha,
            base_sha=base_sha,
            repo_root=root,
        )
        assert metadata["policy"] == "product-development-preflight-job-handoff-v1"
        assert any(
            row["path"].endswith(".clean-command-tts-cache/aa/cache.wav")
            for row in metadata["files"]
        )

        shutil.rmtree(work)
        config.unlink()
        restored = restore_handoff(
            archive_path=archive,
            head_sha=head_sha,
            base_sha=base_sha,
            repo_root=root,
        )
        assert restored["development_manifest_sha256"] == metadata["development_manifest_sha256"]
        assert checkpoint.read_bytes() == b"checkpoint"
        assert clean_cache.read_bytes() == b"RIFFfixture"
        metadata_path = root / "build/product-preflight-handoff/manifest.json"
        verify_materialization(
            metadata_path=metadata_path,
            effective_config_path=effective,
            request_path=request,
            head_sha=head_sha,
            base_sha=base_sha,
            repo_root=root,
        )

        effective.write_text('{"effective":"drift"}\n', encoding="utf-8")
        try:
            verify_materialization(
                metadata_path=metadata_path,
                effective_config_path=effective,
                request_path=request,
                head_sha=head_sha,
                base_sha=base_sha,
                repo_root=root,
            )
        except ValueError as exc:
            assert "materialization changed" in str(exc)
        else:
            raise AssertionError("cross-job materialization drift was accepted")


def main() -> int:
    validate_split_job_handoff()
    with tempfile.TemporaryDirectory(prefix="product-preflight-test-") as tmp:
        root = pathlib.Path(tmp)
        work = root / "work"
        work.mkdir()
        base_path = work / "domain-loop-manifest.json"
        keyword_fixture = root / "keywords.tsv"
        keyword_fixture.write_text(
            "0\twake-zero\t0.5\ta b\n"
            "1\twake-one\t0.5\tb c\n"
            "2\twake-two\t0.5\tc d\n",
            encoding="utf-8",
        )
        config_fixture = root / "preflight.json"
        config_fixture.write_text(
            json.dumps({"keywords": str(keyword_fixture)}),
            encoding="utf-8",
        )
        assert expected_keyword_ids_from_config(config_fixture) == ("0", "1", "2")

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
            expected_keyword_ids=("1", "2"),
        )
        assert result["passed"] is True
        assert result["expected_keyword_ids"] == ["1", "2"]
        assert result["base_epochs"] == [12, 6]
        assert len(result["base_round_metrics"]) == 2
        assert result["base_round_metrics"][0]["test"]["per_keyword"]["1"]["matched"] == 0
        assert result["refinement_source_round"] == -1
        assert result["formal_qualification_used"] is False

        three_keyword_metrics = metrics()
        three_keyword_metrics["per_keyword"]["3"] = {
            "expected": 4,
            "matched": 2,
            "false_rejects": 2,
            "false_accepts": 0,
            "frr": 0.5,
        }
        three_keyword_metrics["expected"] += 4
        three_keyword_metrics["matched"] += 2
        three_keyword_metrics["false_rejects"] += 2
        three_keyword_metrics["frr"] = (
            three_keyword_metrics["false_rejects"] / three_keyword_metrics["expected"]
        )
        three_base = copy.deepcopy(base)
        for row in three_base["records"]:
            row["calibration"] = copy.deepcopy(three_keyword_metrics)
            row["test"] = copy.deepcopy(three_keyword_metrics)
        three_refinement = copy.deepcopy(refinement)
        three_refinement["record"]["calibration"] = copy.deepcopy(three_keyword_metrics)
        three_refinement["record"]["test"] = copy.deepcopy(three_keyword_metrics)
        base_path.write_text(json.dumps(three_base), encoding="utf-8")
        refinement_path.write_text(json.dumps(three_refinement), encoding="utf-8")
        three_result = verify(
            base_manifest_path=base_path,
            refinement_summary_path=refinement_path,
            work_dir=work,
            expected_keyword_ids=("1", "2", "3"),
        )
        assert three_result["expected_keyword_ids"] == ["1", "2", "3"]
        missing_third = copy.deepcopy(three_refinement)
        del missing_third["record"]["test"]["per_keyword"]["3"]
        refinement_path.write_text(json.dumps(missing_third), encoding="utf-8")
        try:
            verify(
                base_manifest_path=base_path,
                refinement_summary_path=refinement_path,
                work_dir=work,
                expected_keyword_ids=("1", "2", "3"),
            )
        except ValueError as exc:
            assert "keyword 3 metrics are missing" in str(exc)
        else:
            raise AssertionError("configured third keyword was not enforced")

        base_path.write_text(json.dumps(base), encoding="utf-8")
        refinement_path.write_text(json.dumps(refinement), encoding="utf-8")

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
                expected_keyword_ids=("1", "2"),
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
                expected_keyword_ids=("1", "2"),
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
    assert "product-development-base-preflight:" in workflow
    assert "product-development-refinement-preflight:" in workflow
    assert workflow.count("timeout-minutes: 120") == 2
    assert "product_preflight_handoff.py pack" in workflow
    assert "product_preflight_handoff.py restore" in workflow
    assert "product_preflight_handoff.py verify-materialization" in workflow
    assert "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c" in workflow
    assert "xiaowo-product-development-base-handoff-" in workflow
    assert '--config "$KWS_PREFLIGHT_CONFIG"' in workflow

    print("product development preflight guard: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
