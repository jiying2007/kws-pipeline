#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def row(name: str, band: str, expected: list[dict], *, azimuth: float | None = None) -> dict:
    return {
        "recording": name,
        "path": f"{name}.wav",
        "duration_s": 10.0,
        "expected": expected,
        "domain": {
            "distance_band": band,
            "distance_m": 0.5 if band == "near" else 4.5,
            "azimuth_deg": (0.0 if band == "near" else 90.0) if azimuth is None else azimuth,
            "rt60_s": 0.2 if band == "near" else 0.7,
            "snr_db": 20.0 if band == "near" else 2.0,
            "noise_profile": "fan" if band == "near" else "motor",
            "playback_sir_db": None if band == "near" else 5.0,
        },
    }


def gate_metric(
    *,
    expected: int = 4,
    negative_hours: float = 0.01,
    frr: float = 0.0,
    far: float = 0.0,
) -> dict:
    return {
        "expected": expected,
        "negative_audio_hours": negative_hours,
        "frr": frr,
        "far_per_hour": far,
        "wake_rate": 1.0 - frr,
    }


def run_gate_case(
    root: pathlib.Path,
    summary: dict,
    config: dict,
    name: str,
) -> tuple[int, dict]:
    summary_path = root / f"{name}-summary.json"
    config_path = root / f"{name}-config.json"
    output_path = root / f"{name}-out.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "eval" / "gate_robustness.py"),
            "--summary",
            str(summary_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        check=False,
    )
    return completed.returncode, json.loads(output_path.read_text(encoding="utf-8"))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        refs = root / "refs.jsonl"
        dets = root / "dets.jsonl"
        out = root / "domains.json"
        references = [
            row("near", "near", [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]),
            row("far", "far", [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]),
            row("far-bg", "far", [], azimuth=180.0),
            # Two overlapping expected windows validate that one detection can be
            # consumed by only one event in the confusion assignment.
            row(
                "overlap",
                "near",
                [
                    {"keyword_id": 1, "start_s": 4.0, "end_s": 5.0},
                    {"keyword_id": 2, "start_s": 4.1, "end_s": 5.1},
                ],
            ),
        ]
        refs.write_text("\n".join(json.dumps(item) for item in references) + "\n", encoding="utf-8")
        detections = [
            {"recording": "near", "keyword_id": 1, "time_s": 2.1, "confidence": 0.9},
            {"recording": "far", "keyword_id": 2, "time_s": 2.1, "confidence": 0.8},
            {"recording": "far-bg", "keyword_id": 1, "time_s": 5.0, "confidence": 0.7},
            {"recording": "overlap", "keyword_id": 2, "time_s": 5.1, "confidence": 0.95},
        ]
        dets.write_text("\n".join(json.dumps(item) for item in detections) + "\n", encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "domain_metrics.py"),
                "--references",
                str(refs),
                "--detections",
                str(dets),
                "--output",
                str(out),
            ]
        )
        result = json.loads(out.read_text(encoding="utf-8"))
        assert result["schema_version"] == 3
        assert result["domains"]["distance:near"]["frr"] > 0.0
        assert result["domains"]["distance:far"]["frr"] == 1.0
        assert result["domains"]["distance_bin:0.5m"]["expected"] == 3
        assert result["domains"]["distance_bin:5m"]["false_accepts"] == 2
        assert result["domains"]["azimuth_deg:0"]["expected"] == 3
        assert result["domains"]["azimuth_deg:180"]["negative_audio_hours"] > 0.0
        assert result["domains"]["snr:critical"]["false_accepts"] == 2
        assert result["domains"]["snr:mid"]["expected"] == 3
        assert 0.0 <= result["domains"]["distance_bin:0.5m"]["wake_rate"] <= 1.0
        assert result["domains"]["azimuth:rear"]["expected"] == 0
        assert result["domains"]["azimuth:rear"]["negative_audio_hours"] > 0.0
        assert result["domains"]["azimuth:rear"]["false_accepts"] == 1
        assert result["worst_domain"] is not None
        assert result["slice_contract"]["azimuth_quantization_deg"] == 30
        confusion = result["keyword_confusion"]
        assert confusion["assignment"] == "global-monotonic-one-to-one-v1"
        assert confusion["expected_events"] == 4
        assert confusion["correct_keyword"] == 2
        assert confusion["wrong_keyword"] == 1
        assert confusion["missed"] == 1
        assert confusion["matrix"]["1"]["2"] >= 1

        gate_config = {
            "robustness_gates": {
                "max_frr": 0.0,
                "max_far_per_hour": 0.0,
                "min_expected_wakes": 4,
                "min_negative_audio_hours": 0.005,
                "required_distance_bins": ["0.5m", "5m"],
                "required_azimuth_deg": [0, 180],
                "required_snr_bands": ["critical", "high"],
            }
        }
        gate_domains = {
            "distance_bin:0.5m": gate_metric(),
            "distance_bin:5m": gate_metric(),
            "azimuth_deg:0": gate_metric(),
            "azimuth_deg:180": gate_metric(),
            "snr:critical": gate_metric(),
            "snr:high": gate_metric(),
        }
        gate_summary = {"qualification_domains": {"domains": gate_domains}}

        code, gate_result = run_gate_case(root, gate_summary, gate_config, "pass")
        assert code == 0
        assert gate_result["qualified"] is True
        assert not gate_result["failures"]
        assert gate_result["slices"]["distance_bin:5m"]["wake_rate"] == 1.0

        failing = json.loads(json.dumps(gate_summary))
        failing["qualification_domains"]["domains"]["azimuth_deg:180"]["frr"] = 0.25
        failing["qualification_domains"]["domains"]["azimuth_deg:180"]["wake_rate"] = 0.75
        code, gate_result = run_gate_case(root, failing, gate_config, "frr-fail")
        assert code == 1
        assert gate_result["qualified"] is False
        assert any(
            item["slice"] == "azimuth_deg:180" and item["reason"] == "frr"
            for item in gate_result["failures"]
        )

        unsupported = json.loads(json.dumps(gate_summary))
        unsupported["qualification_domains"]["domains"]["distance_bin:5m"]["expected"] = 2
        unsupported["qualification_domains"]["domains"]["distance_bin:5m"]["negative_audio_hours"] = 0.001
        code, gate_result = run_gate_case(root, unsupported, gate_config, "support-fail")
        assert code == 1
        reasons = {
            item["reason"]
            for item in gate_result["failures"]
            if item["slice"] == "distance_bin:5m"
        }
        assert reasons == {"insufficient-positive-support", "insufficient-negative-support"}

    print("test_domain_metrics: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
