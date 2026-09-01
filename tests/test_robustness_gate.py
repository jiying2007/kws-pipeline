#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def metric(*, expected: int = 4, negative_hours: float = 0.01, frr: float = 0.0, far: float = 0.0) -> dict:
    return {
        "expected": expected,
        "negative_audio_hours": negative_hours,
        "frr": frr,
        "far_per_hour": far,
        "wake_rate": 1.0 - frr,
    }


def run_case(root: pathlib.Path, summary: dict, config: dict, name: str) -> tuple[int, dict]:
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
    config = {
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
    domains = {
        "distance_bin:0.5m": metric(),
        "distance_bin:5m": metric(),
        "azimuth_deg:0": metric(),
        "azimuth_deg:180": metric(),
        "snr:critical": metric(),
        "snr:high": metric(),
    }
    summary = {"qualification_domains": {"domains": domains}}

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        code, result = run_case(root, summary, config, "pass")
        assert code == 0
        assert result["qualified"] is True
        assert not result["failures"]
        assert result["slices"]["distance_bin:5m"]["wake_rate"] == 1.0

        failing = json.loads(json.dumps(summary))
        failing["qualification_domains"]["domains"]["azimuth_deg:180"]["frr"] = 0.25
        failing["qualification_domains"]["domains"]["azimuth_deg:180"]["wake_rate"] = 0.75
        code, result = run_case(root, failing, config, "frr-fail")
        assert code == 1
        assert result["qualified"] is False
        assert any(
            item["slice"] == "azimuth_deg:180" and item["reason"] == "frr"
            for item in result["failures"]
        )

        unsupported = json.loads(json.dumps(summary))
        unsupported["qualification_domains"]["domains"]["distance_bin:5m"]["expected"] = 2
        unsupported["qualification_domains"]["domains"]["distance_bin:5m"]["negative_audio_hours"] = 0.001
        code, result = run_case(root, unsupported, config, "support-fail")
        assert code == 1
        reasons = {
            item["reason"]
            for item in result["failures"]
            if item["slice"] == "distance_bin:5m"
        }
        assert reasons == {"insufficient-positive-support", "insufficient-negative-support"}

    print("test_robustness_gate: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
