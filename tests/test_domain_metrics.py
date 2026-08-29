#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def row(name: str, band: str, expected: list[dict]) -> dict:
    return {
        "recording": name,
        "path": f"{name}.wav",
        "duration_s": 10.0,
        "expected": expected,
        "domain": {
            "distance_band": band,
            "distance_m": 0.5 if band == "near" else 4.5,
            "azimuth_deg": 0.0 if band == "near" else 90.0,
            "rt60_s": 0.2 if band == "near" else 0.7,
            "snr_db": 20.0 if band == "near" else 2.0,
            "noise_profile": "fan" if band == "near" else "motor",
            "playback_sir_db": None if band == "near" else 5.0,
        },
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        refs = root / "refs.jsonl"
        dets = root / "dets.jsonl"
        out = root / "domains.json"
        references = [
            row("near", "near", [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]),
            row("far", "far", [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]),
            row("far-bg", "far", []),
        ]
        refs.write_text("\n".join(json.dumps(item) for item in references) + "\n", encoding="utf-8")
        detections = [
            {"recording": "near", "keyword_id": 1, "time_s": 2.1, "confidence": 0.9},
            {"recording": "far", "keyword_id": 2, "time_s": 2.1, "confidence": 0.8},
            {"recording": "far-bg", "keyword_id": 1, "time_s": 5.0, "confidence": 0.7},
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
        assert result["domains"]["distance:near"]["frr"] == 0.0
        assert result["domains"]["distance:far"]["frr"] == 1.0
        assert result["worst_domain"] is not None
        confusion = result["keyword_confusion"]
        assert confusion["correct_keyword"] == 1
        assert confusion["wrong_keyword"] == 1
        assert confusion["matrix"]["1"]["2"] == 1

    print("test_domain_metrics: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
