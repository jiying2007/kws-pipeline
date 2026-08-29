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
            {"recording": "overlap", "keyword_id": 2, "time_s": 5.05, "confidence": 0.95},
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
        assert result["schema_version"] == 2
        assert result["domains"]["distance:near"]["frr"] > 0.0
        assert result["domains"]["distance:far"]["frr"] == 1.0
        assert result["domains"]["azimuth:rear"]["expected"] == 0
        assert result["domains"]["azimuth:rear"]["negative_audio_hours"] > 0.0
        assert result["domains"]["azimuth:rear"]["false_accepts"] == 1
        assert result["worst_domain"] is not None
        confusion = result["keyword_confusion"]
        assert confusion["assignment"] == "global-monotonic-one-to-one-v1"
        assert confusion["expected_events"] == 4
        assert confusion["correct_keyword"] == 2
        assert confusion["wrong_keyword"] == 1
        assert confusion["missed"] == 1
        assert confusion["matrix"]["1"]["2"] >= 1

    print("test_domain_metrics: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
