#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORER = ROOT / "eval" / "score_events.py"


def run_score(references: pathlib.Path, detections: pathlib.Path, summary: pathlib.Path):
    return subprocess.run(
        [
            sys.executable,
            str(SCORER),
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--summary",
            str(summary),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        references = root / "references.jsonl"
        detections = root / "detections.jsonl"
        summary = root / "summary.json"
        false_positives = root / "false_positives.jsonl"
        false_rejects = root / "false_rejects.jsonl"

        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": "room-1.wav",
                    "duration_s": 3600.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 9.0, "end_s": 10.0},
                        {"keyword_id": 2, "start_s": 19.0, "end_s": 20.0},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        detections.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recording": "room-1",
                            "keyword_id": 1,
                            "time_s": 10.2,
                            "confidence": 0.8,
                        }
                    ),
                    json.dumps(
                        {
                            "recording": "room-1",
                            "keyword_id": 1,
                            "time_s": 100.0,
                            "confidence": 0.7,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        subprocess.check_call(
            [
                sys.executable,
                str(SCORER),
                "--references",
                str(references),
                "--detections",
                str(detections),
                "--summary",
                str(summary),
                "--false-positives",
                str(false_positives),
                "--false-rejects",
                str(false_rejects),
                "--max-far-per-hour",
                "1.0",
                "--max-frr",
                "0.5",
                "--max-p95-latency-ms",
                "250",
            ]
        )

        result = json.loads(summary.read_text(encoding="utf-8"))
        assert result["expected"] == 2
        assert result["matched"] == 1
        assert result["false_rejects"] == 1
        assert result["false_accepts"] == 1
        assert abs(result["frr"] - 0.5) < 1.0e-9
        assert abs(result["far_per_hour"] - 1.0) < 1.0e-9
        assert result["event_match_pre_tolerance_ms"] == 150.0
        assert result["event_match_post_tolerance_ms"] == 500.0
        assert 199.0 <= result["p95_post_end_latency_ms"] <= 201.0

        early_refs = root / "early-references.jsonl"
        early_dets = root / "early-detections.jsonl"
        early_summary = root / "early-summary.json"
        early_refs.write_text(
            json.dumps({"recording": "early", "duration_s": 3.0,
                        "expected": [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]}) + "\n",
            encoding="utf-8",
        )
        early_dets.write_text(
            json.dumps({"recording": "early", "keyword_id": 1,
                        "time_s": 0.9, "confidence": 0.9}) + "\n",
            encoding="utf-8",
        )
        early_default = run_score(early_refs, early_dets, early_summary)
        assert early_default.returncode == 0, early_default.stderr
        assert json.loads(early_summary.read_text(encoding="utf-8"))["matched"] == 1
        strict = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(early_refs),
             "--detections", str(early_dets), "--pre-tolerance-ms", "0",
             "--summary", str(early_summary)],
            check=False, capture_output=True, text=True,
        )
        assert strict.returncode == 0, strict.stderr
        strict_summary = json.loads(early_summary.read_text(encoding="utf-8"))
        assert strict_summary["matched"] == 0
        assert strict_summary["false_rejects"] == 1
        assert strict_summary["false_accepts"] == 1

        fp = [
            json.loads(line)
            for line in false_positives.read_text(encoding="utf-8").splitlines()
        ]
        assert len(fp) == 1
        assert fp[0]["time_s"] == 100.0
        assert fp[0]["path"] == "room-1.wav"

        fr = [
            json.loads(line)
            for line in false_rejects.read_text(encoding="utf-8").splitlines()
        ]
        assert len(fr) == 1
        assert fr[0]["keyword_id"] == 2
        assert fr[0]["start_s"] == 19.0
        assert fr[0]["end_s"] == 20.0
        assert fr[0]["path"] == "room-1.wav"

        negative_refs = root / "negative-references.jsonl"
        negative_dets = root / "negative-detections.jsonl"
        negative_summary = root / "negative-summary.json"
        negative_refs.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recording": "positive",
                            "duration_s": 1800.0,
                            "expected": [
                                {"keyword_id": 1, "start_s": 10.0, "end_s": 11.0}
                            ],
                        }
                    ),
                    json.dumps(
                        {
                            "recording": "negative",
                            "duration_s": 3600.0,
                            "expected": [],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        negative_dets.write_text(
            json.dumps(
                {
                    "recording": "positive",
                    "keyword_id": 2,
                    "time_s": 100.0,
                    "confidence": 0.7,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_score(negative_refs, negative_dets, negative_summary)
        assert completed.returncode == 0, completed.stderr
        negative_result = json.loads(
            negative_summary.read_text(encoding="utf-8")
        )
        assert abs(negative_result["audio_hours"] - 1.5) < 1.0e-12
        assert negative_result["false_accepts"] == 1
        assert abs(negative_result["far_per_hour"] - (2.0 / 3.0)) < 1.0e-12
        assert negative_result["negative_recording_false_accepts"] == 0
        assert abs(negative_result["negative_recording_audio_hours"] - 1.0) < 1.0e-12
        assert negative_result["longest_negative_recording_seconds"] == 3600.0
        assert negative_result["negative_recording_far_per_hour"] == 0.0
        assert (
            2.9957
            < negative_result["negative_recording_far_upper_95_per_hour"]
            < 2.9958
        )
        assert negative_result["negative_recording_far_confidence"] == 0.95
        assert (
            negative_result["negative_recording_far_policy"]
            == "negative-only-recordings-poisson-upper-v1"
        )

        negative_dets.write_text(
            json.dumps(
                {
                    "recording": "negative",
                    "keyword_id": 1,
                    "time_s": 120.0,
                    "confidence": 0.8,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_score(negative_refs, negative_dets, negative_summary)
        assert completed.returncode == 0, completed.stderr
        negative_result = json.loads(
            negative_summary.read_text(encoding="utf-8")
        )
        assert negative_result["negative_recording_false_accepts"] == 1
        assert negative_result["negative_recording_far_per_hour"] == 1.0
        assert (
            4.7438
            < negative_result["negative_recording_far_upper_95_per_hour"]
            < 4.7440
        )

        positive_only_refs = root / "positive-only-references.jsonl"
        positive_only_dets = root / "positive-only-detections.jsonl"
        positive_only_summary = root / "positive-only-summary.json"
        positive_only_refs.write_text(
            json.dumps(
                {
                    "recording": "only-positive",
                    "duration_s": 10.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        positive_only_dets.write_text("", encoding="utf-8")
        completed = run_score(
            positive_only_refs,
            positive_only_dets,
            positive_only_summary,
        )
        assert completed.returncode == 0, completed.stderr
        positive_only_result = json.loads(
            positive_only_summary.read_text(encoding="utf-8")
        )
        assert positive_only_result["negative_recording_audio_hours"] == 0.0
        assert positive_only_result["negative_recording_far_per_hour"] is None
        assert (
            positive_only_result["negative_recording_far_upper_95_per_hour"]
            is None
        )
        evidence_gate = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(positive_only_refs),
             "--detections", str(positive_only_dets), "--max-frr", "1",
             "--max-p95-latency-ms", "500",
             "--min-expected-per-keyword", "1", "1",
             "--min-expected-per-keyword", "2", "1",
             "--min-negative-hours", "1",
             "--min-continuous-negative-seconds", "60",
             "--max-negative-far-upper-95-per-hour", "3"],
            check=False, capture_output=True, text=True,
        )
        assert evidence_gate.returncode == 1
        assert "latency has no matched events" in evidence_gate.stderr
        assert "keyword 2 expected coverage" in evidence_gate.stderr
        assert "negative recording hours" in evidence_gate.stderr
        assert "continuous negative recording duration" in evidence_gate.stderr
        assert "negative FAR upper 95/hour" in evidence_gate.stderr

        no_positive_refs = root / "no-positive-references.jsonl"
        no_positive_refs.write_text(
            json.dumps({"recording": "negative", "duration_s": 3600.0,
                        "expected": []}) + "\n", encoding="utf-8",
        )
        no_positive_gate = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(no_positive_refs),
             "--detections", str(positive_only_dets), "--max-frr", "1"],
            check=False, capture_output=True, text=True,
        )
        assert no_positive_gate.returncode == 1
        assert "FRR has no positive events" in no_positive_gate.stderr

        sufficient_negative_gate = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(negative_refs),
             "--detections", str(positive_only_dets),
             "--min-expected-per-keyword", "1", "1",
             "--min-negative-hours", "1",
             "--max-negative-far-upper-95-per-hour", "3"],
            check=False, capture_output=True, text=True,
        )
        assert sufficient_negative_gate.returncode == 0, sufficient_negative_gate.stderr

        unlabeled_refs = root / "unlabeled.jsonl"
        unlabeled_refs.write_text(json.dumps({"recording": "unlabeled", "duration_s": 3600.0}) + "\n")
        unlabeled = run_score(unlabeled_refs, positive_only_dets, summary)
        assert unlabeled.returncode == 2
        assert "expected annotation is required" in unlabeled.stderr
        for invalid_args in (
            ["--min-negative-hours", "nan"],
            ["--max-negative-far-upper-95-per-hour", "inf"],
            ["--min-continuous-negative-seconds", "-1"],
            ["--min-expected-per-keyword", "1", "0"],
            ["--min-expected-per-keyword", "1", "1", "--min-expected-per-keyword", "1", "1"],
        ):
            invalid = subprocess.run(
                [sys.executable, str(SCORER), "--references", str(negative_refs),
                 "--detections", str(positive_only_dets), *invalid_args],
                check=False, capture_output=True, text=True,
            )
            assert invalid.returncode == 2, (invalid_args, invalid.stderr)
        insufficient = subprocess.run(
            [sys.executable, str(SCORER), "--references", str(negative_refs),
             "--detections", str(positive_only_dets), "--max-far-per-hour", "0",
             "--max-negative-far-upper-95-per-hour", "2"],
            check=False, capture_output=True, text=True,
        )
        assert insufficient.returncode == 1
        assert "negative FAR upper 95/hour" in insufficient.stderr

        overlap_refs = root / "overlap-references.jsonl"
        overlap_dets = root / "overlap-detections.jsonl"
        overlap_summary = root / "overlap-summary.json"
        overlap_refs.write_text(
            json.dumps(
                {
                    "recording": "overlap",
                    "duration_s": 100.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 4.0, "end_s": 5.45},
                        {"keyword_id": 1, "start_s": 5.4, "end_s": 5.8},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        overlap_dets.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "recording": "overlap",
                            "keyword_id": 1,
                            "time_s": 5.2,
                            "confidence": 0.8,
                        }
                    ),
                    json.dumps(
                        {
                            "recording": "overlap",
                            "keyword_id": 1,
                            "time_s": 5.5,
                            "confidence": 0.8,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_score(overlap_refs, overlap_dets, overlap_summary)
        assert completed.returncode == 0, completed.stderr
        overlap_result = json.loads(overlap_summary.read_text(encoding="utf-8"))
        assert overlap_result["matched"] == 2
        assert overlap_result["false_rejects"] == 0
        assert overlap_result["false_accepts"] == 0

        invalid_refs = root / "invalid-references.jsonl"
        empty_dets = root / "empty-detections.jsonl"
        invalid_summary = root / "invalid-summary.json"
        invalid_refs.write_text(
            '{"recording":"bad","duration_s":NaN,"expected":[]}\n',
            encoding="utf-8",
        )
        empty_dets.write_text("", encoding="utf-8")
        completed = run_score(invalid_refs, empty_dets, invalid_summary)
        assert completed.returncode == 2
        assert "finite" in completed.stderr

    print("test_eval: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
