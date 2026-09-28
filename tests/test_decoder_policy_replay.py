from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_decoder_policy_replay import (  # noqa: E402
    boundary_acceptance,
    boundary_reference_contract,
)
from score_events import score, validate_detections, validate_recordings  # noqa: E402


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def event() -> dict:
    return {
        "keyword_id": 1,
        "start_s": 0.2,
        "end_s": 0.8,
        "match_not_before_s": 0.8,
    }


def expect_failure(call, needle: str) -> None:
    try:
        call()
    except ValueError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected failure containing {needle!r}")


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        positive = root / "within-word.jsonl"
        negative = root / "cross-boundary.jsonl"
        write_jsonl(
            positive,
            [
                {
                    "recording": "within-1",
                    "audio_path": "within-1.wav",
                    "duration_s": 1.2,
                    "expected": [event()],
                },
                {
                    "recording": "within-2",
                    "path": "within-2.wav",
                    "duration_s": 1.3,
                    "expected": [event()],
                },
            ],
        )
        write_jsonl(
            negative,
            [
                {
                    "recording": "cross-1",
                    "audio_path": "cross-1.wav",
                    "duration_s": 1.4,
                    "expected": [],
                }
            ],
        )

        pos = boundary_reference_contract(positive, positive=True)
        neg = boundary_reference_contract(negative, positive=False)
        assert pos["role"] == "within-word-pause-positive-v2"
        assert pos["recordings"] == 2
        assert pos["expected_events"] == 2
        assert len(pos["references_sha256"]) == 64
        assert neg["role"] == "cross-boundary-negative-v1"
        assert neg["recordings"] == 1
        assert neg["expected_events"] == 0

        scored_refs = validate_recordings(
            [
                {
                    "recording": "continuity",
                    "duration_s": 1.2,
                    "expected": [
                        {
                            "keyword_id": 1,
                            "start_s": 0.2,
                            "end_s": 0.8,
                            "match_not_before_s": 0.5,
                        }
                    ],
                }
            ]
        )
        early = validate_detections(
            [
                {
                    "recording": "continuity",
                    "keyword_id": 1,
                    "time_s": 0.49,
                    "confidence": 0.9,
                }
            ],
            scored_refs,
        )
        early_summary, early_false_accepts, early_false_rejects = score(
            scored_refs,
            early,
            0.15,
            0.5,
        )
        assert early_summary["matched"] == 0
        assert len(early_false_accepts) == 1
        assert len(early_false_rejects) == 1

        post_gap = validate_detections(
            [
                {
                    "recording": "continuity",
                    "keyword_id": 1,
                    "time_s": 0.51,
                    "confidence": 0.9,
                }
            ],
            scored_refs,
        )
        post_summary, post_false_accepts, post_false_rejects = score(
            scored_refs,
            post_gap,
            0.15,
            0.5,
        )
        assert post_summary["matched"] == 1
        assert not post_false_accepts
        assert not post_false_rejects

        good = boundary_acceptance(
            {"expected": 2, "false_rejects": 0},
            {
                "recordings": 1,
                "expected": 0,
                "false_accepts": 0,
                "negative_recording_audio_hours": 0.001,
            },
        )
        assert good == {
            "within_word_supported": True,
            "within_word_zero_false_rejects": True,
            "cross_boundary_supported": True,
            "cross_boundary_zero_false_accepts": True,
            "qualified": True,
        }

        for within, cross in (
            (
                {"expected": 0, "false_rejects": 0},
                {
                    "recordings": 1,
                    "expected": 0,
                    "false_accepts": 0,
                    "negative_recording_audio_hours": 0.001,
                },
            ),
            (
                {"expected": 2, "false_rejects": 1},
                {
                    "recordings": 1,
                    "expected": 0,
                    "false_accepts": 0,
                    "negative_recording_audio_hours": 0.001,
                },
            ),
            (
                {"expected": 2, "false_rejects": 0},
                {
                    "recordings": 1,
                    "expected": 0,
                    "false_accepts": 1,
                    "negative_recording_audio_hours": 0.001,
                },
            ),
            (
                {"expected": 2, "false_rejects": 0},
                {
                    "recordings": 0,
                    "expected": 0,
                    "false_accepts": 0,
                    "negative_recording_audio_hours": 0.0,
                },
            ),
        ):
            assert boundary_acceptance(within, cross)["qualified"] is False

        missing_post_gap = root / "missing-post-gap.jsonl"
        write_jsonl(
            missing_post_gap,
            [
                {
                    "recording": "within-no-boundary",
                    "audio_path": "within-no-boundary.wav",
                    "duration_s": 1.0,
                    "expected": [
                        {"keyword_id": 1, "start_s": 0.2, "end_s": 0.8}
                    ],
                }
            ],
        )
        expect_failure(
            lambda: boundary_reference_contract(missing_post_gap, positive=True),
            "post-gap match_not_before_s",
        )

        invalid_positive = root / "invalid-positive.jsonl"
        write_jsonl(
            invalid_positive,
            [
                {
                    "recording": "within-empty",
                    "audio_path": "within-empty.wav",
                    "duration_s": 1.0,
                    "expected": [],
                }
            ],
        )
        expect_failure(
            lambda: boundary_reference_contract(invalid_positive, positive=True),
            "must contain expected wake",
        )

        invalid_negative = root / "invalid-negative.jsonl"
        write_jsonl(
            invalid_negative,
            [
                {
                    "recording": "cross-labeled",
                    "audio_path": "cross-labeled.wav",
                    "duration_s": 1.0,
                    "expected": [event()],
                }
            ],
        )
        expect_failure(
            lambda: boundary_reference_contract(invalid_negative, positive=False),
            "must not contain expected wake",
        )

        missing_audio = root / "missing-audio.jsonl"
        write_jsonl(
            missing_audio,
            [
                {
                    "recording": "missing-audio",
                    "duration_s": 1.0,
                    "expected": [event()],
                }
            ],
        )
        expect_failure(
            lambda: boundary_reference_contract(missing_audio, positive=True),
            "audio path is required",
        )

    print("test_decoder_policy_replay: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
