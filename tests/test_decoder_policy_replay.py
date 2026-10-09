from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_decoder_policy_replay import (  # noqa: E402
    boundary_acceptance,
    boundary_reference_contract,
    joint_strict_verdict,
)
from diagnose_decoder_selected_path_provenance import (  # noqa: E402
    self_test as selected_path_provenance_self_test,
)
from score_events import score, validate_detections, validate_recordings  # noqa: E402
from qualification_fixture import write_model, write_tokens  # noqa: E402


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


def verify_path_runner(runner: pathlib.Path) -> None:
    # Public, deterministic synthetic model/pack/trace fixtures only. No
    # historical acoustic traces or qualification records are rewritten.
    runner = runner.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="decoder-shadow-contract-") as td:
        root = pathlib.Path(td)
        model = root / "model.kwm"
        _, fingerprint = write_tokens(root / "tokens.txt")
        write_model(model, fingerprint)
        model_digest = hashlib.sha256(model.read_bytes()).hexdigest().encode("ascii")
        pack = root / "keywords.kwk"
        trace = root / "trace.kwtr"

        def replay(name: str, tokens: list[int], frames: list[list[float]],
                   *, threshold: float = 0.1, min_blanks: int = 0) -> dict:
            pack.write_bytes(
                struct.pack("<4sHHHHIQ", b"KWKP", 3, 24, 1, 5, 72, fingerprint)
                + struct.pack("<IfHBBBBH16H", 1, threshold, len(tokens),
                              min_blanks, 0, 0, 0, 0,
                              *(tokens + [0] * (16 - len(tokens))))
            )
            trace.write_bytes(
                struct.pack("<8sIHHIIIIQQ64s8x", b"KWTRACE1", 1, 5, 0,
                            16000, 400, 320, 0, fingerprint, len(frames),
                            model_digest)
                + b"".join(struct.pack("<QB7x5f", 400 + index * 320, 1, *frame)
                           for index, frame in enumerate(frames))
            )
            result = subprocess.run([str(runner), str(model), str(pack),
                                     str(trace), name], capture_output=True,
                                    text=True, check=True)
            payload = json.loads(result.stdout)
            assert payload["frames"] == len(frames)
            assert payload["decoder"]["token_boost"] == 1.5
            assert len(payload["keywords"]) == 1
            return payload["keywords"][0]

        root_token = [-8.0, 8.0, -8.0, -8.0, -8.0]
        blank = [8.0, -8.0, -8.0, -8.0, -8.0]
        exact_second = [-8.0, -8.0, 8.0, -8.0, -8.0]
        exact_third = [-8.0, -8.0, -8.0, 8.0, -8.0]
        fuzzy_second = [8.0, -8.0, 7.0, -8.0, -8.0]
        fuzzy_third = [8.0, -8.0, -8.0, 7.0, -8.0]
        exact = replay("exact-default", [1, 2, 3],
                       [root_token, exact_second, exact_third])
        assert exact["detections"] == 1
        one_fuzzy = replay("one-fuzzy", [1, 2, 3],
                           [root_token, fuzzy_second, exact_third])
        assert one_fuzzy["detections"] == 1
        two_fuzzy = replay("two-fuzzy", [1, 2, 3],
                           [root_token, fuzzy_second, fuzzy_third])
        assert two_fuzzy["detections"] == 0
        assert abs(two_fuzzy["max_terminal_retention_log"] + 16.5) < 1e-5
        expired = replay("expired", [1, 2, 3],
                         [root_token] + [blank] * 80 + [exact_second, exact_third])
        assert expired["detections"] == 0
        delayed = replay("immediate-blank-gate", [1], [root_token] + [blank] * 8,
                         min_blanks=8)
        assert delayed["detections"] == 1
        before_gate = replay("before-immediate-blank-gate", [1],
                             [root_token] + [blank] * 7, min_blanks=8)
        assert before_gate["detections"] == 0
        interrupted = replay("interrupted-immediate-blank-gate", [1],
                             [root_token, blank, exact_third] + [blank] * 8,
                             min_blanks=8)
        assert interrupted["detections"] == 0
        fuzzy_qualifies = replay("qualifying-blank-is-not-trailing", [1, 2],
                                 [root_token, fuzzy_second], min_blanks=1)
        assert fuzzy_qualifies["detections"] == 0
        after_fuzzy = replay("blank-after-fuzzy-terminal", [1, 2],
                             [root_token, fuzzy_second, blank], min_blanks=1)
        assert after_fuzzy["detections"] == 1
        below_threshold = replay("blank-gate-still-needs-confidence", [1],
                                 [[0.0] * 5] + [blank] * 8,
                                 threshold=0.5, min_blanks=8)
        assert below_threshold["detections"] == 0

        # Large common offsets must not erase the normalizer in the shadow
        # scorer. All five logits tie, so confidence is 1/5, never 1.
        for offset in (1e9, -1e9, 3.4028234663852886e38):
            tied = replay(f"large-offset-{offset}", [1], [[offset] * 5],
                          threshold=0.5)
            assert tied["detections"] == 0
            assert abs(tied["max_terminal_confidence"] - 0.2) < 1e-6
            assert abs(tied["max_terminal_retention_log"]) < 1e-6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-runner", type=pathlib.Path, required=True,
                        help="built kws_decoder_path_replay executable")
    args = parser.parse_args()
    verify_path_runner(args.path_runner)
    selected_path_provenance_self_test()
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

        product_positive = root / "product-positive.jsonl"
        product_negative = root / "product-negative.jsonl"
        write_jsonl(
            product_positive,
            [
                {
                    "recording": "natural-pause",
                    "audio_path": "natural-pause.wav",
                    "duration_s": 1.2,
                    "boundary_role": "natural-full-phrase-pause-v1",
                    "expected": [
                        {"keyword_id": 1, "start_s": 0.2, "end_s": 0.8}
                    ],
                }
            ],
        )
        write_jsonl(
            product_negative,
            [
                {
                    "recording": "cross-utterance",
                    "audio_path": "cross-utterance.wav",
                    "duration_s": 1.4,
                    "boundary_role": "cross-utterance-long-gap-v1",
                    "expected": [],
                }
            ],
        )
        product_pos = boundary_reference_contract(
            product_positive,
            positive=True,
            required_boundary_role="natural-full-phrase-pause-v1",
        )
        product_neg = boundary_reference_contract(
            product_negative,
            positive=False,
            required_boundary_role="cross-utterance-long-gap-v1",
        )
        assert product_pos["observed_boundary_roles"] == [
            "natural-full-phrase-pause-v1"
        ]
        assert product_neg["observed_boundary_roles"] == [
            "cross-utterance-long-gap-v1"
        ]
        expect_failure(
            lambda: boundary_reference_contract(
                product_positive,
                positive=True,
                required_boundary_role="inserted-silence-ambiguity-positive-v1",
            ),
            "boundary_role must be",
        )

        boundary_failed = {
            "acceptance": {"qualified": False},
        }
        assert joint_strict_verdict(
            True,
            {"acceptance_authority": False},
            boundary_failed,
        ) is True
        assert joint_strict_verdict(
            True,
            {"acceptance_authority": True},
            boundary_failed,
        ) is False

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
