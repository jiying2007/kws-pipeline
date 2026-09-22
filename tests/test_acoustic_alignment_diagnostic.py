#!/usr/bin/env python3
from __future__ import annotations

import math
import pathlib
import struct
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_acoustic_alignment import (  # noqa: E402
    MODEL_HEADER,
    ctc_log_probability,
    decoder_surrogate_log_confidence,
    greedy_collapse,
    infer_logits,
    load_model,
    longest_prefix_subsequence,
    parse_runtime_detections,
    runtime_match_summary,
    stable_sample,
)


def align4(buffer: bytearray) -> None:
    while len(buffer) % 4:
        buffer.append(0)


def write_fixture_model(path: pathlib.Path) -> None:
    feature_dim = 1
    hidden_dim = 1
    vocab_size = 3
    wx = struct.pack("<b", 127)
    wh = struct.pack("<b", 0)
    bh = struct.pack("<f", 0.0)
    wo = struct.pack("<3b", 0, 127, -127)
    bo = struct.pack("<3f", 0.0, 0.0, 0.0)
    buffer = bytearray(b"\x00" * MODEL_HEADER.size)
    offsets = []
    for block in (wx, wh, bh, wo, bo):
        align4(buffer)
        offsets.append(len(buffer))
        buffer.extend(block)
    header = MODEL_HEADER.pack(
        b"KWSP",
        2,
        MODEL_HEADER.size,
        feature_dim,
        hidden_dim,
        vocab_size,
        0,
        16000,
        400,
        320,
        1.0 / 127.0,
        1.0 / 127.0,
        1.0 / 127.0,
        1234,
        *offsets,
        len(buffer),
    )
    buffer[: MODEL_HEADER.size] = header
    path.write_bytes(buffer)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="acoustic-alignment-test-") as tmp:
        model_path = pathlib.Path(tmp) / "fixture.kwm"
        write_fixture_model(model_path)
        model = load_model(model_path)
        assert model["feature_dim"] == 1
        assert model["hidden_dim"] == 1
        assert model["vocab_size"] == 3
        assert model["frontend_name"] == "logmel"

        logits = infer_logits(model, [[1.0], [1.0], [0.0]])
        assert len(logits) == 3
        assert logits[0][1] > logits[0][0] > logits[0][2]
        collapsed = greedy_collapse(logits)
        assert collapsed == (1,)
        assert longest_prefix_subsequence((1, 2), (1, 1, 3)) == 1

        strong = [[0.0, 4.0, -4.0], [0.0, 4.0, -4.0], [4.0, 0.0, -4.0]]
        weak = [[4.0, 0.0, -4.0], [4.0, 0.0, -4.0], [4.0, 0.0, -4.0]]
        strong_logp = ctc_log_probability(strong, (1,))
        weak_logp = ctc_log_probability(weak, (1,))
        assert math.isfinite(strong_logp)
        assert math.isfinite(weak_logp)
        assert strong_logp > weak_logp

        surrogate_log = decoder_surrogate_log_confidence(
            [
                [0.0, 4.0, -4.0],
                [4.0, 0.0, -4.0],
                [0.0, -4.0, 4.0],
            ],
            (1, 2),
        )
        assert math.isfinite(surrogate_log)
        assert 0.0 < math.exp(surrogate_log) < 1.0

        detections = parse_runtime_detections(
            '{"recording":"fixture","keyword_id":1,"time_s":0.42,"confidence":0.73}\n'
            '{"recording":"fixture","keyword_id":2,"time_s":0.61,"confidence":0.66}\n',
            "fixture",
        )
        assert [row["keyword_id"] for row in detections] == [1, 2]
        assert detections[0]["confidence"] == 0.73
        matched = runtime_match_summary(
            [
                {"keyword_id": 1, "time_s": 0.85, "confidence": 0.73},
                {"keyword_id": 1, "time_s": 1.71, "confidence": 0.81},
                {"keyword_id": 2, "time_s": 1.00, "confidence": 0.66},
            ],
            keyword_id=1,
            event_start_frame=16000,
            event_end_frame=19200,
            recording_frames=32000,
            sample_rate_hz=16000,
        )
        assert matched["runtime_expected_detection_count"] == 2
        assert matched["runtime_expected_matched_count"] == 1
        assert matched["runtime_out_of_window_expected_detection_count"] == 1
        assert matched["runtime_wrong_keyword_in_window_count"] == 1
        assert matched["runtime_matched_expected"] is True
        assert matched["runtime_max_matched_expected_confidence"] == 0.73
        assert matched["match_pre_tolerance_ms"] == 150.0
        assert matched["match_post_tolerance_ms"] == 500.0

        try:
            parse_runtime_detections(
                '{"recording":"other","keyword_id":1,"time_s":0.1,"confidence":0.7}\n',
                "fixture",
            )
        except ValueError as exc:
            assert "recording id drifted" in str(exc)
        else:
            raise AssertionError("runtime recording identity drift was accepted")

        rows = [
            {
                "source_wav_sha256": "b",
                "scene_seed": 2,
                "wav_sha256": "b2",
                "path": "b2.wav",
            },
            {
                "source_wav_sha256": "a",
                "scene_seed": 2,
                "wav_sha256": "a2",
                "path": "a2.wav",
            },
            {
                "source_wav_sha256": "a",
                "scene_seed": 1,
                "wav_sha256": "a1",
                "path": "a1.wav",
            },
            {
                "source_wav_sha256": "c",
                "scene_seed": 1,
                "wav_sha256": "c1",
                "path": "c1.wav",
            },
        ]
        sample = stable_sample(rows, 3)
        assert [row["source_wav_sha256"] for row in sample] == ["a", "b", "c"]

    print("acoustic alignment diagnostic: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
