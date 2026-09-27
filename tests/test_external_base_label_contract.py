#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from external_base_dataset import _validate_audio_format, _validate_label  # noqa: E402


def must_reject(row: dict, expected: str, tokens: dict[str, int], wakes: dict[int, dict]) -> None:
    try:
        _validate_label(row, "train", 0, tokens, wakes)
    except ValueError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError(f"invalid external label accepted: {expected}")


def main() -> int:
    tokens = {"<blk>": 0, "a": 1, "b": 2, "c": 3, "d": 4}
    wakes = {1: {"tokens": ["a", "b"]}, 2: {"tokens": ["c", "d"]}}
    positive = {
        "kind": "positive", "keyword_id": 1, "tokens": ["a", "b"],
        "target_ids": [1, 2], "frames": 16000,
        "event_start_frame": 1000, "event_end_frame": 9000,
    }
    _validate_label(positive, "train", 0, tokens, wakes)
    must_reject({**positive, "target_ids": [1, 3]}, "disagree", tokens, wakes)
    must_reject({**positive, "keyword_id": 2}, "differs", tokens, wakes)
    must_reject({**positive, "event_end_frame": 1000}, "bounds", tokens, wakes)
    must_reject({**positive, "keyword_id": True}, "keyword_id", tokens, wakes)

    near_miss = {
        "kind": "confusable", "keyword_id": 1,
        "tokens": ["a", "c"], "target_ids": [1, 3], "frames": 16000,
    }
    _validate_label(near_miss, "train", 1, tokens, wakes)
    must_reject(
        {**near_miss, "tokens": ["a", "c", "b"], "target_ids": [1, 3, 2]},
        "contains a configured wake path", tokens, wakes,
    )
    must_reject(
        {**near_miss, "kind": "negative", "tokens": ["a", "b"], "target_ids": [1, 2]},
        "contains a configured wake path", tokens, wakes,
    )
    must_reject({**near_miss, "tokens": [], "target_ids": []}, "non-empty", tokens, wakes)
    _validate_label({"kind": "background", "tokens": [], "target_ids": [], "frames": 16000}, "train", 2, tokens, wakes)
    must_reject({**near_miss, "kind": "background"}, "empty target", tokens, wakes)
    with tempfile.TemporaryDirectory(prefix="external-label-contract-") as temporary:
        wav = pathlib.Path(temporary) / "clip.wav"
        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * 200)
        _validate_audio_format(wav, 200, "fixture")
        try:
            _validate_audio_format(wav, 199, "fixture")
        except ValueError as exc:
            assert "frames mismatch" in str(exc)
        else:
            raise AssertionError("declared frame count mismatch was accepted")
    print("test_external_base_label_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
