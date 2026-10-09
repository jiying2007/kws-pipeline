from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_decoder_boundary_references as boundary_builder  # noqa: E402
import diagnose_boundary_separability as separability  # noqa: E402
from diagnose_boundary_separability import (  # noqa: E402
    EVIDENCE_CLASS,
    FEATURES,
    POLICY,
    assert_train_split,
    fit_scalar_threshold,
    longest_true_run,
    summarize_feature,
)


def record(voice: str, positive: float, negative: float) -> dict:
    def features(value: float) -> dict:
        return {feature: value for feature in FEATURES}
    return {
        "voice_id": voice,
        "positive": {"features": features(positive)},
        "negative": {"features": features(negative)},
    }


def check_cache_producer_callers() -> None:
    """Exercise both builders with tiny WAVs and a stub cache/feature reader."""
    original_cwd = pathlib.Path.cwd()
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        rows = []
        for name, kind, target in (
            ("kw1-exact", "positive", [1, 2]),
            ("left", "negative", [1]),
            ("right", "negative", [2]),
        ):
            audio = root / (name + ".wav")
            boundary_builder.write_wav(audio, [1] * 640)
            rows.append({
                "kind": kind, "keyword_id": 1 if kind == "positive" else None,
                "target_ids": target, "path": audio.name,
                "wav_sha256": boundary_builder.sha256_file(audio),
                "event_start_frame": 0, "event_end_frame": 640,
                "speech_like_provenance": {
                    "voice_id": "stub-voice",
                    "source_id": "speech-like:train:stub:" + name,
                },
            })
        index = root / "index.jsonl"
        index.write_text("".join(json.dumps(row) + "\n" for row in rows))
        model = root / "model.kwm"
        model.write_bytes(b"synthetic model, never executed")
        producer = root / "dump"
        trace = root / "stub.kwtr"
        trace.write_bytes(b"synthetic trace, decoded by stub")

        for module, expected_calls in ((boundary_builder, 1), (separability, 3)):
            producer.write_bytes(b"producer A, never executed")
            producer_sha = module.sha256_file(producer)
            calls = []

            def cached_trace(**kwargs):
                assert kwargs["posterior_dump_sha256"] == producer_sha
                calls.append(kwargs)
                if len(calls) == expected_calls:
                    producer.write_bytes(b"producer B, changed after producing the trace")
                return trace, {
                    "trace_sha256": module.sha256_file(trace),
                    "posterior_dump_sha256": producer_sha,
                }, True

            args = argparse.Namespace(
                dataset_index=index, model=model, posterior_dump=producer,
                posterior_cache=root / "cache", output_dir=root / module.__name__,
                gap_ms=20, lead_ms=0, tail_ms=20,
            )
            try:
                with mock.patch.object(module, "ROOT", root), \
                        mock.patch.object(module, "ensure_cached_trace", side_effect=cached_trace), \
                        mock.patch.object(module, "internal_split_from_alignment",
                                          return_value={"split_samples": 320}):
                    if module is separability:
                        with mock.patch.object(module, "read_trace_frames",
                                               return_value=[{"logits": [0.0, 1.0, 2.0]}]), \
                                mock.patch.object(module, "extract_features",
                                                  return_value={name: 0.0 for name in FEATURES}):
                            result = module.build(args)
                    else:
                        with mock.patch.object(module, "read_trace_logits", return_value=[]):
                            result = module.build(args)
            finally:
                os.chdir(original_cwd)
            assert len(calls) == expected_calls
            assert result["posterior_dump_sha256"] == producer_sha
            assert module.sha256_file(producer) != producer_sha
            item = result["records"][0]
            if module is separability:
                for role in ("alignment", "positive", "negative"):
                    assert item[role]["posterior_dump_sha256"] == producer_sha
                    assert item[role]["trace_sha256"] == module.sha256_file(trace)
            else:
                assert item["posterior_dump_sha256"] == producer_sha
                assert item["posterior_cache_hit"] is True


def main() -> int:
    check_cache_producer_callers()
    assert POLICY == "decoder-boundary-train-separability-v1"
    assert EVIDENCE_CLASS == "decoder-boundary-separability-development-v1"
    assert longest_true_run([False, True, True, False, True]) == 2

    assert_train_split(
        [
            {
                "speech_like_provenance": {
                    "source_id": "speech-like:train:slot:kw1-exact"
                }
            }
        ]
    )
    try:
        assert_train_split(
            [
                {
                    "speech_like_provenance": {
                        "source_id": "speech-like:calibration:slot:kw1-exact"
                    }
                }
            ]
        )
    except ValueError as exc:
        assert "only speech-like:train" in str(exc)
    else:
        raise AssertionError("calibration evidence entered feedback-allowed diagnostic")

    fitted = fit_scalar_threshold(
        [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    )
    assert fitted["direction"] == "positive_higher"
    assert fitted["accuracy"] == 1.0

    rows = [
        record("voice-a", 0.9, 0.2),
        record("voice-b", 0.8, 0.1),
        record("voice-c", 0.85, 0.15),
    ]
    summary = summarize_feature(rows, FEATURES[0])
    assert summary["range_disjoint"] is True
    assert summary["range_direction"] == "positive_higher"
    assert summary["best_train_scalar_rule"]["accuracy"] == 1.0
    assert summary["leave_one_voice_out"]["accuracy"] == 1.0

    overlap = [
        record("voice-a", 0.9, 0.8),
        record("voice-b", 0.2, 0.3),
        record("voice-c", 0.7, 0.6),
    ]
    mixed = summarize_feature(overlap, FEATURES[0])
    assert mixed["range_disjoint"] is False
    assert mixed["leave_one_voice_out"]["accuracy"] < 1.0

    workflow = (
        ROOT / ".github" / "workflows" / "decoder-boundary-separability.yml"
    ).read_text(encoding="utf-8")
    assert "bundle/train/dataset-index.jsonl" in workflow
    assert "bundle/calibration/dataset-index.jsonl" not in workflow
    assert "selection_feedback_allowed" in workflow
    assert "decoder-boundary-separability-receipt-v1" in workflow
    assert "kws_posterior_dump" in workflow
    assert "kws_decoder_replay" not in workflow

    print("test_decoder_boundary_separability: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
