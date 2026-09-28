from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_boundary_separability import (  # noqa: E402
    EVIDENCE_CLASS,
    FEATURES,
    POLICY,
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


def main() -> int:
    assert POLICY == "decoder-boundary-train-separability-v1"
    assert EVIDENCE_CLASS == "decoder-boundary-separability-development-v1"
    assert longest_true_run([False, True, True, False, True]) == 2

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
