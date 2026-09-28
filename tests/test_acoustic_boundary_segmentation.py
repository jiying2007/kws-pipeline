from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_acoustic_boundary_segmentation import (  # noqa: E402
    EVIDENCE_CLASS,
    POLICY,
    POSTERIOR_ONLY_FEATURES,
    WITH_SPEECH_GATE_FEATURES,
    cross_validate,
    evaluate_feature_set,
    fit_centroid,
    predict,
)


def row(voice: str, gap: int, label: bool, x: float, speech: float) -> dict:
    features = {feature: x for feature in POSTERIOR_ONLY_FEATURES}
    features["gap_speech_active_ratio"] = speech
    return {
        "voice_id": voice,
        "gap_ms": gap,
        "keyword_id": 1,
        "label": label,
        "features": features,
    }


def main() -> int:
    assert POLICY == "acoustic-boundary-segmentation-feasibility-v1"
    assert EVIDENCE_CLASS == "acoustic-boundary-segmentation-development-v1"
    assert "gap_speech_active_ratio" not in POSTERIOR_ONLY_FEATURES
    assert "gap_speech_active_ratio" in WITH_SPEECH_GATE_FEATURES

    rows = []
    for voice_index, voice in enumerate(("a", "b", "c")):
        for gap_index, gap in enumerate((160, 240, 400)):
            jitter = 0.01 * voice_index + 0.001 * gap_index
            rows.append(row(voice, gap, True, 1.0 + jitter, 0.10 + jitter))
            rows.append(row(voice, gap, False, -1.0 + jitter, 0.00 + jitter))

    rule = fit_centroid(rows, POSTERIOR_ONLY_FEATURES)
    assert all(predict(rule, item) is item["label"] for item in rows)
    by_voice = cross_validate(
        rows, features=POSTERIOR_ONLY_FEATURES, group_key="voice_id"
    )
    by_gap = cross_validate(
        rows, features=POSTERIOR_ONLY_FEATURES, group_key="gap_ms"
    )
    assert by_voice["accuracy"] == 1.0
    assert by_gap["accuracy"] == 1.0
    assert evaluate_feature_set(rows, POSTERIOR_ONLY_FEATURES)["stable"] is True

    workflow = (
        ROOT / ".github" / "workflows" / "acoustic-boundary-segmentation.yml"
    ).read_text(encoding="utf-8")
    assert "bundle/train/dataset-index.jsonl" in workflow
    assert "bundle/calibration/dataset-index.jsonl" not in workflow
    assert "--gap-ms 160 240 320 400 480" in workflow
    assert "selection_feedback_allowed" in workflow
    assert "acoustic-boundary-segmentation-receipt-v1" in workflow
    assert "kws_posterior_dump" in workflow
    assert "kws_decoder_replay" not in workflow

    print("test_acoustic_boundary_segmentation: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
