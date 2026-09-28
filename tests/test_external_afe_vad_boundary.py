from __future__ import annotations

import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from diagnose_external_afe_vad_boundary import (  # noqa: E402
    ACTIVE_FEATURES,
    EVIDENCE_CLASS,
    FEATURE_SETS,
    POLICY,
    PROBABILITY_FEATURES,
    extract_features,
    read_source_pin,
)


def main() -> int:
    assert POLICY == "external-afe-vad-boundary-feasibility-v1"
    assert EVIDENCE_CLASS == "external-afe-vad-boundary-development-v1"
    assert "gap_speech_active_ratio" not in PROBABILITY_FEATURES
    assert "gap_speech_active_ratio" in ACTIVE_FEATURES
    assert set(FEATURE_SETS) == {"probability_only", "probability_plus_active"}

    rows = []
    probabilities = [0.8, 0.7, 0.6, 0.1, 0.05, 0.0, 0.02, 0.55, 0.7, 0.8]
    active = [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    for frame, (prob, act) in enumerate(zip(probabilities, active)):
        rows.append(
            {
                "frame": frame,
                "start_sample": frame * 160,
                "end_sample": (frame + 1) * 160,
                "vad_probability": prob,
                "vad_active": act,
            }
        )
    features = extract_features(rows, gap_start=480, gap_end=1120)
    assert features["pre_probability_last"] == 0.6
    assert abs(features["gap_probability_mean"] - 0.0425) < 1.0e-12
    assert features["gap_active_prefix_frames"] == 1.0
    assert features["gap_active_suffix_frames"] == 0.0
    assert features["post_probability_first"] == 0.55

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / "pin.json"
        pin = {
            "schema_version": 1,
            "evidence_class": "external-afe-vad-source-pin-v1",
            "repository": "jiying2007/audio-pipeline",
            "head_sha": "a" * 40,
            "frame_ms": 10,
            "sample_rate_hz": 16000,
            "profiles": ["vad-isolated", "ns-isolated"],
            "metrics_fields": ["vad_probability", "vad_active"],
            "development_only": True,
            "selection_feedback_allowed": True,
            "protected_evidence_used": False,
            "release_authority": False,
        }
        path.write_text(json.dumps(pin), encoding="utf-8")
        assert read_source_pin(path)["head_sha"] == "a" * 40
        pin["source_split"] = "calibration"
        pin["selection_feedback_allowed"] = False
        path.write_text(json.dumps(pin), encoding="utf-8")
        try:
            read_source_pin(path)
        except ValueError as exc:
            assert "selection_feedback_allowed" in str(exc)
        else:
            raise AssertionError("non-feedback external VAD source pin was accepted")

    workflow = (
        ROOT / ".github" / "workflows" / "external-afe-vad-boundary.yml"
    ).read_text(encoding="utf-8")
    assert "bundle/train/dataset-index.jsonl" in workflow
    assert "bundle/calibration/dataset-index.jsonl" not in workflow
    assert "external/audio-pipeline" in workflow
    assert "ap_process_pcm" in workflow
    assert "--capture-profile" not in workflow  # profile sweep lives in the tool
    assert "kws_decoder_replay" not in workflow
    assert "external-afe-vad-boundary-receipt-v1" in workflow

    print("test_external_afe_vad_boundary: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
