#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
V2 = ROOT / "experiments" / "gru_generalization_v2"
sys.path.insert(0, str(V2))

import run_candidate as v2  # noqa: E402

POLICY = "shadow-blind-gru-generalization-v4-cross-domain"
REPLAY_POLICY = "gru-generalization-cross-domain-resynthesis-v1"
REPLAY_EVIDENCE_CLASS = "training-only-gru-generalization-cross-domain-resynthesis"
EXPECTED_EXAMPLES_PER_EVENT = 12
EXPECTED_SOURCE_EVENT_COUNT = 4
EXPECTED_EXAMPLES = EXPECTED_EXAMPLES_PER_EVENT * EXPECTED_SOURCE_EVENT_COUNT


def validate_replay(path: pathlib.Path) -> dict:
    evidence = v2.load_json(path)
    if evidence.get("evidence_class") != REPLAY_EVIDENCE_CLASS:
        raise ValueError("unexpected GRU V4 replay evidence class")
    if evidence.get("policy") != REPLAY_POLICY:
        raise ValueError("unexpected GRU V4 replay policy")
    for key in (
        "formal_qualification_used",
        "calibration_evidence_used",
        "test_evidence_used",
        "shadow_seed_consumed",
        "shadow_wav_bytes_copied",
        "source_scene_used",
    ):
        if evidence.get(key) is not False:
            raise ValueError(f"GRU V4 replay isolation contract failed: {key}")
    if evidence.get("scene_generation_basis") != "config-domain-lattice-only":
        raise ValueError("GRU V4 replay scene-generation basis drifted")
    if int(evidence.get("examples_per_event", -1)) != EXPECTED_EXAMPLES_PER_EVENT:
        raise ValueError("GRU V4 replay examples-per-event drifted")
    if int(evidence.get("source_event_count", -1)) != EXPECTED_SOURCE_EVENT_COUNT:
        raise ValueError("GRU V4 replay source-event count drifted")
    if int(evidence.get("examples", -1)) != EXPECTED_EXAMPLES:
        raise ValueError("GRU V4 replay example count drifted")

    coverage = evidence.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("GRU V4 replay coverage summary missing")
    if set(coverage.get("distance_bands", {})) != {"near", "mid", "far"}:
        raise ValueError("GRU V4 replay distance coverage incomplete")
    if set(coverage.get("playback_states", {})) != {"playback", "no-playback"}:
        raise ValueError("GRU V4 replay playback coverage incomplete")
    if set(coverage.get("noise_profiles", {})) != {"white", "fan", "motor", "media"}:
        raise ValueError("GRU V4 replay noise coverage incomplete")
    if len(coverage.get("azimuth_deg", {})) != 12:
        raise ValueError("GRU V4 replay azimuth coverage incomplete")

    manifest = pathlib.Path(str(evidence["manifest"]))
    if not manifest.is_file() or v2.sha256_file(manifest) != str(evidence["manifest_sha256"]):
        raise ValueError("GRU V4 replay manifest binding failed")
    return evidence


def main() -> int:
    if v2.POSITIVE_EXAMPLE_WEIGHT != 2.45:
        raise ValueError("GRU V2/V4 positive-example weight drifted")
    if v2.EPOCHS != 12 or v2.LR_SCALE != 1.0:
        raise ValueError("GRU V2/V4 training schedule drifted")
    if v2.TRAINING_SEED_OFFSET != 9_000_019 or v2.EXPOSURE_REPEAT != 3:
        raise ValueError("GRU V2/V4 seed/exposure contract drifted")
    v2.POLICY = POLICY
    v2.validate_replay = validate_replay
    return v2.main()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
