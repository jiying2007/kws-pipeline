#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "run_gru_development.py"
POLICY = ROOT / "configs" / "training" / "xiaowo.gru-development-loop.json"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stress_config() -> dict:
    azimuths = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150, 180]
    required_stress = [
        "distance_azimuth:distance_bin=5m|azimuth=rear",
        "distance_snr:distance_bin=5m|snr=critical",
        "azimuth_snr:azimuth=rear|snr=critical",
        "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical",
    ]
    return {
        "domains": {
            "distance_bands": {
                "near": {"distance_m": [0.3, 1.0], "weight": 1.0},
                "mid": {"distance_m": [1.0, 3.0], "weight": 1.5},
                "far": {"distance_m": [3.0, 5.0], "weight": 2.5},
            },
            "azimuth_deg": azimuths,
            "rt60_s": [0.15, 0.80],
            "snr_db": [3.0, 30.0],
            "noise_profiles": ["white", "fan", "motor", "media"],
            "mic_spacing_m": 0.06,
            "playback": {"probability": 0.35, "sir_db": [-8.0, 20.0]},
            "afe": {"backend": "proxy"},
            "scenes_per_example": {
                "train": 1,
                "calibration": 1,
                "test": 1,
                "qualification": 1,
            },
        },
        "robustness_gates": {
            "min_expected_wakes": 9,
            "min_negative_recordings": 4,
            "required_distance_bins": ["0.5m", "1m", "2m", "3m", "5m"],
            "required_azimuth_deg": azimuths,
            "required_snr_bands": ["critical", "low", "mid", "high"],
            "required_stress_slices": required_stress,
        },
    }


def main() -> int:
    module = load_module(SCRIPT, "gru_train_acoustic_rotation_contract")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))

    assert policy["training_acoustic_seed_namespace"] == 193000019
    assert policy["training_acoustic_seed_stride"] == 1009
    first, stride = module._rotation(policy, 0)
    second, _ = module._rotation(policy, 1)
    third, _ = module._rotation(policy, 2)
    assert stride == 1009
    assert [first, second, third] == [193000019, 193001028, 193002037]
    assert len({first, second, third}) == 3
    assert first != int(policy["training_seed_namespace"])
    assert first != int(policy["candidate_freeze"]["fresh_validation_seed_namespace"])

    canonical = [
        {"split": "train", "wav_sha256": "a", "scene": {"distance_band": "near"}},
        {"split": "calibration", "wav_sha256": "b", "scene": {"distance_band": "mid"}},
        {"split": "test", "wav_sha256": "c", "scene": {"distance_band": "far"}},
        {"split": "train", "wav_sha256": "d", "scene": {"distance_band": "far"}},
        {"split": "qualification", "wav_sha256": "e", "scene": {"distance_band": "near"}},
    ]
    rotated = [
        {"split": "train", "wav_sha256": "x", "scene": {"distance_band": "mid"}},
        {"split": "calibration", "wav_sha256": "y", "scene": {"distance_band": "far"}},
        {"split": "train", "wav_sha256": "z", "scene": {"distance_band": "near"}},
    ]
    merged = module._merge_domain_rows(canonical, rotated)
    assert [row["wav_sha256"] for row in merged] == ["x", "b", "c", "z", "e"]
    for index in (1, 2, 4):
        assert merged[index] is canonical[index]

    bad = dict(policy)
    bad["training_acoustic_seed_namespace"] = policy["candidate_freeze"][
        "fresh_validation_seed_namespace"
    ]
    try:
        module._rotation(bad, 0)
    except ValueError as exc:
        assert "overlaps model/fresh namespace" in str(exc)
    else:
        raise AssertionError("fresh namespace overlap was accepted")

    config = stress_config()
    domains = module.renderer.validate_domains(config)
    axes = module.renderer._evaluation_axes(config, domains)
    assert axes is not None
    negative_plan = module._build_negative_stress_plan(
        config,
        axes,
        split="calibration",
        negative_scene_count=64,
    )
    triple = "distance_azimuth_snr:distance_bin=5m|azimuth=rear|snr=critical"
    assert negative_plan["support_kind"] == "negative"
    assert negative_plan["target_per_slice"] == 4
    assert negative_plan["planned_support"][triple] >= 4
    assert negative_plan["reserved_scenes"] > 0
    assert config["robustness_gates"]["min_expected_wakes"] == 9
    assert config["robustness_gates"]["min_negative_recordings"] == 4

    source = SCRIPT.read_text(encoding="utf-8")
    assert "renderer.generate_dataset = _reuse_canonical_base" in source
    assert 'shutil.copy2(rotated / "train.tsv", output / "train.tsv")' in source
    assert '"base_utterance_reused": True' in source
    assert '"evaluation_seed_rotated": False' in source
    assert "training_code_sha256" in source
    assert module.NEGATIVE_STRESS_POLICY == "gru-development-negative-stress-support-v1"
    assert "development_negative_stress_support" in source
    assert '"qualification_overridden": False' in source
    assert '"validation_feedback_used": False' in source

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        path = root / "rows.jsonl"
        module._write_jsonl(path, merged)
        roundtrip = module._read_jsonl(path)
        assert [row["wav_sha256"] for row in roundtrip] == ["x", "b", "c", "z", "e"]

    print("GRU train acoustic rotation contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
