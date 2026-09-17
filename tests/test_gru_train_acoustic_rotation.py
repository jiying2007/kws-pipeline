#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "training" / "run_gru_development.py"
POLICY = ROOT / "configs" / "training" / "xiaowo.gru-development-loop.json"
MULTISEED_SCRIPT = ROOT / "training" / "run_gru_development_multiseed.py"
MULTISEED_POLICY = ROOT / "configs" / "training" / "xiaowo.gru-development-multiseed-v1.json"
CONFIG = ROOT / "configs" / "training" / "xiaowo.torch-domain.json"
FRESH_REGISTRY = ROOT / "experiments" / "model_family" / "fresh_validation_registry.json"
SHADOW_REGISTRY = ROOT / "experiments" / "model_family" / "shadow_arena_registry.json"
RECONCILE = ROOT / "tools" / "reconcile_gru_development_gate.py"


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
    multiseed = load_module(MULTISEED_SCRIPT, "gru_train_acoustic_multiseed_contract")
    reconcile = load_module(RECONCILE, "gru_reconcile_wrapper_contract")
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
    assert "_render_rotated_train_rows(" in source
    assert '"acoustic_rendering": "direct-train-scene-v1"' in source
    assert "_persist_rotation_round(output" in source
    assert '"base_utterance_reused": True' in source
    assert '"evaluation_seed_rotated": False' in source
    assert "training_code_sha256" in source
    assert module.NEGATIVE_STRESS_POLICY == "gru-development-negative-stress-support-v1"
    assert "development_negative_stress_support" in source
    assert '"qualification_overridden": False' in source
    assert '"validation_feedback_used": False' in source

    multi_policy = json.loads(MULTISEED_POLICY.read_text(encoding="utf-8"))
    assert multi_policy["training_acoustic_exposures"] == 3
    assert multi_policy["training_acoustic_exposure_stride"] == 104729
    offsets0, round_stride, exposure_stride = multiseed.exposure_offsets(multi_policy, 0)
    offsets1, _, _ = multiseed.exposure_offsets(multi_policy, 1)
    assert offsets0 == [193000019, 193104748, 193209477]
    assert offsets1 == [193001028, 193105757, 193210486]
    assert round_stride == 1009
    assert exposure_stride == 104729
    all_offsets = multiseed.validate_exposure_namespace(multi_policy)
    assert len(all_offsets) == int(multi_policy["max_rounds"]) * 3
    assert len(all_offsets) == len(set(all_offsets))

    real_config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fresh_registry = json.loads(FRESH_REGISTRY.read_text(encoding="utf-8"))
    shadow_registry = json.loads(SHADOW_REGISTRY.read_text(encoding="utf-8"))
    try:
        multiseed.validate_protected_seed_independence(
            multi_policy,
            real_config,
            fresh_registry,
            shadow_registry,
        )
    except ValueError as exc:
        assert "current fresh validation namespace is not reserved for GRU" in str(exc)
    else:
        raise AssertionError("consumed Fresh v4 namespace was accepted for new development")

    reserved_fresh_registry = json.loads(json.dumps(fresh_registry))
    for row in reserved_fresh_registry["namespaces"]:
        if int(row.get("namespace", -1)) == int(
            multi_policy["candidate_freeze"]["fresh_validation_seed_namespace"]
        ):
            row["status"] = "reserved-untouched"
    protection = multiseed.validate_protected_seed_independence(
        multi_policy,
        real_config,
        reserved_fresh_registry,
        shadow_registry,
    )
    assert protection["fresh_registry_entry"] == "gru-fresh-validation-v4"
    assert protection["fresh_registry_status"] == "reserved-untouched"
    assert protection["shadow_arena"] == "gru-independent-shadow-v4"
    assert protection["shadow_arena_status"] == "reserved-untouched"
    assert protection["effective_exposure_seed_count"] == len(all_offsets)
    assert protection["overlap_count"] == 0

    bad_config = json.loads(json.dumps(real_config))
    bad_config["qualification_holdout_seed"] = int(real_config.get("seed", 1337)) + offsets0[0]
    try:
        multiseed.validate_protected_seed_independence(
            multi_policy,
            bad_config,
            reserved_fresh_registry,
            shadow_registry,
        )
    except ValueError as exc:
        assert "overlaps protected/model seed" in str(exc)
    else:
        raise AssertionError("formal qualification seed overlap was accepted")

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        source_wav = root / "source.wav"
        target_wav = root / "dataset" / "clips" / "train" / "d00000-s00.wav"
        target_wav.parent.mkdir(parents=True)
        module.renderer.write_wav(source_wav, [0, 1000, -1000, 500] * 4000)
        module.renderer.write_wav(target_wav, [0] * 16000)
        direct_config = stress_config()
        direct_config["seed"] = 1337
        direct_path = root / "direct.json"
        direct_path.write_text(json.dumps(direct_config), encoding="utf-8")
        canonical_row = {
            "split": "train",
            "source_path": str(source_wav),
            "path": str(target_wav),
            "scene_seed": 9001,
            "target_ids": [1, 2, 3, 4],
        }
        offset = 12345
        rendered, manifest = module._render_rotated_train_rows(
            direct_path,
            root / "dataset",
            [canonical_row],
            offset,
            curriculum_weights=None,
        )
        row = rendered[0]
        expected_seed = int(canonical_row["scene_seed"]) + offset
        domains = module.renderer.validate_domains(direct_config)
        rng = random.Random(expected_seed)
        scene = module.renderer.sample_scene(domains, rng, curriculum_weights=None, forced_band=None)
        expected_pcm, expected_meta = module.renderer.render_scene(
            module.renderer.read_wav(source_wav), scene, seed=expected_seed, afe=domains["afe"]
        )
        assert module.renderer.read_wav(pathlib.Path(row["path"])) == expected_pcm
        assert row["scene_seed"] == expected_seed
        assert row["scene"] == expected_meta
        assert row["wav_sha256"] == module.sha256_file(pathlib.Path(row["path"]))
        assert "train-acoustic-realization/clips/train" in pathlib.Path(row["path"]).as_posix()
        assert manifest.read_text(encoding="utf-8").split("\t", 1)[0] == row["path"]

    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        path = root / "rows.jsonl"
        module._write_jsonl(path, merged)
        roundtrip = module._read_jsonl(path)
        assert [row["wav_sha256"] for row in roundtrip] == ["x", "b", "c", "z", "e"]

        direct_config = stress_config()
        direct_config_path = root / "direct-config.json"
        direct_config_path.write_text(json.dumps(direct_config), encoding="utf-8")
        source = root / "clean.wav"
        clean_samples = [800 if index % 2 == 0 else -800 for index in range(3200)]
        module.renderer.write_wav(source, clean_samples)
        canonical_rows = [
            {
                "split": "train",
                "path": str(root / f"target-{index}.wav"),
                "source_path": str(source),
                "scene_seed": 1000 + index * 101,
                "wav_sha256": "canonical",
            }
            for index in range(4)
        ]
        offsets = [11, 23, 37]
        selected, evidence = multiseed._render_selected_train_rows(
            direct_config_path,
            canonical_rows,
            offsets,
            curriculum_weights=None,
        )
        assert [item["selected_scene_count"] for item in evidence] == [2, 1, 1]
        assert [item["rendered_scene_count"] for item in evidence] == [2, 1, 1]
        assert all(item["direct_selected_scene_render"] is True for item in evidence)
        domains = module.renderer.validate_domains(direct_config)
        clean = module.renderer.read_wav(source)
        for index, row in enumerate(selected):
            exposure = index % 3
            expected_seed = canonical_rows[index]["scene_seed"] + offsets[exposure]
            assert row["scene_seed"] == expected_seed
            rng = multiseed.random.Random(expected_seed)
            scene = module.renderer.sample_scene(
                domains, rng, curriculum_weights=None, forced_band=None
            )
            expected_mono, expected_meta = module.renderer.render_scene(
                clean, scene, seed=expected_seed, afe=domains["afe"]
            )
            target = root / f"target-{index}.wav"
            assert module.renderer.read_wav(target) == expected_mono
            assert row["scene"] == expected_meta
            assert row["domain_id"] == module._scene_domain_id(expected_meta)
            assert row["wav_sha256"] == module.sha256_file(target)

    wrapper_sha = module.sha256_file(MULTISEED_SCRIPT)
    bound = reconcile._bound_training_wrapper(
        {
            "development_training_wrapper": {
                "policy": multiseed.POLICY,
                "path": "training/run_gru_development_multiseed.py",
                "sha256": wrapper_sha,
            }
        }
    )
    assert bound is not None
    assert bound["policy"] == multiseed.POLICY
    assert bound["sha256"] == wrapper_sha
    assert bound["path"] == "training/run_gru_development_multiseed.py"
    try:
        reconcile._bound_training_wrapper(
            {
                "development_training_wrapper": {
                    "policy": multiseed.POLICY,
                    "path": "training/run_gru_development_multiseed.py",
                    "sha256": "0" * 64,
                }
            }
        )
    except ValueError as exc:
        assert "drifted before freeze" in str(exc)
    else:
        raise AssertionError("tampered multiseed wrapper provenance was accepted")

    multiseed_source = MULTISEED_SCRIPT.read_text(encoding="utf-8")
    reconcile_source = RECONCILE.read_text(encoding="utf-8")
    assert '"training_example_count_preserved": True' in multiseed_source
    assert '"evaluation_seed_rotated": False' in multiseed_source
    assert '"exposure_rendering": "direct-selected-scene-v1"' in multiseed_source
    assert "train-acoustic-realization-" not in multiseed_source
    assert 'manifest["development_training_wrapper"] = wrapper' in multiseed_source
    assert 'code[wrapper["path"]] = wrapper["sha256"]' in multiseed_source
    assert 'code[str(wrapper["path"])] = str(wrapper["sha256"])' in reconcile_source

    print("GRU train acoustic rotation contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
