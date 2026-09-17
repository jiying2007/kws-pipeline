#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import run_gru_development as base  # noqa: E402

POLICY = "development-train-acoustic-multiseed-v1"
EXPECTED_EXPOSURES = 3
MODEL_SEED_STRIDE = 1009
FRESH_REGISTRY = ROOT / "experiments" / "model_family" / "fresh_validation_registry.json"
SHADOW_REGISTRY = ROOT / "experiments" / "model_family" / "shadow_arena_registry.json"


def exposure_offsets(policy: dict, round_index: int) -> tuple[list[int], int, int]:
    first, round_stride = base._rotation(policy, round_index)
    count = int(policy.get("training_acoustic_exposures", 0))
    exposure_stride = int(policy.get("training_acoustic_exposure_stride", 0))
    if count != EXPECTED_EXPOSURES:
        raise ValueError(f"training_acoustic_exposures must be {EXPECTED_EXPOSURES}")
    if exposure_stride <= 0:
        raise ValueError("training_acoustic_exposure_stride must be positive")
    offsets = [first + exposure * exposure_stride for exposure in range(count)]
    if len(offsets) != len(set(offsets)):
        raise ValueError("training acoustic exposure seeds are not unique within round")
    return offsets, round_stride, exposure_stride


def validate_exposure_namespace(policy: dict) -> list[int]:
    rounds = int(policy.get("max_rounds", 0))
    if rounds <= 0:
        raise ValueError("max_rounds must be positive")
    all_offsets: list[int] = []
    for round_index in range(rounds):
        offsets, _, _ = exposure_offsets(policy, round_index)
        all_offsets.extend(offsets)
    if len(all_offsets) != len(set(all_offsets)):
        raise ValueError("training acoustic exposure seeds collide across rounds")
    protected_namespaces = {
        int(policy.get("training_seed_namespace", 0)),
        int(policy.get("candidate_freeze", {}).get("fresh_validation_seed_namespace", 0)),
    }
    if any(offset in protected_namespaces for offset in all_offsets):
        raise ValueError("training acoustic exposure overlaps model/fresh namespace")
    return all_offsets


def validate_protected_seed_independence(
    policy: dict,
    config: dict,
    fresh_registry: dict,
    shadow_registry: dict,
) -> dict:
    offsets = validate_exposure_namespace(policy)
    base_seed = int(config.get("seed", 1337))
    effective = {base_seed + offset for offset in offsets}
    if len(effective) != len(offsets):
        raise ValueError("effective training acoustic exposure seeds are not unique")

    freeze = policy.get("candidate_freeze")
    if not isinstance(freeze, dict):
        raise ValueError("candidate_freeze policy is missing")
    fresh_namespace = int(freeze.get("fresh_validation_seed_namespace", 0))
    fresh_rows = [row for row in fresh_registry.get("namespaces", []) if isinstance(row, dict)]
    fresh_matches = [row for row in fresh_rows if int(row.get("namespace", -1)) == fresh_namespace]
    if len(fresh_matches) != 1:
        raise ValueError("current fresh validation namespace is not uniquely registered")
    fresh = fresh_matches[0]
    if fresh.get("model_family") != "gru" or fresh.get("status") != "reserved-untouched":
        raise ValueError("current fresh validation namespace is not reserved for GRU")

    arena_name = str(freeze.get("shadow_arena", ""))
    arenas = {
        str(row.get("name")): row
        for row in shadow_registry.get("arenas", [])
        if isinstance(row, dict)
    }
    arena = arenas.get(arena_name)
    if not isinstance(arena, dict):
        raise ValueError("current shadow arena is not registered")
    if arena.get("model_family") != "gru" or arena.get("status") != "reserved-untouched":
        raise ValueError("current shadow arena is not reserved for GRU")

    protected: set[int] = {
        int(config.get("qualification_holdout_seed", -1)),
        *[int(value) for value in config.get("retired_qualification_holdout_seeds", [])],
    }
    for row in fresh_rows:
        namespace = int(row.get("namespace", -1))
        if namespace > 0:
            protected.add(base_seed + namespace)
    for row in arenas.values():
        protected.update(int(value) for value in row.get("seeds", []))

    model_namespace = int(policy.get("training_seed_namespace", 0))
    model_seeds = {
        base_seed + model_namespace + round_index * MODEL_SEED_STRIDE
        for round_index in range(int(policy.get("max_rounds", 0)))
    }
    overlap = sorted(effective & (protected | model_seeds))
    if overlap:
        raise ValueError(f"training acoustic exposure overlaps protected/model seed(s): {overlap}")
    return {
        "fresh_registry_entry": str(fresh.get("name", "")),
        "fresh_registry_status": str(fresh.get("status", "")),
        "shadow_arena": arena_name,
        "shadow_arena_status": str(arena.get("status", "")),
        "effective_exposure_seed_count": len(effective),
        "protected_seed_count": len(protected),
        "model_seed_count": len(model_seeds),
        "overlap_count": 0,
    }


def exposure_for_train_ordinal(ordinal: int, exposure_count: int) -> int:
    if ordinal < 0 or exposure_count <= 0:
        raise ValueError("train ordinal/exposure count is invalid")
    return ordinal % exposure_count


def _render_selected_train_row(
    canonical_row: dict,
    domains: dict,
    seed_offset: int,
    *,
    curriculum_weights: dict | None,
) -> dict:
    if str(canonical_row.get("split")) != "train":
        raise ValueError("direct multiseed renderer only accepts train rows")
    source = pathlib.Path(str(canonical_row.get("source_path", "")))
    target = pathlib.Path(str(canonical_row.get("path", "")))
    if not source.is_file() or not target.parent.is_dir():
        raise ValueError("canonical train source/target is missing")
    clean = base.renderer.read_wav(source)
    scene_seed = int(canonical_row["scene_seed"]) + int(seed_offset)
    rng = random.Random(scene_seed)
    scene = base.renderer.sample_scene(
        domains,
        rng,
        curriculum_weights=curriculum_weights,
        forced_band=None,
    )
    mono, scene_meta = base.renderer.render_scene(
        clean, scene, seed=scene_seed, afe=domains["afe"]
    )
    base.renderer.write_wav(target, mono)
    replacement = dict(canonical_row)
    replacement["scene_seed"] = scene_seed
    replacement["wav_sha256"] = base.sha256_file(target)
    replacement["scene"] = scene_meta
    replacement["domain_id"] = base._scene_domain_id(scene_meta)
    return replacement


def _render_selected_train_rows(
    config_path: pathlib.Path,
    canonical: list[dict],
    offsets: list[int],
    *,
    curriculum_weights: dict | None,
) -> tuple[list[dict], list[dict]]:
    if not offsets:
        raise ValueError("no acoustic exposure offsets")
    config = base.load_object(config_path.resolve())
    domains = base.renderer.validate_domains(config)
    base_seed = int(config.get("seed", 1337))
    selected_counts = [0] * len(offsets)
    merged: list[dict] = []
    train_ordinal = 0
    for canonical_row in canonical:
        if str(canonical_row.get("split")) != "train":
            merged.append(canonical_row)
            continue
        exposure = exposure_for_train_ordinal(train_ordinal, len(offsets))
        replacement = _render_selected_train_row(
            canonical_row,
            domains,
            offsets[exposure],
            curriculum_weights=curriculum_weights,
        )
        merged.append(replacement)
        selected_counts[exposure] += 1
        train_ordinal += 1
    expected = sum(str(row.get("split")) == "train" for row in canonical)
    if train_ordinal != expected:
        raise ValueError("train acoustic exposure merge count drifted")
    evidence = [
        {
            "exposure": exposure,
            "seed_offset": int(seed_offset),
            "effective_seed": base_seed + int(seed_offset),
            "selected_scene_count": selected_counts[exposure],
            "rendered_scene_count": selected_counts[exposure],
            "direct_selected_scene_render": True,
        }
        for exposure, seed_offset in enumerate(offsets)
    ]
    return merged, evidence


def install_multiseed_rotation(policy_path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    policy = base.load_object(policy_path)
    validate_exposure_namespace(policy)
    original_render = base.loop.render_domain_dataset
    rotations: list[dict] = []
    negative_stress_rounds: list[dict] = []

    def render_with_multiseed_train(
        config_path: pathlib.Path,
        output: pathlib.Path,
        *,
        curriculum_weights: dict | None = None,
    ) -> dict:
        output = output.resolve()
        round_index = base._round_index(output)
        offsets, round_stride, exposure_stride = exposure_offsets(policy, round_index)
        original_render(config_path, output, curriculum_weights=curriculum_weights)

        config = base.load_object(config_path.resolve())
        base_seed = int(config.get("seed", 1337))
        canonical_index = output / "domain-index.jsonl"
        canonical_rows = base._read_jsonl(canonical_index)
        merged_rows, exposure_evidence = _render_selected_train_rows(
            config_path,
            canonical_rows,
            offsets,
            curriculum_weights=curriculum_weights,
        )

        negative_stress = base._apply_negative_stress_support(config_path, output, merged_rows)
        negative_stress["round"] = round_index
        negative_stress_rounds.append(negative_stress)
        base._write_jsonl(canonical_index, merged_rows)

        total, by_split = base._histograms(merged_rows)
        summary_path = output / "domain-summary.json"
        summary = base.load_object(summary_path)
        summary["domain_index_sha256"] = base.sha256_file(canonical_index)
        summary["distance_histogram"] = total
        summary["distance_histogram_by_split"] = by_split
        summary["splits"]["train"]["manifest_sha256"] = base.sha256_file(output / "train.tsv")
        for split in ("calibration", "test"):
            references = output / f"{split}.references.jsonl"
            summary["splits"][split]["references_sha256"] = base.sha256_file(references)
        sampling = summary.get("evaluation_sampling")
        if isinstance(sampling, dict):
            sampling["development_negative_override"] = negative_stress

        rotation = {
            "policy": POLICY,
            "round": round_index,
            "seed_offset": offsets[0],
            "effective_seed": base_seed + offsets[0],
            "seed_stride": round_stride,
            "exposure_count": len(offsets),
            "exposure_stride": exposure_stride,
            "exposure_selection": "train-scene-ordinal-round-robin-v1",
            "exposures": exposure_evidence,
            "training_example_count_preserved": True,
            "base_utterance_reused": True,
            "canonical_evaluation_rows_reused": True,
            "exposure_rendering": "direct-selected-scene-v1",
            "auxiliary_full_dataset_rendered": False,
            "evaluation_seed_rotated": False,
        }
        summary["train_acoustic_rotation"] = rotation
        base.write_object(summary_path, summary)
        base._persist_rotation_round(output, rotation, negative_stress)
        rotations.append(dict(rotation))
        return summary

    base.loop.render_domain_dataset = render_with_multiseed_train
    return rotations, negative_stress_rounds


def retain_multiseed_wrapper_evidence(work: pathlib.Path, protection: dict) -> None:
    manifest_path = work / "development-loop-manifest.json"
    freeze_path = work / "frozen-candidate" / "freeze-manifest.json"
    if not manifest_path.is_file() or not freeze_path.is_file():
        raise ValueError("multiseed development evidence is incomplete")
    implementation = pathlib.Path(__file__).resolve()
    wrapper = {
        "policy": POLICY,
        "path": implementation.relative_to(ROOT).as_posix(),
        "sha256": base.sha256_file(implementation),
    }
    manifest = base.load_object(manifest_path)
    freeze = base.load_object(freeze_path)
    manifest["development_training_wrapper"] = wrapper
    manifest["training_acoustic_protected_seed_binding"] = protection
    freeze["development_training_wrapper"] = wrapper
    freeze["training_acoustic_protected_seed_binding"] = protection
    code = freeze.setdefault("training_code_sha256", {})
    if not isinstance(code, dict):
        raise ValueError("freeze training_code_sha256 must be an object")
    code[wrapper["path"]] = wrapper["sha256"]
    base.write_object(manifest_path, manifest)
    base.write_object(freeze_path, freeze)


def main() -> int:
    policy_path = base._argument_path("--policy")
    config_path = base._argument_path("--config")
    work = base._argument_path("--work-dir")
    policy = base.load_object(policy_path)
    config = base.load_object(config_path)
    protection = validate_protected_seed_independence(
        policy,
        config,
        base.load_object(FRESH_REGISTRY),
        base.load_object(SHADOW_REGISTRY),
    )
    base.POLICY = POLICY
    base.install_rotation = install_multiseed_rotation
    code = int(base.main())
    if code == 0:
        retain_multiseed_wrapper_evidence(work, protection)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
