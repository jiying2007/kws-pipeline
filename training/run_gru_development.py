#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import feature_cached_trainer as feature_cache  # noqa: E402
import iterate_gru_development as loop  # noqa: E402
import render_domains as renderer  # noqa: E402

POLICY = "development-train-acoustic-round-rotation-v1"
NEGATIVE_STRESS_POLICY = "gru-development-negative-stress-support-v1"
ROUND_STRIDE_FALLBACK = 1009


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _argument_path(name: str) -> pathlib.Path:
    try:
        index = sys.argv.index(name)
        value = sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"missing required argument: {name}") from exc
    return pathlib.Path(value).resolve()


def _round_index(output: pathlib.Path) -> int:
    name = output.name
    if not name.startswith("round-"):
        raise ValueError(f"cannot resolve development round from dataset path: {output}")
    return int(name.removeprefix("round-"))


def _rotation(policy: dict, round_index: int) -> tuple[int, int]:
    namespace = int(policy.get("training_acoustic_seed_namespace", 0))
    stride = int(policy.get("training_acoustic_seed_stride", ROUND_STRIDE_FALLBACK))
    if namespace <= 0 or stride <= 0:
        raise ValueError("training acoustic seed namespace/stride must be positive")
    model_namespace = int(policy.get("training_seed_namespace", 0))
    fresh_namespace = int(policy.get("candidate_freeze", {}).get("fresh_validation_seed_namespace", 0))
    if namespace in {model_namespace, fresh_namespace}:
        raise ValueError("training acoustic seed namespace overlaps model/fresh namespace")
    return namespace + round_index * stride, stride


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def _merge_domain_rows(canonical: list[dict], rotated: list[dict]) -> list[dict]:
    replacement = [row for row in rotated if str(row.get("split")) == "train"]
    expected = sum(str(row.get("split")) == "train" for row in canonical)
    if len(replacement) != expected:
        raise ValueError(
            f"train acoustic realization count drifted: expected {expected}, got {len(replacement)}"
        )
    iterator = iter(replacement)
    merged: list[dict] = []
    for row in canonical:
        merged.append(next(iterator) if str(row.get("split")) == "train" else row)
    return merged


def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _histograms(rows: list[dict]) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    total: dict[str, int] = {}
    by_split: dict[str, dict[str, int]] = {split: {} for split in renderer.SPLITS}
    for row in rows:
        split = str(row["split"])
        band = str(row["scene"]["distance_band"])
        total[band] = total.get(band, 0) + 1
        current = by_split.setdefault(split, {})
        current[band] = current.get(band, 0) + 1
    return total, by_split


def _scene_domain_id(scene: dict) -> str:
    playback = "on" if scene.get("playback_sir_db") is not None else "off"
    return (
        f"{scene['distance_band']}|az={int(round(float(scene['azimuth_deg'])))}|"
        f"rt60={float(scene['rt60_s']):.2f}|noise={scene['noise_profile']}|"
        f"playback={playback}"
    )


def _build_negative_stress_plan(
    config: dict,
    axes: dict,
    *,
    split: str,
    negative_scene_count: int,
) -> dict:
    gates = config.get("robustness_gates")
    if not isinstance(gates, dict):
        raise ValueError("robustness_gates config is required")
    target = int(gates.get("min_negative_recordings", 1))
    if target <= 0:
        raise ValueError("robustness_gates.min_negative_recordings must be > 0")
    effective = json.loads(json.dumps(config))
    effective_gates = effective.get("robustness_gates")
    if not isinstance(effective_gates, dict):
        raise ValueError("robustness_gates config is required")
    effective_gates["min_expected_wakes"] = target
    plan = renderer.build_development_stress_plan(
        effective,
        axes,
        split=split,
        positive_scene_count=negative_scene_count,
    )
    plan["support_kind"] = "negative"
    plan["negative_scene_count"] = negative_scene_count
    return plan


def _apply_negative_stress_support(
    config_path: pathlib.Path,
    output: pathlib.Path,
    rows: list[dict],
) -> dict:
    config = renderer.load_config(config_path.resolve())
    domains = renderer.validate_domains(config)
    axes = renderer._evaluation_axes(config, domains)
    evidence = {
        "policy": NEGATIVE_STRESS_POLICY,
        "development_only": True,
        "qualification_overridden": False,
        "validation_feedback_used": False,
        "splits": {},
    }
    if axes is None:
        evidence["applied"] = False
        evidence["reason"] = "evaluation-axes-unavailable"
        return evidence

    modified_by_split: dict[str, list[dict]] = {}
    for split in ("calibration", "test"):
        negative_indices = [
            index
            for index, row in enumerate(rows)
            if str(row.get("split")) == split and str(row.get("kind")) != "positive"
        ]
        plan = _build_negative_stress_plan(
            config,
            axes,
            split=split,
            negative_scene_count=len(negative_indices),
        )
        modified: list[dict] = []
        for ordinal, row_index in enumerate(negative_indices):
            stress_spec = plan["slots"].get(ordinal)
            if stress_spec is None:
                continue
            row = dict(rows[row_index])
            source = pathlib.Path(str(row["source_path"]))
            target = pathlib.Path(str(row["path"]))
            clean = renderer.read_wav(source)
            scene = renderer.apply_development_stress_scene(
                dict(row["scene"]),
                domains,
                axes,
                stress_spec,
            )
            mono, scene_meta = renderer.render_scene(
                clean,
                scene,
                seed=int(row["scene_seed"]),
                afe=domains["afe"],
            )
            renderer.write_wav(target, mono)
            row["wav_sha256"] = sha256_file(target)
            row["scene"] = scene_meta
            row["domain_id"] = _scene_domain_id(scene_meta)
            rows[row_index] = row
            modified.append(
                {
                    "ordinal": ordinal,
                    "path": str(target),
                    "wav_sha256": row["wav_sha256"],
                    "source_slice": str(stress_spec["source_slice"]),
                }
            )
        modified_by_split[split] = modified
        evidence["splits"][split] = {
            "negative_scene_capacity": len(negative_indices),
            "target_per_slice": int(plan["target_per_slice"]),
            "required_slices": list(plan["required_slices"]),
            "baseline_support": dict(plan["baseline_support"]),
            "planned_support": dict(plan["planned_support"]),
            "reserved_scenes": int(plan["reserved_scenes"]),
            "reserved_ordinals": sorted(int(value) for value in plan["slots"]),
            "pairwise_unique_before": dict(plan["pairwise_unique_before"]),
            "pairwise_unique_after": dict(plan["pairwise_unique_after"]),
        }

    for split, modified in modified_by_split.items():
        if not modified:
            continue
        references_path = output / f"{split}.references.jsonl"
        references = _read_jsonl(references_path)
        by_audio_path = {
            str(row.get("audio_path")): index for index, row in enumerate(references)
        }
        for item in modified:
            audio_path = str(item["path"])
            if audio_path not in by_audio_path:
                raise ValueError(f"negative stress reference missing for {audio_path}")
            reference = dict(references[by_audio_path[audio_path]])
            if reference.get("expected"):
                raise ValueError("negative stress override targeted positive reference")
            domain_row = next(
                row for row in rows if str(row.get("path")) == audio_path
            )
            reference["domain"] = domain_row["scene"]
            reference["domain_id"] = domain_row["domain_id"]
            reference["duration_s"] = (
                len(renderer.read_wav(pathlib.Path(audio_path))) / renderer.SAMPLE_RATE_HZ
            )
            references[by_audio_path[audio_path]] = reference
        _write_jsonl(references_path, references)

    evidence["applied"] = True
    return evidence


def _render_rotated_train_rows(
    config_path: pathlib.Path,
    output: pathlib.Path,
    canonical_rows: list[dict],
    seed_offset: int,
    *,
    curriculum_weights: dict | None,
) -> tuple[list[dict], pathlib.Path]:
    config = load_object(config_path.resolve())
    domains = renderer.validate_domains(config)
    rotated = output / "train-acoustic-realization"
    merged: list[dict] = []
    manifest_rows: list[tuple[pathlib.Path, list[int]]] = []
    for canonical_row in canonical_rows:
        if str(canonical_row.get("split")) != "train":
            merged.append(canonical_row)
            continue
        source = pathlib.Path(str(canonical_row.get("source_path", "")))
        canonical_target = pathlib.Path(str(canonical_row.get("path", ""))).resolve()
        if not source.is_file():
            raise ValueError("canonical train source is missing")
        try:
            relative = canonical_target.relative_to(output.resolve())
        except ValueError as exc:
            raise ValueError("canonical train target escaped development dataset") from exc
        target = rotated / relative
        clean = renderer.read_wav(source)
        scene_seed = int(canonical_row["scene_seed"]) + int(seed_offset)
        rng = random.Random(scene_seed)
        scene = renderer.sample_scene(
            domains,
            rng,
            curriculum_weights=curriculum_weights,
            forced_band=None,
        )
        mono, scene_meta = renderer.render_scene(
            clean, scene, seed=scene_seed, afe=domains["afe"]
        )
        renderer.write_wav(target, mono)
        replacement = dict(canonical_row)
        replacement["path"] = str(target.resolve())
        replacement["scene_seed"] = scene_seed
        replacement["wav_sha256"] = sha256_file(target)
        replacement["scene"] = scene_meta
        replacement["domain_id"] = _scene_domain_id(scene_meta)
        merged.append(replacement)
        manifest_rows.append((target.resolve(), [int(v) for v in replacement["target_ids"]]))
    manifest = rotated / "train.tsv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(
            f"{target}\t{' '.join(str(v) for v in target_ids)}\n"
            for target, target_ids in manifest_rows
        ),
        encoding="utf-8",
    )
    return merged, manifest


def _reuse_canonical_base(canonical_base: pathlib.Path):
    def reuse(_config: pathlib.Path, target: pathlib.Path) -> dict:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("dataset-index.jsonl", "dataset-summary.json"):
            shutil.copy2(canonical_base / name, target / name)
        return load_object(target / "dataset-summary.json")

    return reuse


def install_rotation(policy_path: pathlib.Path) -> tuple[list[dict], list[dict]]:
    policy = load_object(policy_path)
    original_render = loop.render_domain_dataset
    rotations: list[dict] = []
    negative_stress_rounds: list[dict] = []

    def render_with_rotated_train(
        config_path: pathlib.Path,
        output: pathlib.Path,
        *,
        curriculum_weights: dict | None = None,
    ) -> dict:
        output = output.resolve()
        round_index = _round_index(output)
        seed_offset, stride = _rotation(policy, round_index)
        original_render(
            config_path,
            output,
            curriculum_weights=curriculum_weights,
        )

        config = load_object(config_path.resolve())
        base_seed = int(config.get("seed", 1337))
        config["seed"] = base_seed + seed_offset
        rotated_config = output / "train-acoustic-effective-config.json"
        write_object(rotated_config, config)
        canonical_index = output / "domain-index.jsonl"
        merged_rows, rotated_train_manifest = _render_rotated_train_rows(
            config_path,
            output,
            _read_jsonl(canonical_index),
            seed_offset,
            curriculum_weights=curriculum_weights,
        )
        shutil.copy2(rotated_train_manifest, output / "train.tsv")
        negative_stress = _apply_negative_stress_support(
            config_path,
            output,
            merged_rows,
        )
        negative_stress["round"] = round_index
        negative_stress_rounds.append(negative_stress)
        _write_jsonl(canonical_index, merged_rows)

        total, by_split = _histograms(merged_rows)
        summary_path = output / "domain-summary.json"
        summary = load_object(summary_path)
        summary["domain_index_sha256"] = sha256_file(canonical_index)
        summary["distance_histogram"] = total
        summary["distance_histogram_by_split"] = by_split
        summary["splits"]["train"]["manifest_sha256"] = sha256_file(output / "train.tsv")
        for split in ("calibration", "test"):
            references = output / f"{split}.references.jsonl"
            summary["splits"][split]["references_sha256"] = sha256_file(references)
        sampling = summary.get("evaluation_sampling")
        if isinstance(sampling, dict):
            sampling["development_negative_override"] = negative_stress
        summary["train_acoustic_rotation"] = {
            "policy": POLICY,
            "round": round_index,
            "seed_offset": seed_offset,
            "effective_seed": base_seed + seed_offset,
            "seed_stride": stride,
            "base_utterance_reused": True,
            "canonical_evaluation_rows_reused": True,
            "acoustic_rendering": "direct-train-scene-v1",
            "auxiliary_full_dataset_rendered": False,
            "evaluation_seed_rotated": False,
        }
        write_object(summary_path, summary)
        rotations.append(dict(summary["train_acoustic_rotation"]))
        return summary

    loop.render_domain_dataset = render_with_rotated_train
    return rotations, negative_stress_rounds


def retain_rotation_evidence(
    work: pathlib.Path,
    rotations: list[dict],
    negative_stress_rounds: list[dict],
) -> None:
    if not rotations:
        raise ValueError("development loop produced no train acoustic rotation evidence")
    if len(negative_stress_rounds) != len(rotations):
        raise ValueError("development negative stress evidence is incomplete")
    manifest_path = work / "development-loop-manifest.json"
    freeze_path = work / "frozen-candidate" / "freeze-manifest.json"
    manifest = load_object(manifest_path)
    freeze = load_object(freeze_path)
    manifest["training_acoustic_rotation"] = {
        "policy": POLICY,
        "rounds": rotations,
        "evaluation_seed_rotated": False,
        "base_utterance_reused": True,
    }
    manifest["development_negative_stress_support"] = {
        "policy": NEGATIVE_STRESS_POLICY,
        "development_only": True,
        "qualification_overridden": False,
        "rounds": negative_stress_rounds,
    }
    by_round = {int(row["round"]): row for row in rotations}
    negative_by_round = {int(row["round"]): row for row in negative_stress_rounds}
    for record in manifest.get("records", []):
        round_index = int(record["round"])
        if round_index not in by_round:
            raise ValueError(f"missing train acoustic rotation evidence for round {round_index}")
        if round_index not in negative_by_round:
            raise ValueError(f"missing negative stress evidence for round {round_index}")
        record.setdefault("training", {})["acoustic_seed_offset"] = int(
            by_round[round_index]["seed_offset"]
        )
        record["training"]["acoustic_effective_seed"] = int(
            by_round[round_index]["effective_seed"]
        )
        record["development_negative_stress_support"] = negative_by_round[round_index]
    write_object(manifest_path, manifest)

    code = freeze.setdefault("training_code_sha256", {})
    if not isinstance(code, dict):
        raise ValueError("freeze training_code_sha256 must be an object")
    code[pathlib.Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha256_file(
        pathlib.Path(__file__).resolve()
    )
    freeze["training_acoustic_rotation"] = manifest["training_acoustic_rotation"]
    freeze["development_negative_stress_support"] = manifest[
        "development_negative_stress_support"
    ]
    write_object(freeze_path, freeze)


def main() -> int:
    policy_path = _argument_path("--policy")
    work = _argument_path("--work-dir")
    feature_cache.install_development_feature_cache(loop, load_object(policy_path))
    rotations, negative_stress_rounds = install_rotation(policy_path)
    code = int(loop.main())
    if code == 0:
        retain_rotation_evidence(work, rotations, negative_stress_rounds)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
