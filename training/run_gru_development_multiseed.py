#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import run_gru_development as base  # noqa: E402

POLICY = "development-train-acoustic-multiseed-v1"
EXPECTED_EXPOSURES = 3


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


def validate_exposure_namespace(policy: dict) -> None:
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


def exposure_for_train_ordinal(ordinal: int, exposure_count: int) -> int:
    if ordinal < 0 or exposure_count <= 0:
        raise ValueError("train ordinal/exposure count is invalid")
    return ordinal % exposure_count


def _selected_train_rows(
    canonical: list[dict],
    exposure_rows: list[list[dict]],
) -> tuple[list[dict], list[int]]:
    if not exposure_rows:
        raise ValueError("no acoustic exposure rows")
    expected = sum(str(row.get("split")) == "train" for row in canonical)
    for rows in exposure_rows:
        if len(rows) != expected:
            raise ValueError(
                f"train acoustic realization count drifted: expected {expected}, got {len(rows)}"
            )
    selected_counts = [0] * len(exposure_rows)
    merged: list[dict] = []
    train_ordinal = 0
    for canonical_row in canonical:
        if str(canonical_row.get("split")) != "train":
            merged.append(canonical_row)
            continue
        exposure = exposure_for_train_ordinal(train_ordinal, len(exposure_rows))
        replacement = dict(exposure_rows[exposure][train_ordinal])
        source = pathlib.Path(str(replacement["path"]))
        target = pathlib.Path(str(canonical_row["path"]))
        if not source.is_file() or not target.parent.is_dir():
            raise ValueError("acoustic exposure source/canonical target is missing")
        shutil.copy2(source, target)
        replacement["path"] = str(target)
        replacement["wav_sha256"] = base.sha256_file(target)
        merged.append(replacement)
        selected_counts[exposure] += 1
        train_ordinal += 1
    if train_ordinal != expected:
        raise ValueError("train acoustic exposure merge count drifted")
    return merged, selected_counts


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
        original_render(
            config_path,
            output,
            curriculum_weights=curriculum_weights,
        )

        config = base.load_object(config_path.resolve())
        base_seed = int(config.get("seed", 1337))
        exposure_train_rows: list[list[dict]] = []
        exposure_evidence: list[dict] = []
        for exposure_index, seed_offset in enumerate(offsets):
            effective = json.loads(json.dumps(config))
            effective["seed"] = base_seed + seed_offset
            exposure_root = output / f"train-acoustic-realization-{exposure_index:02d}"
            effective_path = output / f"train-acoustic-effective-config-{exposure_index:02d}.json"
            base.write_object(effective_path, effective)

            original_generate = base.renderer.generate_dataset
            base.renderer.generate_dataset = base._reuse_canonical_base(output / "base")
            try:
                original_render(
                    effective_path,
                    exposure_root,
                    curriculum_weights=curriculum_weights,
                )
            finally:
                base.renderer.generate_dataset = original_generate

            rows = base._read_jsonl(exposure_root / "domain-index.jsonl")
            train_rows = [row for row in rows if str(row.get("split")) == "train"]
            exposure_train_rows.append(train_rows)
            exposure_evidence.append(
                {
                    "exposure": exposure_index,
                    "seed_offset": seed_offset,
                    "effective_seed": base_seed + seed_offset,
                    "train_scene_count": len(train_rows),
                }
            )

        canonical_index = output / "domain-index.jsonl"
        canonical_rows = base._read_jsonl(canonical_index)
        merged_rows, selected_counts = _selected_train_rows(
            canonical_rows,
            exposure_train_rows,
        )
        for item, selected_count in zip(exposure_evidence, selected_counts):
            item["selected_scene_count"] = selected_count

        negative_stress = base._apply_negative_stress_support(
            config_path,
            output,
            merged_rows,
        )
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
            "evaluation_seed_rotated": False,
        }
        summary["train_acoustic_rotation"] = rotation
        base.write_object(summary_path, summary)
        rotations.append(dict(rotation))
        return summary

    base.loop.render_domain_dataset = render_with_multiseed_train
    return rotations, negative_stress_rounds


def main() -> int:
    base.POLICY = POLICY
    base.install_rotation = install_multiseed_rotation
    return int(base.main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
