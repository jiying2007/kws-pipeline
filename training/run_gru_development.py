#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import iterate_gru_development as loop  # noqa: E402
import render_domains as renderer  # noqa: E402

POLICY = "development-train-acoustic-round-rotation-v1"
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


def _reuse_canonical_base(canonical_base: pathlib.Path):
    def reuse(_config: pathlib.Path, target: pathlib.Path) -> dict:
        target.mkdir(parents=True, exist_ok=True)
        for name in ("dataset-index.jsonl", "dataset-summary.json"):
            shutil.copy2(canonical_base / name, target / name)
        return load_object(target / "dataset-summary.json")

    return reuse


def install_rotation(policy_path: pathlib.Path) -> list[dict]:
    policy = load_object(policy_path)
    original_render = loop.render_domain_dataset
    rotations: list[dict] = []

    def render_with_rotated_train(
        config_path: pathlib.Path,
        output: pathlib.Path,
        *,
        curriculum_weights: dict | None = None,
    ) -> dict:
        output = output.resolve()
        round_index = _round_index(output)
        seed_offset, stride = _rotation(policy, round_index)
        canonical = original_render(
            config_path,
            output,
            curriculum_weights=curriculum_weights,
        )

        config = load_object(config_path.resolve())
        base_seed = int(config.get("seed", 1337))
        config["seed"] = base_seed + seed_offset
        rotated_config = output / "train-acoustic-effective-config.json"
        write_object(rotated_config, config)
        rotated = output / "train-acoustic-realization"

        original_generate = renderer.generate_dataset
        renderer.generate_dataset = _reuse_canonical_base(output / "base")
        try:
            original_render(
                rotated_config,
                rotated,
                curriculum_weights=curriculum_weights,
            )
        finally:
            renderer.generate_dataset = original_generate

        shutil.copy2(rotated / "train.tsv", output / "train.tsv")
        canonical_index = output / "domain-index.jsonl"
        rotated_index = rotated / "domain-index.jsonl"
        merged_rows = _merge_domain_rows(
            _read_jsonl(canonical_index),
            _read_jsonl(rotated_index),
        )
        _write_jsonl(canonical_index, merged_rows)

        total, by_split = _histograms(merged_rows)
        summary_path = output / "domain-summary.json"
        summary = load_object(summary_path)
        summary["domain_index_sha256"] = sha256_file(canonical_index)
        summary["distance_histogram"] = total
        summary["distance_histogram_by_split"] = by_split
        summary["splits"]["train"]["manifest_sha256"] = sha256_file(output / "train.tsv")
        summary["train_acoustic_rotation"] = {
            "policy": POLICY,
            "round": round_index,
            "seed_offset": seed_offset,
            "effective_seed": base_seed + seed_offset,
            "seed_stride": stride,
            "base_utterance_reused": True,
            "evaluation_seed_rotated": False,
        }
        write_object(summary_path, summary)
        rotations.append(dict(summary["train_acoustic_rotation"]))
        return summary

    loop.render_domain_dataset = render_with_rotated_train
    return rotations


def retain_rotation_evidence(work: pathlib.Path, rotations: list[dict]) -> None:
    if not rotations:
        raise ValueError("development loop produced no train acoustic rotation evidence")
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
    by_round = {int(row["round"]): row for row in rotations}
    for record in manifest.get("records", []):
        round_index = int(record["round"])
        if round_index not in by_round:
            raise ValueError(f"missing train acoustic rotation evidence for round {round_index}")
        record.setdefault("training", {})["acoustic_seed_offset"] = int(
            by_round[round_index]["seed_offset"]
        )
        record["training"]["acoustic_effective_seed"] = int(
            by_round[round_index]["effective_seed"]
        )
    write_object(manifest_path, manifest)

    code = freeze.setdefault("training_code_sha256", {})
    if not isinstance(code, dict):
        raise ValueError("freeze training_code_sha256 must be an object")
    code[pathlib.Path(__file__).resolve().relative_to(ROOT).as_posix()] = sha256_file(
        pathlib.Path(__file__).resolve()
    )
    freeze["training_acoustic_rotation"] = manifest["training_acoustic_rotation"]
    write_object(freeze_path, freeze)


def main() -> int:
    policy_path = _argument_path("--policy")
    work = _argument_path("--work-dir")
    rotations = install_rotation(policy_path)
    code = int(loop.main())
    if code == 0:
        retain_rotation_evidence(work, rotations)
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
