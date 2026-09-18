#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from external_base_dataset import SPLITS, load_external_base_bundle  # noqa: E402

MATRIX_CLASS = "kws-v2-staged-experiment-matrix-v1"
MATRIX_POLICY = "kws-v2-staged-experiments-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def relative_ref(target: pathlib.Path, config_dir: pathlib.Path) -> str:
    return pathlib.Path(os.path.relpath(target.resolve(), config_dir.resolve())).as_posix()


def validate_matrix(path: pathlib.Path) -> dict:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError("experiment matrix schema_version must be 1")
    if value.get("policy") != MATRIX_POLICY or value.get("evidence_class") != MATRIX_CLASS:
        raise ValueError("experiment matrix identity mismatch")
    if value.get("evidence_scope") != "development-only":
        raise ValueError("experiment matrix must be development-only")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if value.get(key) is not False:
            raise ValueError(f"protected evidence flag {key} must be false")
    if value.get("requires_predeclared_generalization_plan") is not True:
        raise ValueError("experiment matrix must require predeclared generalization plan")
    cohort_id = value.get("generalization_cohort_id")
    if not isinstance(cohort_id, str) or not cohort_id.strip() or len(cohort_id) > 128:
        raise ValueError("experiment matrix generalization_cohort_id is invalid")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-" for ch in cohort_id):
        raise ValueError("experiment matrix generalization_cohort_id contains unsupported characters")
    return value


def candidate_from_matrix(matrix_path: pathlib.Path, matrix: dict, candidate_id: str) -> tuple[dict, pathlib.Path]:
    rows = matrix.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("experiment matrix candidates must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"candidate_id must match exactly one matrix row: {candidate_id}")
    row = matches[0]
    if row.get("protected_evidence_used") is not False:
        raise ValueError("candidate row protected_evidence_used must be false")
    if row.get("generalization_cohort_id") != matrix["generalization_cohort_id"]:
        raise ValueError("candidate generalization_cohort_id must match experiment matrix")
    if row.get("config_path_contract") != "matrix-relative-v1":
        raise ValueError("candidate config path contract mismatch")
    raw = row.get("config")
    if not isinstance(raw, str) or not raw or pathlib.Path(raw).is_absolute():
        raise ValueError("candidate config must be matrix-relative")
    path = (matrix_path.parent / raw).resolve()
    if not path.is_file():
        raise ValueError(f"candidate config is missing: {path}")
    expected = row.get("config_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError("candidate config sha256 mismatch")
    return row, path


def split_inputs(args: argparse.Namespace) -> dict[str, tuple[pathlib.Path, pathlib.Path]]:
    result: dict[str, tuple[pathlib.Path, pathlib.Path]] = {}
    for split in SPLITS:
        index = getattr(args, f"{split}_index").resolve()
        summary = getattr(args, f"{split}_summary").resolve()
        if not index.is_file() or not summary.is_file():
            raise ValueError(f"{split} index/summary must exist")
        result[split] = (index, summary)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bind one staged KWS v2 candidate to one immutable speech-like external base bundle."
    )
    parser.add_argument("--matrix", required=True, type=pathlib.Path)
    parser.add_argument("--candidate-id", required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}-index", required=True, type=pathlib.Path)
        parser.add_argument(f"--{split}-summary", required=True, type=pathlib.Path)
    parser.add_argument("--output-config", required=True, type=pathlib.Path)
    parser.add_argument("--output-binding", required=True, type=pathlib.Path)
    args = parser.parse_args()

    matrix_path = args.matrix.resolve()
    matrix = validate_matrix(matrix_path)
    row, candidate_config_path = candidate_from_matrix(matrix_path, matrix, args.candidate_id)
    candidate = load_object(candidate_config_path)
    speech_like = candidate.get("kws_v2_speech_like_training")
    if not isinstance(speech_like, dict):
        raise ValueError("candidate is missing kws_v2_speech_like_training contract")
    if speech_like.get("external_base_required") is not True:
        raise ValueError("candidate must require external speech-like base")
    if speech_like.get("tone_replay_allowed") is not False:
        raise ValueError("candidate must forbid tone replay")
    if speech_like.get("fixed_hard_negative_replay") != "disabled":
        raise ValueError("candidate fixed hard-negative replay must be disabled")
    if speech_like.get("positive_stress_replay") != "disabled":
        raise ValueError("candidate positive-stress replay must be disabled")
    if speech_like.get("failure_replay") != "disabled":
        raise ValueError("candidate failure replay must be disabled")
    if candidate.get("data_augmentation_v3", {}).get("failure_replay_enabled") is not False:
        raise ValueError("candidate failure replay must be disabled in data_augmentation_v3")
    iteration = candidate.get("domain_iteration")
    if not isinstance(iteration, dict):
        raise ValueError("candidate domain_iteration must be an object")
    if iteration.get("hard_negative_replay") != [] or iteration.get("positive_stress_replay") != []:
        raise ValueError("candidate synthetic replay lists must be empty")
    inputs = split_inputs(args)

    output_config = args.output_config.resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)
    effective = json.loads(json.dumps(candidate))
    generator = effective.setdefault("generator", {})
    if not isinstance(generator, dict):
        raise ValueError("candidate generator must be an object")
    if "external_base_dataset" in generator:
        raise ValueError("candidate already declares external_base_dataset")
    spec: dict[str, dict] = {}
    for split, (index, summary) in inputs.items():
        spec[split] = {
            "index": relative_ref(index, output_config.parent),
            "summary": relative_ref(summary, output_config.parent),
            "index_sha256": sha256_file(index),
            "summary_sha256": sha256_file(summary),
        }
    generator["external_base_dataset"] = spec
    write_json(output_config, effective)

    loaded = load_object(output_config)
    validated = load_external_base_bundle(output_config, loaded)
    if validated is None:
        raise ValueError("effective config failed to declare external base dataset")
    _, bundle = validated
    if bundle.get("tone_backend_used") is not False:
        raise ValueError("speech-like bundle must not use tone backend")

    binding = {
        "schema_version": 1,
        "evidence_class": "kws-v2-runnable-candidate-binding-v1",
        "evidence_scope": "development-only",
        "candidate_id": args.candidate_id,
        "stage": matrix["stage"],
        "model_family": row["model_family"],
        "frontend": row["frontend"],
        "hidden_dim": int(row["hidden_dim"]),
        "matrix_sha256": sha256_file(matrix_path),
        "candidate_config_sha256": sha256_file(candidate_config_path),
        "effective_config_sha256": sha256_file(output_config),
        "external_base_bundle_sha256": bundle["bundle_sha256"],
        "split_corpus_sha256": {
            split: bundle["splits"][split]["corpus_sha256"] for split in SPLITS
        },
        "resource_contract_candidate": row["resource_contract_candidate"],
        "generalization_tier": row["generalization_tier"],
        "generalization_cohort_id": row["generalization_cohort_id"],
        "requires_predeclared_generalization_plan": True,
        "speech_like_external_base_required": True,
        "tone_replay_allowed": False,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
    }
    write_json(args.output_binding.resolve(), binding)
    print(
        f"kws-v2-runnable-binding: candidate={args.candidate_id} "
        f"bundle={bundle['bundle_sha256']} effective={binding['effective_config_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
