#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

import iterate_gru_development as shared  # noqa: E402

POLICY = "rnn-development-curriculum-loop-v1"
FREEZE_POLICY = "rnn-frozen-candidate-v1"
MODEL_FAMILY = "rnn"
ARCHITECTURE = "tiny-streaming-rnn-v1"

_original_run = shared.run


def _translated_run(command: list[str]) -> None:
    translated = list(command)
    if len(translated) >= 2:
        script = pathlib.Path(translated[1]).name
        if script == "train_gru_ctc.py":
            translated[1] = str(TRAINING / "train_ctc.py")
        elif script == "export_gru_model.py":
            translated[1] = str(TRAINING / "export_model.py")
    _original_run(translated)


def _copy_frozen_candidate(
    selected: dict,
    output: pathlib.Path,
    config: pathlib.Path,
    policy: pathlib.Path,
) -> dict:
    frozen = output / "frozen-candidate"
    frozen.mkdir(parents=True, exist_ok=True)
    mapping = (
        (pathlib.Path(selected["model"]), "model.kwm"),
        (pathlib.Path(selected["checkpoint"]), "model.pt"),
        (pathlib.Path(selected["provenance"]), "model-provenance.json"),
        (pathlib.Path(selected["pack"]), "keywords.kwk"),
        (pathlib.Path(selected["keywords"]), "keywords.tsv"),
    )
    for source, name in mapping:
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"selected RNN candidate member missing: {source}")
        shutil.copy2(source, frozen / name)

    provenance_path = frozen / "model-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if isinstance(provenance, dict) and isinstance(provenance.get("model"), dict):
        provenance["model"]["name"] = "model.kwm"
        provenance["model"]["model_family"] = MODEL_FAMILY
        provenance["model"]["architecture"] = ARCHITECTURE
        provenance_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    code_paths = [
        pathlib.Path(__file__).resolve(),
        TRAINING / "train_ctc.py",
        TRAINING / "model.py",
        TRAINING / "export_model.py",
        TRAINING / "domain_curriculum.py",
        TRAINING / "hard_negative_replay.py",
        TRAINING / "development_failure_replay.py",
    ]
    freeze = {
        "schema_version": 1,
        "policy": FREEZE_POLICY,
        "source_policy": POLICY,
        "model_family": MODEL_FAMILY,
        "architecture": ARCHITECTURE,
        "evidence_scope": shared.EVIDENCE_SCOPE,
        "selection_policy": shared.SELECTION_POLICY,
        "selected_round": int(selected["round"]),
        "selected_score": float(selected["score"]),
        "model_sha256": shared.sha256_file(frozen / "model.kwm"),
        "checkpoint_sha256": shared.sha256_file(frozen / "model.pt"),
        "pack_sha256": shared.sha256_file(frozen / "keywords.kwk"),
        "keywords_sha256": shared.sha256_file(frozen / "keywords.tsv"),
        "provenance_sha256": shared.sha256_file(provenance_path),
        "config_sha256": shared.sha256_file(config),
        "development_policy_sha256": shared.sha256_file(policy),
        "training_code_sha256": {
            path.relative_to(ROOT).as_posix(): shared.sha256_file(path)
            for path in code_paths
        },
        "selection_evidence": ["development-calibration", "development-test"],
        "qualification_used_for_selection": False,
        "shadow_used_for_selection": False,
        "formal_qualification_used_for_selection": False,
        "candidate_stage": {
            "fresh_validation_required": True,
            "shadow_required": True,
            "formal_qualification_required": True,
            "bounded_repair_only": True,
            "validation_feedback_allowed": False,
            "threshold_feedback_allowed": False,
            "training_rule_feedback_allowed": False,
        },
    }
    (frozen / "freeze-manifest.json").write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return freeze


def main() -> int:
    shared.POLICY = POLICY
    shared.FREEZE_POLICY = FREEZE_POLICY
    shared.run = _translated_run
    shared.copy_frozen_candidate = _copy_frozen_candidate
    return int(shared.main())


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
