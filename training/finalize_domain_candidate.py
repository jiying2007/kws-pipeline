#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from iterate_domain import base_gate, domain_gate, evaluate, gate_values  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize the latest calibration/test strict-gate-passing domain candidate "
            "and bind it to the hard-negative replay used to train that same round."
        )
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runner", required=True)
    parser.add_argument("--iteration-exit-code", type=int, required=True)
    return parser.parse_args()


def require_file(path: pathlib.Path, label: str) -> pathlib.Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    return path


def main() -> int:
    args = parse_args()
    root = repo_path(args.work_dir)
    config_path = repo_path(args.config)
    runner = repo_path(args.runner)
    manifest_path = root / "domain-loop-manifest.json"
    require_file(manifest_path, "domain-loop manifest")
    require_file(config_path, "training config")
    require_file(runner, "qualification runner")

    iteration_exit = int(args.iteration_exit_code)
    if iteration_exit not in (0, 1):
        raise RuntimeError(f"training infrastructure failed with exit code {iteration_exit}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    eligible = [
        row
        for row in manifest["records"]
        if bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not eligible:
        raise RuntimeError("no calibration/test gate-passing torch candidate")

    # Each full warm-start round consumes the replay rendered for that exact
    # round before training. Never select an earlier model against a newer
    # replay merely because its objective is slightly lower.
    latest_round = max(int(row["round"]) for row in eligible)
    latest = [row for row in eligible if int(row["round"]) == latest_round]
    selected = min(latest, key=lambda row: (float(row["score"]), str(row["frontend"])))
    selected_round = int(selected["round"])
    selected_frontend = str(selected["frontend"])
    selected_score = float(selected["score"])
    score_floor = min(float(row["score"]) for row in eligible)
    original_round = int(manifest["best_round"])
    original_frontend = str(manifest["best_frontend"])
    eligible_rounds = sorted({int(row["round"]) for row in eligible})
    reselected = (selected_round, selected_frontend) != (original_round, original_frontend)

    print(
        f"candidate-selection eligible_rounds={eligible_rounds} "
        f"score_floor={score_floor:.9f} original={original_round}/{original_frontend} "
        f"selected={selected_round}/{selected_frontend} score={selected_score:.9f}"
    )

    best_dir = root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        repo_path(str(selected["model"])): best_dir / "model.kwm",
        repo_path(str(selected["checkpoint"])): best_dir / "model.pt",
        repo_path(str(selected["pack"])): best_dir / "keywords.kwk",
        repo_path(str(selected["keywords"])): best_dir / "keywords.tsv",
        repo_path(str(selected["provenance"])): best_dir / "model-provenance.json",
    }
    for source, destination in copies.items():
        require_file(source, "selected candidate artifact")
        shutil.copy2(source, destination)

    # Replay round-N is rendered before training round N and its SHA is written
    # into that candidate record. Bind the final model to this exact replay and
    # freeze the pair under best/ for every downstream FAR gate.
    replay = root / "hard-negative-replay" / f"round-{selected_round:02d}" / "hard-negatives.tsv"
    replay_evidence = replay.with_name("hard-negatives.json")
    require_file(replay, "selected-round hard-negative replay")
    require_file(replay_evidence, "selected-round hard-negative replay evidence")
    replay_meta = json.loads(replay_evidence.read_text(encoding="utf-8"))
    replay_sha = sha256_file(replay)
    expected_replay_sha = str(selected.get("hard_negative_replay_manifest_sha256") or "")
    if int(replay_meta.get("round", -1)) != selected_round:
        raise RuntimeError(
            f"hard-negative replay round mismatch: selected={selected_round} "
            f"evidence={replay_meta.get('round')}"
        )
    if str(replay_meta.get("manifest_sha256") or "") != replay_sha:
        raise RuntimeError("hard-negative replay evidence SHA does not match manifest")
    if not expected_replay_sha or expected_replay_sha != replay_sha:
        raise RuntimeError(
            "selected candidate hard-negative replay SHA does not match its training record"
        )
    frozen_replay = best_dir / "hard-negatives.tsv"
    shutil.copy2(replay, frozen_replay)

    # Selection above uses calibration/test evidence only. Re-run untouched
    # qualification after selection so qualification cannot influence candidate
    # choice, curriculum, or replay generation.
    qualification_dir = best_dir / "qualification"
    if qualification_dir.exists():
        shutil.rmtree(qualification_dir)
    references = root / "qualification-dataset" / "qualification.references.jsonl"
    require_file(references, "qualification holdout")
    qualification, qualification_domains = evaluate(
        runner=runner,
        model=(best_dir / "model.kwm").resolve(),
        pack=(best_dir / "keywords.kwk").resolve(),
        references=references.resolve(),
        output=qualification_dir.resolve(),
    )
    gates = gate_values(config.get("domain_gates", {}))
    qualified = base_gate(qualification, gates) and domain_gate(qualification_domains, gates)

    candidate_selection = {
        "policy": "latest-strict-gate-passing-round",
        "eligible_rounds": eligible_rounds,
        "score_floor": score_floor,
        "original_round": original_round,
        "original_frontend": original_frontend,
        "selected_round": selected_round,
        "selected_frontend": selected_frontend,
        "selected_score": selected_score,
        "reselected": reselected,
        "qualification_used_for_selection": False,
        "replay_round": selected_round,
        "replay_source": str(replay.relative_to(ROOT)),
        "replay_sha256": replay_sha,
        "model_replay_generation_match": True,
    }

    manifest["best_round"] = selected_round
    manifest["best_frontend"] = selected_frontend
    manifest["best_score"] = selected_score
    manifest["best_model_sha256"] = sha256_file(best_dir / "model.kwm")
    manifest["best_pack_sha256"] = sha256_file(best_dir / "keywords.kwk")
    manifest["best_hard_negative_manifest_sha256"] = replay_sha
    manifest["qualification"] = qualification
    manifest["qualification_domains"] = qualification_domains
    manifest["qualified"] = bool(qualified)
    manifest["candidate_selection"] = candidate_selection
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    required = [
        best_dir / "model.kwm",
        best_dir / "model.pt",
        best_dir / "keywords.kwk",
        best_dir / "keywords.tsv",
        best_dir / "model-provenance.json",
        best_dir / "hard-negatives.tsv",
    ]
    for path in required:
        require_file(path, "final training artifact")

    summary = {
        "schema_version": 2,
        "iteration_exit_code": iteration_exit,
        "training_exit_code": 0 if qualified else 1,
        "qualified": bool(qualified),
        "best_round": selected_round,
        "best_frontend": selected_frontend,
        "best_score": selected_score,
        "candidate_selection": candidate_selection,
        "qualification": qualification,
        "qualification_domains": qualification_domains,
        "artifacts": {path.name: sha256_file(path) for path in required},
        "limitations": manifest["limitations"],
    }
    summary_path = root / "training-run-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
