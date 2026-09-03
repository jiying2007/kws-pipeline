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

from hard_negative_replay import render_hard_negative_replay  # noqa: E402
from iterate_domain import base_gate, domain_gate, evaluate, gate_values  # noqa: E402

# render_hard_negative_replay uses round_index only as deterministic seed input
# plus evidence metadata. Keep FAR holdout synthesis in a disjoint seed namespace
# while preserving the selected round's acoustic curriculum distribution.
FAR_HOLDOUT_ROUND_NAMESPACE = 1_000_000


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
            "Finalize the latest calibration/test strict-gate-passing domain candidate, "
            "prove its training-replay generation, and render an independent FAR holdout."
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


def manifest_clip_hashes(path: pathlib.Path) -> set[str]:
    hashes: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if not cols or not cols[0].strip():
            raise RuntimeError(f"{path}:{line_no}: missing hard-negative WAV path")
        source = pathlib.Path(cols[0])
        if not source.is_absolute():
            source = (path.parent / source).resolve()
        else:
            source = source.resolve()
        require_file(source, "hard-negative WAV")
        digest = sha256_file(source)
        if digest in hashes:
            raise RuntimeError(f"duplicate hard-negative WAV content in {path}: {digest}")
        hashes.add(digest)
    if not hashes:
        raise RuntimeError(f"hard-negative manifest has no clips: {path}")
    return hashes


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
    # training replay merely because its objective is slightly lower.
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

    # Training replay round N is rendered before training round N and its SHA is
    # recorded in the candidate. Freeze it only as generation/provenance proof;
    # it must never serve as the final continuous-FAR validation corpus.
    training_replay = (
        root / "hard-negative-replay" / f"round-{selected_round:02d}" / "hard-negatives.tsv"
    )
    training_replay_evidence = training_replay.with_name("hard-negatives.json")
    require_file(training_replay, "selected-round training hard-negative replay")
    require_file(training_replay_evidence, "selected-round training replay evidence")
    training_replay_meta = json.loads(training_replay_evidence.read_text(encoding="utf-8"))
    training_replay_sha = sha256_file(training_replay)
    expected_replay_sha = str(selected.get("hard_negative_replay_manifest_sha256") or "")
    if int(training_replay_meta.get("round", -1)) != selected_round:
        raise RuntimeError(
            f"training replay round mismatch: selected={selected_round} "
            f"evidence={training_replay_meta.get('round')}"
        )
    if str(training_replay_meta.get("manifest_sha256") or "") != training_replay_sha:
        raise RuntimeError("training replay evidence SHA does not match manifest")
    if not expected_replay_sha or expected_replay_sha != training_replay_sha:
        raise RuntimeError("selected candidate replay SHA does not match its training record")
    training_clip_hashes = manifest_clip_hashes(training_replay)
    shutil.copy2(training_replay, best_dir / "training-hard-negatives.tsv")
    shutil.copy2(training_replay_evidence, best_dir / "training-hard-negatives.json")

    # Render a deterministic but seed-disjoint hard-negative holdout only after
    # candidate selection. Match the selected round's input curriculum while
    # ensuring no underlying WAV bytes overlap training replay. This holdout is
    # never fed back into training, calibration/test selection, or qualification.
    replay_curriculum = None
    if selected_round > 0:
        curriculum_path = root / "curriculum" / f"round-{selected_round - 1:02d}.json"
        require_file(curriculum_path, "selected-round input curriculum")
        replay_curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
    holdout_generation_index = FAR_HOLDOUT_ROUND_NAMESPACE + selected_round
    holdout_dir = root / "hard-negative-holdout" / f"round-{selected_round:02d}"
    holdout_meta = render_hard_negative_replay(
        config_path,
        holdout_dir,
        round_index=holdout_generation_index,
        curriculum_weights=replay_curriculum,
    )
    holdout_manifest = pathlib.Path(str(holdout_meta["manifest"]))
    holdout_evidence = pathlib.Path(str(holdout_meta["evidence"]))
    require_file(holdout_manifest, "FAR hard-negative holdout manifest")
    require_file(holdout_evidence, "FAR hard-negative holdout evidence")
    holdout_sha = sha256_file(holdout_manifest)
    if str(holdout_meta.get("manifest_sha256") or "") != holdout_sha:
        raise RuntimeError("FAR holdout evidence SHA does not match manifest")
    holdout_clip_hashes = manifest_clip_hashes(holdout_manifest)
    overlapping_clip_hashes = training_clip_hashes & holdout_clip_hashes
    if overlapping_clip_hashes:
        raise RuntimeError(
            f"FAR holdout overlaps {len(overlapping_clip_hashes)} training replay WAV(s)"
        )
    if len(holdout_clip_hashes) != len(training_clip_hashes):
        raise RuntimeError(
            "FAR holdout/training replay clip-count mismatch: "
            f"training={len(training_clip_hashes)} holdout={len(holdout_clip_hashes)}"
        )
    if holdout_sha == training_replay_sha:
        raise RuntimeError("FAR holdout manifest is byte-identical to training replay manifest")
    shutil.copy2(holdout_manifest, best_dir / "hard-negatives.tsv")
    shutil.copy2(holdout_evidence, best_dir / "hard-negative-holdout.json")

    # Selection above uses calibration/test evidence only. Re-run untouched
    # qualification after selection so qualification cannot influence candidate
    # choice, curriculum, training replay, or FAR holdout generation.
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
        "replay_source": str(training_replay.relative_to(ROOT)),
        "replay_sha256": training_replay_sha,
        "model_replay_generation_match": True,
        "far_holdout_source": str(holdout_manifest.relative_to(ROOT)),
        "far_holdout_sha256": holdout_sha,
        "far_holdout_generation_index": holdout_generation_index,
        "far_holdout_used_for_training": False,
        "far_holdout_used_for_selection": False,
        "training_replay_clip_count": len(training_clip_hashes),
        "far_holdout_clip_count": len(holdout_clip_hashes),
        "overlapping_training_holdout_wav_sha256": 0,
    }

    manifest["best_round"] = selected_round
    manifest["best_frontend"] = selected_frontend
    manifest["best_score"] = selected_score
    manifest["best_model_sha256"] = sha256_file(best_dir / "model.kwm")
    manifest["best_pack_sha256"] = sha256_file(best_dir / "keywords.kwk")
    manifest["best_training_hard_negative_manifest_sha256"] = training_replay_sha
    manifest["best_hard_negative_manifest_sha256"] = holdout_sha
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
        best_dir / "training-hard-negatives.tsv",
        best_dir / "training-hard-negatives.json",
        best_dir / "hard-negatives.tsv",
        best_dir / "hard-negative-holdout.json",
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
