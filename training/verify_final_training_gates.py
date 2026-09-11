from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

PREFLIGHT_POLICY = "shadow-adversarial-failure-formal-preflight-v2"


def read_json(path: pathlib.Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def verify(config_path: pathlib.Path, root: pathlib.Path) -> dict:
    config = read_json(config_path, "training config")
    summary = read_json(root / "training-run-summary.json", "training summary")
    cohort = read_json(root / "qualification-dataset/qualification-cohort.json", "formal qualification cohort")
    far_cohort_path = root / "best/far-holdout-cohort.json"
    far_cohort = read_json(far_cohort_path, "FAR holdout cohort")
    shadow = read_json(root / "shadow-qualification/summary.json", "shadow qualification summary")
    refinement = read_json(root / "adversarial-refinement/summary.json", "adversarial refinement summary")
    qualified = bool(summary["qualified"])
    train_exit = int(summary["training_exit_code"])
    iteration_exit = int(summary["iteration_exit_code"])
    selection = summary.get("candidate_selection", {})
    if not isinstance(selection, dict):
        raise ValueError("candidate selection must be an object")

    if not bool(refinement.get("qualified")) or bool(refinement.get("formal_qualification_used", True)):
        raise ValueError("formal qualification ran without qualified development-only adversarial refinement")
    if not bool(shadow.get("qualified")):
        raise ValueError("formal qualification ran without a qualified shadow arena")
    if bool(shadow.get("formal_qualification_seed_consumed", True)):
        raise ValueError("shadow qualification claims formal seed consumption")
    if str(cohort.get("formal_preflight_policy")) != PREFLIGHT_POLICY:
        raise ValueError("formal qualification cohort lacks current guarded preflight policy")
    if not bool(cohort.get("shadow_qualification_required")) or not bool(cohort.get("shadow_qualification_qualified")):
        raise ValueError("formal qualification cohort lacks qualified shadow evidence")
    if not bool(cohort.get("adversarial_refinement_required")):
        raise ValueError("formal qualification cohort lacks adversarial refinement prerequisite")
    if bool(cohort.get("adversarial_formal_qualification_used", True)):
        raise ValueError("formal cohort says adversarial training used formal qualification")
    if not bool(cohort.get("model_provenance_adversarial_manifest_verified")):
        raise ValueError("formal cohort lacks adversarial training provenance proof")
    if not bool(cohort.get("adversarial_overlap_guard_included")):
        raise ValueError("formal cohort did not include adversarial WAVs in overlap guard")
    if not bool(cohort.get("failure_replay_required")):
        raise ValueError("formal cohort lacks failure-replay prerequisite")
    if bool(cohort.get("failure_replay_formal_qualification_used", True)):
        raise ValueError("formal cohort says failure replay used formal qualification")
    if not bool(cohort.get("model_provenance_failure_replay_manifest_verified")):
        raise ValueError("formal cohort lacks failure-replay training provenance proof")
    if iteration_exit not in (0, 1):
        raise ValueError(f"training infrastructure failed with exit code {iteration_exit}")
    if qualified != (train_exit == 0):
        raise ValueError(f"qualification/exit-code contract mismatch: qualified={qualified} exit={train_exit}")
    if not bool(cohort.get("seed_disjoint")):
        raise ValueError("active qualification seed is not disjoint from training seed")
    if int(cohort.get("qualification_seed", -1)) == int(cohort.get("training_seed", -1)):
        raise ValueError("active qualification reuses training seed")
    for field, label in (
        ("overlapping_retired_active_wav_sha256", "retired qualification WAV content"),
        ("overlapping_exposed_active_wav_sha256", "exposed retired qualification WAV content"),
        ("overlapping_development_active_wav_sha256", "development/training WAV content"),
    ):
        if int(cohort.get(field, -1)) != 0:
            raise ValueError(f"active qualification overlaps {label}")
    if int(cohort.get("expected_wakes", 0)) != 256:
        raise ValueError("active qualification cohort does not contain 256 expected wakes")
    retired_qualification = [int(value) for value in config["retired_qualification_holdout_seeds"]]
    if [int(value) for value in cohort.get("retired_exposed_qualification_seeds", [])] != retired_qualification:
        raise ValueError("qualification retired seed lineage differs from config")
    if bool(selection.get("qualification_used_for_selection", True)):
        raise ValueError("qualification leaked into candidate selection")
    if bool(selection.get("far_holdout_used_for_training", True)):
        raise ValueError("FAR holdout leaked into model training")
    if bool(selection.get("far_holdout_used_for_selection", True)):
        raise ValueError("FAR holdout leaked into candidate selection")
    if int(selection.get("overlapping_training_holdout_wav_sha256", -1)) != 0:
        raise ValueError("active FAR holdout overlaps training replay WAV content")
    if int(selection.get("overlapping_retired_active_far_holdout_wav_sha256", -1)) != 0:
        raise ValueError("active FAR holdout overlaps retired FAR holdout WAV content")

    active_namespace = int(config["far_holdout_round_namespace"])
    retired_namespaces = [int(value) for value in config["retired_far_holdout_round_namespaces"]]
    if int(selection.get("far_holdout_round_namespace", -1)) != active_namespace:
        raise ValueError("candidate-selection FAR namespace differs from config")
    if [int(value) for value in selection.get("retired_far_holdout_round_namespaces", [])] != retired_namespaces:
        raise ValueError("candidate-selection retired FAR namespaces differ from config")
    if int(far_cohort.get("active_namespace", -1)) != active_namespace:
        raise ValueError("FAR cohort active namespace differs from config")
    if [int(row["namespace"]) for row in far_cohort.get("retired", [])] != retired_namespaces:
        raise ValueError("FAR cohort retired namespaces differ from config")
    if int(far_cohort.get("overlapping_retired_active_wav_sha256", -1)) != 0:
        raise ValueError("FAR cohort attests retired/active WAV overlap")
    if int(far_cohort.get("active_clip_count", 0)) != int(selection.get("far_holdout_clip_count", -1)):
        raise ValueError("FAR cohort active clip count differs from candidate-selection evidence")
    if str(far_cohort.get("active_manifest_sha256", "")) != str(selection.get("far_holdout_sha256", "")):
        raise ValueError("FAR cohort active manifest SHA differs from candidate-selection evidence")
    digest = hashlib.sha256(far_cohort_path.read_bytes()).hexdigest()
    if digest != str(selection.get("far_holdout_cohort_sha256", "")):
        raise ValueError("FAR cohort SHA differs from candidate-selection evidence")
    if digest != str(summary.get("artifacts", {}).get("far-holdout-cohort.json", "")):
        raise ValueError("FAR cohort SHA differs from final artifact evidence")
    if not qualified:
        raise ValueError("synthetic qualification gates were not met")
    return {
        "qualified": qualified,
        "best_round": summary["best_round"],
        "best_frontend": summary["best_frontend"],
        "qualification_seed": int(cohort["qualification_seed"]),
        "far_holdout_clip_count": int(selection["far_holdout_clip_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = verify(args.config.resolve(), args.work_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
