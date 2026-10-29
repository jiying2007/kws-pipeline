from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from model_provenance import boolean  # noqa: E402
from qualification_common import json_int  # noqa: E402

PREFLIGHT_POLICY = "shadow-adversarial-failure-formal-preflight-v2"


def read_json(path: pathlib.Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def integer_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return [json_int(item, f"{label}[{index}]") for index, item in enumerate(value)]


def retired_namespaces(value: object) -> list[int]:
    if not isinstance(value, list):
        raise ValueError("FAR cohort retired must be an array")
    result = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"FAR cohort retired[{index}] must be an object")
        result.append(json_int(row.get("namespace"), f"FAR cohort retired[{index}].namespace"))
    return result


def verify(config_path: pathlib.Path, root: pathlib.Path) -> dict:
    config = read_json(config_path, "training config")
    summary = read_json(root / "training-run-summary.json", "training summary")
    cohort = read_json(root / "qualification-dataset/qualification-cohort.json", "formal qualification cohort")
    far_cohort_path = root / "best/far-holdout-cohort.json"
    far_cohort = read_json(far_cohort_path, "FAR holdout cohort")
    shadow = read_json(root / "shadow-qualification/summary.json", "shadow qualification summary")
    refinement = read_json(root / "adversarial-refinement/summary.json", "adversarial refinement summary")
    qualified = boolean(summary.get("qualified"), "training summary qualified")
    train_exit = json_int(summary.get("training_exit_code"), "training_exit_code", maximum=1)
    iteration_exit = json_int(summary.get("iteration_exit_code"), "iteration_exit_code", maximum=1)
    selection = summary.get("candidate_selection", {})
    if not isinstance(selection, dict):
        raise ValueError("candidate selection must be an object")

    if not boolean(refinement.get("qualified"), "refinement qualified") or boolean(refinement.get("formal_qualification_used"), "refinement formal_qualification_used"):
        raise ValueError("formal qualification ran without qualified development-only adversarial refinement")
    if not boolean(shadow.get("qualified"), "shadow qualified"):
        raise ValueError("formal qualification ran without a qualified shadow arena")
    if boolean(shadow.get("formal_qualification_seed_consumed"), "shadow formal_qualification_seed_consumed"):
        raise ValueError("shadow qualification claims formal seed consumption")
    if cohort.get("formal_preflight_policy") != PREFLIGHT_POLICY:
        raise ValueError("formal qualification cohort lacks current guarded preflight policy")
    if not boolean(cohort.get("shadow_qualification_required"), "cohort shadow_qualification_required") or not boolean(cohort.get("shadow_qualification_qualified"), "cohort shadow_qualification_qualified"):
        raise ValueError("formal qualification cohort lacks qualified shadow evidence")
    if not boolean(cohort.get("adversarial_refinement_required"), "cohort adversarial_refinement_required"):
        raise ValueError("formal qualification cohort lacks adversarial refinement prerequisite")
    if boolean(cohort.get("adversarial_formal_qualification_used"), "cohort adversarial_formal_qualification_used"):
        raise ValueError("formal cohort says adversarial training used formal qualification")
    if not boolean(cohort.get("model_provenance_adversarial_manifest_verified"), "cohort model_provenance_adversarial_manifest_verified"):
        raise ValueError("formal cohort lacks adversarial training provenance proof")
    if not boolean(cohort.get("adversarial_overlap_guard_included"), "cohort adversarial_overlap_guard_included"):
        raise ValueError("formal cohort did not include adversarial WAVs in overlap guard")
    if not boolean(cohort.get("failure_replay_required"), "cohort failure_replay_required"):
        raise ValueError("formal cohort lacks failure-replay prerequisite")
    if boolean(cohort.get("failure_replay_formal_qualification_used"), "cohort failure_replay_formal_qualification_used"):
        raise ValueError("formal cohort says failure replay used formal qualification")
    if not boolean(cohort.get("model_provenance_failure_replay_manifest_verified"), "cohort model_provenance_failure_replay_manifest_verified"):
        raise ValueError("formal cohort lacks failure-replay training provenance proof")
    if iteration_exit not in (0, 1):
        raise ValueError(f"training infrastructure failed with exit code {iteration_exit}")
    if qualified != (train_exit == 0):
        raise ValueError(f"qualification/exit-code contract mismatch: qualified={qualified} exit={train_exit}")
    if not boolean(cohort.get("seed_disjoint"), "cohort seed_disjoint"):
        raise ValueError("active qualification seed is not disjoint from training seed")
    if json_int(cohort.get("qualification_seed"), "qualification_seed") == json_int(cohort.get("training_seed"), "training_seed"):
        raise ValueError("active qualification reuses training seed")
    for field, label in (
        ("overlapping_retired_active_wav_sha256", "retired qualification WAV content"),
        ("overlapping_exposed_active_wav_sha256", "exposed retired qualification WAV content"),
        ("overlapping_development_active_wav_sha256", "development/training WAV content"),
    ):
        if json_int(cohort.get(field), field) != 0:
            raise ValueError(f"active qualification overlaps {label}")
    if json_int(cohort.get("expected_wakes"), "expected_wakes") != 256:
        raise ValueError("active qualification cohort does not contain 256 expected wakes")
    retired_qualification = integer_list(config.get("retired_qualification_holdout_seeds"), "retired_qualification_holdout_seeds")
    if integer_list(cohort.get("retired_exposed_qualification_seeds"), "retired_exposed_qualification_seeds") != retired_qualification:
        raise ValueError("qualification retired seed lineage differs from config")
    if boolean(selection.get("qualification_used_for_selection"), "selection qualification_used_for_selection"):
        raise ValueError("qualification leaked into candidate selection")
    if boolean(selection.get("far_holdout_used_for_training"), "selection far_holdout_used_for_training"):
        raise ValueError("FAR holdout leaked into model training")
    if boolean(selection.get("far_holdout_used_for_selection"), "selection far_holdout_used_for_selection"):
        raise ValueError("FAR holdout leaked into candidate selection")
    if json_int(selection.get("overlapping_training_holdout_wav_sha256"), "selection overlapping_training_holdout_wav_sha256") != 0:
        raise ValueError("active FAR holdout overlaps training replay WAV content")
    if json_int(selection.get("overlapping_retired_active_far_holdout_wav_sha256"), "selection overlapping_retired_active_far_holdout_wav_sha256") != 0:
        raise ValueError("active FAR holdout overlaps retired FAR holdout WAV content")

    active_namespace = json_int(config.get("far_holdout_round_namespace"), "far_holdout_round_namespace")
    retired_namespace_values = integer_list(config.get("retired_far_holdout_round_namespaces"), "retired_far_holdout_round_namespaces")
    if json_int(selection.get("far_holdout_round_namespace"), "selection far_holdout_round_namespace") != active_namespace:
        raise ValueError("candidate-selection FAR namespace differs from config")
    if integer_list(selection.get("retired_far_holdout_round_namespaces"), "selection retired_far_holdout_round_namespaces") != retired_namespace_values:
        raise ValueError("candidate-selection retired FAR namespaces differ from config")
    if json_int(far_cohort.get("active_namespace"), "FAR cohort active_namespace") != active_namespace:
        raise ValueError("FAR cohort active namespace differs from config")
    if retired_namespaces(far_cohort.get("retired")) != retired_namespace_values:
        raise ValueError("FAR cohort retired namespaces differ from config")
    if json_int(far_cohort.get("overlapping_retired_active_wav_sha256"), "FAR cohort overlapping_retired_active_wav_sha256") != 0:
        raise ValueError("FAR cohort attests retired/active WAV overlap")
    if json_int(far_cohort.get("active_clip_count"), "FAR cohort active_clip_count", minimum=1) != json_int(selection.get("far_holdout_clip_count"), "selection far_holdout_clip_count", minimum=1):
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
        "best_round": json_int(summary.get("best_round"), "best_round"),
        "best_frontend": summary["best_frontend"],
        "qualification_seed": json_int(cohort.get("qualification_seed"), "qualification_seed"),
        "far_holdout_clip_count": json_int(selection.get("far_holdout_clip_count"), "far_holdout_clip_count", minimum=1),
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
