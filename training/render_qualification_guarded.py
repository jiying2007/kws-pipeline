from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"

from render_qualification_holdout import require_strict_development_candidate, sha256_file
from shadow_qualification import validate_shadow_policy
from synthetic_audio import load_config

POLICY = "shadow-adversarial-failure-formal-preflight-v2"


def _read_json(path: pathlib.Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _selected_record(manifest: dict) -> dict:
    selection = manifest.get("candidate_selection", {})
    round_index = int(selection.get("selected_round", -1))
    frontend = str(selection.get("selected_frontend") or "")
    rows = [
        row
        for row in manifest.get("records", [])
        if isinstance(row, dict)
        and int(row.get("round", -1)) == round_index
        and str(row.get("frontend") or "") == frontend
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not rows:
        raise ValueError("formal preflight cannot resolve selected strict candidate")
    return min(rows, key=lambda row: (float(row["score"]), str(row["checkpoint"])))


def _provenance_manifest_matches(provenance: dict, name: str, digest: str) -> int:
    manifests = provenance.get("training", {}).get("manifests", [])
    if not isinstance(manifests, list):
        raise ValueError("selected model provenance lacks training manifests")
    return sum(
        1
        for row in manifests
        if isinstance(row, dict)
        and str(row.get("name")) == name
        and str(row.get("sha256")) == digest
    )


def validate_preflight(config_path: pathlib.Path, work: pathlib.Path) -> dict:
    cfg = load_config(config_path)
    shadow_policy = validate_shadow_policy(cfg)
    development = require_strict_development_candidate(work)
    manifest_path = work / "domain-loop-manifest.json"
    manifest_sha = sha256_file(manifest_path)
    selection = development["candidate_selection"]
    selected = _selected_record(development)

    if not bool(selection.get("adversarial_refinement_used")):
        raise ValueError("formal qualification requires adversarial refinement evidence")
    if str(selection.get("adversarial_refinement_policy")) != "post-domain-adversarial-refinement-v1":
        raise ValueError("formal qualification adversarial refinement policy drifted")
    if str(selected.get("stage")) != "post-domain-adversarial-refinement-v1":
        raise ValueError("selected candidate is not the adversarial refinement candidate")
    if bool(selected.get("adversarial_formal_qualification_used", True)):
        raise ValueError("adversarial refinement illegally used formal qualification")
    if bool(selected.get("failure_replay_formal_qualification_used", True)):
        raise ValueError("failure replay illegally used formal qualification")
    if bool(selected.get("failure_replay_development_source_wav_bytes_copied", True)):
        raise ValueError("failure replay copied development evaluation WAV bytes")

    refinement_path = work / "adversarial-refinement" / "summary.json"
    refinement = _read_json(refinement_path, "adversarial refinement summary")
    if str(refinement.get("policy")) != "post-domain-adversarial-refinement-v1":
        raise ValueError("adversarial refinement summary policy drifted")
    if not bool(refinement.get("qualified")):
        raise ValueError("adversarial refinement is not development-qualified")
    if bool(refinement.get("formal_qualification_used", True)):
        raise ValueError("adversarial refinement summary reports formal qualification use")
    if str(refinement.get("output_development_manifest_sha256") or "") != manifest_sha:
        raise ValueError("adversarial refinement output manifest SHA differs from current development manifest")
    if int(refinement.get("refinement_round", -1)) != int(selection["selected_round"]):
        raise ValueError("adversarial refinement round differs from selected development round")
    if str(refinement.get("frontend") or "") != str(selection["selected_frontend"]):
        raise ValueError("adversarial refinement frontend differs from selected development frontend")

    adversarial_path = work / "best" / "adversarial-lexicon.json"
    adversarial = _read_json(adversarial_path, "adversarial lexicon evidence")
    adversarial_sha = sha256_file(adversarial_path)
    if str(adversarial.get("evidence_class")) != "development-only-adversarial-lexicon":
        raise ValueError("adversarial lexicon evidence class drifted")
    if bool(adversarial.get("formal_qualification_used", True)):
        raise ValueError("adversarial lexicon used formal qualification")
    if str(adversarial.get("manifest_sha256") or "") != str(selected.get("adversarial_manifest_sha256") or ""):
        raise ValueError("selected adversarial manifest SHA differs from retained evidence")
    if adversarial_sha != str(selected.get("adversarial_evidence_sha256") or ""):
        raise ValueError("selected adversarial evidence SHA differs from retained evidence")
    for evidence_key, record_key, label in (
        ("enumerated_sequences", "adversarial_enumerated_sequences", "enumeration count"),
        ("top_k", "adversarial_top_k", "Top-K"),
        ("probes_per_sequence", "adversarial_probes_per_sequence", "probe count"),
        ("replay_examples_per_sequence", "adversarial_replay_examples_per_sequence", "per-sequence replay count"),
        ("replay_examples", "adversarial_replay_examples", "replay count"),
        ("min_per_keyword", "adversarial_min_per_keyword", "keyword quota"),
    ):
        if int(adversarial.get(evidence_key, -1)) != int(selected.get(record_key, -2)):
            raise ValueError(f"adversarial {label} differs from selected record")
    if dict(adversarial.get("per_keyword_selected", {})) != dict(
        selected.get("adversarial_per_keyword_selected", {})
    ):
        raise ValueError("adversarial per-keyword selection differs from selected record")
    if list(adversarial.get("strict_prefix_anchors", [])) != list(
        selected.get("adversarial_strict_prefix_anchors", [])
    ):
        raise ValueError("adversarial strict-prefix anchors differ from selected record")
    if str(adversarial.get("selection_policy") or "") != str(selected.get("adversarial_policy") or ""):
        raise ValueError("adversarial selection policy differs from selected record")
    if str(adversarial.get("data_augmentation_policy") or "") != str(
        selected.get("adversarial_data_augmentation_policy") or ""
    ):
        raise ValueError("adversarial data policy differs from selected record")
    adversarial_manifest = pathlib.Path(str(adversarial.get("manifest") or ""))
    if not adversarial_manifest.is_file() or sha256_file(adversarial_manifest) != str(adversarial["manifest_sha256"]):
        raise ValueError("adversarial replay manifest is missing or its SHA drifted")

    failure_path = work / "best" / "development-failure-replay.json"
    failure = _read_json(failure_path, "development failure replay evidence")
    failure_sha = sha256_file(failure_path)
    if str(failure.get("evidence_class")) != "development-only-failure-resynthesis":
        raise ValueError("failure replay evidence class drifted")
    if str(failure.get("policy")) != "development-failure-resynthesis-v1":
        raise ValueError("failure replay policy drifted")
    if bool(failure.get("formal_qualification_used", True)):
        raise ValueError("failure replay evidence reports formal qualification use")
    if bool(failure.get("development_source_wav_bytes_copied", True)):
        raise ValueError("failure replay evidence reports copied development WAV bytes")
    if list(failure.get("source_splits", [])) != ["calibration", "test"]:
        raise ValueError("failure replay source splits must be development calibration/test only")
    if str(failure.get("manifest_sha256") or "") != str(selected.get("failure_replay_manifest_sha256") or ""):
        raise ValueError("selected failure replay manifest SHA differs from retained evidence")
    if failure_sha != str(selected.get("failure_replay_evidence_sha256") or ""):
        raise ValueError("selected failure replay evidence SHA differs from retained evidence")
    if int(failure.get("examples", -1)) != int(selected.get("failure_replay_examples", -2)):
        raise ValueError("failure replay example count differs from selected record")
    if int(failure.get("selected_unique_failures", -1)) != int(
        selected.get("failure_replay_selected_unique_failures", -2)
    ):
        raise ValueError("failure replay selected-failure count differs from selected record")
    failure_manifest = pathlib.Path(str(failure.get("manifest") or ""))
    failure_examples = int(failure.get("examples", 0))
    if not failure_manifest.is_file():
        raise ValueError("failure replay manifest is missing")
    if sha256_file(failure_manifest) != str(failure["manifest_sha256"]):
        raise ValueError("failure replay manifest SHA drifted")
    if failure_examples > 0 and failure_manifest.stat().st_size == 0:
        raise ValueError("failure replay evidence has examples but empty manifest")

    provenance_path = work / "best" / "model-provenance.json"
    provenance = _read_json(provenance_path, "selected model provenance")
    adversarial_manifest_sha = str(selected.get("adversarial_manifest_sha256") or "")
    if _provenance_manifest_matches(
        provenance, "adversarial-hard-negatives.tsv", adversarial_manifest_sha
    ) != 1:
        raise ValueError("selected model provenance does not prove adversarial replay training")
    failure_manifest_sha = str(selected.get("failure_replay_manifest_sha256") or "")
    failure_provenance_verified = failure_examples == 0 or _provenance_manifest_matches(
        provenance, "development-failure-replay.tsv", failure_manifest_sha
    ) == 1
    if not failure_provenance_verified:
        raise ValueError("selected model provenance does not prove development failure replay training")

    shadow_path = work / "shadow-qualification" / "summary.json"
    shadow = _read_json(shadow_path, "shadow qualification summary")
    shadow_sha = sha256_file(shadow_path)
    if int(shadow.get("schema_version", 0)) != 1:
        raise ValueError("shadow qualification summary schema drifted")
    if str(shadow.get("evidence_class")) != "development-only-shadow-qualification":
        raise ValueError("shadow qualification evidence class drifted")
    if not bool(shadow.get("qualified")):
        raise ValueError("shadow qualification arena is not qualified")
    if bool(shadow.get("formal_qualification_seed_consumed", True)):
        raise ValueError("shadow qualification reports formal seed consumption")
    if str(shadow.get("development_manifest_sha256") or "") != manifest_sha:
        raise ValueError("shadow qualification development manifest SHA differs from current manifest")
    if int(shadow.get("development_selected_round", -1)) != int(selection["selected_round"]):
        raise ValueError("shadow qualification round differs from selected development round")
    if str(shadow.get("development_selected_frontend") or "") != str(selection["selected_frontend"]):
        raise ValueError("shadow qualification frontend differs from selected development frontend")
    if [int(value) for value in shadow.get("seeds", [])] != list(shadow_policy["seeds"]):
        raise ValueError("shadow qualification seed arena differs from config")
    if int(shadow.get("expected_wakes_per_seed", -1)) != int(shadow_policy["expected_wakes_per_seed"]):
        raise ValueError("shadow qualification support differs from config")
    if abs(float(shadow.get("min_surrogate_separation", -1.0)) - float(shadow_policy["min_surrogate_separation"])) > 1.0e-12:
        raise ValueError("shadow surrogate separation gate differs from config")
    results = shadow.get("results")
    if not isinstance(results, list) or len(results) != len(shadow_policy["seeds"]):
        raise ValueError("shadow qualification result count differs from seed arena")
    if any(
        not isinstance(row, dict)
        or not bool(row.get("qualified"))
        or not bool(row.get("runtime_qualified"))
        or not bool(row.get("surrogate_separation_qualified"))
        for row in results
    ):
        raise ValueError("shadow qualification contains a non-qualified seed")

    return {
        "policy": POLICY,
        "development_manifest_sha256": manifest_sha,
        "development_selected_round": int(selection["selected_round"]),
        "development_selected_frontend": str(selection["selected_frontend"]),
        "shadow_summary_sha256": shadow_sha,
        "shadow_seed_count": len(shadow_policy["seeds"]),
        "shadow_min_surrogate_separation": float(shadow_policy["min_surrogate_separation"]),
        "adversarial_refinement_summary_sha256": sha256_file(refinement_path),
        "adversarial_lexicon_evidence_sha256": adversarial_sha,
        "adversarial_data_augmentation_policy": str(adversarial["data_augmentation_policy"]),
        "adversarial_selection_policy": str(adversarial["selection_policy"]),
        "adversarial_manifest_sha256": adversarial_manifest_sha,
        "adversarial_manifest": str(adversarial_manifest),
        "adversarial_enumerated_sequences": int(adversarial["enumerated_sequences"]),
        "adversarial_top_k": int(adversarial["top_k"]),
        "adversarial_probes_per_sequence": int(adversarial["probes_per_sequence"]),
        "adversarial_replay_examples_per_sequence": int(adversarial["replay_examples_per_sequence"]),
        "adversarial_replay_examples": int(adversarial["replay_examples"]),
        "adversarial_min_per_keyword": int(adversarial["min_per_keyword"]),
        "adversarial_per_keyword_selected": dict(adversarial["per_keyword_selected"]),
        "adversarial_strict_prefix_anchors": list(adversarial["strict_prefix_anchors"]),
        "model_provenance_adversarial_manifest_verified": True,
        "failure_replay_evidence_sha256": failure_sha,
        "failure_replay_policy": str(failure["policy"]),
        "failure_replay_manifest_sha256": failure_manifest_sha,
        "failure_replay_manifest": str(failure_manifest),
        "failure_replay_examples": failure_examples,
        "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
        "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
        "failure_replay_development_source_wav_bytes_copied": False,
        "model_provenance_failure_replay_manifest_verified": bool(failure_provenance_verified),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed shadow/adversarial/failure preflight followed by formal qualification rotation."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config = args.config.resolve()
    work = args.work_dir.resolve()
    output = args.output.resolve()
    preflight = validate_preflight(config, work)
    preflight_path = work / "formal-preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    guards: list[pathlib.Path] = []
    adversarial_guard = work / "hard-negative-replay" / "round-adversarial-overlap-guard"
    adversarial_guard.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(preflight["adversarial_manifest"]), adversarial_guard / "hard-negatives.tsv")
    guards.append(adversarial_guard)
    if int(preflight["failure_replay_examples"]) > 0:
        failure_guard = work / "hard-negative-replay" / "round-failure-replay-overlap-guard"
        failure_guard.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pathlib.Path(preflight["failure_replay_manifest"]), failure_guard / "hard-negatives.tsv")
        guards.append(failure_guard)
    try:
        subprocess.run(
            [
                sys.executable,
                str(TRAINING / "render_qualification_holdout.py"),
                "--config",
                str(config),
                "--work-dir",
                str(work),
                "--output",
                str(output),
            ],
            check=True,
        )
    finally:
        for guard in guards:
            shutil.rmtree(guard, ignore_errors=True)

    cohort_path = output / "qualification-cohort.json"
    cohort = _read_json(cohort_path, "formal qualification cohort")
    if str(cohort.get("development_manifest_sha256") or "") != preflight["development_manifest_sha256"]:
        raise ValueError("formal cohort development manifest SHA differs from preflight")
    cohort.update(
        {
            "formal_preflight_policy": POLICY,
            "formal_preflight_sha256": sha256_file(preflight_path),
            "adversarial_overlap_guard_included": True,
            "failure_replay_overlap_guard_included": int(preflight["failure_replay_examples"]) > 0,
            "shadow_qualification_required": True,
            "shadow_qualification_qualified": True,
            "shadow_summary_sha256": preflight["shadow_summary_sha256"],
            "shadow_seed_count": preflight["shadow_seed_count"],
            "shadow_min_surrogate_separation": preflight["shadow_min_surrogate_separation"],
            "adversarial_refinement_required": True,
            "adversarial_refinement_policy": "post-domain-adversarial-refinement-v1",
            "adversarial_refinement_summary_sha256": preflight["adversarial_refinement_summary_sha256"],
            "adversarial_lexicon_evidence_sha256": preflight["adversarial_lexicon_evidence_sha256"],
            "adversarial_data_augmentation_policy": preflight["adversarial_data_augmentation_policy"],
            "adversarial_selection_policy": preflight["adversarial_selection_policy"],
            "adversarial_manifest_sha256": preflight["adversarial_manifest_sha256"],
            "adversarial_enumerated_sequences": preflight["adversarial_enumerated_sequences"],
            "adversarial_top_k": preflight["adversarial_top_k"],
            "adversarial_probes_per_sequence": preflight["adversarial_probes_per_sequence"],
            "adversarial_replay_examples_per_sequence": preflight["adversarial_replay_examples_per_sequence"],
            "adversarial_replay_examples": preflight["adversarial_replay_examples"],
            "adversarial_min_per_keyword": preflight["adversarial_min_per_keyword"],
            "adversarial_per_keyword_selected": preflight["adversarial_per_keyword_selected"],
            "adversarial_strict_prefix_anchors": preflight["adversarial_strict_prefix_anchors"],
            "adversarial_formal_qualification_used": False,
            "model_provenance_adversarial_manifest_verified": True,
            "failure_replay_required": True,
            "failure_replay_policy": preflight["failure_replay_policy"],
            "failure_replay_evidence_sha256": preflight["failure_replay_evidence_sha256"],
            "failure_replay_manifest_sha256": preflight["failure_replay_manifest_sha256"],
            "failure_replay_examples": preflight["failure_replay_examples"],
            "failure_replay_observed_unique_failures": preflight["failure_replay_observed_unique_failures"],
            "failure_replay_selected_unique_failures": preflight["failure_replay_selected_unique_failures"],
            "failure_replay_formal_qualification_used": False,
            "failure_replay_development_source_wav_bytes_copied": False,
            "model_provenance_failure_replay_manifest_verified": preflight[
                "model_provenance_failure_replay_manifest_verified"
            ],
        }
    )
    cohort_path.write_text(
        json.dumps(cohort, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(preflight, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
