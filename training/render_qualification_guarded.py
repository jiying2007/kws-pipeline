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

POLICY = "shadow-adversarial-formal-preflight-v1"


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
    if int(adversarial.get("enumerated_sequences", -1)) != int(selected.get("adversarial_enumerated_sequences", -2)):
        raise ValueError("adversarial enumeration count differs from selected record")
    if int(adversarial.get("top_k", -1)) != int(selected.get("adversarial_top_k", -2)):
        raise ValueError("adversarial Top-K differs from selected record")
    if int(adversarial.get("replay_examples", -1)) != int(selected.get("adversarial_replay_examples", -2)):
        raise ValueError("adversarial replay count differs from selected record")
    adversarial_manifest = pathlib.Path(str(adversarial.get("manifest") or ""))
    if not adversarial_manifest.is_file() or sha256_file(adversarial_manifest) != str(adversarial["manifest_sha256"]):
        raise ValueError("adversarial replay manifest is missing or its SHA drifted")

    provenance_path = work / "best" / "model-provenance.json"
    provenance = _read_json(provenance_path, "selected model provenance")
    manifests = provenance.get("training", {}).get("manifests", [])
    if not isinstance(manifests, list):
        raise ValueError("selected model provenance lacks training manifests")
    adversarial_manifest_sha = str(selected.get("adversarial_manifest_sha256") or "")
    matching = [
        row
        for row in manifests
        if isinstance(row, dict)
        and str(row.get("name")) == "adversarial-hard-negatives.tsv"
        and str(row.get("sha256")) == adversarial_manifest_sha
    ]
    if len(matching) != 1:
        raise ValueError("selected model provenance does not prove adversarial replay training")

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
        "adversarial_manifest_sha256": adversarial_manifest_sha,
        "adversarial_manifest": str(adversarial_manifest),
        "adversarial_enumerated_sequences": int(adversarial["enumerated_sequences"]),
        "adversarial_top_k": int(adversarial["top_k"]),
        "adversarial_replay_examples": int(adversarial["replay_examples"]),
        "model_provenance_adversarial_manifest_verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed shadow/adversarial preflight followed by formal qualification rotation."
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

    # The legacy renderer's development overlap scan intentionally only follows
    # hard-negative-replay/round-*/hard-negatives.tsv. Inject the model-mined
    # adversarial manifest into that scan namespace for the duration of formal
    # rendering, so the active formal cohort proves zero WAV-byte overlap with
    # every training source without changing finalizer FAR-replay semantics.
    overlap_guard = work / "hard-negative-replay" / "round-adversarial-overlap-guard"
    overlap_guard.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pathlib.Path(preflight["adversarial_manifest"]), overlap_guard / "hard-negatives.tsv")
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
        shutil.rmtree(overlap_guard, ignore_errors=True)

    cohort_path = output / "qualification-cohort.json"
    cohort = _read_json(cohort_path, "formal qualification cohort")
    if str(cohort.get("development_manifest_sha256") or "") != preflight["development_manifest_sha256"]:
        raise ValueError("formal cohort development manifest SHA differs from preflight")
    cohort.update(
        {
            "formal_preflight_policy": POLICY,
            "formal_preflight_sha256": sha256_file(preflight_path),
            "adversarial_overlap_guard_included": True,
            "shadow_qualification_required": True,
            "shadow_qualification_qualified": True,
            "shadow_summary_sha256": preflight["shadow_summary_sha256"],
            "shadow_seed_count": preflight["shadow_seed_count"],
            "shadow_min_surrogate_separation": preflight["shadow_min_surrogate_separation"],
            "adversarial_refinement_required": True,
            "adversarial_refinement_policy": "post-domain-adversarial-refinement-v1",
            "adversarial_refinement_summary_sha256": preflight["adversarial_refinement_summary_sha256"],
            "adversarial_lexicon_evidence_sha256": preflight["adversarial_lexicon_evidence_sha256"],
            "adversarial_manifest_sha256": preflight["adversarial_manifest_sha256"],
            "adversarial_enumerated_sequences": preflight["adversarial_enumerated_sequences"],
            "adversarial_top_k": preflight["adversarial_top_k"],
            "adversarial_replay_examples": preflight["adversarial_replay_examples"],
            "adversarial_formal_qualification_used": False,
            "model_provenance_adversarial_manifest_verified": True,
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
