#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_status(root: pathlib.Path) -> dict:
    shipping_path = root / "configs" / "shipping.xiaowo.json"
    closure_path = root / "configs" / "training" / "kws-v2-efficient-encoder-closure-v1.json"
    shipping = load_json(shipping_path)
    closure = load_json(closure_path)

    model = shipping.get("model")
    if not isinstance(model, dict):
        raise ValueError("shipping config has no model object")
    tag = str(model.get("release_tag") or "")
    if not tag.startswith("model-"):
        raise ValueError("shipping model.release_tag is invalid")
    registry_dir = root / "models" / "registry" / tag
    registry_path = registry_dir / "registry.json"
    if not registry_path.is_file():
        raise ValueError(f"pinned model is not mirrored in Git registry: {tag}")
    registry = load_json(registry_path)

    pins = {
        "xiaowo-model.kwm": str(model.get("model_sha256") or ""),
        "xiaowo-model.pt": str(model.get("checkpoint_sha256") or ""),
        "xiaowo-keywords.kwk": str(model.get("keyword_pack_sha256") or ""),
        "xiaowo-keywords.tsv": str(model.get("keyword_tsv_sha256") or ""),
    }
    for name, expected in pins.items():
        path = registry_dir / name
        if not path.is_file():
            raise ValueError(f"registry pinned asset missing: {name}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"registry pinned asset digest mismatch: {name}")

    if registry.get("release_tag") != tag:
        raise ValueError("registry release tag differs from shipping config")
    training = registry.get("training")
    if not isinstance(training, dict):
        raise ValueError("registry training identity missing")
    if int(training.get("run_id", 0)) != int(model.get("training_run_id", 0)):
        raise ValueError("registry training run differs from shipping config")
    if str(training.get("head_sha") or "") != str(model.get("trained_head_sha") or ""):
        raise ValueError("registry training head differs from shipping config")

    pending = list(shipping.get("shipping_evidence_boundary", {}).get("pending") or [])
    expected_pending = [
        "real-human-final-afe-acoustic-qualification",
        "physical-target-board-performance-and-soak",
    ]
    if pending != expected_pending:
        raise ValueError(f"unexpected product blocker set: {pending}")

    required_paths = [
        root / ".github" / "workflows" / "dataset-driven-iteration.yml",
        root / ".github" / "workflows" / "real-human-qualification.yml",
        root / "commercial" / "real-human-qualification.policy.json",
        root / "commercial" / "target-qualification.policy.json",
    ]
    missing = [path.relative_to(root).as_posix() for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"landing control-plane files missing: {missing}")

    if closure.get("policy") != "kws-v2-efficient-encoder-closure-v1":
        raise ValueError("efficient encoder closure policy mismatch")
    decision = closure.get("decision")
    if not isinstance(decision, dict) or decision.get("architecture_search_paused") is not True:
        raise ValueError("architecture search is not explicitly paused")

    calibration = shipping.get("threshold_calibration")
    if not isinstance(calibration, dict) or type(calibration.get("recalibration_required")) is not bool:
        raise ValueError("threshold calibration must explicitly declare recalibration_required")
    if type(shipping.get("shipping_approved")) is not bool:
        raise ValueError("shipping_approved must be a boolean")

    # Link retained observations instead of maintaining another outcome ledger.
    # Reading these local files is not a live regression/research status check.
    snapshot_path = root / "research" / "consolidation" / "CURRENT_STATUS_2026-10-09.md"
    research_entry = root / "research" / "README.md"
    admission_entry = root / "research" / "d20-diagnostic-admission-v1" / "PLAN-SCHEMA.md"
    for path in (snapshot_path, research_entry, admission_entry):
        if not path.is_file():
            raise ValueError(f"status evidence/navigation missing: {path.relative_to(root)}")

    qualification = model.get("qualification")
    synthetic_qualified = (
        shipping.get("evidence_status") == "synthetic-qualified"
        and isinstance(qualification, dict)
        and int(qualification.get("expected_wakes", 0)) > 0
        and int(qualification.get("matched_wakes", -1)) == int(qualification.get("expected_wakes", 0))
        and int(qualification.get("false_rejects", -1)) == 0
        and int(qualification.get("false_accepts", -1)) == 0
        and shipping.get("model", {}).get("robustness_qualified") is True
    )

    status = {
        "schema_version": 2,
        "policy": "kws-product-landing-status-v2",
        "assessment_scope": "repository-source-contract-only",
        "live_qualification_checked": False,
        "model": {
            "available": True,
            "release_tag": tag,
            "training_run_id": int(model["training_run_id"]),
            "trained_head_sha": str(model["trained_head_sha"]),
            "deployable_model_format": "KWSP-v2",
            "model_sha256": str(model["model_sha256"]),
            "git_registry_mirrored": True,
            "git_registry_path": registry_dir.relative_to(root).as_posix(),
            "registry_bytes": int(registry.get("storage", {}).get("release_assets_bytes", 0)),
        },
        "historical_release_qualification": {
            "scope": "frozen-model-release-only",
            "source_path": shipping_path.relative_to(root).as_posix(),
            "release_tag": tag,
            "status": str(shipping.get("evidence_status") or ""),
            "synthetic_qualification_passed": synthetic_qualified,
            "qualification": qualification,
            "applies_to_current_source": False,
        },
        "current_source": {
            "scope": "source-contract-only-no-acoustic-evaluation",
            "contract_path": shipping_path.relative_to(root).as_posix(),
            "contract_sha256": sha256(shipping_path),
            "recalibration_required": calibration["recalibration_required"],
            "recalibration_reason": str(calibration.get("recalibration_reason") or ""),
            "synthetic_qualification_checked": False,
        },
        "dated_regression_and_research": {
            "scope": "retained-dated-observations-not-live-status",
            "snapshot_path": snapshot_path.relative_to(root).as_posix(),
            "snapshot_sha256": sha256(snapshot_path),
            "live_status_checked": False,
            "current_research_entry": research_entry.relative_to(root).as_posix(),
            "current_admission_entry": admission_entry.relative_to(root).as_posix()
            + "#current-admission-checklist",
        },
        "external_qualification": {
            "scope": "pending-in-source-contract-not-live-checked",
            "real_human_final_afe_passed": False,
            "physical_target_board_passed": False,
        },
        "product": {
            "shipping_approved": shipping["shipping_approved"],
            "blockers": pending,
            "next_gate": pending[0] if pending else None,
        },
        "historical_research_closure": {
            "scope": "retained-closed-research-line-not-current-authorization",
            "source_path": closure_path.relative_to(root).as_posix(),
            "architecture_search_paused": True,
            "next_authorized_lane": str(decision["next_authorized_lane"]),
            "reopen_condition": "product-data-implicates-model-capacity",
        },
        "control_plane": {
            "scope": "repository-file-presence-only",
            "dataset_iteration_ready": True,
            "real_human_phase_a_ready": True,
            "physical_target_phase_b_policy_ready": True,
            "private_real_audio_required": True,
            "physical_dut_evidence_required": True,
        },
    }
    return status


def verify(status: dict) -> None:
    if status["model"]["available"] is not True:
        raise ValueError("trained deployable model is unavailable")
    if status["model"]["git_registry_mirrored"] is not True:
        raise ValueError("trained model is not in Git registry")
    if status["historical_release_qualification"]["synthetic_qualification_passed"] is not True:
        raise ValueError("pinned model has no historical synthetic qualification")
    if status["historical_release_qualification"]["applies_to_current_source"] is not False:
        raise ValueError("historical qualification must not qualify current source")
    if status["current_source"]["synthetic_qualification_checked"] is not False:
        raise ValueError("source status must not claim acoustic evaluation")
    if status["live_qualification_checked"] is not False:
        raise ValueError("source status must not claim live qualification")
    if status["dated_regression_and_research"]["live_status_checked"] is not False:
        raise ValueError("retained observations must not claim live status")
    if any(status["external_qualification"][key] is not False for key in (
        "real_human_final_afe_passed", "physical_target_board_passed"
    )):
        raise ValueError("source status must not claim external qualification")
    if status["product"]["shipping_approved"] is not False:
        raise ValueError("status unexpectedly claims shipping approval")
    if status["product"]["next_gate"] != "real-human-final-afe-acoustic-qualification":
        raise ValueError("unexpected next product gate")
    if status["historical_research_closure"]["architecture_search_paused"] is not True:
        raise ValueError("historical synthetic architecture closure should remain paused")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    status = build_status(args.root.resolve())
    if args.verify:
        verify(status)
    text = json.dumps(status, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
