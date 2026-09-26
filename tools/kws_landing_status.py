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
        "schema_version": 1,
        "policy": "kws-product-landing-status-v1",
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
        "evidence": {
            "status": str(shipping.get("evidence_status") or ""),
            "synthetic_qualification_passed": synthetic_qualified,
            "real_human_final_afe_passed": False,
            "physical_target_board_passed": False,
        },
        "product": {
            "shipping_approved": bool(shipping.get("shipping_approved", False)),
            "blockers": pending,
            "next_gate": pending[0] if pending else None,
        },
        "research": {
            "architecture_search_paused": True,
            "next_authorized_lane": str(decision["next_authorized_lane"]),
            "reopen_condition": "product-data-implicates-model-capacity",
        },
        "control_plane": {
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
    if status["evidence"]["synthetic_qualification_passed"] is not True:
        raise ValueError("pinned model is not synthetic-qualified")
    if status["product"]["shipping_approved"] is not False:
        raise ValueError("status unexpectedly claims shipping approval")
    if status["product"]["next_gate"] != "real-human-final-afe-acoustic-qualification":
        raise ValueError("unexpected next product gate")
    if status["research"]["architecture_search_paused"] is not True:
        raise ValueError("synthetic architecture search should be paused")


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
