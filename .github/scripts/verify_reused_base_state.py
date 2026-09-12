#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess

POLICY = "cross-run-development-base-reuse-v1"
BASE_POLICY = "staged-domain-base-handoff-v1"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: pathlib.Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def fetch_run(repo: str, run_id: int) -> dict:
    raw = subprocess.check_output(
        ["gh", "api", f"repos/{repo}/actions/runs/{run_id}"], text=True
    )
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("source workflow run API payload is invalid")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True, type=int)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    repo = str(os.environ.get("GITHUB_REPOSITORY") or "")
    if "/" not in repo:
        raise ValueError("GITHUB_REPOSITORY is missing")
    source_head = str(args.source_head)
    if len(source_head) != 40:
        raise ValueError("source head must be a full 40-hex commit SHA")

    run = fetch_run(repo, int(args.source_run_id))
    if int(run.get("id", -1)) != int(args.source_run_id):
        raise ValueError("source workflow run id mismatch")
    if str(run.get("name")) != "model-training":
        raise ValueError("source run is not model-training")
    if str(run.get("head_sha")) != source_head:
        raise ValueError("source run head_sha differs from expected development head")
    if str(run.get("status")) != "completed":
        raise ValueError("source model-training run is not terminal")

    root = args.work_dir.resolve()
    config = args.config.resolve()
    receipt = load_json(root / "base-stage-receipt.json", "base-stage receipt")
    manifest_path = root / "domain-loop-manifest.json"
    manifest = load_json(manifest_path, "domain manifest")

    if str(receipt.get("policy")) != BASE_POLICY or int(receipt.get("schema_version", 0)) != 1:
        raise ValueError("source base-stage receipt policy/schema drifted")
    if str(receipt.get("training_config_sha256")) != sha256(config):
        raise ValueError("source base-stage training config SHA differs from experiment config")
    if str(receipt.get("domain_manifest_sha256")) != sha256(manifest_path):
        raise ValueError("source base-stage domain manifest SHA is inconsistent")
    if int(receipt.get("qualification_holdout_seed", -1)) != 271843:
        raise ValueError("source base-state formal seed identity drifted")
    if bool(receipt.get("formal_seed_consumed", True)):
        raise ValueError("source base-state claims formal seed consumption")
    if int(receipt.get("iteration_exit_code", -1)) not in (0, 1):
        raise ValueError("source base-stage iteration exit code is invalid")
    if not bool(manifest.get("development_qualified")):
        raise ValueError("source base-state is not development-qualified")
    selection = manifest.get("candidate_selection")
    if not isinstance(selection, dict) or str(selection.get("policy")) != "latest-strict-gate-passing-round":
        raise ValueError("source base-state selection policy is not strict/latest")

    # pull_request runs use a workflow/check-out SHA identity that can differ from
    # Actions REST head_sha (PR head). Keep both identities explicit rather than
    # conflating them. The artifact is selected by exact source run id, while the
    # REST API above binds that run to source_head. The receipt remains internally
    # self-consistent and binds config + manifest + formal-seed ownership.
    workflow_sha = str(receipt.get("workflow_sha") or "")
    checkout_sha = str(receipt.get("checkout_commit_sha") or "")
    if len(workflow_sha) != 40 or len(checkout_sha) != 40:
        raise ValueError("source receipt workflow/checkout SHA identity is invalid")
    if workflow_sha != checkout_sha:
        raise ValueError("source receipt workflow SHA and checkout SHA disagree")

    result = {
        "schema_version": 1,
        "policy": POLICY,
        "source_run_id": int(args.source_run_id),
        "source_api_head_sha": source_head,
        "source_receipt_workflow_sha": workflow_sha,
        "source_receipt_checkout_sha": checkout_sha,
        "training_config_sha256": sha256(config),
        "domain_manifest_sha256": sha256(manifest_path),
        "qualification_holdout_seed": 271843,
        "formal_seed_consumed": False,
        "selected_round": int(selection["selected_round"]),
        "selected_frontend": str(selection["selected_frontend"]),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
