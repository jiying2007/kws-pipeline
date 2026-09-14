from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess

POLICY = "staged-domain-base-handoff-v1"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def load_json(path: pathlib.Path, label: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def write_receipt(
    *, config_path: pathlib.Path, work: pathlib.Path, output: pathlib.Path, iteration_exit_code: int
) -> dict:
    if iteration_exit_code not in (0, 1):
        raise ValueError(f"base domain iteration failed with infrastructure exit code {iteration_exit_code}")
    config = load_json(config_path, "training config")
    manifest_path = work / "domain-loop-manifest.json"
    manifest = load_json(manifest_path, "base domain manifest")
    if not isinstance(manifest.get("records"), list) or not manifest["records"]:
        raise ValueError("base domain manifest has no training records")
    workflow_sha = str(os.environ.get("GITHUB_SHA") or "")
    if len(workflow_sha) != 40:
        raise ValueError("GITHUB_SHA is missing or invalid")
    receipt = {
        "schema_version": 1,
        "evidence_class": "model-training-base-stage",
        "policy": POLICY,
        "workflow_sha": workflow_sha,
        "checkout_commit_sha": git_head(),
        "training_config_sha256": sha256_file(config_path),
        "domain_manifest_sha256": sha256_file(manifest_path),
        "iteration_exit_code": iteration_exit_code,
        "qualification_holdout_seed": int(config["qualification_holdout_seed"]),
        "formal_seed_consumed": False,
        "record_count": len(manifest["records"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def verify_receipt(
    *, config_path: pathlib.Path, work: pathlib.Path, receipt_path: pathlib.Path, github_output: pathlib.Path | None
) -> dict:
    config = load_json(config_path, "training config")
    manifest_path = work / "domain-loop-manifest.json"
    load_json(manifest_path, "base domain manifest")
    receipt = load_json(receipt_path, "base stage receipt")
    if int(receipt.get("schema_version", 0)) != 1 or str(receipt.get("policy")) != POLICY:
        raise ValueError("base stage receipt policy/schema drifted")
    workflow_sha = str(os.environ.get("GITHUB_SHA") or "")
    if str(receipt.get("workflow_sha")) != workflow_sha:
        raise ValueError("base stage workflow SHA differs from continuation workflow SHA")
    if str(receipt.get("checkout_commit_sha")) != git_head():
        raise ValueError("base stage checkout commit differs from continuation checkout")
    if str(receipt.get("training_config_sha256")) != sha256_file(config_path):
        raise ValueError("base stage training config SHA differs from continuation config")
    if str(receipt.get("domain_manifest_sha256")) != sha256_file(manifest_path):
        raise ValueError("base stage domain manifest SHA differs from downloaded state")
    if int(receipt.get("qualification_holdout_seed", -1)) != int(config["qualification_holdout_seed"]):
        raise ValueError("base stage formal seed differs from continuation config")
    if bool(receipt.get("formal_seed_consumed", True)):
        raise ValueError("base stage receipt claims formal seed consumption")
    exit_code = int(receipt.get("iteration_exit_code", -1))
    if exit_code not in (0, 1):
        raise ValueError("base stage iteration exit code is invalid")
    if github_output is not None:
        with github_output.open("a", encoding="utf-8") as stream:
            stream.write(f"iteration_exit_code={exit_code}\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    write = sub.add_parser("write")
    write.add_argument("--config", required=True, type=pathlib.Path)
    write.add_argument("--work-dir", required=True, type=pathlib.Path)
    write.add_argument("--output", required=True, type=pathlib.Path)
    write.add_argument("--iteration-exit-code", required=True, type=int)
    verify = sub.add_parser("verify")
    verify.add_argument("--config", required=True, type=pathlib.Path)
    verify.add_argument("--work-dir", required=True, type=pathlib.Path)
    verify.add_argument("--receipt", required=True, type=pathlib.Path)
    verify.add_argument("--github-output", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "write":
        result = write_receipt(
            config_path=args.config.resolve(),
            work=args.work_dir.resolve(),
            output=args.output.resolve(),
            iteration_exit_code=int(args.iteration_exit_code),
        )
    else:
        result = verify_receipt(
            config_path=args.config.resolve(),
            work=args.work_dir.resolve(),
            receipt_path=args.receipt.resolve(),
            github_output=args.github_output.resolve() if args.github_output else None,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
