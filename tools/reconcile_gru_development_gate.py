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
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

import iterate_gru_development as loop  # noqa: E402
from gru_development_gate import (  # noqa: E402
    POLICY as DEVELOPMENT_GATE_POLICY,
    evaluate_development_split,
    terminal_strict_streak,
)

WRAPPER_EVIDENCE_KEY = "development_training_wrapper"
RETAINED_WRAPPER_KEYS = (
    "training_acoustic_rotation",
    "development_negative_stress_support",
    WRAPPER_EVIDENCE_KEY,
)


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _bound_training_wrapper(manifest: dict) -> dict | None:
    raw = manifest.get(WRAPPER_EVIDENCE_KEY)
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("development training wrapper evidence must be an object")
    policy = str(raw.get("policy", ""))
    relative_text = str(raw.get("path", ""))
    expected_sha = str(raw.get("sha256", ""))
    if not policy or not relative_text:
        raise ValueError("development training wrapper policy/path is missing")
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        raise ValueError("development training wrapper SHA256 is invalid")
    relative = pathlib.PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("development training wrapper path must be repository-relative")
    path = (ROOT / pathlib.Path(*relative.parts)).resolve()
    try:
        normalized = path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("development training wrapper escapes repository root") from exc
    if not path.is_file():
        raise ValueError("development training wrapper file is missing")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError("development training wrapper SHA256 drifted before freeze")
    return {
        "policy": policy,
        "path": normalized,
        "sha256": actual_sha,
    }


def _recreate_frozen_candidate(
    selected: dict,
    manifest: dict,
    work: pathlib.Path,
    config_path: pathlib.Path,
    policy_path: pathlib.Path,
) -> dict:
    frozen = work / "frozen-candidate"
    shutil.rmtree(frozen, ignore_errors=True)
    freeze = loop.copy_frozen_candidate(selected, work, config_path, policy_path)
    freeze["development_gate_policy"] = DEVELOPMENT_GATE_POLICY
    for key in RETAINED_WRAPPER_KEYS:
        if key in manifest:
            freeze[key] = manifest[key]
    code = freeze.setdefault("training_code_sha256", {})
    if not isinstance(code, dict):
        raise ValueError("freeze training_code_sha256 must be an object")
    for path in (
        ROOT / "training" / "run_gru_development.py",
        ROOT / "tools" / "gru_development_gate.py",
        pathlib.Path(__file__).resolve(),
    ):
        code[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    wrapper = _bound_training_wrapper(manifest)
    if wrapper is not None:
        freeze[WRAPPER_EVIDENCE_KEY] = wrapper
        code[str(wrapper["path"])] = str(wrapper["sha256"])
    write_object(frozen / "freeze-manifest.json", freeze)
    return freeze


def reconcile(
    work: pathlib.Path,
    config_path: pathlib.Path,
    policy_path: pathlib.Path,
) -> dict:
    work = work.resolve()
    config_path = config_path.resolve()
    policy_path = policy_path.resolve()
    manifest_path = work / "development-loop-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("development loop manifest is missing")
    manifest = load_object(manifest_path)
    config = load_object(config_path)
    policy = load_object(policy_path)
    if manifest.get("policy") != "gru-development-curriculum-loop-v1":
        raise ValueError("unexpected development loop policy")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("development manifest has no records")

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("development record must be an object")
        calibration = evaluate_development_split(
            record["calibration"], record["calibration_domains"], config
        )
        test = evaluate_development_split(
            record["test"], record["test_domains"], config
        )
        record["calibration_development_gate"] = calibration
        record["test_development_gate"] = test
        record["calibration_gate"] = bool(calibration["qualified"])
        record["test_gate"] = bool(test["qualified"])

    required = positive_int(
        policy.get("stable_strict_pass_rounds"), "stable_strict_pass_rounds"
    )
    observed = terminal_strict_streak(records)
    stable = observed >= required
    selected = loop.select_best_strict_candidate(records) if stable else None
    qualified = stable and selected is not None

    manifest["development_gate_policy"] = DEVELOPMENT_GATE_POLICY
    manifest["stable_strict_pass_rounds_required"] = required
    manifest["stable_strict_pass_rounds_observed"] = observed
    manifest["development_qualified"] = qualified
    manifest["selected_round"] = int(selected["round"]) if selected is not None else None
    manifest["selected_score"] = float(selected["score"]) if selected is not None else None

    if selected is None:
        manifest.pop("frozen_candidate", None)
        shutil.rmtree(work / "frozen-candidate", ignore_errors=True)
    else:
        manifest["frozen_candidate"] = _recreate_frozen_candidate(
            selected, manifest, work, config_path, policy_path
        )

    write_object(manifest_path, manifest)
    return {
        "schema_version": 1,
        "policy": DEVELOPMENT_GATE_POLICY,
        "qualified": qualified,
        "stable_strict_pass_rounds_required": required,
        "stable_strict_pass_rounds_observed": observed,
        "selected_round": manifest["selected_round"],
        "selected_score": manifest["selected_score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    args = parser.parse_args()
    result = reconcile(args.work_dir, args.config, args.policy)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
