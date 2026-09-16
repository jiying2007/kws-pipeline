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

import iterate_rnn_development as loop  # noqa: E402
from rnn_development_gate import POLICY as DEVELOPMENT_GATE_POLICY, evaluate_development_split, terminal_strict_streak  # noqa: E402

LOOP_POLICY = "rnn-development-curriculum-loop-v1"
RETAINED_WRAPPER_KEYS = ("training_acoustic_rotation", "development_negative_stress_support")


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


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


def _recreate_frozen_candidate(selected: dict, manifest: dict, work: pathlib.Path, config_path: pathlib.Path, policy_path: pathlib.Path, required: int, observed: int) -> dict:
    frozen = work / "frozen-candidate"
    shutil.rmtree(frozen, ignore_errors=True)
    freeze = loop._copy_frozen_candidate(selected, work, config_path, policy_path)
    freeze["development_gate_policy"] = DEVELOPMENT_GATE_POLICY
    freeze["stable_strict_pass_rounds_required"] = required
    freeze["stable_strict_pass_rounds_observed"] = observed
    for key in RETAINED_WRAPPER_KEYS:
        if key in manifest:
            freeze[key] = manifest[key]
    code = freeze.setdefault("training_code_sha256", {})
    if not isinstance(code, dict):
        raise ValueError("freeze training_code_sha256 must be an object")
    for path in (ROOT / "training" / "run_rnn_development.py", ROOT / "tools" / "rnn_development_gate.py", pathlib.Path(__file__).resolve()):
        code[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    write_object(frozen / "freeze-manifest.json", freeze)
    selection_path = frozen / "selection-evidence.json"
    selection = load_object(selection_path)
    selection["development_gate_policy"] = DEVELOPMENT_GATE_POLICY
    selection["stable_strict_pass_rounds_required"] = required
    selection["stable_strict_pass_rounds_observed"] = observed
    write_object(selection_path, selection)
    freeze["selection_evidence_sha256"] = sha256_file(selection_path)
    write_object(frozen / "freeze-manifest.json", freeze)
    return freeze


def reconcile(work: pathlib.Path, config_path: pathlib.Path, policy_path: pathlib.Path) -> dict:
    work = work.resolve()
    config_path = config_path.resolve()
    policy_path = policy_path.resolve()
    manifest_path = work / "development-loop-manifest.json"
    if not manifest_path.is_file():
        raise ValueError("RNN development loop manifest is missing")
    manifest = load_object(manifest_path)
    config = load_object(config_path)
    policy = load_object(policy_path)
    if manifest.get("policy") != LOOP_POLICY or manifest.get("model_family") != "rnn":
        raise ValueError("unexpected RNN development identity")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("RNN development manifest has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("RNN development record must be an object")
        calibration = evaluate_development_split(record["calibration"], record["calibration_domains"], config)
        test = evaluate_development_split(record["test"], record["test_domains"], config)
        record["calibration_development_gate"] = calibration
        record["test_development_gate"] = test
        record["calibration_gate"] = bool(calibration["qualified"])
        record["test_gate"] = bool(test["qualified"])
    required = positive_int(policy.get("stable_strict_pass_rounds"), "stable_strict_pass_rounds")
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
        manifest["frozen_candidate"] = _recreate_frozen_candidate(selected, manifest, work, config_path, policy_path, required, observed)
    write_object(manifest_path, manifest)
    return {"schema_version": 1, "policy": DEVELOPMENT_GATE_POLICY, "model_family": "rnn", "qualified": qualified, "stable_strict_pass_rounds_required": required, "stable_strict_pass_rounds_observed": observed, "selected_round": manifest["selected_round"], "selected_score": manifest["selected_score"]}


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
