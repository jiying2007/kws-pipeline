#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

import iterate_rnn_development as loop  # noqa: E402
from rnn_development_gate import POLICY as DEVELOPMENT_GATE_POLICY, evaluate_development_split, terminal_strict_streak  # noqa: E402


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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


def _verify_record(record: dict, config: dict) -> None:
    calibration = evaluate_development_split(record["calibration"], record["calibration_domains"], config)
    test = evaluate_development_split(record["test"], record["test_domains"], config)
    if record.get("calibration_development_gate") != calibration:
        raise ValueError("RNN calibration development gate evidence mismatch")
    if record.get("test_development_gate") != test:
        raise ValueError("RNN test development gate evidence mismatch")
    if record.get("calibration_gate") is not bool(calibration["qualified"]):
        raise ValueError("RNN calibration gate flag mismatch")
    if record.get("test_gate") is not bool(test["qualified"]):
        raise ValueError("RNN test gate flag mismatch")


def verify_manifest(work: pathlib.Path, config_path: pathlib.Path, policy_path: pathlib.Path) -> dict:
    work = work.resolve()
    manifest = load_object(work / "development-loop-manifest.json")
    config = load_object(config_path.resolve())
    policy = load_object(policy_path.resolve())
    if manifest.get("policy") != loop.POLICY or manifest.get("model_family") != "rnn":
        raise ValueError("RNN development identity mismatch")
    if manifest.get("architecture") != loop.ARCHITECTURE:
        raise ValueError("RNN development architecture mismatch")
    if manifest.get("development_gate_policy") != DEVELOPMENT_GATE_POLICY:
        raise ValueError("RNN development gate policy mismatch")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("RNN development manifest has no records")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("RNN development record must be an object")
        _verify_record(record, config)
    required = positive_int(policy.get("stable_strict_pass_rounds"), "stable_strict_pass_rounds")
    observed = terminal_strict_streak(records)
    stable = observed >= required
    selected = loop.select_best_strict_candidate(records) if stable else None
    qualified = stable and selected is not None
    if manifest.get("development_qualified") is not qualified:
        raise ValueError("RNN development_qualified does not match robustness/stability gate")
    if int(manifest.get("stable_strict_pass_rounds_required", -1)) != required:
        raise ValueError("RNN required stable strict-pass streak mismatch")
    if int(manifest.get("stable_strict_pass_rounds_observed", -1)) != observed:
        raise ValueError("RNN observed stable strict-pass streak mismatch")
    selected_round = int(selected["round"]) if selected is not None else None
    selected_score = float(selected["score"]) if selected is not None else None
    if manifest.get("selected_round") != selected_round or manifest.get("selected_score") != selected_score:
        raise ValueError("RNN selected candidate mismatch")
    freeze_path = work / "frozen-candidate" / "freeze-manifest.json"
    if qualified:
        if not freeze_path.is_file():
            raise ValueError("qualified RNN development manifest is missing frozen candidate")
        freeze = load_object(freeze_path)
        if freeze.get("model_family") != "rnn" or freeze.get("architecture") != loop.ARCHITECTURE:
            raise ValueError("frozen RNN identity mismatch")
        if freeze.get("development_gate_policy") != DEVELOPMENT_GATE_POLICY:
            raise ValueError("frozen RNN development gate policy mismatch")
        if str(freeze.get("model_sha256")) != str(selected.get("model_sha256")):
            raise ValueError("frozen RNN model does not match selected round")
        if int(freeze.get("stable_strict_pass_rounds_required", -1)) != required or int(freeze.get("stable_strict_pass_rounds_observed", -1)) != observed:
            raise ValueError("frozen RNN stability evidence mismatch")
    elif freeze_path.exists():
        raise ValueError("unqualified RNN development retained a frozen candidate")
    return {"verified": True, "mode": "manifest", "policy": DEVELOPMENT_GATE_POLICY, "model_family": "rnn", "qualified": qualified, "stable_strict_pass_rounds_required": required, "stable_strict_pass_rounds_observed": observed, "selected_round": selected_round, "selected_score": selected_score}


def verify_candidate(candidate: pathlib.Path) -> dict:
    candidate = candidate.resolve()
    freeze = load_object(candidate / "freeze-manifest.json")
    config = load_object(candidate / "source-config.json")
    policy = load_object(candidate / "source-development-policy.json")
    selection_path = candidate / "selection-evidence.json"
    selection = load_object(selection_path)
    if freeze.get("model_family") != "rnn" or freeze.get("architecture") != loop.ARCHITECTURE:
        raise ValueError("frozen RNN candidate identity mismatch")
    if freeze.get("development_gate_policy") != DEVELOPMENT_GATE_POLICY:
        raise ValueError("frozen RNN development gate policy mismatch")
    for name, field in (("model.kwm", "model_sha256"), ("model.pt", "checkpoint_sha256"), ("keywords.kwk", "pack_sha256"), ("keywords.tsv", "keywords_sha256"), ("model-provenance.json", "provenance_sha256")):
        path = candidate / name
        if not path.is_file() or sha256_file(path) != str(freeze.get(field)):
            raise ValueError(f"frozen RNN member hash mismatch: {name}")
    if sha256_file(selection_path) != str(freeze.get("selection_evidence_sha256")):
        raise ValueError("frozen RNN selection evidence hash mismatch")
    calibration = evaluate_development_split(selection["calibration"], selection["calibration_domains"], config)
    test = evaluate_development_split(selection["test"], selection["test_domains"], config)
    if not calibration["qualified"] or not test["qualified"]:
        raise ValueError("frozen RNN selection evidence does not pass robustness gate")
    required = positive_int(policy.get("stable_strict_pass_rounds"), "stable_strict_pass_rounds")
    if int(freeze.get("stable_strict_pass_rounds_required", -1)) != required:
        raise ValueError("frozen RNN required stability evidence mismatch")
    if int(freeze.get("stable_strict_pass_rounds_observed", -1)) < required:
        raise ValueError("frozen RNN candidate lacks stable terminal strict evidence")
    stage = freeze.get("candidate_stage")
    if not isinstance(stage, dict):
        raise ValueError("frozen RNN candidate stage is missing")
    source_stage = policy.get("candidate_freeze", {})
    if int(stage.get("fresh_validation_seed_namespace", 0)) != int(source_stage.get("fresh_validation_seed_namespace", -1)):
        raise ValueError("frozen RNN fresh namespace mismatch")
    if str(stage.get("shadow_arena")) != str(source_stage.get("shadow_arena")):
        raise ValueError("frozen RNN shadow arena mismatch")
    if int(stage.get("formal_qualification_seed", 0)) != int(source_stage.get("formal_qualification_seed", -1)):
        raise ValueError("frozen RNN formal seed mismatch")
    return {"verified": True, "mode": "candidate", "policy": DEVELOPMENT_GATE_POLICY, "model_family": "rnn", "selected_round": int(selection["selected_round"]), "model_sha256": str(selection["model_sha256"])}


def self_test() -> dict:
    config = {"domain_gates": {"max_frr": 0.0, "max_far_per_hour": 0.0, "max_p95_latency_ms": 800.0, "max_far_frr": 0.0}, "robustness_gates": {"max_frr": 0.0, "max_far_per_hour": 0.0, "min_expected_wakes": 1, "min_negative_recordings": 1, "min_negative_audio_hours": 0.0, "required_distance_bins": ["5m"], "required_azimuth_deg": [], "required_snr_bands": [], "required_rt60_bands": [], "required_noise_profiles": [], "required_playback_states": [], "required_stress_slices": []}}
    base = {"frr": 0.0, "far_per_hour": 0.0, "p95_post_end_latency_ms": 0.0}
    domains = {"domains": {"distance:far": {"frr": 0.0}, "distance_bin:5m": {"expected": 1, "positive_recordings": 1, "negative_recordings": 1, "negative_audio_hours": 0.0, "wake_rate": 1.0, "frr": 0.0, "far_per_hour": 0.0}}}
    if evaluate_development_split(base, domains, config)["qualified"] is not True:
        raise ValueError("RNN self-test valid robustness gate did not pass")
    broken = json.loads(json.dumps(domains))
    broken["domains"]["distance_bin:5m"]["negative_recordings"] = 0
    if evaluate_development_split(base, broken, config)["qualified"] is not False:
        raise ValueError("RNN self-test insufficient negative support did not fail")
    return {"verified": True, "mode": "self-test", "policy": DEVELOPMENT_GATE_POLICY}


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--work-dir", type=pathlib.Path)
    group.add_argument("--candidate", type=pathlib.Path)
    group.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--policy", type=pathlib.Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.candidate is not None:
        result = verify_candidate(args.candidate)
    else:
        if args.config is None or args.policy is None:
            raise ValueError("--work-dir requires --config and --policy")
        result = verify_manifest(args.work_dir, args.config, args.policy)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
