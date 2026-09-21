#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

import development_resume as development_resume  # noqa: E402
from development_loss_controller import (  # noqa: E402
    initial_controller,
    next_controller as next_loss_controller,
    validate_controller_config,
)
from development_failure_replay import render_development_failure_replay  # noqa: E402
from domain_curriculum import metric_hardness, update_curriculum  # noqa: E402
from hard_negative_replay import render_hard_negative_replay  # noqa: E402
from frontend_spec import FRONTEND_IDS  # noqa: E402
from iterate_domain import calibrate, evaluate, gate_values, objective, repo_path, run, safe_reset, sha256_file  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from rnn_development_gate import evaluate_development_split, terminal_strict_streak  # noqa: E402
from synthetic_audio import load_config  # noqa: E402

POLICY = "rnn-development-curriculum-loop-v1"
EVIDENCE_SCOPE = "development-only"
FREEZE_POLICY = "rnn-frozen-candidate-v1"
SELECTION_POLICY = "best-strict-development-objective-round"
MODEL_FAMILY = "rnn"
ARCHITECTURE = "tiny-streaming-rnn-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def development_frontend(model_cfg: dict) -> str:
    frontends = model_cfg.get("frontends")
    if not isinstance(frontends, list) or len(frontends) != 1:
        raise ValueError("RNN development requires exactly one frontend")
    frontend = str(frontends[0])
    if frontend not in FRONTEND_IDS:
        raise ValueError(f"unsupported RNN development frontend: {frontend}")
    return frontend


def validate_policy(path: pathlib.Path) -> dict:
    policy = load_object(path)
    if policy.get("policy") != POLICY or policy.get("model_family") != MODEL_FAMILY:
        raise ValueError("RNN development policy identity mismatch")
    if policy.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError("RNN development evidence scope mismatch")
    for field in ("formal_qualification_used", "shadow_used", "qualification_used"):
        if policy.get(field) is not False:
            raise ValueError(f"RNN development loop requires {field}=false")
    max_rounds = int(policy.get("max_rounds", 0))
    min_rounds = int(policy.get("min_rounds", 0))
    patience = int(policy.get("patience", -1))
    stable = int(policy.get("stable_strict_pass_rounds", 0))
    epochs = int(policy.get("epochs_per_round", 0))
    if not 1 <= min_rounds <= max_rounds <= 24:
        raise ValueError("RNN development round bounds are invalid")
    if patience < 0 or stable <= 0 or stable > max_rounds or epochs <= 0:
        raise ValueError("RNN development stopping policy is invalid")
    decay = finite(policy.get("lr_decay_per_round"), "lr_decay_per_round")
    if not 0.0 < decay <= 1.0:
        raise ValueError("lr_decay_per_round must be in (0,1]")
    if int(policy.get("fixed_replay_repeat", 0)) < 1:
        raise ValueError("fixed_replay_repeat must be >= 1")
    if not 1 <= int(policy.get("failure_replay_repeat_max", 0)) <= 8:
        raise ValueError("failure_replay_repeat_max must be 1..8")
    if policy.get("failure_replay_latch_after_failure") is not True:
        raise ValueError("RNN development requires failure_replay_latch_after_failure=true")
    if int(policy.get("training_seed_namespace", 0)) <= 0:
        raise ValueError("training_seed_namespace must be positive")
    if int(policy.get("training_acoustic_seed_namespace", 0)) <= 0:
        raise ValueError("training_acoustic_seed_namespace must be positive")
    controller = policy.get("loss_controller")
    validate_controller_config(controller)
    freeze = policy.get("candidate_freeze")
    if not isinstance(freeze, dict):
        raise ValueError("candidate_freeze must be an object")
    if freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("RNN candidate freeze selection policy drifted")
    for field in ("fresh_validation_required", "shadow_required", "formal_qualification_required", "bounded_repair_only_after_freeze"):
        if freeze.get(field) is not True:
            raise ValueError(f"RNN candidate freeze requires {field}=true")
    for field in ("validation_feedback_allowed", "threshold_feedback_allowed", "training_rule_feedback_allowed"):
        if freeze.get(field) is not False:
            raise ValueError(f"RNN candidate freeze requires {field}=false")
    if int(freeze.get("fresh_validation_seed_namespace", 0)) <= 0:
        raise ValueError("RNN fresh namespace must be positive")
    if int(freeze.get("formal_qualification_seed", 0)) <= 0:
        raise ValueError("RNN formal seed must be positive")
    if not str(freeze.get("shadow_arena", "")).startswith("rnn-"):
        raise ValueError("RNN shadow arena identity mismatch")
    return policy


def read_jsonl_count(path_value: object) -> int:
    if not isinstance(path_value, str) or not path_value:
        return 0
    path = pathlib.Path(path_value)
    if not path.is_file():
        return 0
    return sum(1 for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip())


def failure_counts(calibration: dict, test: dict) -> tuple[int, int]:
    fr = read_jsonl_count(calibration.get("false_rejects_path")) + read_jsonl_count(test.get("false_rejects_path"))
    fa = read_jsonl_count(calibration.get("false_positives_path")) + read_jsonl_count(test.get("false_positives_path"))
    return fr, fa


def choose_harder(left: dict | None, right: dict | None, label: str) -> dict:
    if not isinstance(left, dict):
        return dict(right or {})
    if not isinstance(right, dict):
        return dict(left)
    return dict(right) if metric_hardness(right, label) > metric_hardness(left, label) else dict(left)


def merge_domain_metrics(calibration: dict, test: dict) -> dict:
    result: dict = {"domains": {}, "keyword_domains": {}}
    cal_domains = calibration.get("domains", {})
    test_domains = test.get("domains", {})
    if not isinstance(cal_domains, dict) or not isinstance(test_domains, dict):
        raise ValueError("domain metrics are missing domains")
    for key in sorted(set(cal_domains) | set(test_domains)):
        result["domains"][key] = choose_harder(cal_domains.get(key), test_domains.get(key), f"merged.{key}")
    cal_keywords = calibration.get("keyword_domains", {}) or {}
    test_keywords = test.get("keyword_domains", {}) or {}
    if not isinstance(cal_keywords, dict) or not isinstance(test_keywords, dict):
        raise ValueError("keyword domain metrics must be objects")
    for keyword_id in sorted(set(cal_keywords) | set(test_keywords), key=str):
        cal_value = cal_keywords.get(keyword_id, {})
        test_value = test_keywords.get(keyword_id, {})
        cal_map = cal_value.get("domains", {}) if isinstance(cal_value, dict) else {}
        test_map = test_value.get("domains", {}) if isinstance(test_value, dict) else {}
        merged: dict[str, dict] = {}
        for key in sorted(set(cal_map) | set(test_map)):
            merged[key] = choose_harder(cal_map.get(key), test_map.get(key), f"merged.keyword.{keyword_id}.{key}")
        result["keyword_domains"][str(keyword_id)] = {"domains": merged}
    return result


def controller_initial(policy: dict) -> dict:
    return initial_controller(policy)


def controller_next(
    policy: dict,
    current: dict,
    false_rejects: int,
    false_accepts: int,
    *,
    frr: float | None = None,
    far_per_hour: float | None = None,
) -> dict:
    return next_loss_controller(
        policy,
        current,
        false_rejects,
        false_accepts,
        frr=frr,
        far_per_hour=far_per_hour,
        latch_after_failure=policy.get("failure_replay_latch_after_failure", False) is True,
    )


def repeated(path: pathlib.Path | None, count: int) -> list[pathlib.Path]:
    return [] if path is None or count <= 0 else [path] * count


def strict(record: dict) -> bool:
    return record.get("calibration_gate") is True and record.get("test_gate") is True


def select_best_strict_candidate(records: list[dict]) -> dict | None:
    eligible = [record for record in records if strict(record)]
    if not eligible:
        return None
    return min(eligible, key=lambda record: (finite(record.get("score"), f"round-{record.get('round')}.score"), -int(record["round"]), str(record.get("frontend", ""))))


def _write_object(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _copy_frozen_candidate(selected: dict, output: pathlib.Path, config: pathlib.Path, policy: pathlib.Path) -> dict:
    frozen = output / "frozen-candidate"
    frozen.mkdir(parents=True, exist_ok=True)
    mapping = ((pathlib.Path(selected["model"]), "model.kwm"), (pathlib.Path(selected["checkpoint"]), "model.pt"), (pathlib.Path(selected["provenance"]), "model-provenance.json"), (pathlib.Path(selected["pack"]), "keywords.kwk"), (pathlib.Path(selected["keywords"]), "keywords.tsv"))
    for source, name in mapping:
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"selected RNN candidate member missing: {source}")
        shutil.copy2(source, frozen / name)
    config_value = load_object(config)
    policy_value = load_object(policy)
    _write_object(frozen / "source-config.json", config_value)
    _write_object(frozen / "source-development-policy.json", policy_value)
    selection = dict(selected)
    selection.update({"schema_version": 1, "policy": SELECTION_POLICY, "model_family": MODEL_FAMILY, "architecture": ARCHITECTURE, "evidence_scope": EVIDENCE_SCOPE, "selected_round": int(selected["round"]), "selected_score": float(selected["score"]), "qualification_used": False, "shadow_used": False, "formal_qualification_used": False})
    _write_object(frozen / "selection-evidence.json", selection)
    code_paths = [pathlib.Path(__file__).resolve(), TRAINING / "train_ctc.py", TRAINING / "feature_cached_trainer.py",
        TRAINING / "development_resume.py", TRAINING / "model.py", TRAINING / "export_model.py", TRAINING / "domain_curriculum.py", TRAINING / "hard_negative_replay.py", TRAINING / "development_failure_replay.py", TOOLS / "rnn_development_gate.py"]
    freeze_cfg = policy_value["candidate_freeze"]
    freeze = {"schema_version": 1, "policy": FREEZE_POLICY, "source_policy": POLICY, "model_family": MODEL_FAMILY, "architecture": ARCHITECTURE, "evidence_scope": EVIDENCE_SCOPE, "selection_policy": SELECTION_POLICY, "selected_round": int(selected["round"]), "selected_score": float(selected["score"]), "model_sha256": sha256_file(frozen / "model.kwm"), "checkpoint_sha256": sha256_file(frozen / "model.pt"), "pack_sha256": sha256_file(frozen / "keywords.kwk"), "keywords_sha256": sha256_file(frozen / "keywords.tsv"), "provenance_sha256": sha256_file(frozen / "model-provenance.json"), "config_sha256": sha256_file(config), "development_policy_sha256": sha256_file(policy), "selection_evidence_sha256": sha256_file(frozen / "selection-evidence.json"), "training_code_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in code_paths if p.is_file()}, "selection_evidence": ["development-calibration", "development-test"], "qualification_used_for_selection": False, "shadow_used_for_selection": False, "formal_qualification_used_for_selection": False, "candidate_stage": {"fresh_validation_required": True, "fresh_validation_seed_namespace": int(freeze_cfg["fresh_validation_seed_namespace"]), "shadow_required": True, "shadow_arena": str(freeze_cfg["shadow_arena"]), "formal_qualification_required": True, "formal_qualification_seed": int(freeze_cfg["formal_qualification_seed"]), "bounded_repair_only": True, "validation_feedback_allowed": False, "threshold_feedback_allowed": False, "training_rule_feedback_allowed": False}}
    _write_object(frozen / "freeze-manifest.json", freeze)
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Governed development-only TinyStreamingRNN curriculum loop.")
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--resume-state", type=pathlib.Path)
    parser.add_argument("--round-budget", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    policy_path = args.policy.resolve()
    cfg = load_config(config_path)
    policy = validate_policy(policy_path)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("RNN runtime runner does not exist")
    work = args.work_dir.resolve()
    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    if not isinstance(model_cfg, dict):
        raise ValueError("model config must be an object")
    frontend = development_frontend(model_cfg)
    gates = gate_values(cfg.get("domain_gates", {}))
    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    calibration_parallel_trials = int(
        cfg.get("calibration", {}).get("max_parallel_trials", 1)
    )
    if not 1 <= calibration_parallel_trials <= 4:
        raise ValueError("calibration.max_parallel_trials must be 1..4")
    if not thresholds:
        raise ValueError("calibration threshold grid is empty")
    if args.round_budget is not None and args.round_budget <= 0:
        raise ValueError("round budget must be positive")
    state_path = work / "development-resume-state.json"
    if args.resume_state is None:
        work = safe_reset(work)
        records: list[dict] = []
        curriculum: dict | None = None
        controller = controller_initial(policy)
        previous_checkpoint: pathlib.Path | None = None
        best_objective: float | None = None
        stale_rounds = 0
        start_round = 0
    else:
        work.mkdir(parents=True, exist_ok=True)
        restored = development_resume.load_state(
            args.resume_state.resolve(),
            work=work,
            model_family=MODEL_FAMILY,
            architecture=ARCHITECTURE,
            source_policy=POLICY,
            config_path=config_path,
            policy_path=policy_path,
        )
        records = restored["records"]
        start_round = int(restored["next_round"])
        if restored["complete"]:
            manifest = load_object(work / "development-loop-manifest.json")
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
            return 0 if manifest.get("development_qualified") is True else 1
        curriculum = (
            load_object(work / "curriculum" / f"round-{start_round - 1:02d}.json")
            if start_round > 0
            else None
        )
        controller, best_objective, stale_rounds, _ = development_resume.rebuild_progress(
            records, policy, controller_initial, controller_next, strict
        )
        previous_checkpoint = pathlib.Path(str(records[-1]["checkpoint"])) if records else None
    rounds_run = 0
    for round_index in range(start_round, int(policy["max_rounds"])):
        dataset = work / "datasets" / f"round-{round_index:02d}"
        render_domain_dataset(
            config_path,
            dataset,
            curriculum_weights=curriculum,
            splits=("train", "calibration", "test"),
        )
        run([sys.executable, str(TRAINING / "audit_dataset.py"), "--split", f"train={dataset / 'train.tsv'}", "--split", f"calibration={dataset / 'calibration.tsv'}", "--split", f"test={dataset / 'test.tsv'}", "--report", str(dataset / "development-audit.json"), "--fail-within-split"])
        fixed_replay = render_hard_negative_replay(config_path, work / "fixed-replay" / f"round-{round_index:02d}", round_index=round_index, curriculum_weights=curriculum)
        failure_replay = render_development_failure_replay(config_path, records, work, work / "failure-replay" / f"round-{round_index:02d}")
        fixed_manifest = pathlib.Path(str(fixed_replay["manifest"])) if int(fixed_replay.get("examples", 0)) > 0 else None
        failure_manifest = pathlib.Path(str(failure_replay["manifest"])) if int(failure_replay.get("examples", 0)) > 0 else None
        manifests = [dataset / "train.tsv"]
        manifests.extend(repeated(fixed_manifest, int(policy["fixed_replay_repeat"])))
        manifests.extend(repeated(failure_manifest, int(controller["failure_replay_repeat"])))
        for manifest in manifests:
            if not manifest.is_file() or manifest.stat().st_size <= 0:
                raise ValueError(f"RNN development manifest missing: {manifest}")
        candidate = work / "candidates" / f"round-{round_index:02d}"
        candidate.mkdir(parents=True, exist_ok=True)
        checkpoint = candidate / "model.pt"
        learning_rate = float(train_cfg.get("lr", 0.001)) * float(policy["lr_decay_per_round"]) ** round_index
        command = [sys.executable, str(TRAINING / "train_ctc.py")]
        for manifest in manifests:
            command.extend(["--manifest", str(manifest)])
        command.extend(["--tokens", str(tokens), "--keywords", str(keywords), "--frontend", frontend, "--feature-dim", str(int(model_cfg.get("feature_dim", 32))), "--hidden-dim", str(int(model_cfg.get("hidden_dim", 64))), "--epochs", str(int(policy["epochs_per_round"])), "--batch-size", str(int(train_cfg.get("batch_size", 16))), "--lr", str(learning_rate), "--seed", str(int(cfg.get("seed", 1337)) + int(policy["training_seed_namespace"]) + round_index * 1009), "--positive-example-weight", str(float(controller["positive_example_weight"])), "--wake-example-weight", str(float(controller["wake_example_weight"])), "--ordered-token-loss-weight", str(float(controller["ordered_token_loss_weight"])), "--output", str(checkpoint)])
        if previous_checkpoint is not None:
            command.extend(["--warm-start", str(previous_checkpoint)])
        run(command)
        model = candidate / "model.kwm"
        run([sys.executable, str(TRAINING / "export_model.py"), "--checkpoint", str(checkpoint), "--tokens", str(tokens), "--output", str(model)])
        provenance = pathlib.Path(str(model) + ".provenance.json")
        calibrated, pack, cal_base, cal_domains = calibrate(
            runner=runner,
            model=model,
            tokens=tokens,
            source_keywords=keywords,
            references=dataset / "calibration.references.jsonl",
            output=candidate / "calibration",
            thresholds=thresholds,
            rounds=coordinate_rounds,
            gates=gates,
            parallel_trials=calibration_parallel_trials,
        )
        test_base, test_domains = evaluate(runner=runner, model=model, pack=pack, references=dataset / "test.references.jsonl", output=candidate / "test")
        cal_contract = evaluate_development_split(cal_base, cal_domains, cfg)
        test_contract = evaluate_development_split(test_base, test_domains, cfg)
        score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)
        fr, fa = failure_counts(cal_base, test_base)
        record = {"round": round_index, "model_family": MODEL_FAMILY, "architecture": ARCHITECTURE, "frontend": frontend, "candidate": 0, "score": score, "model": str(model), "model_sha256": sha256_file(model), "checkpoint": str(checkpoint), "provenance": str(provenance), "provenance_sha256": sha256_file(provenance), "keywords": str(calibrated), "pack": str(pack), "calibration": cal_base, "calibration_domains": cal_domains, "calibration_development_gate": cal_contract, "test": test_base, "test_domains": test_domains, "test_development_gate": test_contract, "calibration_gate": bool(cal_contract["qualified"]), "test_gate": bool(test_contract["qualified"]), "false_rejects": fr, "false_accepts": fa, "training": {"architecture": ARCHITECTURE, "warm_started": previous_checkpoint is not None, "epochs": int(policy["epochs_per_round"]), "learning_rate": learning_rate, "positive_example_weight": float(controller["positive_example_weight"]), "wake_example_weight": float(controller["wake_example_weight"]), "ordered_token_loss_weight": float(controller["ordered_token_loss_weight"]), "fixed_replay_repeat": int(policy["fixed_replay_repeat"]), "fixed_replay_examples": int(fixed_replay.get("examples", 0)), "failure_replay_repeat": int(controller["failure_replay_repeat"]), "failure_replay_examples": int(failure_replay.get("examples", 0)), "manifest_count": len(manifests)}}
        records.append(record)
        previous_checkpoint = checkpoint
        merged = merge_domain_metrics(cal_domains, test_domains)
        curriculum = update_curriculum(merged, previous=curriculum, strength=float(cfg.get("domain_iteration", {}).get("curriculum_strength", 2.5)), max_weight=float(cfg.get("domain_iteration", {}).get("max_domain_weight", 6.0)))
        curriculum_path = work / "curriculum" / f"round-{round_index:02d}.json"
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        _write_object(curriculum_path, curriculum)
        controller_frr = max(float(cal_base["frr"]), float(test_base["frr"]))
        controller_far_per_hour = max(
            float(cal_base["far_per_hour"]),
            float(test_base["far_per_hour"]),
        )
        controller = controller_next(
            policy,
            controller,
            fr,
            fa,
            frr=controller_frr,
            far_per_hour=controller_far_per_hour,
        )
        record["controller_feedback"] = {
            "signal_mode": str(controller["controller_signal_mode"]),
            "frr": controller_frr,
            "far_per_hour": controller_far_per_hour,
            "severity": float(controller["controller_severity"]),
            "next_positive_example_weight": float(controller["positive_example_weight"]),
            "next_wake_example_weight": float(controller["wake_example_weight"]),
            "next_ordered_token_loss_weight": float(controller["ordered_token_loss_weight"]),
            "next_failure_replay_repeat": int(controller["failure_replay_repeat"]),
        }
        if best_objective is None or score < best_objective - 1.0e-12:
            best_objective = score
            stale_rounds = 0
        else:
            stale_rounds += 1
        completed = round_index + 1
        observed = terminal_strict_streak(records)
        rounds_run += 1
        development_resume.write_state(
            state_path,
            work=work,
            model_family=MODEL_FAMILY,
            architecture=ARCHITECTURE,
            source_policy=POLICY,
            config_path=config_path,
            policy_path=policy_path,
            records=records,
            complete=False,
        )
        terminal_stop = (
            completed >= int(policy["min_rounds"])
            and (
                observed >= int(policy["stable_strict_pass_rounds"])
                or stale_rounds >= int(policy["patience"])
            )
        )
        if terminal_stop:
            break
        if (
            args.round_budget is not None
            and rounds_run >= args.round_budget
            and completed < int(policy["max_rounds"])
        ):
            print(json.dumps({"segment_complete": True, "next_round": completed}, sort_keys=True))
            return development_resume.SEGMENT_CONTINUE_EXIT_CODE
    required = int(policy["stable_strict_pass_rounds"])
    observed = terminal_strict_streak(records)
    selected = select_best_strict_candidate(records) if observed >= required else None
    manifest = {"schema_version": 1, "policy": POLICY, "model_family": MODEL_FAMILY, "architecture": ARCHITECTURE, "evidence_scope": EVIDENCE_SCOPE, "development_gate_policy": "rnn-development-full-robustness-gate-v1", "development_qualified": selected is not None, "qualification_used": False, "shadow_used": False, "formal_qualification_used": False, "selection_policy": SELECTION_POLICY, "stable_strict_pass_rounds_required": required, "stable_strict_pass_rounds_observed": observed, "selected_round": int(selected["round"]) if selected is not None else None, "selected_score": float(selected["score"]) if selected is not None else None, "candidate_stage_feedback_allowed": False, "records": records, "final_curriculum": curriculum or {}, "next_controller": controller, "config_sha256": sha256_file(config_path), "development_policy_sha256": sha256_file(policy_path)}
    if selected is not None:
        manifest["frozen_candidate"] = _copy_frozen_candidate(selected, work, config_path, policy_path)
    manifest_path = work / "development-loop-manifest.json"
    _write_object(manifest_path, manifest)
    development_resume.write_state(
        state_path,
        work=work,
        model_family=MODEL_FAMILY,
        architecture=ARCHITECTURE,
        source_policy=POLICY,
        config_path=config_path,
        policy_path=policy_path,
        records=records,
        complete=True,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if selected is not None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
