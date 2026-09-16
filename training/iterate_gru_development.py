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

from development_failure_replay import render_development_failure_replay  # noqa: E402
from domain_curriculum import metric_hardness, update_curriculum  # noqa: E402
from hard_negative_replay import render_hard_negative_replay  # noqa: E402
from iterate_domain import (  # noqa: E402
    calibrate,
    evaluate,
    gate_values,
    objective,
    repo_path,
    run,
    safe_reset,
    sha256_file,
)
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402

POLICY = "gru-development-curriculum-loop-v1"
EVIDENCE_SCOPE = "development-only"
FREEZE_POLICY = "gru-frozen-candidate-v1"
SELECTION_POLICY = "best-strict-development-objective-round"


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


def validate_policy(path: pathlib.Path) -> dict:
    policy = load_object(path)
    if policy.get("policy") != POLICY or policy.get("evidence_scope") != EVIDENCE_SCOPE:
        raise ValueError("GRU development policy identity mismatch")
    for field in ("formal_qualification_used", "shadow_used", "qualification_used"):
        if policy.get(field) is not False:
            raise ValueError(f"development loop requires {field}=false")
    max_rounds = int(policy.get("max_rounds", 0))
    min_rounds = int(policy.get("min_rounds", 0))
    patience = int(policy.get("patience", -1))
    stable = int(policy.get("stable_strict_pass_rounds", 0))
    epochs = int(policy.get("epochs_per_round", 0))
    if not 1 <= min_rounds <= max_rounds <= 24:
        raise ValueError("development round bounds are invalid")
    if patience < 0 or stable <= 0 or stable > max_rounds or epochs <= 0:
        raise ValueError("development stopping policy is invalid")
    decay = finite(policy.get("lr_decay_per_round"), "lr_decay_per_round")
    if not 0.0 < decay <= 1.0:
        raise ValueError("lr_decay_per_round must be in (0,1]")
    if int(policy.get("fixed_replay_repeat", 0)) < 1:
        raise ValueError("fixed_replay_repeat must be >= 1")
    if not 1 <= int(policy.get("failure_replay_repeat_max", 0)) <= 8:
        raise ValueError("failure_replay_repeat_max must be 1..8")
    controller = policy.get("loss_controller")
    if not isinstance(controller, dict):
        raise ValueError("loss_controller must be an object")
    for prefix in ("positive_example_weight", "ordered_token_loss_weight"):
        initial = finite(controller.get(f"{prefix}_initial"), f"{prefix}_initial")
        low = finite(controller.get(f"{prefix}_min"), f"{prefix}_min")
        high = finite(controller.get(f"{prefix}_max"), f"{prefix}_max")
        step = finite(controller.get(f"{prefix}_step"), f"{prefix}_step")
        if not 0.0 < low <= initial <= high or step <= 0.0:
            raise ValueError(f"{prefix} controller bounds are invalid")
    freeze = policy.get("candidate_freeze")
    if not isinstance(freeze, dict):
        raise ValueError("candidate_freeze must be an object")
    if freeze.get("selection_policy") != SELECTION_POLICY:
        raise ValueError("candidate freeze selection policy drifted")
    for field in (
        "fresh_validation_required",
        "shadow_required",
        "formal_qualification_required",
        "bounded_repair_only_after_freeze",
    ):
        if freeze.get(field) is not True:
            raise ValueError(f"candidate freeze requires {field}=true")
    for field in (
        "validation_feedback_allowed",
        "threshold_feedback_allowed",
        "training_rule_feedback_allowed",
    ):
        if freeze.get(field) is not False:
            raise ValueError(f"candidate freeze requires {field}=false")
    return policy


def read_jsonl_count(path_value: object) -> int:
    if not isinstance(path_value, str) or not path_value:
        return 0
    path = pathlib.Path(path_value)
    if not path.is_file():
        return 0
    return sum(1 for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip())


def failure_counts(calibration: dict, test: dict) -> tuple[int, int]:
    false_rejects = read_jsonl_count(calibration.get("false_rejects_path")) + read_jsonl_count(
        test.get("false_rejects_path")
    )
    false_accepts = read_jsonl_count(calibration.get("false_positives_path")) + read_jsonl_count(
        test.get("false_positives_path")
    )
    return false_rejects, false_accepts


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
        result["domains"][key] = choose_harder(
            cal_domains.get(key), test_domains.get(key), f"merged.{key}"
        )
    cal_keywords = calibration.get("keyword_domains", {}) or {}
    test_keywords = test.get("keyword_domains", {}) or {}
    if not isinstance(cal_keywords, dict) or not isinstance(test_keywords, dict):
        raise ValueError("keyword domain metrics must be objects")
    for keyword_id in sorted(set(cal_keywords) | set(test_keywords), key=str):
        cal_value = cal_keywords.get(keyword_id, {})
        test_value = test_keywords.get(keyword_id, {})
        cal_map = cal_value.get("domains", {}) if isinstance(cal_value, dict) else {}
        test_map = test_value.get("domains", {}) if isinstance(test_value, dict) else {}
        if not isinstance(cal_map, dict) or not isinstance(test_map, dict):
            raise ValueError("keyword domain maps must be objects")
        merged: dict[str, dict] = {}
        for key in sorted(set(cal_map) | set(test_map)):
            merged[key] = choose_harder(
                cal_map.get(key), test_map.get(key), f"merged.keyword.{keyword_id}.{key}"
            )
        result["keyword_domains"][str(keyword_id)] = {"domains": merged}
    return result


def controller_initial(policy: dict) -> dict:
    raw = policy["loss_controller"]
    return {
        "positive_example_weight": float(raw["positive_example_weight_initial"]),
        "ordered_token_loss_weight": float(raw["ordered_token_loss_weight_initial"]),
        "failure_replay_repeat": 0,
    }


def controller_next(policy: dict, current: dict, false_rejects: int, false_accepts: int) -> dict:
    raw = policy["loss_controller"]
    positive = float(current["positive_example_weight"])
    ordered = float(current["ordered_token_loss_weight"])
    p_step = float(raw["positive_example_weight_step"])
    o_step = float(raw["ordered_token_loss_weight_step"])
    if false_rejects > false_accepts:
        positive += p_step
        ordered -= 0.5 * o_step
    elif false_accepts > false_rejects:
        positive -= p_step
        ordered += o_step
    elif false_accepts > 0:
        ordered += 0.5 * o_step
    positive = clamp(
        positive,
        float(raw["positive_example_weight_min"]),
        float(raw["positive_example_weight_max"]),
    )
    ordered = clamp(
        ordered,
        float(raw["ordered_token_loss_weight_min"]),
        float(raw["ordered_token_loss_weight_max"]),
    )
    failures = false_rejects + false_accepts
    if failures <= 0:
        repeat = 0
    elif failures <= 4:
        repeat = 1
    elif failures <= 16:
        repeat = 2
    else:
        repeat = int(policy["failure_replay_repeat_max"])
    repeat = min(repeat, int(policy["failure_replay_repeat_max"]))
    return {
        "positive_example_weight": positive,
        "ordered_token_loss_weight": ordered,
        "failure_replay_repeat": repeat,
    }


def repeated(path: pathlib.Path | None, count: int) -> list[pathlib.Path]:
    if path is None or count <= 0:
        return []
    return [path] * count


def strict(record: dict) -> bool:
    return bool(record.get("calibration_gate")) and bool(record.get("test_gate"))


def select_best_strict_candidate(records: list[dict]) -> dict | None:
    eligible = [record for record in records if strict(record)]
    if not eligible:
        return None
    for record in eligible:
        finite(record.get("score"), f"round-{record.get('round')}.score")
    return min(
        eligible,
        key=lambda record: (
            float(record["score"]),
            -int(record["round"]),
            str(record.get("frontend", "")),
        ),
    )


def copy_frozen_candidate(selected: dict, output: pathlib.Path, config: pathlib.Path, policy: pathlib.Path) -> dict:
    frozen = output / "frozen-candidate"
    frozen.mkdir(parents=True, exist_ok=True)
    mapping = (
        (pathlib.Path(selected["model"]), "model.kwm"),
        (pathlib.Path(selected["checkpoint"]), "model.pt"),
        (pathlib.Path(selected["provenance"]), "model-provenance.json"),
        (pathlib.Path(selected["pack"]), "keywords.kwk"),
        (pathlib.Path(selected["keywords"]), "keywords.tsv"),
    )
    for source, name in mapping:
        if not source.is_file() or source.stat().st_size <= 0:
            raise ValueError(f"selected candidate member missing: {source}")
        shutil.copy2(source, frozen / name)
    code_paths = [
        pathlib.Path(__file__).resolve(),
        TRAINING / "train_gru_ctc.py",
        TRAINING / "feature_cached_trainer.py",
        TRAINING / "gru_model.py",
        TRAINING / "domain_curriculum.py",
        TRAINING / "hard_negative_replay.py",
        TRAINING / "development_failure_replay.py",
    ]
    freeze = {
        "schema_version": 1,
        "policy": FREEZE_POLICY,
        "source_policy": POLICY,
        "evidence_scope": EVIDENCE_SCOPE,
        "selection_policy": SELECTION_POLICY,
        "selected_round": int(selected["round"]),
        "selected_score": float(selected["score"]),
        "model_sha256": sha256_file(frozen / "model.kwm"),
        "checkpoint_sha256": sha256_file(frozen / "model.pt"),
        "pack_sha256": sha256_file(frozen / "keywords.kwk"),
        "keywords_sha256": sha256_file(frozen / "keywords.tsv"),
        "provenance_sha256": sha256_file(frozen / "model-provenance.json"),
        "config_sha256": sha256_file(config),
        "development_policy_sha256": sha256_file(policy),
        "training_code_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path) for path in code_paths
        },
        "selection_evidence": ["development-calibration", "development-test"],
        "qualification_used_for_selection": False,
        "shadow_used_for_selection": False,
        "formal_qualification_used_for_selection": False,
        "candidate_stage": {
            "fresh_validation_required": True,
            "shadow_required": True,
            "formal_qualification_required": True,
            "bounded_repair_only": True,
            "validation_feedback_allowed": False,
            "threshold_feedback_allowed": False,
            "training_rule_feedback_allowed": False,
        },
    }
    path = frozen / "freeze-manifest.json"
    path.write_text(
        json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return freeze


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Development-only multi-round Tiny-GRU curriculum loop."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    policy_path = args.policy.resolve()
    cfg = load_config(config_path)
    policy = validate_policy(policy_path)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("GRU runtime runner does not exist")
    work = safe_reset(args.work_dir)
    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    gates = gate_values(cfg.get("domain_gates", {}))
    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    if not thresholds:
        raise ValueError("calibration threshold grid is empty")

    records: list[dict] = []
    curriculum: dict | None = None
    controller = controller_initial(policy)
    previous_checkpoint: pathlib.Path | None = None
    best_objective: float | None = None
    stale_rounds = 0
    strict_streak = 0

    for round_index in range(int(policy["max_rounds"])):
        dataset = work / "datasets" / f"round-{round_index:02d}"
        render_domain_dataset(config_path, dataset, curriculum_weights=curriculum)
        run(
            [
                sys.executable,
                str(TRAINING / "audit_dataset.py"),
                "--split",
                f"train={dataset / 'train.tsv'}",
                "--split",
                f"calibration={dataset / 'calibration.tsv'}",
                "--split",
                f"test={dataset / 'test.tsv'}",
                "--report",
                str(dataset / "development-audit.json"),
                "--fail-within-split",
            ]
        )
        fixed_replay = render_hard_negative_replay(
            config_path,
            work / "fixed-replay" / f"round-{round_index:02d}",
            round_index=round_index,
            curriculum_weights=curriculum,
        )
        failure_replay = render_development_failure_replay(
            config_path,
            records,
            work,
            work / "failure-replay" / f"round-{round_index:02d}",
        )
        fixed_manifest = pathlib.Path(str(fixed_replay["manifest"])) if int(fixed_replay.get("examples", 0)) > 0 else None
        failure_manifest = pathlib.Path(str(failure_replay["manifest"])) if int(failure_replay.get("examples", 0)) > 0 else None
        manifests = [dataset / "train.tsv"]
        manifests.extend(repeated(fixed_manifest, int(policy["fixed_replay_repeat"])))
        manifests.extend(repeated(failure_manifest, int(controller["failure_replay_repeat"])))
        for manifest in manifests:
            if not manifest.is_file() or manifest.stat().st_size <= 0:
                raise ValueError(f"GRU development manifest missing: {manifest}")

        candidate = work / "candidates" / f"round-{round_index:02d}"
        candidate.mkdir(parents=True, exist_ok=True)
        checkpoint = candidate / "model.pt"
        learning_rate = float(train_cfg.get("lr", 0.001)) * float(policy["lr_decay_per_round"]) ** round_index
        command = [sys.executable, str(TRAINING / "train_gru_ctc.py")]
        for manifest in manifests:
            command.extend(["--manifest", str(manifest)])
        command.extend(
            [
                "--tokens",
                str(tokens),
                "--keywords",
                str(keywords),
                "--frontend",
                "logmel",
                "--feature-dim",
                str(int(model_cfg.get("feature_dim", 32))),
                "--hidden-dim",
                str(int(model_cfg.get("hidden_dim", 64))),
                "--epochs",
                str(int(policy["epochs_per_round"])),
                "--batch-size",
                str(int(train_cfg.get("batch_size", 16))),
                "--lr",
                str(learning_rate),
                "--seed",
                str(int(cfg.get("seed", 1337)) + int(policy["training_seed_namespace"]) + round_index * 1009),
                "--positive-example-weight",
                str(float(controller["positive_example_weight"])),
                "--ordered-token-loss-weight",
                str(float(controller["ordered_token_loss_weight"])),
                "--output",
                str(checkpoint),
            ]
        )
        if previous_checkpoint is not None:
            command.extend(["--warm-start", str(previous_checkpoint)])
        run(command)
        model = candidate / "model.kwg"
        run(
            [
                sys.executable,
                str(TRAINING / "export_gru_model.py"),
                "--checkpoint",
                str(checkpoint),
                "--tokens",
                str(tokens),
                "--output",
                str(model),
            ]
        )
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
        )
        test_base, test_domains = evaluate(
            runner=runner,
            model=model,
            pack=pack,
            references=dataset / "test.references.jsonl",
            output=candidate / "test",
        )
        cal_gate = (
            float(cal_base["frr"]) <= gates["max_frr"]
            and float(cal_base["far_per_hour"]) <= gates["max_far_per_hour"]
            and float(cal_base["p95_post_end_latency_ms"]) <= gates["max_p95_latency_ms"]
            and float(cal_domains.get("domains", {}).get("distance:far", {}).get("frr", 1.0)) <= gates["max_far_frr"]
        )
        test_gate = (
            float(test_base["frr"]) <= gates["max_frr"]
            and float(test_base["far_per_hour"]) <= gates["max_far_per_hour"]
            and float(test_base["p95_post_end_latency_ms"]) <= gates["max_p95_latency_ms"]
            and float(test_domains.get("domains", {}).get("distance:far", {}).get("frr", 1.0)) <= gates["max_far_frr"]
        )
        score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)
        false_rejects, false_accepts = failure_counts(cal_base, test_base)
        record = {
            "round": round_index,
            "frontend": "logmel",
            "candidate": 0,
            "score": score,
            "model": str(model),
            "model_sha256": sha256_file(model),
            "checkpoint": str(checkpoint),
            "provenance": str(provenance),
            "provenance_sha256": sha256_file(provenance),
            "keywords": str(calibrated),
            "pack": str(pack),
            "calibration": cal_base,
            "calibration_domains": cal_domains,
            "test": test_base,
            "test_domains": test_domains,
            "calibration_gate": cal_gate,
            "test_gate": test_gate,
            "false_rejects": false_rejects,
            "false_accepts": false_accepts,
            "training": {
                "warm_started": previous_checkpoint is not None,
                "epochs": int(policy["epochs_per_round"]),
                "learning_rate": learning_rate,
                "positive_example_weight": float(controller["positive_example_weight"]),
                "ordered_token_loss_weight": float(controller["ordered_token_loss_weight"]),
                "fixed_replay_repeat": int(policy["fixed_replay_repeat"]),
                "fixed_replay_examples": int(fixed_replay.get("examples", 0)),
                "failure_replay_repeat": int(controller["failure_replay_repeat"]),
                "failure_replay_examples": int(failure_replay.get("examples", 0)),
                "manifest_count": len(manifests),
            },
        }
        records.append(record)
        previous_checkpoint = checkpoint

        merged = merge_domain_metrics(cal_domains, test_domains)
        curriculum = update_curriculum(
            merged,
            previous=curriculum,
            strength=float(cfg.get("domain_iteration", {}).get("curriculum_strength", 2.5)),
            max_weight=float(cfg.get("domain_iteration", {}).get("max_domain_weight", 6.0)),
        )
        curriculum_path = work / "curriculum" / f"round-{round_index:02d}.json"
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        curriculum_path.write_text(
            json.dumps(curriculum, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        controller = controller_next(policy, controller, false_rejects, false_accepts)
        if strict(record):
            strict_streak += 1
        else:
            strict_streak = 0
        if best_objective is None or score < best_objective - 1.0e-12:
            best_objective = score
            stale_rounds = 0
        else:
            stale_rounds += 1
        completed = round_index + 1
        if completed >= int(policy["min_rounds"]) and strict_streak >= int(policy["stable_strict_pass_rounds"]):
            break
        if completed >= int(policy["min_rounds"]) and stale_rounds >= int(policy["patience"]):
            break

    selected = select_best_strict_candidate(records)
    manifest = {
        "schema_version": 1,
        "policy": POLICY,
        "evidence_scope": EVIDENCE_SCOPE,
        "development_qualified": selected is not None,
        "qualification_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "selection_policy": SELECTION_POLICY,
        "selected_round": int(selected["round"]) if selected is not None else None,
        "selected_score": float(selected["score"]) if selected is not None else None,
        "candidate_stage_feedback_allowed": False,
        "records": records,
        "final_curriculum": curriculum or {},
        "next_controller": controller,
        "config_sha256": sha256_file(config_path),
        "development_policy_sha256": sha256_file(policy_path),
    }
    if selected is not None:
        manifest["frozen_candidate"] = copy_frozen_candidate(
            selected, work, config_path, policy_path
        )
    manifest_path = work / "development-loop-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if selected is not None else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
