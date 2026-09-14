from __future__ import annotations

import argparse
import copy
import json
import math
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"

from adversarial_lexicon import mine_adversarial_lexicon
from development_failure_replay import render_development_failure_replay
from hard_negative_replay import render_hard_negative_replay
from iterate_domain import (
    base_gate,
    calibrate,
    domain_gate,
    evaluate,
    gate_values,
    objective,
    repo_path,
    run,
    sha256_file,
)
from qualification_failure_replay import (
    POLICY as QUALIFICATION_REPAIR_POLICY,
    REPAIR_EPOCHS,
    REPAIR_LR_SCALE,
    render_qualification_failure_replay,
)
from render_domains import render_domain_dataset
from render_qualification_holdout import require_strict_development_candidate
from synthetic_audio import load_config

POLICY = "post-domain-adversarial-refinement-v1"
REPAIR_VALIDATION_SEED_NAMESPACE = 171_000_003


def _selected_record(manifest: dict) -> dict:
    selection = manifest.get("candidate_selection", {})
    selected_round = int(selection.get("selected_round", -1))
    selected_frontend = str(selection.get("selected_frontend") or "")
    rows = [
        row
        for row in manifest.get("records", [])
        if isinstance(row, dict)
        and int(row.get("round", -1)) == selected_round
        and str(row.get("frontend") or "") == selected_frontend
        and bool(row.get("calibration_gate"))
        and bool(row.get("test_gate"))
        and "checkpoint" in row
    ]
    if not rows:
        raise ValueError("adversarial refinement cannot resolve selected strict checkpoint")
    return min(rows, key=lambda row: (float(row["score"]), str(row["checkpoint"])))


def _refinement_policy(cfg: dict) -> dict:
    iteration = cfg.get("domain_iteration", {})
    if not isinstance(iteration, dict):
        raise ValueError("domain_iteration must be an object")
    raw = iteration.get("adversarial_lexicon")
    if not isinstance(raw, dict) or not bool(raw.get("enabled", False)):
        raise ValueError("adversarial lexicon must be enabled for refinement")
    epochs = int(raw.get("refinement_epochs", 12))
    lr_scale = float(raw.get("refinement_lr_scale", 0.5))
    if epochs <= 0 or not math.isfinite(lr_scale) or not 0.0 < lr_scale <= 1.0:
        raise ValueError("adversarial refinement epochs/lr scale are invalid")
    return {"epochs": epochs, "lr_scale": lr_scale}


def _audit(dataset: pathlib.Path) -> None:
    argv = [sys.executable, str(TRAINING / "audit_dataset.py")]
    for split in ("train", "calibration", "test", "qualification"):
        argv.extend(["--split", f"{split}={dataset / (split + '.tsv')}"])
    argv.extend(["--report", str(dataset / "audit.json"), "--fail-within-split"])
    run(argv)


def _write_effective_seed_config(cfg: dict, seed: int, path: pathlib.Path) -> pathlib.Path:
    rendered = copy.deepcopy(cfg)
    rendered["seed"] = seed
    rendered.pop("qualification_holdout_seed", None)
    rendered.pop("retired_qualification_holdout_seeds", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _train_refinement(
    *,
    cfg: dict,
    frontend: str,
    tokens: pathlib.Path,
    keywords: pathlib.Path,
    dataset_manifest: pathlib.Path,
    static_manifest: pathlib.Path,
    adversarial_manifest: pathlib.Path,
    failure_manifest: pathlib.Path | None,
    warm_start: pathlib.Path,
    output: pathlib.Path,
    epochs: int,
    lr_scale: float,
    refinement_round: int,
    seed_offset: int = 0,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    train = cfg.get("train", {})
    if not isinstance(train, dict):
        raise ValueError("train config must be an object")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    model = output / "model.kwm"
    provenance = pathlib.Path(str(model) + ".provenance.json")
    learning_rate = float(train.get("lr", 0.001)) * lr_scale
    seed = int(cfg.get("seed", 1337)) + 4_000_003 + refinement_round * 1009 + seed_offset
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(dataset_manifest),
        "--manifest",
        str(static_manifest),
        "--manifest",
        str(adversarial_manifest),
    ]
    if failure_manifest is not None:
        if not failure_manifest.is_file() or failure_manifest.stat().st_size == 0:
            raise ValueError("non-empty failure replay manifest was requested but is missing")
        command.extend(["--manifest", str(failure_manifest)])
    command.extend(
        [
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--frontend",
            frontend,
            "--feature-dim",
            str(int(cfg.get("model", {}).get("feature_dim", 32))),
            "--hidden-dim",
            str(int(cfg.get("model", {}).get("hidden_dim", 48))),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(int(train.get("batch_size", 16))),
            "--lr",
            str(learning_rate),
            "--seed",
            str(seed),
            "--warm-start",
            str(warm_start),
            "--output",
            str(checkpoint),
        ]
    )
    run(command)
    run(
        [
            sys.executable,
            str(TRAINING / "export_model.py"),
            "--checkpoint",
            str(checkpoint),
            "--tokens",
            str(tokens),
            "--output",
            str(model),
        ]
    )
    return model, checkpoint, provenance


def _strict(base: dict, domains: dict, gates: dict) -> bool:
    return base_gate(base, gates) and domain_gate(domains, gates)


def _update_record_candidate(
    record: dict,
    *,
    model: pathlib.Path,
    checkpoint: pathlib.Path,
    provenance: pathlib.Path,
    calibrated: pathlib.Path,
    pack: pathlib.Path,
    cal_base: dict,
    cal_domains: dict,
    test_base: dict,
    test_domains: dict,
    failure: dict,
    failure_evidence: pathlib.Path,
    score: float,
    qualification_repair_used: bool,
) -> None:
    gates = record["_gates"]
    record.update(
        {
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
            "calibration_gate": _strict(cal_base, cal_domains, gates),
            "test_gate": _strict(test_base, test_domains, gates),
            "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
            "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
            "failure_replay_examples": int(failure["examples"]),
            "failure_replay_manifest_sha256": str(failure["manifest_sha256"]),
            "failure_replay_evidence_sha256": sha256_file(failure_evidence),
            "qualification_repair_used": qualification_repair_used,
        }
    )


def _write_failed_summary(work: pathlib.Path, value: dict) -> None:
    out = work / "adversarial-refinement" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mine development-only lexical adversaries and refine the latest strict candidate."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    runner = args.runner.resolve()
    work = args.work_dir.resolve()
    cfg = load_config(config_path)
    if str(cfg.get("domain_iteration", {}).get("backend")) != "torch_ctc":
        raise ValueError("adversarial refinement requires torch_ctc backend")
    policy = _refinement_policy(cfg)
    manifest_path = work / "domain-loop-manifest.json"
    input_manifest_sha = sha256_file(manifest_path)
    manifest = require_strict_development_candidate(work)
    source = _selected_record(manifest)
    source_checkpoint = repo_path(str(source["checkpoint"]))
    source_round = int(source["round"])
    frontend = str(source["frontend"])
    refinement_round = max(int(row["round"]) for row in manifest["records"]) + 1
    if refinement_round <= source_round:
        raise ValueError("adversarial refinement round must follow development rounds")

    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    final_curriculum = manifest.get("final_curriculum")
    curriculum = final_curriculum if isinstance(final_curriculum, dict) else None

    dataset = work / "datasets" / f"round-{refinement_round:02d}"
    render_domain_dataset(config_path, dataset, curriculum_weights=curriculum)
    _audit(dataset)

    static = render_hard_negative_replay(
        config_path,
        work / "hard-negative-replay" / f"round-{refinement_round:02d}",
        round_index=refinement_round,
        curriculum_weights=curriculum,
    )
    static_manifest = pathlib.Path(str(static["manifest"]))

    adversarial = mine_adversarial_lexicon(
        config_path,
        source_checkpoint,
        work / "adversarial-lexicon" / f"round-{refinement_round:02d}",
        round_index=refinement_round,
        frontend=frontend,
    )
    adversarial_manifest = pathlib.Path(str(adversarial["manifest"]))
    adversarial_evidence = pathlib.Path(str(adversarial["evidence"]))
    if bool(adversarial.get("formal_qualification_used", True)):
        raise ValueError("adversarial mining must not use formal qualification")
    adversarial_selection_policy = str(adversarial.get("selection_policy") or "")
    adversarial_data_policy = str(adversarial.get("data_augmentation_policy") or "")
    if not adversarial_selection_policy or not adversarial_data_policy:
        raise ValueError("adversarial evidence is missing data/selection policy provenance")

    failure = render_development_failure_replay(
        config_path,
        list(manifest.get("records", [])),
        work,
        work / "development-failure-replay" / f"round-{refinement_round:02d}",
    )
    failure_manifest_path = pathlib.Path(str(failure["manifest"]))
    failure_evidence = pathlib.Path(str(failure["evidence"]))
    if bool(failure.get("formal_qualification_used", True)):
        raise ValueError("development failure replay must not use formal qualification")
    if bool(failure.get("development_source_wav_bytes_copied", True)):
        raise ValueError("development failure replay copied evaluation WAV bytes")
    failure_manifest = failure_manifest_path if int(failure.get("examples", 0)) > 0 else None

    candidate_dir = work / "candidates" / f"r{refinement_round:02d}-{frontend}-adversarial"
    model, checkpoint, provenance = _train_refinement(
        cfg=cfg,
        frontend=frontend,
        tokens=tokens,
        keywords=keywords,
        dataset_manifest=dataset / "train.tsv",
        static_manifest=static_manifest,
        adversarial_manifest=adversarial_manifest,
        failure_manifest=failure_manifest,
        warm_start=source_checkpoint,
        output=candidate_dir,
        epochs=int(policy["epochs"]),
        lr_scale=float(policy["lr_scale"]),
        refinement_round=refinement_round,
    )

    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    gates = gate_values(cfg.get("domain_gates", {}))
    calibrated, pack, cal_base, cal_domains = calibrate(
        runner=runner,
        model=model,
        tokens=tokens,
        source_keywords=keywords,
        references=dataset / "calibration.references.jsonl",
        output=candidate_dir / "calibration",
        thresholds=thresholds,
        rounds=coordinate_rounds,
        gates=gates,
    )
    test_base, test_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=dataset / "test.references.jsonl",
        output=candidate_dir / "test",
    )
    cal_gate = _strict(cal_base, cal_domains, gates)
    test_gate = _strict(test_base, test_domains, gates)
    score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)

    record = {
        "round": refinement_round,
        "stage": POLICY,
        "frontend": frontend,
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
        "warm_started": True,
        "warm_start_strategy": "full",
        "source_round": source_round,
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "hard_negative_replay_examples": int(static.get("examples", 0)),
        "hard_negative_replay_manifest_sha256": str(static["manifest_sha256"]),
        "adversarial_policy": adversarial_selection_policy,
        "adversarial_data_augmentation_policy": adversarial_data_policy,
        "adversarial_enumerated_sequences": int(adversarial["enumerated_sequences"]),
        "adversarial_top_k": int(adversarial["top_k"]),
        "adversarial_probes_per_sequence": int(adversarial["probes_per_sequence"]),
        "adversarial_replay_examples_per_sequence": int(adversarial["replay_examples_per_sequence"]),
        "adversarial_replay_examples": int(adversarial["replay_examples"]),
        "adversarial_min_per_keyword": int(adversarial["min_per_keyword"]),
        "adversarial_per_keyword_selected": dict(adversarial["per_keyword_selected"]),
        "adversarial_strict_prefix_anchors": list(adversarial["strict_prefix_anchors"]),
        "adversarial_manifest_sha256": str(adversarial["manifest_sha256"]),
        "adversarial_evidence_sha256": sha256_file(adversarial_evidence),
        "adversarial_formal_qualification_used": False,
        "failure_replay_policy": str(failure["policy"]),
        "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
        "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
        "failure_replay_examples": int(failure["examples"]),
        "failure_replay_manifest_sha256": str(failure["manifest_sha256"]),
        "failure_replay_evidence_sha256": sha256_file(failure_evidence),
        "failure_replay_formal_qualification_used": False,
        "failure_replay_development_source_wav_bytes_copied": False,
        "qualification_repair_used": False,
        "_gates": gates,
    }
    if not cal_gate or not test_gate:
        _write_failed_summary(
            work,
            {
                "schema_version": 3,
                "policy": POLICY,
                "qualified": False,
                "input_development_manifest_sha256": input_manifest_sha,
                "source_round": source_round,
                "refinement_round": refinement_round,
                "adversarial_data_augmentation_policy": adversarial_data_policy,
                "adversarial_selection_policy": adversarial_selection_policy,
                "failure_replay_policy": str(failure["policy"]),
                "failure_replay_examples": int(failure["examples"]),
                "qualification_repair_used": False,
                "record": {key: value for key, value in record.items() if key != "_gates"},
            },
        )
        raise ValueError("adversarial refinement did not retain calibration/test strict dual-pass")

    mining_qualification = work / "development-qualification-mining"
    render_domain_dataset(config_path, mining_qualification, curriculum_weights=None)
    mining_eval = candidate_dir / "development-qualification-mining"
    mining_qual_base, mining_qual_domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=mining_qualification / "qualification.references.jsonl",
        output=mining_eval,
    )
    mining_qualification_qualified = _strict(mining_qual_base, mining_qual_domains, gates)
    qualification_repair = None
    validation_seed = int(cfg.get("seed", 1337))
    validation_config = config_path

    if not mining_qualification_qualified:
        failure = render_qualification_failure_replay(
            config_path,
            mining_qualification,
            mining_eval,
            failure_evidence,
            work / "development-failure-replay" / f"round-{refinement_round:02d}-qualification-repair",
        )
        failure_manifest_path = pathlib.Path(str(failure["manifest"]))
        failure_evidence = pathlib.Path(str(failure["evidence"]))
        repair_dir = candidate_dir / "qualification-repair"
        repaired_model, repaired_checkpoint, repaired_provenance = _train_refinement(
            cfg=cfg,
            frontend=frontend,
            tokens=tokens,
            keywords=keywords,
            dataset_manifest=dataset / "train.tsv",
            static_manifest=static_manifest,
            adversarial_manifest=adversarial_manifest,
            failure_manifest=failure_manifest_path,
            warm_start=checkpoint,
            output=repair_dir,
            epochs=REPAIR_EPOCHS,
            lr_scale=REPAIR_LR_SCALE,
            refinement_round=refinement_round,
            seed_offset=700_001,
        )
        repaired_keywords, repaired_pack, repaired_cal_base, repaired_cal_domains = calibrate(
            runner=runner,
            model=repaired_model,
            tokens=tokens,
            source_keywords=keywords,
            references=dataset / "calibration.references.jsonl",
            output=repair_dir / "calibration",
            thresholds=thresholds,
            rounds=coordinate_rounds,
            gates=gates,
        )
        repaired_test_base, repaired_test_domains = evaluate(
            runner=runner,
            model=repaired_model,
            pack=repaired_pack,
            references=dataset / "test.references.jsonl",
            output=repair_dir / "test",
        )
        repaired_cal_gate = _strict(repaired_cal_base, repaired_cal_domains, gates)
        repaired_test_gate = _strict(repaired_test_base, repaired_test_domains, gates)
        repaired_score = objective(repaired_cal_base, repaired_cal_domains, gates) + objective(
            repaired_test_base, repaired_test_domains, gates
        )

        validation_seed = int(cfg.get("seed", 1337)) + REPAIR_VALIDATION_SEED_NAMESPACE
        validation_config = _write_effective_seed_config(
            cfg,
            validation_seed,
            work / "development-qualification-repair-validation-config.json",
        )
        development_qualification = work / "qualification-dataset"
        render_domain_dataset(validation_config, development_qualification, curriculum_weights=None)
        repaired_qual_base, repaired_qual_domains = evaluate(
            runner=runner,
            model=repaired_model,
            pack=repaired_pack,
            references=development_qualification / "qualification.references.jsonl",
            output=repair_dir / "development-qualification-validation",
        )
        repaired_qual_gate = _strict(repaired_qual_base, repaired_qual_domains, gates)
        qualification_repair = {
            "policy": QUALIFICATION_REPAIR_POLICY,
            "epochs": REPAIR_EPOCHS,
            "lr_scale": REPAIR_LR_SCALE,
            "mining_qualification": mining_qual_base,
            "mining_seed": int(cfg.get("seed", 1337)),
            "validation_seed": validation_seed,
            "validation_config_sha256": sha256_file(validation_config),
            "calibration_gate": repaired_cal_gate,
            "test_gate": repaired_test_gate,
            "qualification_gate": repaired_qual_gate,
            "failure_replay_examples": int(failure["examples"]),
            "qualification_repair_examples": int(failure.get("qualification_repair_examples", 0)),
            "qualification_repair_selected_unique_failures": int(
                failure.get("qualification_repair_selected_unique_failures", 0)
            ),
            "mining_cohort_used_for_training": True,
            "validation_cohort_used_for_training": False,
            "formal_qualification_used": False,
        }
        record["qualification_repair_used"] = True
        record["qualification_repair"] = qualification_repair
        _update_record_candidate(
            record,
            model=repaired_model,
            checkpoint=repaired_checkpoint,
            provenance=repaired_provenance,
            calibrated=repaired_keywords,
            pack=repaired_pack,
            cal_base=repaired_cal_base,
            cal_domains=repaired_cal_domains,
            test_base=repaired_test_base,
            test_domains=repaired_test_domains,
            failure=failure,
            failure_evidence=failure_evidence,
            score=repaired_score,
            qualification_repair_used=True,
        )
        if not repaired_cal_gate or not repaired_test_gate or not repaired_qual_gate:
            _write_failed_summary(
                work,
                {
                    "schema_version": 3,
                    "policy": POLICY,
                    "qualified": False,
                    "input_development_manifest_sha256": input_manifest_sha,
                    "source_round": source_round,
                    "refinement_round": refinement_round,
                    "adversarial_data_augmentation_policy": adversarial_data_policy,
                    "adversarial_selection_policy": adversarial_selection_policy,
                    "failure_replay_policy": str(failure["policy"]),
                    "failure_replay_examples": int(failure["examples"]),
                    "qualification_repair_used": True,
                    "qualification_repair": qualification_repair,
                    "record": {key: value for key, value in record.items() if key != "_gates"},
                    "development_qualification": repaired_qual_base,
                },
            )
            raise ValueError("development qualification repair did not reach strict triple-pass")

        model = repaired_model
        checkpoint = repaired_checkpoint
        provenance = repaired_provenance
        calibrated = repaired_keywords
        pack = repaired_pack
        cal_base = repaired_cal_base
        cal_domains = repaired_cal_domains
        test_base = repaired_test_base
        test_domains = repaired_test_domains
        cal_gate = repaired_cal_gate
        test_gate = repaired_test_gate
        score = repaired_score
    else:
        development_qualification = work / "qualification-dataset"
        render_domain_dataset(config_path, development_qualification, curriculum_weights=None)

    record.pop("_gates", None)
    manifest["records"].append(record)
    eligible_rounds = sorted(
        {
            int(row["round"])
            for row in manifest["records"]
            if bool(row.get("calibration_gate")) and bool(row.get("test_gate"))
        }
    )
    selection = manifest["candidate_selection"]
    selection.update(
        {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": eligible_rounds,
            "selected_round": refinement_round,
            "selected_frontend": frontend,
            "selected_score": score,
            "objective_fallback_used": False,
            "adversarial_refinement_used": True,
            "adversarial_refinement_policy": POLICY,
            "adversarial_data_augmentation_policy": adversarial_data_policy,
            "adversarial_selection_policy": adversarial_selection_policy,
            "development_failure_replay_used": int(failure["examples"]) > 0,
            "development_failure_replay_policy": str(failure["policy"]),
            "development_qualification_repair_used": qualification_repair is not None,
            "development_qualification_repair_policy": (
                QUALIFICATION_REPAIR_POLICY if qualification_repair is not None else None
            ),
            "development_qualification_validation_seed": validation_seed,
            "development_qualification_validation_config_sha256": sha256_file(validation_config),
            "development_qualification_validation_used_for_training": False,
        }
    )
    manifest["best_round"] = refinement_round
    manifest["best_frontend"] = frontend
    manifest["best_score"] = score
    manifest["best_model_sha256"] = sha256_file(model)
    manifest["best_pack_sha256"] = sha256_file(pack)
    manifest["development_qualified"] = True

    best = work / "best"
    best.mkdir(parents=True, exist_ok=True)
    for source_path, name in (
        (model, "model.kwm"),
        (checkpoint, "model.pt"),
        (pack, "keywords.kwk"),
        (calibrated, "keywords.tsv"),
        (provenance, "model-provenance.json"),
        (adversarial_evidence, "adversarial-lexicon.json"),
        (failure_evidence, "development-failure-replay.json"),
    ):
        shutil.copy2(source_path, best / name)

    canonical_qual_base, canonical_qual_domains = evaluate(
        runner=runner,
        model=best / "model.kwm",
        pack=best / "keywords.kwk",
        references=development_qualification / "qualification.references.jsonl",
        output=best / "qualification",
    )
    qualification_qualified = _strict(canonical_qual_base, canonical_qual_domains, gates)
    manifest["qualification"] = canonical_qual_base
    manifest["qualification_domains"] = canonical_qual_domains
    manifest["qualification_qualified"] = qualification_qualified
    manifest["qualified"] = bool(qualification_qualified)
    manifest["evidence_class"] = (
        "synthetic-domain-qualified" if qualification_qualified else "synthetic-domain-unqualified"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 3,
        "policy": POLICY,
        "qualified": bool(qualification_qualified),
        "input_development_manifest_sha256": input_manifest_sha,
        "output_development_manifest_sha256": sha256_file(manifest_path),
        "source_round": source_round,
        "refinement_round": refinement_round,
        "frontend": frontend,
        "adversarial_data_augmentation_policy": adversarial_data_policy,
        "adversarial_selection_policy": adversarial_selection_policy,
        "adversarial_evidence_sha256": sha256_file(adversarial_evidence),
        "adversarial_manifest_sha256": str(adversarial["manifest_sha256"]),
        "adversarial_enumerated_sequences": int(adversarial["enumerated_sequences"]),
        "adversarial_top_k": int(adversarial["top_k"]),
        "adversarial_probes_per_sequence": int(adversarial["probes_per_sequence"]),
        "adversarial_replay_examples_per_sequence": int(adversarial["replay_examples_per_sequence"]),
        "adversarial_replay_examples": int(adversarial["replay_examples"]),
        "adversarial_min_per_keyword": int(adversarial["min_per_keyword"]),
        "adversarial_per_keyword_selected": dict(adversarial["per_keyword_selected"]),
        "strict_prefix_anchors": list(adversarial["strict_prefix_anchors"]),
        "failure_replay_policy": str(failure["policy"]),
        "failure_replay_evidence_sha256": sha256_file(failure_evidence),
        "failure_replay_manifest_sha256": str(failure["manifest_sha256"]),
        "failure_replay_observed_unique_failures": int(failure["observed_unique_failures"]),
        "failure_replay_selected_unique_failures": int(failure["selected_unique_failures"]),
        "failure_replay_examples": int(failure["examples"]),
        "failure_replay_development_source_wav_bytes_copied": False,
        "qualification_repair_used": qualification_repair is not None,
        "qualification_repair": qualification_repair,
        "development_qualification_validation_seed": validation_seed,
        "development_qualification_validation_config_sha256": sha256_file(validation_config),
        "development_qualification_validation_used_for_training": False,
        "formal_qualification_used": False,
        "record": record,
        "development_qualification": canonical_qual_base,
    }
    out = work / "adversarial-refinement" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualification_qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
