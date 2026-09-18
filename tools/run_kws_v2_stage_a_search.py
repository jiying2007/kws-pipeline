#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
TRAINING = ROOT / "training"

MATRIX_CLASS = "kws-v2-staged-experiment-matrix-v1"
BINDING_CLASS = "kws-v2-runnable-candidate-binding-v1"
STAGE_CLASS = "kws-v2-stage-a-search-run-v1"
SHARD_CLASS = "kws-v2-stage-a-search-shard-v1"
EXPECTED_STAGE_A = {
    ("rnn", "logmel", 64),
    ("rnn", "pcen-lite", 64),
    ("gru", "logmel", 64),
    ("gru", "pcen-lite", 64),
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_checked(command: list[str], *, log: pathlib.Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        suffix = f"; see {log}" if log is not None else ""
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}{suffix}")
    return completed.stdout


def current_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    value = completed.stdout.strip().lower()
    if len(value) != 40:
        raise ValueError("repository HEAD is not a full SHA")
    return value


def require_file(path: pathlib.Path, label: str) -> pathlib.Path:
    result = path.resolve()
    if not result.is_file():
        raise ValueError(f"{label} does not exist: {result}")
    return result


def validate_matrix(path: pathlib.Path) -> tuple[dict, list[dict]]:
    matrix = load_object(path)
    if int(matrix.get("schema_version", 0)) != 1 or matrix.get("evidence_class") != MATRIX_CLASS:
        raise ValueError("Stage A matrix identity mismatch")
    if matrix.get("evidence_scope") != "development-only" or matrix.get("stage") != "A":
        raise ValueError("orchestrator accepts only development-only Stage A matrices")
    if matrix.get("requires_predeclared_generalization_plan") is not True:
        raise ValueError("Stage A matrix must require predeclared generalization plans")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if matrix.get(key) is not False:
            raise ValueError(f"Stage A matrix protected flag {key} must be false")
    cohort = str(matrix.get("generalization_cohort_id", "")).strip()
    if not cohort:
        raise ValueError("Stage A matrix generalization_cohort_id is required")
    rows = matrix.get("candidates")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Stage A matrix must contain exactly four candidates")
    identities = {
        (str(row.get("model_family", "")), str(row.get("frontend", "")), int(row.get("hidden_dim", 0)))
        for row in rows
        if isinstance(row, dict)
    }
    if identities != EXPECTED_STAGE_A:
        raise ValueError(f"Stage A candidate identity drifted: {sorted(identities)}")
    candidate_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Stage A candidate row must be an object")
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in candidate_ids:
            raise ValueError("Stage A candidate ids must be non-empty and unique")
        candidate_ids.add(candidate_id)
        if row.get("generalization_cohort_id") != cohort:
            raise ValueError(f"{candidate_id}: candidate cohort does not match matrix")
        if row.get("generalization_tier") != "search":
            raise ValueError(f"{candidate_id}: Stage A generalization tier must be search")
        if row.get("protected_evidence_used") is not False:
            raise ValueError(f"{candidate_id}: protected evidence must be false")
        if row.get("development_policy_path_contract") != "repository-relative-v1":
            raise ValueError(f"{candidate_id}: development policy path contract mismatch")
        raw_policy = str(row.get("development_policy", ""))
        if not raw_policy or pathlib.Path(raw_policy).is_absolute():
            raise ValueError(f"{candidate_id}: development policy must be repository-relative")
        policy = require_file(ROOT / raw_policy, f"{candidate_id} development policy")
        if sha256_file(policy) != str(row.get("development_policy_sha256", "")):
            raise ValueError(f"{candidate_id}: development policy sha256 mismatch")
        policy_value = load_object(policy)
        if int(policy_value.get("min_rounds", 0)) != 8 or int(policy_value.get("max_rounds", 0)) != 8:
            raise ValueError(f"{candidate_id}: Stage A development policy must run exactly eight rounds")
        if int(policy_value.get("epochs_per_round", 0)) != 8:
            raise ValueError(f"{candidate_id}: Stage A development policy must use eight epochs per round")
        if int(policy_value.get("patience", 0)) < 8:
            raise ValueError(f"{candidate_id}: Stage A patience must not terminate before round eight")
        for key in ("qualification_used", "shadow_used", "formal_qualification_used"):
            if policy_value.get(key) is not False:
                raise ValueError(f"{candidate_id}: development policy protected flag {key} must be false")
    return matrix, rows


def bundle_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for split in ("train", "calibration", "test", "qualification"):
        index = require_file(getattr(args, f"{split}_index"), f"{split} index")
        summary = require_file(getattr(args, f"{split}_summary"), f"{split} summary")
        result.extend([f"--{split}-index", str(index), f"--{split}-summary", str(summary)])
    return result


def materialize_stage_a(args: argparse.Namespace, root: pathlib.Path) -> pathlib.Path:
    matrix_dir = root / "matrix"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "kws_v2_experiment_plan.py"),
            "--policy",
            str(require_file(args.experiment_policy, "experiment policy")),
            "--stage",
            "A",
            "--output-dir",
            str(matrix_dir),
        ],
        log=root / "logs" / "materialize-matrix.log",
    )
    return matrix_dir / "experiment-matrix.json"


def prepare(args: argparse.Namespace, root: pathlib.Path) -> tuple[pathlib.Path, dict]:
    matrix_path = materialize_stage_a(args, root)
    matrix, rows = validate_matrix(matrix_path)
    shared_bundle_args = bundle_args(args)
    prepared: list[dict] = []
    bundle_sha: str | None = None

    for row in rows:
        candidate_id = str(row["candidate_id"])
        candidate_root = root / "candidates" / candidate_id
        effective = candidate_root / "effective-config.json"
        binding_path = candidate_root / "binding.json"
        run_checked(
            [
                sys.executable,
                str(TOOLS / "bind_kws_v2_candidate.py"),
                "--matrix",
                str(matrix_path),
                "--candidate-id",
                candidate_id,
                *shared_bundle_args,
                "--output-config",
                str(effective),
                "--output-binding",
                str(binding_path),
            ],
            log=candidate_root / "bind.log",
        )
        binding = load_object(binding_path)
        if binding.get("evidence_class") != BINDING_CLASS:
            raise ValueError(f"{candidate_id}: runnable binding identity mismatch")
        if binding.get("candidate_id") != candidate_id:
            raise ValueError(f"{candidate_id}: runnable binding candidate mismatch")
        if binding.get("generalization_cohort_id") != matrix["generalization_cohort_id"]:
            raise ValueError(f"{candidate_id}: runnable binding cohort mismatch")
        if binding.get("generalization_tier") != "search":
            raise ValueError(f"{candidate_id}: runnable binding tier mismatch")
        current_bundle = str(binding.get("external_base_bundle_sha256", ""))
        if not current_bundle:
            raise ValueError(f"{candidate_id}: external base bundle identity is missing")
        if bundle_sha is None:
            bundle_sha = current_bundle
        elif current_bundle != bundle_sha:
            raise ValueError("Stage A candidates do not share one immutable external base bundle")

        family = str(row["model_family"])
        script = TRAINING / ("run_gru_development.py" if family == "gru" else "run_rnn_development.py")
        policy = ROOT / str(row["development_policy"])
        prepared.append(
            {
                "candidate_id": candidate_id,
                "model_family": family,
                "frontend": str(row["frontend"]),
                "hidden_dim": int(row["hidden_dim"]),
                "resource_contract_candidate": str(row["resource_contract_candidate"]),
                "generalization_tier": "search",
                "generalization_cohort_id": str(binding["generalization_cohort_id"]),
                "effective_config": str(effective),
                "effective_config_sha256": sha256_file(effective),
                "binding": str(binding_path),
                "binding_sha256": sha256_file(binding_path),
                "development_policy": str(policy.resolve()),
                "development_policy_sha256": sha256_file(policy),
                "development_runner_script": str(script.resolve()),
                "development_runner_script_sha256": sha256_file(script),
                "runtime_runner_role": f"{family}-runner",
                "development_work_dir": str(candidate_root / "development"),
                "generalization_plan": str(candidate_root / "generalization-plan.json"),
                "generalization_report": str(candidate_root / "generalization-report.json"),
                "status": "prepared",
            }
        )

    manifest = {
        "schema_version": 1,
        "evidence_class": STAGE_CLASS,
        "evidence_scope": "development-only",
        "stage": "A",
        "status": "prepared",
        "product_head": current_head(),
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "generalization_cohort_id": str(matrix["generalization_cohort_id"]),
        "generalization_tier": "search",
        "external_base_bundle_sha256": bundle_sha,
        "candidate_count": len(prepared),
        "candidates": prepared,
        "full_training_round_budget": True,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
    }
    manifest_path = root / "stage-a-search-manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest


def resolve_runner(family: str, args: argparse.Namespace) -> pathlib.Path:
    raw = args.gru_runner if family == "gru" else args.rnn_runner
    if raw is None:
        raise ValueError(f"{family} runner is required unless --prepare-only is used")
    return require_file(raw, f"{family} runner")


def execute_candidate(
    *,
    row: dict,
    args: argparse.Namespace,
    generalization_policy: pathlib.Path,
    product_head: str,
) -> dict:
    candidate_id = str(row["candidate_id"])
    family = str(row["model_family"])
    candidate_root = pathlib.Path(str(row["binding"])).resolve().parent
    runner = resolve_runner(family, args)
    effective = require_file(pathlib.Path(str(row["effective_config"])), f"{candidate_id} effective config")
    binding = require_file(pathlib.Path(str(row["binding"])), f"{candidate_id} binding")
    development_policy = require_file(
        pathlib.Path(str(row["development_policy"])), f"{candidate_id} development policy"
    )
    training_script = require_file(
        pathlib.Path(str(row["development_runner_script"])), f"{candidate_id} development runner"
    )
    development_work = candidate_root / "development"

    run_checked(
        [
            sys.executable,
            str(training_script),
            "--config",
            str(effective),
            "--policy",
            str(development_policy),
            "--runner",
            str(runner),
            "--work-dir",
            str(development_work),
        ],
        log=candidate_root / "development.log",
    )

    frozen = development_work / "frozen-candidate"
    model = require_file(frozen / "model.kwm", f"{candidate_id} frozen model")
    keywords = require_file(frozen / "keywords.kwk", f"{candidate_id} frozen keyword pack")
    development_manifest = require_file(
        development_work / "development-loop-manifest.json", f"{candidate_id} development manifest"
    )
    development_value = load_object(development_manifest)
    if development_value.get("development_qualified") is not True:
        raise ValueError(f"{candidate_id}: development loop did not produce a qualified frozen candidate")

    plan = candidate_root / "generalization-plan.json"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "build_kws_v2_generalization_plan.py"),
            "--binding",
            str(binding),
            "--policy",
            str(generalization_policy),
            "--model-sha256",
            sha256_file(model),
            "--product-head",
            product_head,
            "--output",
            str(plan),
        ],
        log=candidate_root / "generalization-plan.log",
    )

    report = candidate_root / "generalization-report.json"
    arena_root = candidate_root / "generalization-arena"
    run_checked(
        [
            sys.executable,
            str(TOOLS / "run_development_generalization_arena.py"),
            "--plan",
            str(plan),
            "--policy",
            str(generalization_policy),
            "--config",
            str(effective),
            "--runner",
            str(runner),
            "--model",
            str(model),
            "--keywords",
            str(keywords),
            "--product-root",
            str(ROOT),
            "--split",
            "test",
            "--workers",
            str(args.generalization_workers),
            "--work-dir",
            str(arena_root),
            "--output",
            str(report),
        ],
        log=candidate_root / "generalization-arena.log",
    )

    plan_value = load_object(plan)
    report_value = load_object(report)
    if report_value.get("plan_sha256") != plan_value.get("plan_sha256"):
        raise ValueError(f"{candidate_id}: generalization report/plan binding mismatch")
    if report_value.get("plan_complete") is not True:
        raise ValueError(f"{candidate_id}: generalization cohort is incomplete")
    if report_value.get("candidate_id") != candidate_id:
        raise ValueError(f"{candidate_id}: generalization report candidate mismatch")
    if report_value.get("model_family") != family:
        raise ValueError(f"{candidate_id}: generalization report family mismatch")
    if report_value.get("tier") != "search":
        raise ValueError(f"{candidate_id}: generalization report tier mismatch")
    if report_value.get("protected_evidence_used") is not False:
        raise ValueError(f"{candidate_id}: generalization report used protected evidence")
    metrics = {
        "pooled": report_value["pooled"],
        "worst_seed": report_value["worst_seed"],
        "coverage_passed": bool(report_value["coverage_passed"]),
        "seed_ids": report_value["seed_ids"],
        "confidence_level": report_value["confidence_level"],
        "keywords": report_value["keywords"],
        "statistical_gate_mode": report_value["statistical_gate_mode"],
        "eligible_for_threshold_calibration": bool(
            report_value["eligible_for_threshold_calibration"]
        ),
    }

    result = dict(row)
    result.update(
        {
            "status": "complete",
            "runtime_runner": str(runner),
            "runtime_runner_sha256": sha256_file(runner),
            "development_manifest": str(development_manifest),
            "development_manifest_sha256": sha256_file(development_manifest),
            "frozen_model": str(model),
            "frozen_model_sha256": sha256_file(model),
            "frozen_keywords": str(keywords),
            "frozen_keywords_sha256": sha256_file(keywords),
            "generalization_plan_sha256": str(plan_value["plan_sha256"]),
            "generalization_seed_plan": plan_value["seed_plan"],
            "generalization_report_sha256": sha256_file(report),
            "generalization_metrics": metrics,
            "independent_seed_count": int(report_value["independent_seed_count"]),
            "generalization_plan_complete": True,
        }
    )
    return result


def execute_shard(
    args: argparse.Namespace,
    manifest: dict,
    candidate_id: str,
    output: pathlib.Path,
) -> dict:
    generalization_policy = require_file(
        args.generalization_policy, "development generalization policy"
    )
    product_head = current_head()
    if product_head != str(manifest["product_head"]):
        raise ValueError("repository HEAD moved after Stage A preparation")
    matches = [
        row for row in manifest["candidates"]
        if str(row.get("candidate_id", "")) == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Stage A candidate must match exactly once: {candidate_id}")
    result = execute_candidate(
        row=matches[0],
        args=args,
        generalization_policy=generalization_policy,
        product_head=product_head,
    )
    shard = {
        "schema_version": 1,
        "evidence_class": SHARD_CLASS,
        "evidence_scope": "development-only",
        "stage": "A",
        "status": "complete",
        "product_head": product_head,
        "matrix_sha256": str(manifest["matrix_sha256"]),
        "generalization_cohort_id": str(manifest["generalization_cohort_id"]),
        "generalization_tier": "search",
        "external_base_bundle_sha256": str(manifest["external_base_bundle_sha256"]),
        "full_training_round_budget": True,
        "generalization_policy": str(generalization_policy),
        "generalization_policy_sha256": sha256_file(generalization_policy),
        "candidate": result,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
        "protected_evidence_used": False,
    }
    write_json(output, shard)
    return shard


def execute(args: argparse.Namespace, manifest_path: pathlib.Path, manifest: dict) -> dict:
    generalization_policy = require_file(args.generalization_policy, "development generalization policy")
    product_head = current_head()
    if product_head != str(manifest["product_head"]):
        raise ValueError("repository HEAD moved after Stage A preparation")
    completed: list[dict] = []

    for prepared in manifest["candidates"]:
        try:
            result = execute_candidate(
                row=prepared,
                args=args,
                generalization_policy=generalization_policy,
                product_head=product_head,
            )
        except Exception as exc:
            failed = dict(prepared)
            failed["status"] = "failed"
            failed["failure"] = str(exc)
            completed.append(failed)
            manifest["candidates"] = completed + [
                dict(row) for row in manifest["candidates"][len(completed):]
            ]
            manifest["status"] = "failed"
            write_json(manifest_path, manifest)
            raise
        completed.append(result)
        manifest["candidates"] = completed + [
            dict(row) for row in manifest["candidates"][len(completed):]
        ]
        write_json(manifest_path, manifest)

    seed_plans = [row["generalization_seed_plan"] for row in completed]
    if any(plan != seed_plans[0] for plan in seed_plans[1:]):
        raise ValueError("Stage A candidates did not execute the identical shared generalization seed cohort")
    plan_shas = {str(row["generalization_plan_sha256"]) for row in completed}
    if len(plan_shas) != len(completed):
        raise ValueError("Stage A candidate-specific generalization plan hashes must be unique")

    manifest["status"] = "complete"
    manifest["candidates"] = completed
    manifest["generalization_policy"] = str(generalization_policy)
    manifest["generalization_policy_sha256"] = sha256_file(generalization_policy)
    manifest["shared_seed_plan"] = seed_plans[0]
    manifest["all_candidates_complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or execute the complete speech-like KWS v2 Stage A search without protected evidence."
        )
    )
    parser.add_argument(
        "--experiment-policy",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "kws-v2-staged-experiments-v1.json",
    )
    parser.add_argument(
        "--generalization-policy",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "development-generalization-v1.json",
    )
    for split in ("train", "calibration", "test", "qualification"):
        parser.add_argument(f"--{split}-index", required=True, type=pathlib.Path)
        parser.add_argument(f"--{split}-summary", required=True, type=pathlib.Path)
    parser.add_argument("--rnn-runner", type=pathlib.Path)
    parser.add_argument("--gru-runner", type=pathlib.Path)
    parser.add_argument("--generalization-workers", type=int, default=4)
    parser.add_argument("--candidate-id")
    parser.add_argument("--candidate-output", type=pathlib.Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.generalization_workers <= 0 or args.generalization_workers > 8:
        raise ValueError("generalization-workers must be in [1,8]")
    root = args.work_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Stage A work-dir must be empty for a fresh governed search")
    root.mkdir(parents=True, exist_ok=True)

    manifest_path, manifest = prepare(args, root)
    if args.candidate_output is not None and args.candidate_id is None:
        raise ValueError("--candidate-output requires --candidate-id")
    if args.prepare_only:
        if args.candidate_id is not None:
            matches = [
                row for row in manifest["candidates"]
                if str(row.get("candidate_id", "")) == args.candidate_id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Stage A candidate must match exactly once: {args.candidate_id}"
                )
        print(
            f"kws-v2-stage-a-search prepared: candidates={manifest['candidate_count']} "
            f"cohort={manifest['generalization_cohort_id']} bundle={manifest['external_base_bundle_sha256']}"
        )
        return 0

    if args.candidate_id is not None:
        output = (
            args.candidate_output.resolve()
            if args.candidate_output is not None
            else root / f"{args.candidate_id}-result.json"
        )
        shard = execute_shard(args, manifest, args.candidate_id, output)
        candidate = shard["candidate"]
        print(
            f"kws-v2-stage-a-shard complete: candidate={args.candidate_id} "
            f"family={candidate['model_family']} "
            f"seeds={candidate['independent_seed_count']} "
            f"bundle={shard['external_base_bundle_sha256']}"
        )
        return 0

    result = execute(args, manifest_path, manifest)
    print(
        f"kws-v2-stage-a-search complete: candidates={result['candidate_count']} "
        f"cohort={result['generalization_cohort_id']} seeds={len(result['shared_seed_plan'])}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
