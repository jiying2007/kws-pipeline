#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from build_development_generalization_seed import load_object, plan_sha256  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_one(
    *,
    executor: pathlib.Path,
    plan: pathlib.Path,
    ordinal: int,
    config: pathlib.Path,
    runner: pathlib.Path,
    model: pathlib.Path,
    keywords: pathlib.Path,
    product_root: pathlib.Path,
    split: str,
    work_root: pathlib.Path,
    pre_tolerance_ms: float,
    post_tolerance_ms: float,
) -> pathlib.Path:
    seed_dir = work_root / f"seed-{ordinal:03d}"
    output = seed_dir / "seed-summary.json"
    seed_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(executor),
        "--plan", str(plan),
        "--plan-ordinal", str(ordinal),
        "--config", str(config),
        "--runner", str(runner),
        "--model", str(model),
        "--keywords", str(keywords),
        "--split", split,
        "--work-dir", str(seed_dir / "work"),
        "--output", str(output),
        "--product-root", str(product_root),
        "--pre-tolerance-ms", str(pre_tolerance_ms),
        "--post-tolerance-ms", str(post_tolerance_ms),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (seed_dir / "executor.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"seed ordinal {ordinal} failed with exit {completed.returncode}; "
            f"see {seed_dir / 'executor.log'}"
        )
    if not output.is_file():
        raise RuntimeError(f"seed ordinal {ordinal} did not produce {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a complete predeclared development-generalization cohort in parallel."
    )
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--product-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--split", choices=("calibration", "test", "qualification"), default="test")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument(
        "--seed-executor",
        type=pathlib.Path,
        default=ROOT / "tools" / "run_development_generalization_seed.py",
    )
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    args = parser.parse_args()

    plan = args.plan.resolve()
    policy = args.policy.resolve()
    config = args.config.resolve()
    runner = args.runner.resolve()
    model = args.model.resolve()
    keywords = args.keywords.resolve()
    product_root = args.product_root.resolve()
    executor = args.seed_executor.resolve()
    for path, label in (
        (plan, "plan"),
        (policy, "policy"),
        (config, "config"),
        (runner, "runner"),
        (model, "model"),
        (keywords, "keywords"),
        (executor, "seed executor"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    plan_value = load_object(plan)
    digest = plan_sha256(plan_value)
    if plan_value.get("selection_before_results") is not True:
        raise ValueError("arena requires a plan selected before results")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if plan_value.get(key) is not False:
            raise ValueError(f"protected evidence flag {key} must be false")
    entries = plan_value.get("seed_plan")
    if not isinstance(entries, list) or not entries:
        raise ValueError("plan seed cohort must be non-empty")
    count = int(plan_value.get("independent_seed_count", -1))
    if count != len(entries):
        raise ValueError("plan independent_seed_count mismatch")
    ordinals = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or int(entry.get("ordinal", -1)) != index:
            raise ValueError("plan ordinals must be contiguous from zero")
        ordinals.append(index)

    workers = min(4, count) if args.workers is None else int(args.workers)
    if workers <= 0 or workers > min(32, count):
        raise ValueError(f"workers must be in [1,{min(32, count)}]")

    work_root = args.work_dir.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    seed_paths: dict[int, pathlib.Path] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                run_one,
                executor=executor,
                plan=plan,
                ordinal=ordinal,
                config=config,
                runner=runner,
                model=model,
                keywords=keywords,
                product_root=product_root,
                split=args.split,
                work_root=work_root,
                pre_tolerance_ms=args.pre_tolerance_ms,
                post_tolerance_ms=args.post_tolerance_ms,
            ): ordinal
            for ordinal in ordinals
        }
        for future in as_completed(future_map):
            ordinal = future_map[future]
            try:
                seed_paths[ordinal] = future.result()
            except Exception as exc:  # noqa: BLE001
                failures.append(f"ordinal {ordinal}: {exc}")

    if failures:
        raise RuntimeError("arena seed execution failed: " + "; ".join(sorted(failures)))
    if set(seed_paths) != set(ordinals):
        raise RuntimeError("arena did not produce the complete predeclared cohort")

    output = args.output.resolve()
    command = [
        sys.executable,
        str(ROOT / "tools" / "development_generalization.py"),
        "--policy", str(policy),
        "--tier", str(plan_value["tier"]),
        "--plan", str(plan),
    ]
    for ordinal in ordinals:
        command.extend(["--seed-summary", str(seed_paths[ordinal])])
    command.extend(["--output", str(output)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (work_root / "aggregate.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"generalization aggregation failed with exit {completed.returncode}; "
            f"see {work_root / 'aggregate.log'}"
        )
    if not output.is_file():
        raise RuntimeError("generalization aggregator did not produce the report")

    report = load_object(output)
    if report.get("plan_sha256") != digest or report.get("plan_complete") is not True:
        raise ValueError("aggregated report is not bound to the complete plan")
    if int(report.get("independent_seed_count", -1)) != count:
        raise ValueError("aggregated report seed count mismatch")
    manifest = {
        "schema_version": 1,
        "evidence_class": "development-generalization-arena-run-v1",
        "evidence_scope": "development-only",
        "plan_sha256": digest,
        "tier": str(plan_value["tier"]),
        "model_family": str(plan_value["model_family"]),
        "candidate_id": str(plan_value["candidate_id"]),
        "independent_seed_count": count,
        "workers": workers,
        "split": args.split,
        "report_sha256": sha256_file(output),
        "seed_summaries": [
            {
                "ordinal": ordinal,
                "path": str(seed_paths[ordinal]),
                "sha256": sha256_file(seed_paths[ordinal]),
            }
            for ordinal in ordinals
        ],
        "protected_evidence_used": False,
    }
    write_json(work_root / "arena-manifest.json", manifest)
    print(
        f"development-generalization-arena: tier={manifest['tier']} "
        f"family={manifest['model_family']} candidate={manifest['candidate_id']} "
        f"seeds={count} workers={workers} plan={digest}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
