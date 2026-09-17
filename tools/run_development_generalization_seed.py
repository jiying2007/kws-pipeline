#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "eval"))

from build_development_generalization_seed import build_seed, load_object, plan_sha256  # noqa: E402
from external_base_dataset import load_external_base_bundle  # noqa: E402
from gate_robustness import evaluate as evaluate_robustness  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402


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


def run_checked(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def current_head(root: pathlib.Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip().lower()


def absolutize_external_base(config_path: pathlib.Path, config: dict) -> None:
    generator = config.get("generator")
    if not isinstance(generator, dict):
        raise ValueError("generator config is required")
    external = generator.get("external_base_dataset")
    if not isinstance(external, dict):
        raise ValueError("development generalization requires generator.external_base_dataset")
    root = config_path.parent.resolve()
    for split, item in external.items():
        if not isinstance(item, dict):
            raise ValueError(f"external base split {split} must be an object")
        for key in ("index", "summary"):
            raw = item.get(key)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"external base {split}.{key} is required")
            path = pathlib.Path(raw)
            item[key] = str(path.resolve() if path.is_absolute() else (root / path).resolve())


def validate_plan(plan: dict, ordinal: int, model: pathlib.Path, product_root: pathlib.Path) -> tuple[dict, str]:
    digest = plan_sha256(plan)
    if plan.get("evidence_scope") != "development-only":
        raise ValueError("plan evidence_scope must be development-only")
    for key in ("fresh_used", "shadow_used", "formal_qualification_used"):
        if plan.get(key) is not False:
            raise ValueError(f"protected evidence flag {key} must be false")
    entries = plan.get("seed_plan")
    if not isinstance(entries, list) or int(plan.get("independent_seed_count", -1)) != len(entries):
        raise ValueError("plan seed cohort is invalid")
    if ordinal < 0 or ordinal >= len(entries):
        raise ValueError("plan ordinal is out of range")
    entry = entries[ordinal]
    if not isinstance(entry, dict) or int(entry.get("ordinal", -1)) != ordinal:
        raise ValueError("plan ordinal identity mismatch")
    expected_model = str(plan.get("model_sha256", ""))
    actual_model = sha256_file(model)
    if actual_model != expected_model:
        raise ValueError(f"model sha256 mismatch: expected {expected_model}, got {actual_model}")
    expected_head = str(plan.get("product_head", "")).lower()
    actual_head = current_head(product_root)
    if actual_head != expected_head:
        raise ValueError(f"product HEAD mismatch: expected {expected_head}, got {actual_head}")
    return entry, digest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one predeclared development-generalization acoustic seed without protected evidence."
    )
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--plan-ordinal", required=True, type=int)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--split", choices=("calibration", "test", "qualification"), default="test")
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--product-root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--pre-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--post-tolerance-ms", type=float, default=500.0)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    config_path = args.config.resolve()
    runner = args.runner.resolve()
    model = args.model.resolve()
    keywords = args.keywords.resolve()
    product_root = args.product_root.resolve()
    for path, label in ((plan_path, "plan"), (config_path, "config"), (runner, "runner"), (model, "model"), (keywords, "keywords")):
        if not path.is_file():
            raise ValueError(f"{label} does not exist: {path}")

    plan = load_object(plan_path)
    entry, plan_digest = validate_plan(plan, args.plan_ordinal, model, product_root)
    tier = str(plan.get("tier", ""))
    model_family = str(plan.get("model_family", ""))
    candidate_id = str(plan.get("candidate_id", ""))
    seed = int(entry.get("seed", -1))
    source_identity = str(entry.get("source_identity", ""))
    if tier not in {"search", "freeze"} or model_family not in {"rnn", "gru"}:
        raise ValueError("plan tier/model_family is invalid")
    if seed < 0 or not candidate_id or not source_identity:
        raise ValueError("plan candidate/seed/source identity is invalid")

    config = load_object(config_path)
    external_result = load_external_base_bundle(config_path, config)
    if external_result is None:
        raise ValueError("development generalization requires an immutable external speech-like base bundle")
    _, external_summary = external_result
    if external_summary.get("tone_backend_used") is not False:
        raise ValueError("tone backend is forbidden for development generalization execution")

    effective = json.loads(json.dumps(config))
    effective["seed"] = seed
    absolutize_external_base(config_path, effective)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    effective_config = work_dir / "effective-config.json"
    write_json(effective_config, effective)

    render_root = work_dir / "rendered"
    domain_summary = render_domain_dataset(effective_config, render_root)
    if not isinstance(domain_summary.get("external_base_dataset"), dict):
        raise ValueError("renderer did not retain external speech-like base identity")
    if domain_summary["external_base_dataset"].get("bundle_sha256") != external_summary.get("bundle_sha256"):
        raise ValueError("renderer external base bundle identity drifted")

    references = render_root / f"{args.split}.references.jsonl"
    detections = work_dir / "detections.jsonl"
    provenance = work_dir / "run-corpus-provenance.json"
    corpus_identity = work_dir / "corpus-identity.json"
    run_checked(
        [
            sys.executable,
            str(ROOT / "eval" / "run_corpus.py"),
            "--runner", str(runner),
            "--model", str(model),
            "--keywords", str(keywords),
            "--references", str(references),
            "--audio-root", str(render_root),
            "--detections", str(detections),
            "--provenance", str(provenance),
            "--corpus-identity", str(corpus_identity),
        ]
    )

    metrics_path = work_dir / "domain-metrics.json"
    run_checked(
        [
            sys.executable,
            str(ROOT / "eval" / "domain_metrics.py"),
            "--references", str(references),
            "--detections", str(detections),
            "--output", str(metrics_path),
            "--pre-tolerance-ms", str(args.pre_tolerance_ms),
            "--post-tolerance-ms", str(args.post_tolerance_ms),
        ]
    )
    metrics = load_object(metrics_path)
    robustness = evaluate_robustness({"qualification_domains": metrics}, effective)
    robustness_path = work_dir / "robustness.json"
    write_json(robustness_path, robustness)

    seed_summary = build_seed(
        domain_metrics=metrics,
        robustness=robustness,
        tier=tier,
        model_family=model_family,
        candidate_id=candidate_id,
        source_identity=source_identity,
        seed=seed,
        plan=plan,
        plan_ordinal=args.plan_ordinal,
    )
    seed_summary["execution_binding"] = {
        "schema_version": 1,
        "plan_sha256": plan_digest,
        "plan_ordinal": args.plan_ordinal,
        "product_head": str(plan["product_head"]),
        "model_sha256": sha256_file(model),
        "runner_sha256": sha256_file(runner),
        "keyword_pack_sha256": sha256_file(keywords),
        "source_config_sha256": sha256_file(config_path),
        "effective_config_sha256": sha256_file(effective_config),
        "external_base_bundle_sha256": str(external_summary["bundle_sha256"]),
        "domain_summary_sha256": sha256_file(render_root / "domain-summary.json"),
        "references_sha256": sha256_file(references),
        "detections_sha256": sha256_file(detections),
        "domain_metrics_sha256": sha256_file(metrics_path),
        "robustness_sha256": sha256_file(robustness_path),
        "split": args.split,
        "protected_evidence_used": False,
    }
    write_json(args.output.resolve(), seed_summary)
    write_json(work_dir / "execution-binding.json", seed_summary["execution_binding"])
    print(
        f"development-generalization-executor: tier={tier} family={model_family} "
        f"candidate={candidate_id} ordinal={args.plan_ordinal} seed={seed} "
        f"coverage={str(seed_summary['coverage_passed']).lower()}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
