#!/usr/bin/env python3
"""Run one dataset-iteration evaluation and emit a comparable scorecard.

This is the measurement half of the dataset x model iteration lane. It reuses
the repository's existing evaluation chain verbatim --
``eval/run_corpus.py`` -> ``eval/score_events.py`` -> ``eval/domain_metrics.py``
-- so the numbers are the same numbers the governed path already trusts. What it
adds is the part that made iteration impossible before:

an **identity block** that binds a result to (dataset, model, code, protocol),
and **exposure-aware false-accept reporting**, so two runs can actually be
compared.

Why exposure matters: the governed gate asks for ``far_per_hour <= 0.0``, which
is not a measurable quantity. Observing zero false accepts over ``T`` hours does
not establish a rate of zero; it establishes an upper bound of about ``3/T`` per
hour at 95% confidence. Two runs with different exposure therefore produce
FAR numbers that cannot be placed next to each other, which is one reason the
gate could never be reasoned about. This tool records the exposure and the
bound alongside the point estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import subprocess
import sys

from dataset_iteration_inputs import (
    bundle_system_evidence, dataset_identity, dataset_input_paths, load_jsonl, system_evidence,
    validate_output_paths, validate_runner_context, verify_system_evidence,
)
from export_artifacts import write_export_artifacts
from dataset_iteration_policy import (
    CONFIDENCE, DATASET_IDENTITY_POLICY, FAR_POLICY, PROTOCOL, SCHEMA_VERSION,
    far_upper_bound_95,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
sys.path.insert(0, str(EVAL))
from domain_metrics import domain_keys  # noqa: E402

VARIABLES = ("dataset", "model", "system")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_identity(model: pathlib.Path, keywords: pathlib.Path) -> str:
    """The model tuple is model + keyword pack: the pack carries the threshold,
    so changing it changes the operating point and therefore the identity."""
    return canonical_hash(
        {"model_sha256": sha256_file(model), "pack_sha256": sha256_file(keywords)}
    )


def exposure_hours(references: pathlib.Path) -> tuple[float, int]:
    """Negative exposure: recordings with no expected keyword.

    This population excludes entire positive recordings, including unmatched
    detections there. Global false accepts remain separate diagnostics.
    """
    rows = load_jsonl(references)
    total = 0.0
    negatives = 0
    for row in rows:
        expected = row.get("expected", []) or []
        if expected:
            continue
        duration = float(row.get("duration_s", 0.0) or 0.0)
        if duration <= 0.0:
            raise ValueError(f"{row.get('recording')}: duration_s must be > 0")
        total += duration
        negatives += 1
    return total / 3600.0, negatives


def negative_far_rate(summary: dict) -> tuple[float, dict]:
    """Use negative-only exposure and the shared exact one-sided 95% bound."""
    fields = ("negative_recording_false_accepts", "negative_recording_audio_hours",
              "negative_recording_far_per_hour")
    for field in fields:
        if field not in summary:
            raise ValueError(f"summary is missing required '{field}'")
    count, raw_hours, raw_point = (summary[field] for field in fields)
    if type(count) is not int or count < 0:
        raise ValueError("negative false accepts must be a non-negative integer")
    if isinstance(raw_hours, bool) or isinstance(raw_point, bool):
        raise ValueError("negative FAR hours and point must be numbers, not booleans")
    try:
        hours, point = float(raw_hours), float(raw_point)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("negative FAR requires positive exposure and a finite point") from exc
    if not math.isfinite(hours) or hours <= 0.0:
        raise ValueError("negative exposure must be finite and positive")
    if not math.isfinite(point) or point < 0.0 or point != count / hours:
        raise ValueError("negative FAR point must equal negative count / exposure")
    bound = far_upper_bound_95(count, hours)
    if not math.isfinite(bound):
        raise ValueError("negative FAR upper bound must be finite")
    return hours, {
        "observed_false_accepts": count,
        "point_estimate_per_hour": point,
        "upper_bound_95_per_hour": bound,
        "confidence_level": CONFIDENCE,
        "basis": "exact one-sided Poisson upper confidence bound",
    }


def run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise ValueError(
            f"{' '.join(argv[1:2])} failed ({completed.returncode}): "
            f"{completed.stderr.strip()[:400]}"
        )


def git_head() -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure one dataset iteration and bind its inputs.")
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--variable", required=True, choices=list(VARIABLES),
                        help="controlled variable, or system for an explicit system comparison")
    parser.add_argument("--mode", choices=("ablation", "system"), default="ablation")
    parser.add_argument("--system-context", type=pathlib.Path,
                        help="common declared runner input/event context JSON; system mode only")
    parser.add_argument("--native-frontend", default="",
                        help="native frontend description; system mode only, not forced equal")
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--code-sha", default="", help="defaults to git HEAD")
    parser.add_argument("--protocol", default=PROTOCOL)
    args = parser.parse_args()

    code_sha = args.code_sha or git_head()
    context = None
    if args.mode == "system":
        if args.variable != "system" or args.system_context is None or not args.native_frontend.strip():
            raise ValueError("system mode requires --variable system, --system-context and --native-frontend")
        if re.fullmatch(r"[0-9a-f]{40}", code_sha) is None:
            raise ValueError("system mode requires a full source revision in --code-sha")
        context = validate_runner_context(json.loads(args.system_context.read_text(encoding="utf-8")))
        context_sha256 = sha256_file(args.system_context)
    elif args.variable == "system" or args.system_context is not None or args.native_frontend:
        raise ValueError("system declarations require --mode system")

    for label, path in (("runner", args.runner), ("model", args.model),
                        ("keywords", args.keywords), ("references", args.references)):
        if not path.is_file():
            raise ValueError(f"{label} not found: {path}")
    execution_files = {str(path): sha256_file(path) for path in
                       (args.runner, args.model, args.keywords)} if context is not None else {}

    # eval/run_corpus.py resolves audio through `_execution_path`, an internal
    # field that the domain renderer happens to add. A plain references file
    # does not have it, so fill it in here rather than making callers learn an
    # implementation detail of the evaluation chain.
    rows = load_jsonl(args.references)
    work = args.work_dir.resolve()
    effective_references = work / "references.effective.jsonl"
    detections = work / "detections.jsonl"
    provenance = work / "detections.provenance.json"
    summary = work / "summary.json"
    domains = work / "domains.json"
    false_positives = work / "false-positives.jsonl"
    false_rejects = work / "false-rejects.jsonl"
    outputs = {
        "effective references": effective_references, "detections": detections,
        "provenance": provenance, "summary": summary, "domains": domains,
        "false positives": false_positives, "false rejects": false_rejects,
        "scorecard": args.output,
    }
    inputs = [args.runner, args.model, args.keywords, args.references,
              *dataset_input_paths(rows, args.audio_root)]
    if args.system_context is not None:
        inputs.append(args.system_context)
    validate_output_paths(outputs, inputs)
    work.mkdir(parents=True, exist_ok=True)
    enriched = []
    for row in rows:
        item = dict(row)
        if not str(item.get("_execution_path") or ""):
            item["_execution_path"] = str(item.get("path") or "")
        enriched.append(item)
    effective_references.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for item in enriched
        ),
        encoding="utf-8",
    )

    # Bind verified inputs before execution. Recheck the effective and original
    # references after evaluation before issuing a card. Corpus retains its own
    # per-file pre/post hash checks; none of these checks is an atomic snapshot.
    dataset_id, dataset_info = dataset_identity(effective_references, args.audio_root)

    run([sys.executable, str(EVAL / "run_corpus.py"),
         "--runner", str(args.runner.resolve()),
         "--model", str(args.model.resolve()),
         "--keywords", str(args.keywords.resolve()),
         "--references", str(effective_references),
         "--audio-root", str(args.audio_root.resolve()),
         "--detections", str(detections),
         "--provenance", str(provenance)])
    run([sys.executable, str(EVAL / "score_events.py"),
         "--references", str(effective_references),
         "--detections", str(detections),
         "--summary", str(summary),
         "--false-positives", str(false_positives),
         "--false-rejects", str(false_rejects)])
    run([sys.executable, str(EVAL / "domain_metrics.py"),
         "--references", str(effective_references),
         "--detections", str(detections),
         "--output", str(domains)])

    for reference in (effective_references, args.references):
        if dataset_identity(reference, args.audio_root)[0] != dataset_id:
            raise ValueError("dataset inputs changed during iteration evaluation")
    model_id = model_identity(args.model, args.keywords)
    reference_hours, negatives = exposure_hours(args.references)
    base = json.loads(summary.read_text(encoding="utf-8"))
    domain_rows = json.loads(domains.read_text(encoding="utf-8"))
    hours, far_rate = negative_far_rate(base)
    if not math.isclose(hours, reference_hours, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("negative scorer exposure differs from references")

    false_accepts = float(base.get("false_accepts", 0.0) or 0.0)
    point_far = float(base.get("far_per_hour", 0.0) or 0.0)
    identity = {
        "protocol": args.protocol,
        "far_policy": FAR_POLICY,
        "dataset_identity_policy": DATASET_IDENTITY_POLICY,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "code_sha": code_sha,
        "seed": args.seed,
        "variable": args.variable,
        "comparison_mode": args.mode,
    }
    scorecard = {
        "schema_version": SCHEMA_VERSION,
        "protocol": args.protocol,
        "run_id": canonical_hash(identity)[:16],
        "identity": identity,
        "labels": {"dataset": args.dataset_label, "model": args.model_label},
        "dataset": dataset_info,
        "exposure": {
            "negative_recordings": negatives,
            "exposure_hours": hours,
        },
        "metrics": {
            "expected": base.get("expected"),
            "matched": base.get("matched"),
            "false_rejects": base.get("false_rejects"),
            "false_accepts": false_accepts,
            "frr": base.get("frr"),
            "far_per_hour": point_far,
            "p95_post_end_latency_ms": base.get("p95_post_end_latency_ms"),
        },
        "far_rate": far_rate,
        "domains": domain_rows,
        "artifacts": {
            "detections_sha256": sha256_file(detections),
            "references_sha256": sha256_file(args.references),
        },
    }
    if context is not None:
        if (sha256_file(args.system_context) != context_sha256 or
                any(sha256_file(pathlib.Path(path)) != digest for path, digest in execution_files.items())):
            raise ValueError("system context or runner/model/keyword bytes changed during evaluation")
        evidence = system_evidence(
            references=effective_references, source_references=args.references,
            detections=detections, provenance=provenance, audio_root=args.audio_root, context=context,
        )
        identity.update(native_frontend=args.native_frontend.strip(),
                        runner_sha256=execution_files[str(args.runner)],
                        evaluation_protocol_id=canonical_hash(evidence["protocol"]))
        scorecard["system_evaluation"] = evidence
        scorecard["artifacts"]["provenance_sha256"] = sha256_file(provenance)
        scorecard["run_id"] = canonical_hash(identity)[:16]
        verify_system_evidence(scorecard, "produced")
        bundle_system_evidence(scorecard, args.output.parent / "evidence" / scorecard["run_id"])
    validate_output_paths(outputs, inputs)
    text = json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_export_artifacts({args.output: text.encode("utf-8")})
    print(
        "run_dataset_iteration: ok "
        f"run={scorecard['run_id']} dataset={dataset_id[:12]} model={model_id[:12]} "
        f"variable={args.variable} frr={base.get('frr')} "
        f"far_ub95={scorecard['far_rate']['upper_bound_95_per_hour']:.3f}/h "
        f"exposure={hours:.4f}h"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
