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
import subprocess
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
PROTOCOL = "dataset-iteration-v1"
VARIABLES = ("dataset", "model")

# chi-square with 2 degrees of freedom at 95% is 5.991, and the Poisson upper
# bound for zero observed events over exposure T is chi2/(2T) ~= 3/T.
ZERO_COUNT_UPPER_BOUND = 2.9957
# Normal approximation to Poisson for a non-zero count. Deliberately an
# approximation, and labelled as one in the output: the exact bound needs an
# incomplete gamma function, and the value that actually drives decisions is the
# zero-count case below.
NORMAL_Z95 = 1.959964


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


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON: {exc}") from None
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def wav_duration_s(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as reader:
        return reader.getnframes() / float(reader.getframerate())


def dataset_identity(references: pathlib.Path, audio_root: pathlib.Path) -> tuple[str, dict]:
    """Content hash of the dataset: audio bytes + what is expected of them.

    Two datasets that differ only in expectations are different datasets, and so
    are two that share expectations but differ in audio. Both halves are in the
    hash on purpose.
    """
    rows = load_jsonl(references)
    entries = []
    for row in rows:
        name = str(row.get("recording", ""))
        if not name:
            raise ValueError("reference row is missing 'recording'")
        relative = str(row.get("path") or "")
        audio = pathlib.Path(relative)
        if not audio.is_absolute():
            audio = audio_root / audio
        if not audio.is_file():
            raise ValueError(f"{name}: audio file is missing: {audio}")
        entries.append(
            {
                "recording": name,
                "wav_sha256": sha256_file(audio),
                "duration_s": round(wav_duration_s(audio), 6),
                "expected": row.get("expected", []),
                "domains": {key: row[key] for key in sorted(row) if key.startswith("distance")},
            }
        )
    if not entries:
        raise ValueError("reference file contains no recordings")
    entries.sort(key=lambda item: item["recording"])
    return canonical_hash(entries), {"recordings": len(entries)}


def model_identity(model: pathlib.Path, keywords: pathlib.Path) -> str:
    """The model tuple is model + keyword pack: the pack carries the threshold,
    so changing it changes the operating point and therefore the identity."""
    return canonical_hash(
        {"model_sha256": sha256_file(model), "pack_sha256": sha256_file(keywords)}
    )


def exposure_hours(references: pathlib.Path) -> tuple[float, int]:
    """Negative exposure: recordings with no expected keyword.

    False accepts can only happen where no wake is expected, so this -- not the
    total corpus duration -- is the denominator of FAR/hour.
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


def far_upper_bound_95(false_accepts: float, hours: float) -> float:
    if hours <= 0.0:
        raise ValueError("negative exposure is zero: FAR/hour is undefined")
    if false_accepts <= 0.0:
        return ZERO_COUNT_UPPER_BOUND / hours
    return (false_accepts + NORMAL_Z95 * math.sqrt(false_accepts)) / hours


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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--variable", required=True, choices=list(VARIABLES),
                        help="which of the two controlled variables is under test")
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--code-sha", default="", help="defaults to git HEAD")
    parser.add_argument("--protocol", default=PROTOCOL)
    args = parser.parse_args()

    for label, path in (("runner", args.runner), ("model", args.model),
                        ("keywords", args.keywords), ("references", args.references)):
        if not path.is_file():
            raise ValueError(f"{label} not found: {path}")

    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    # eval/run_corpus.py resolves audio through `_execution_path`, an internal
    # field that the domain renderer happens to add. A plain references file
    # does not have it, so fill it in here rather than making callers learn an
    # implementation detail of the evaluation chain.
    rows = load_jsonl(args.references)
    enriched = []
    for row in rows:
        item = dict(row)
        if not str(item.get("_execution_path") or ""):
            item["_execution_path"] = str(item.get("path") or "")
        enriched.append(item)
    effective_references = work / "references.effective.jsonl"
    effective_references.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for item in enriched
        ),
        encoding="utf-8",
    )

    detections = work / "detections.jsonl"
    provenance = work / "detections.provenance.json"
    summary = work / "summary.json"
    domains = work / "domains.json"
    false_positives = work / "false-positives.jsonl"
    false_rejects = work / "false-rejects.jsonl"

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

    dataset_id, dataset_info = dataset_identity(args.references, args.audio_root)
    model_id = model_identity(args.model, args.keywords)
    hours, negatives = exposure_hours(args.references)
    base = json.loads(summary.read_text(encoding="utf-8"))
    domain_rows = json.loads(domains.read_text(encoding="utf-8"))

    false_accepts = float(base.get("false_accepts", 0.0) or 0.0)
    point_far = float(base.get("far_per_hour", 0.0) or 0.0)
    identity = {
        "protocol": args.protocol,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "code_sha": args.code_sha or git_head(),
        "seed": args.seed,
        "variable": args.variable,
    }
    scorecard = {
        "schema_version": 1,
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
        "far_rate": {
            "point_estimate_per_hour": point_far,
            "upper_bound_95_per_hour": far_upper_bound_95(false_accepts, hours),
            "basis": (
                "zero observed: chi-square(2) 95% / (2T)"
                if false_accepts <= 0.0
                else "non-zero count: normal approximation to Poisson"
            ),
        },
        "domains": domain_rows,
        "artifacts": {
            "detections_sha256": sha256_file(detections),
            "references_sha256": sha256_file(args.references),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
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
