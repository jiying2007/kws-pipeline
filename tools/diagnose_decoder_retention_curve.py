#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from iterate_domain import base_gate, domain_gate, gate_values  # noqa: E402

EVIDENCE_CLASS = "decoder-retention-operating-curve-development-v1"
POLICY = "development-only-posterior-replay-retention-sweep-v1"


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


def run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}"
        )


def compact_metrics(base: dict, domains: dict) -> dict:
    far_domain = domains.get("domains", {}).get("distance:far", {})
    return {
        "frr": float(base.get("frr", 1.0)),
        "far_per_hour": float(base.get("far_per_hour", math.inf)),
        "negative_recording_far_per_hour": base.get(
            "negative_recording_far_per_hour"
        ),
        "negative_recording_far_upper_95_per_hour": base.get(
            "negative_recording_far_upper_95_per_hour"
        ),
        "p95_post_end_latency_ms": float(
            base.get("p95_post_end_latency_ms", math.inf)
        ),
        "per_keyword": base.get("per_keyword", {}),
        "far_field_frr": (
            float(far_domain["frr"]) if isinstance(far_domain, dict) and "frr" in far_domain
            else None
        ),
        "worst_domain_score": domains.get("worst_domain_score"),
    }


def evaluate(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    keywords: pathlib.Path,
    references: pathlib.Path,
    posterior_dump: pathlib.Path,
    decoder_replay: pathlib.Path,
    posterior_cache: pathlib.Path,
    retention: float,
    output: pathlib.Path,
) -> tuple[dict, dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    detections = output / "detections.jsonl"
    provenance = output / "detections.provenance.json"
    summary = output / "summary.json"
    false_positives = output / "false-positives.jsonl"
    false_rejects = output / "false-rejects.jsonl"
    domains = output / "domains.json"

    run(
        [
            sys.executable,
            str(EVAL / "run_corpus.py"),
            "--runner",
            str(runner),
            "--model",
            str(model),
            "--keywords",
            str(keywords),
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--provenance",
            str(provenance),
            "--posterior-dump",
            str(posterior_dump),
            "--decoder-replay",
            str(decoder_replay),
            "--posterior-cache",
            str(posterior_cache),
            "--decoder-state-retention",
            str(retention),
        ]
    )
    run(
        [
            sys.executable,
            str(EVAL / "score_events.py"),
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--summary",
            str(summary),
            "--false-positives",
            str(false_positives),
            "--false-rejects",
            str(false_rejects),
        ]
    )
    run(
        [
            sys.executable,
            str(EVAL / "domain_metrics.py"),
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--output",
            str(domains),
        ]
    )
    return load_object(summary), load_object(domains), load_object(provenance)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only state-retention operating-curve diagnostic. "
            "Uses exact posterior replay and never changes shipping settings."
        )
    )
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-references", required=True, type=pathlib.Path)
    parser.add_argument("--test-references", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--decoder-replay", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--retentions", required=True, nargs="+", type=float)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    for path, label in (
        (args.runner, "runner"),
        (args.model, "model"),
        (args.keywords, "keywords"),
        (args.config, "config"),
        (args.calibration_references, "calibration references"),
        (args.test_references, "test references"),
        (args.posterior_dump, "posterior dump"),
        (args.decoder_replay, "decoder replay"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    retentions = sorted(set(float(value) for value in args.retentions))
    if (
        not retentions
        or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in retentions
        )
    ):
        raise ValueError("retentions must be unique finite values in (0,1)")

    cfg = load_object(args.config)
    gates = gate_values(cfg.get("domain_gates", {}))
    work = args.work_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cache = args.posterior_cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    total_hits = 0
    total_misses = 0
    for retention in retentions:
        trial = work / f"retention-{retention:.6f}"
        cal_base, cal_domains, cal_provenance = evaluate(
            runner=args.runner.resolve(),
            model=args.model.resolve(),
            keywords=args.keywords.resolve(),
            references=args.calibration_references.resolve(),
            posterior_dump=args.posterior_dump.resolve(),
            decoder_replay=args.decoder_replay.resolve(),
            posterior_cache=cache,
            retention=retention,
            output=trial / "calibration",
        )
        test_base, test_domains, test_provenance = evaluate(
            runner=args.runner.resolve(),
            model=args.model.resolve(),
            keywords=args.keywords.resolve(),
            references=args.test_references.resolve(),
            posterior_dump=args.posterior_dump.resolve(),
            decoder_replay=args.decoder_replay.resolve(),
            posterior_cache=cache,
            retention=retention,
            output=trial / "test",
        )
        total_hits += int(cal_provenance.get("posterior_cache_hits", 0))
        total_hits += int(test_provenance.get("posterior_cache_hits", 0))
        total_misses += int(cal_provenance.get("posterior_cache_misses", 0))
        total_misses += int(test_provenance.get("posterior_cache_misses", 0))
        rows.append(
            {
                "state_retention": retention,
                "calibration": compact_metrics(cal_base, cal_domains),
                "test": compact_metrics(test_base, test_domains),
                "strict": (
                    base_gate(cal_base, gates)
                    and domain_gate(cal_domains, gates)
                    and base_gate(test_base, gates)
                    and domain_gate(test_domains, gates)
                ),
                "posterior_cache": {
                    "calibration_hits": int(
                        cal_provenance.get("posterior_cache_hits", 0)
                    ),
                    "calibration_misses": int(
                        cal_provenance.get("posterior_cache_misses", 0)
                    ),
                    "test_hits": int(
                        test_provenance.get("posterior_cache_hits", 0)
                    ),
                    "test_misses": int(
                        test_provenance.get("posterior_cache_misses", 0)
                    ),
                },
            }
        )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "model_sha256": sha256_file(args.model),
        "keyword_pack_sha256": sha256_file(args.keywords),
        "config_sha256": sha256_file(args.config),
        "calibration_references_sha256": sha256_file(args.calibration_references),
        "test_references_sha256": sha256_file(args.test_references),
        "retentions": retentions,
        "posterior_cache_hits": total_hits,
        "posterior_cache_misses": total_misses,
        "operating_curve": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "retentions": len(retentions),
                "cache_hits": total_hits,
                "cache_misses": total_misses,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
