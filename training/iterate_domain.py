#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
EVAL = ROOT / "eval"
TRAINING = ROOT / "training"
sys.path.insert(0, str(TOOLS))

from domain_curriculum import merge_domain_metrics, update_curriculum  # noqa: E402
from domain_progress import append_round_progress, build_round_progress  # noqa: E402
from development_failure_replay import (  # noqa: E402
    failure_replay_focus_rows,
    render_development_failure_replay,
)
from fit_domain_prototype import fit_domain_prototype  # noqa: E402
from frontend_spec import FRONTEND_IDS, FRONTEND_LOGMEL  # noqa: E402
from hard_negative_replay import render_hard_negative_replay  # noqa: E402
from feature_cached_trainer import feature_cache_max_items, rewrite_training_command  # noqa: E402
from objective_config import optional_objective_cli_args  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402
from wake_pressure_balance import (  # noqa: E402
    DEFAULT_POSITIVE_EXAMPLE_WEIGHT,
    WAKE_BALANCE_POLICY,
    derive_wake_pressure_balance,
    static_replay_focus_rows,
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str], *, suppress_stdout: bool = False) -> None:
    completed = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.DEVNULL if suppress_stdout else None,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(argv)}")


def safe_reset(path: pathlib.Path) -> pathlib.Path:
    value = path.resolve()
    forbidden = {
        pathlib.Path(value.anchor).resolve(),
        pathlib.Path.home().resolve(),
        ROOT.resolve(),
        ROOT.parent.resolve(),
    }
    if value in forbidden or len(value.parts) < 3:
        raise ValueError(f"refusing unsafe domain work directory: {value}")
    if value.exists() and not value.is_dir():
        raise ValueError(f"domain work path is not a directory: {value}")
    if value.exists():
        shutil.rmtree(value)
    value.mkdir(parents=True)
    return value


def repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def resolve_posterior_replay(
    posterior_dump: pathlib.Path | None,
    decoder_replay: pathlib.Path | None,
    posterior_cache: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path] | None:
    values = (posterior_dump, decoder_replay, posterior_cache)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "posterior replay requires --posterior-dump, --decoder-replay, "
            "and --posterior-cache together"
        )
    dump = posterior_dump.resolve()
    replay = decoder_replay.resolve()
    cache = posterior_cache.resolve()
    if not dump.is_file():
        raise ValueError("posterior dump tool does not exist")
    if not replay.is_file():
        raise ValueError("decoder replay tool does not exist")
    cache.mkdir(parents=True, exist_ok=True)
    return dump, replay, cache


def posterior_replay_cli_args(
    posterior_replay: tuple[pathlib.Path, pathlib.Path, pathlib.Path] | None,
    *,
    decoder_state_retention: float | None = None,
    decoder_refractory_ms: int | None = None,
    decoder_blank_retention: float | None = None,
    decoder_fuzzy_child_cost_log: float | None = None,
) -> list[str]:
    if posterior_replay is None:
        if (
            decoder_state_retention is not None
            or decoder_refractory_ms is not None
            or decoder_blank_retention is not None
            or decoder_fuzzy_child_cost_log is not None
        ):
            raise ValueError("decoder replay overrides require posterior replay")
        return []
    dump, replay, cache = posterior_replay
    result = [
        "--posterior-dump",
        str(dump),
        "--decoder-replay",
        str(replay),
        "--posterior-cache",
        str(cache),
    ]
    if decoder_state_retention is not None:
        if (
            not math.isfinite(decoder_state_retention)
            or not 0.0 < decoder_state_retention < 1.0
        ):
            raise ValueError("decoder_state_retention must be finite and in (0,1)")
        result.extend(["--decoder-state-retention", str(decoder_state_retention)])
    if decoder_refractory_ms is not None:
        if isinstance(decoder_refractory_ms, bool) or not 0 <= int(decoder_refractory_ms) <= 10000:
            raise ValueError("decoder_refractory_ms must be in [0,10000]")
        result.extend(["--decoder-refractory-ms", str(int(decoder_refractory_ms))])
    if decoder_blank_retention is not None:
        if (
            not math.isfinite(decoder_blank_retention)
            or not 0.0 < decoder_blank_retention < 1.0
        ):
            raise ValueError("decoder_blank_retention must be finite and in (0,1)")
        result.extend(["--decoder-blank-retention", str(decoder_blank_retention)])
    if decoder_fuzzy_child_cost_log is not None:
        if (
            not math.isfinite(decoder_fuzzy_child_cost_log)
            or not -16.0 <= decoder_fuzzy_child_cost_log <= 0.0
        ):
            raise ValueError("decoder_fuzzy_child_cost_log must be finite and in [-16,0]")
        result.extend(
            ["--decoder-fuzzy-child-cost-log", str(decoder_fuzzy_child_cost_log)]
        )
    return result


def keyword_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) < 4 or len(cols) > 8:
            raise ValueError(f"{path}:{line_no}: expected 4..8 TSV columns")
        threshold = float(cols[2])
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"{path}:{line_no}: threshold must be finite and in (0,1)")
        extras = cols[4:]
        while len(extras) < 4:
            extras.append("")
        rows.append(
            {
                "id": int(cols[0]),
                "text": cols[1].strip(),
                "threshold": threshold,
                "tokens": cols[3].strip(),
                "min_trailing_blanks": extras[0],
                "priority": extras[1],
                "prefix_policy": extras[2],
                "grace_frames": extras[3],
            }
        )
    if not rows:
        raise ValueError("keyword TSV has no rows")
    return rows


def write_keywords(rows: list[dict], path: pathlib.Path) -> None:
    lines = [
        "# id\ttext\tthreshold\texplicit-pinyin-tokens\tmin_trailing_blanks\tpriority\tprefix_policy\tgrace_frames"
    ]
    for row in rows:
        fields = [
            str(int(row["id"])),
            str(row["text"]),
            f"{float(row['threshold']):.6f}",
            str(row["tokens"]),
            str(row.get("min_trailing_blanks", "")),
            str(row.get("priority", "")),
            str(row.get("prefix_policy", "")),
            str(row.get("grace_frames", "")),
        ]
        while fields[-1] == "":
            fields.pop()
        lines.append("\t".join(fields))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_pack(tokens: pathlib.Path, keywords: pathlib.Path, output: pathlib.Path) -> None:
    run(
        [
            sys.executable,
            str(TOOLS / "compile_keywords.py"),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--out-pack",
            str(output),
        ]
    )


def evaluate(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    references: pathlib.Path,
    output: pathlib.Path,
    posterior_replay: tuple[pathlib.Path, pathlib.Path, pathlib.Path] | None = None,
    decoder_state_retention: float | None = None,
    decoder_refractory_ms: int | None = None,
    decoder_blank_retention: float | None = None,
    decoder_fuzzy_child_cost_log: float | None = None,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=True)
    detections = output / "detections.jsonl"
    provenance = output / "detections.provenance.json"
    summary = output / "summary.json"
    false_positives = output / "false-positives.jsonl"
    false_rejects = output / "false-rejects.jsonl"
    domains = output / "domains.json"
    corpus_command = [
        sys.executable,
        str(EVAL / "run_corpus.py"),
        "--runner",
        str(runner),
        "--model",
        str(model),
        "--keywords",
        str(pack),
        "--references",
        str(references),
        "--detections",
        str(detections),
        "--provenance",
        str(provenance),
    ]
    corpus_command.extend(
        posterior_replay_cli_args(
            posterior_replay,
            decoder_state_retention=decoder_state_retention,
            decoder_refractory_ms=decoder_refractory_ms,
            decoder_blank_retention=decoder_blank_retention,
            decoder_fuzzy_child_cost_log=decoder_fuzzy_child_cost_log,
        )
    )
    run(corpus_command)
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
        ],
        suppress_stdout=True,
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
        ],
        suppress_stdout=True,
    )
    base = json.loads(summary.read_text(encoding="utf-8"))
    base["false_positives_path"] = str(false_positives)
    base["false_rejects_path"] = str(false_rejects)
    return base, json.loads(domains.read_text(encoding="utf-8"))


def gate_values(raw: dict) -> dict[str, float]:
    keys = ("max_frr", "max_far_per_hour", "max_p95_latency_ms", "max_far_frr")
    result = {key: float(raw[key]) for key in keys}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("domain gates must be finite and non-negative")
    if result["max_frr"] > 1.0 or result["max_far_frr"] > 1.0:
        raise ValueError("domain FRR gates must be <= 1")
    return result


def base_gate(metrics: dict, gates: dict) -> bool:
    return (
        float(metrics["frr"]) <= gates["max_frr"]
        and float(metrics["far_per_hour"]) <= gates["max_far_per_hour"]
        and float(metrics["p95_post_end_latency_ms"]) <= gates["max_p95_latency_ms"]
    )


def domain_gate(metrics: dict, gates: dict) -> bool:
    far = metrics.get("domains", {}).get("distance:far")
    return isinstance(far, dict) and float(far["frr"]) <= gates["max_far_frr"]


def strict_gate_candidate(record: dict) -> bool:
    return record.get("calibration_gate") is True and record.get("test_gate") is True


def select_strict_candidate(records: list[dict]) -> dict | None:
    eligible = [record for record in records if strict_gate_candidate(record)]
    if not eligible:
        return None
    latest_round = max(int(record["round"]) for record in eligible)
    latest = [record for record in eligible if int(record["round"]) == latest_round]
    return min(latest, key=lambda record: (float(record["score"]), str(record["frontend"])))


def objective(base: dict, domains: dict, gates: dict) -> float:
    far = domains.get("domains", {}).get("distance:far", {})
    far_frr = float(far.get("frr", 1.0))
    worst = float(domains.get("worst_domain_score", 1000.0))
    violation = max(0.0, float(base["frr"]) - gates["max_frr"]) * 10000.0
    violation += max(0.0, float(base["far_per_hour"]) - gates["max_far_per_hour"]) * 100.0
    violation += max(0.0, far_frr - gates["max_far_frr"]) * 12000.0
    return (
        violation
        + float(base["frr"]) * 100.0
        + float(base["far_per_hour"]) * 0.1
        + far_frr * 150.0
        + worst * 0.02
        + float(base["p95_post_end_latency_ms"]) * 0.001
    )


def calibration_behavior_key(base: dict, domains: dict, gates: dict) -> tuple[float, ...]:
    far = domains.get("domains", {}).get("distance:far", {})
    frr = float(base["frr"])
    far_per_hour = float(base["far_per_hour"])
    latency = float(base["p95_post_end_latency_ms"])
    far_frr = float(far.get("frr", 1.0))
    strict = base_gate(base, gates) and domain_gate(domains, gates)
    if strict:
        # Preserve the historical strict-pass plateau semantics. Once all hard
        # gates are satisfied, threshold calibration should not overfit minor
        # objective/latency differences; select_calibration_threshold() can keep
        # choosing the lower median of equivalent strict operating points.
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # When no threshold is strict, FAR-first lexicographic ordering degenerates
    # toward all-reject under the product 0-FAR/0-FRR gates. Use the same
    # balanced development objective that ranks model candidates instead.
    score = objective(base, domains, gates)
    return (
        1.0,
        score,
        frr,
        far_per_hour,
        far_frr,
        latency,
        max(0.0, frr - gates["max_frr"]),
        max(0.0, far_per_hour - gates["max_far_per_hour"]),
    )


def select_calibration_threshold(
    candidates: list[tuple[float, tuple[float, ...]]],
) -> float:
    if not candidates:
        raise ValueError("calibration threshold candidates must not be empty")
    ordered = sorted(candidates, key=lambda item: item[0])
    if any(not math.isfinite(threshold) or not 0.0 < threshold < 1.0 for threshold, _ in ordered):
        raise ValueError("calibration threshold candidates must be finite and in (0,1)")
    best_key = min(key for _, key in ordered)
    plateau = [threshold for threshold, key in ordered if key == best_key]
    # Choose the lower median of the empirically equivalent operating plateau.
    # This preserves both false-accept margin and recall reserve instead of
    # pinning calibration to the highest-threshold edge of the plateau.
    return plateau[(len(plateau) - 1) // 2]


CALIBRATION_TRIAL_POLICY = "coordinate-threshold-trials-v1"


def calibration_threshold_vector(keywords: list[dict]) -> tuple[tuple[int, float], ...]:
    rows: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in keywords:
        keyword_id = int(item["id"])
        threshold = float(item["threshold"])
        if keyword_id in seen:
            raise ValueError("calibration threshold vector contains duplicate keyword ids")
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("calibration threshold vector contains invalid threshold")
        seen.add(keyword_id)
        rows.append((keyword_id, threshold))
    if not rows:
        raise ValueError("calibration threshold vector must not be empty")
    return tuple(sorted(rows))


def calibration_trial_evidence(
    *,
    coordinate: int,
    keyword_id: int,
    threshold: float,
    trial_keywords: list[dict],
    base: dict,
    domains: dict,
    gates: dict,
) -> dict:
    far = domains.get("domains", {}).get("distance:far", {})
    per_keyword = base.get("per_keyword", {})
    if not isinstance(per_keyword, dict):
        raise ValueError("calibration trial per-keyword metrics are missing")
    return {
        "coordinate": int(coordinate),
        "keyword_id": int(keyword_id),
        "threshold": float(threshold),
        "keyword_thresholds": {
            str(item["id"]): float(item["threshold"]) for item in trial_keywords
        },
        "strict": base_gate(base, gates) and domain_gate(domains, gates),
        "behavior_key": list(calibration_behavior_key(base, domains, gates)),
        "metrics": {
            "frr": float(base["frr"]),
            "far_per_hour": float(base["far_per_hour"]),
            "p95_post_end_latency_ms": float(base["p95_post_end_latency_ms"]),
            "far_domain_frr": float(far.get("frr", 1.0)),
            "per_keyword_frr": {
                str(key): float(value["frr"])
                for key, value in sorted(per_keyword.items(), key=lambda item: str(item[0]))
                if isinstance(value, dict) and "frr" in value
            },
        },
    }


def calibration_operating_curve_summary(
    trials: list[dict],
    *,
    selected_thresholds: dict[str, float],
    threshold_grid: list[float],
    coordinates_executed: int,
) -> dict:
    if not trials:
        raise ValueError("calibration operating curve must contain trials")
    grid = sorted(float(value) for value in threshold_grid)
    if not grid:
        raise ValueError("calibration threshold grid must not be empty")
    per_keyword: dict[str, dict] = {}
    keyword_ids = sorted({str(int(row["keyword_id"])) for row in trials}, key=int)
    for keyword_id in keyword_ids:
        rows = [row for row in trials if str(int(row["keyword_id"])) == keyword_id]
        selected = float(selected_thresholds[keyword_id])
        per_keyword[keyword_id] = {
            "trials": len(rows),
            "selected_threshold": selected,
            "selected_on_grid_min": math.isclose(
                selected, grid[0], rel_tol=0.0, abs_tol=1.0e-12
            ),
            "selected_on_grid_max": math.isclose(
                selected, grid[-1], rel_tol=0.0, abs_tol=1.0e-12
            ),
            "strict_trials": sum(bool(row["strict"]) for row in rows),
            "min_frr": min(float(row["metrics"]["frr"]) for row in rows),
            "min_far_per_hour": min(
                float(row["metrics"]["far_per_hour"]) for row in rows
            ),
            "min_far_domain_frr": min(
                float(row["metrics"]["far_domain_frr"]) for row in rows
            ),
        }
    return {
        "policy": CALIBRATION_TRIAL_POLICY,
        "trial_count": len(trials),
        "coordinates_executed": int(coordinates_executed),
        "grid_min": grid[0],
        "grid_max": grid[-1],
        "strict_trial_count": sum(bool(row["strict"]) for row in trials),
        "grid_saturated": any(
            row["selected_on_grid_min"] or row["selected_on_grid_max"]
            for row in per_keyword.values()
        ),
        "per_keyword": per_keyword,
    }


def calibrate(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    source_keywords: pathlib.Path,
    references: pathlib.Path,
    output: pathlib.Path,
    thresholds: list[float],
    rounds: int,
    gates: dict,
    parallel_trials: int = 1,
    posterior_replay: tuple[pathlib.Path, pathlib.Path, pathlib.Path] | None = None,
    decoder_state_retention: float | None = None,
    decoder_refractory_ms: int | None = None,
    decoder_blank_retention: float | None = None,
    decoder_fuzzy_child_cost_log: float | None = None,
) -> tuple[pathlib.Path, pathlib.Path, dict, dict]:
    current = keyword_rows(source_keywords)
    if isinstance(parallel_trials, bool) or not 1 <= int(parallel_trials) <= 4:
        raise ValueError("calibration parallel_trials must be 1..4")
    parallel_trials = int(parallel_trials)
    trial_evidence: list[dict] = []
    trial_cache: dict[
        tuple[tuple[int, float], ...],
        tuple[dict, dict],
    ] = {}
    trial_cache_lock = threading.Lock()
    trial_cache_hits = 0
    coordinates_executed = 0

    for coordinate in range(rounds):
        coordinates_executed = coordinate + 1
        changed = False
        for index, row in enumerate(current):
            base_trial = [dict(item) for item in current]

            def evaluate_threshold(
                threshold: float,
            ) -> tuple[float, tuple[float, ...], dict]:
                nonlocal trial_cache_hits
                trial = [dict(item) for item in base_trial]
                trial[index]["threshold"] = threshold
                vector = calibration_threshold_vector(trial)
                with trial_cache_lock:
                    cached = trial_cache.get(vector)
                if cached is None:
                    trial_dir = output / f"coord{coordinate}-kw{row['id']}-t{threshold:.3f}"
                    tsv = trial_dir / "keywords.tsv"
                    pack = trial_dir / "keywords.kwk"
                    write_keywords(trial, tsv)
                    compile_pack(tokens, tsv, pack)
                    base, domains = evaluate(
                        runner=runner,
                        model=model,
                        pack=pack,
                        references=references,
                        output=trial_dir / "eval",
                        posterior_replay=posterior_replay,
                        decoder_state_retention=decoder_state_retention,
                        decoder_refractory_ms=decoder_refractory_ms,
                        decoder_blank_retention=decoder_blank_retention,
                        decoder_fuzzy_child_cost_log=decoder_fuzzy_child_cost_log,
                    )
                    with trial_cache_lock:
                        existing = trial_cache.get(vector)
                        if existing is None:
                            trial_cache[vector] = (base, domains)
                        else:
                            base, domains = existing
                            trial_cache_hits += 1
                else:
                    base, domains = cached
                    with trial_cache_lock:
                        trial_cache_hits += 1
                key = calibration_behavior_key(base, domains, gates)
                evidence = calibration_trial_evidence(
                    coordinate=coordinate,
                    keyword_id=int(row["id"]),
                    threshold=threshold,
                    trial_keywords=trial,
                    base=base,
                    domains=domains,
                    gates=gates,
                )
                return threshold, key, evidence

            if parallel_trials == 1:
                evaluated = [evaluate_threshold(threshold) for threshold in thresholds]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(parallel_trials, len(thresholds))
                ) as executor:
                    evaluated = list(executor.map(evaluate_threshold, thresholds))
            evaluated.sort(key=lambda item: item[0])
            trial_evidence.extend(item[2] for item in evaluated)
            candidates = [(item[0], item[1]) for item in evaluated]
            selected = select_calibration_threshold(candidates)
            if not math.isclose(float(current[index]["threshold"]), selected):
                changed = True
            current[index]["threshold"] = selected
        if not changed:
            break
    tsv = output / "calibrated-keywords.tsv"
    pack = output / "calibrated-keywords.kwk"
    write_keywords(current, tsv)
    compile_pack(tokens, tsv, pack)
    base, domains = evaluate(
        runner=runner,
        model=model,
        pack=pack,
        references=references,
        output=output / "final-eval",
        posterior_replay=posterior_replay,
        decoder_state_retention=decoder_state_retention,
        decoder_refractory_ms=decoder_refractory_ms,
        decoder_blank_retention=decoder_blank_retention,
        decoder_fuzzy_child_cost_log=decoder_fuzzy_child_cost_log,
    )
    base["calibrated_thresholds"] = {
        str(row["id"]): float(row["threshold"]) for row in current
    }
    base["calibration_threshold_grid"] = [float(value) for value in thresholds]
    base["calibration_coordinate_rounds"] = int(rounds)
    base["calibration_coordinate_rounds_executed"] = int(coordinates_executed)
    base["calibration_parallel_trials"] = parallel_trials
    base["calibration_trial_count"] = len(trial_evidence)
    base["calibration_unique_trial_vectors"] = len(trial_cache)
    base["calibration_trial_cache_hits"] = trial_cache_hits
    if len(trial_cache) + trial_cache_hits != len(trial_evidence):
        raise RuntimeError("calibration trial cache accounting drifted")

    curve_path = output / "calibration-operating-curve.json"
    curve = {
        "schema_version": 1,
        "evidence_class": CALIBRATION_TRIAL_POLICY,
        "selected_thresholds": dict(base["calibrated_thresholds"]),
        "threshold_grid": list(base["calibration_threshold_grid"]),
        "coordinate_rounds_configured": int(rounds),
        "coordinate_rounds_executed": int(coordinates_executed),
        "parallel_trials": parallel_trials,
        "trial_count": len(trial_evidence),
        "unique_trial_vectors": len(trial_cache),
        "cache_hits": trial_cache_hits,
        "cache_policy": "exact-keyword-threshold-vector-v1",
        "trials": trial_evidence,
    }
    curve["summary"] = calibration_operating_curve_summary(
        trial_evidence,
        selected_thresholds=base["calibrated_thresholds"],
        threshold_grid=base["calibration_threshold_grid"],
        coordinates_executed=coordinates_executed,
    )
    curve["summary"].update(
        {
            "unique_trial_vectors": len(trial_cache),
            "cache_hits": trial_cache_hits,
            "cache_policy": "exact-keyword-threshold-vector-v1",
        }
    )
    curve_path.write_text(
        json.dumps(curve, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    base["calibration_operating_curve_path"] = str(curve_path)
    base["calibration_operating_curve_sha256"] = sha256_file(curve_path)
    base["calibration_operating_curve_summary"] = curve["summary"]
    return tsv, pack, base, domains


def parse_warm_start_strategy(iteration: dict) -> str:
    value = str(iteration.get("warm_start_strategy", "full"))
    if value not in {"full", "head-only"}:
        raise ValueError("domain_iteration.warm_start_strategy must be full or head-only")
    return value


def warm_start_args(previous: pathlib.Path | None, strategy: str) -> list[str]:
    if strategy not in {"full", "head-only"}:
        raise ValueError("warm-start strategy must be full or head-only")
    if previous is None:
        return []
    result = ["--warm-start", str(previous)]
    if strategy == "head-only":
        result.append("--head-only")
    return result


def train_acoustic_seed_offset(iteration: dict, round_index: int) -> int:
    if round_index < 0:
        raise ValueError("round index must be non-negative")
    if not isinstance(iteration, dict):
        raise ValueError("domain_iteration must be an object")
    raw = iteration.get("training_acoustic_seed_stride", 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError(
            "domain_iteration.training_acoustic_seed_stride must be a non-negative integer"
        )
    return round_index * raw


def torch_round_training_values(
    cfg: dict,
    *,
    round_index: int,
    warm_started: bool,
) -> tuple[int, float, int]:
    if round_index < 0:
        raise ValueError("round index must be non-negative")
    train = cfg.get("train", {})
    iteration = cfg.get("domain_iteration", {})
    if not isinstance(train, dict) or not isinstance(iteration, dict):
        raise ValueError("train/domain_iteration config must be objects")

    cold_epochs = int(train.get("epochs", 10))
    warm_epochs = int(train.get("warm_start_epochs", cold_epochs))
    if cold_epochs <= 0 or warm_epochs <= 0 or warm_epochs > cold_epochs:
        raise ValueError("train epochs/warm_start_epochs are invalid")

    base_lr = float(train.get("lr", 0.001))
    lr_decay = float(iteration.get("lr_decay_per_round", 1.0))
    seed_stride = int(iteration.get("training_seed_stride", 0))
    if (
        not math.isfinite(base_lr)
        or base_lr <= 0.0
        or not math.isfinite(lr_decay)
        or not 0.0 < lr_decay <= 1.0
        or seed_stride < 0
    ):
        raise ValueError("torch round learning-rate/seed policy is invalid")

    epochs = warm_epochs if warm_started else cold_epochs
    learning_rate = base_lr * (lr_decay ** round_index)
    seed = int(cfg.get("seed", 1337)) + round_index * seed_stride
    return epochs, learning_rate, seed


def build_torch(
    *,
    cfg: dict,
    frontend: str,
    tokens: pathlib.Path,
    keywords: pathlib.Path,
    manifest: pathlib.Path,
    output: pathlib.Path,
    previous: pathlib.Path | None,
    hard_negative_manifest: pathlib.Path | None,
    failure_replay_manifest: pathlib.Path | None,
    wake_balance: dict | None,
    warm_start_strategy: str,
    round_index: int,
) -> tuple[pathlib.Path, pathlib.Path]:
    checkpoint = output / "model.pt"
    model = output / "model.kwm"
    train = cfg.get("train", {})
    epochs, learning_rate, seed = torch_round_training_values(
        cfg,
        round_index=round_index,
        warm_started=previous is not None,
    )
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(manifest),
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
        "--output",
        str(checkpoint),
    ]
    if (
        hard_negative_manifest is not None
        and hard_negative_manifest.is_file()
        and hard_negative_manifest.stat().st_size > 0
    ):
        command.extend(["--manifest", str(hard_negative_manifest)])
    if (
        failure_replay_manifest is not None
        and failure_replay_manifest.is_file()
        and failure_replay_manifest.stat().st_size > 0
    ):
        command.extend(["--manifest", str(failure_replay_manifest)])
    if wake_balance is not None:
        if wake_balance.get("policy") != WAKE_BALANCE_POLICY:
            raise ValueError("base wake-balance policy identity mismatch")
        command.extend(
            [
                "--positive-example-weight",
                str(float(wake_balance["positive_example_weight"])),
                "--wake-example-weight",
                str(float(wake_balance["default_wake_example_weight"])),
                "--wake-keyword-weights",
                json.dumps(wake_balance["wake_keyword_weights"], sort_keys=True),
            ]
        )
    command.extend(optional_objective_cli_args(train))
    command.extend(warm_start_args(previous, warm_start_strategy))
    command = rewrite_training_command(command, feature_cache_max_items(train))
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
    return model, checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", type=pathlib.Path)
    parser.add_argument("--posterior-dump", type=pathlib.Path)
    parser.add_argument("--decoder-replay", type=pathlib.Path)
    parser.add_argument("--posterior-cache", type=pathlib.Path)
    parser.add_argument(
        "--defer-qualification",
        action="store_true",
        help="stop after development candidate selection; later staged jobs own qualification",
    )
    parser.add_argument(
        "--compact-log",
        action="store_true",
        help="emit compact round/final summaries instead of the full manifest on stdout",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("runtime runner does not exist")
    work = safe_reset(args.work_dir or pathlib.Path(cfg.get("domain_work_dir", "build/domain-loop")))
    posterior_replay = resolve_posterior_replay(
        args.posterior_dump,
        args.decoder_replay,
        args.posterior_cache,
    )
    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    iteration = cfg.get("domain_iteration", {})
    backend = str(iteration.get("backend", "prototype"))
    if backend not in {"prototype", "torch_ctc"}:
        raise ValueError("domain_iteration.backend must be prototype or torch_ctc")
    warm_start_strategy = parse_warm_start_strategy(iteration)
    base_failure_replay_enabled = iteration.get("base_failure_replay_enabled", False)
    if not isinstance(base_failure_replay_enabled, bool):
        raise ValueError("domain_iteration.base_failure_replay_enabled must be boolean")
    max_rounds = int(iteration.get("max_rounds", 3))
    min_rounds = int(iteration.get("min_rounds", 2))
    patience = int(iteration.get("patience", 2))
    if not 0 < min_rounds <= max_rounds or patience < 0:
        raise ValueError("domain iteration round settings are invalid")
    frontends = cfg.get("model", {}).get("frontends", [FRONTEND_LOGMEL])
    if not isinstance(frontends, list) or not frontends or any(str(value) not in FRONTEND_IDS for value in frontends):
        raise ValueError("model.frontends must contain supported frontend names")
    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    if not thresholds or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("calibration.thresholds is invalid")
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    calibration_parallel_trials = int(
        cfg.get("calibration", {}).get("max_parallel_trials", 1)
    )
    if not 1 <= calibration_parallel_trials <= 4:
        raise ValueError("calibration.max_parallel_trials must be 1..4")
    gates = gate_values(cfg.get("domain_gates", {}))
    prototype_candidates = cfg.get("model", {}).get("prototype_candidates", [])
    if backend == "prototype" and (not isinstance(prototype_candidates, list) or not prototype_candidates):
        raise ValueError("prototype backend needs prototype_candidates")

    best = None
    records: list[dict] = []
    completed_round_best_records: list[dict] = []
    curriculum: dict | None = None
    stale = 0
    previous_checkpoints: dict[str, pathlib.Path] = {}
    for round_index in range(max_rounds):
        dataset_dir = work / "datasets" / f"round-{round_index:02d}"
        acoustic_seed_offset = train_acoustic_seed_offset(iteration, round_index)
        render_domain_dataset(
            config_path,
            dataset_dir,
            curriculum_weights=curriculum,
            splits=("train", "calibration", "test"),
            train_seed_offset=acoustic_seed_offset,
        )
        run(
            [
                sys.executable,
                str(TRAINING / "audit_dataset.py"),
                "--split",
                f"train={dataset_dir / 'train.tsv'}",
                "--split",
                f"calibration={dataset_dir / 'calibration.tsv'}",
                "--split",
                f"test={dataset_dir / 'test.tsv'}",
                "--report",
                str(dataset_dir / "audit.json"),
                "--fail-within-split",
            ]
        )
        replay = None
        base_failure_replay = None
        wake_balance = None
        if backend == "torch_ctc":
            replay = render_hard_negative_replay(
                config_path,
                work / "hard-negative-replay" / f"round-{round_index:02d}",
                round_index=round_index,
                curriculum_weights=curriculum,
            )
            if base_failure_replay_enabled and round_index > 0:
                if not completed_round_best_records:
                    raise ValueError(
                        "base failure replay requires completed prior round-best records"
                    )
                base_failure_replay = render_development_failure_replay(
                    config_path,
                    list(completed_round_best_records),
                    work,
                    work / "base-failure-replay" / f"round-{round_index:02d}",
                )
                if base_failure_replay.get("formal_qualification_used") is not False:
                    raise ValueError("base failure replay must not use formal qualification")
                source_rounds = sorted(
                    {
                        int(value)
                        for item in base_failure_replay.get("selected", [])
                        if isinstance(item, dict)
                        for value in item.get("source_rounds", [])
                    }
                )
                if source_rounds and max(source_rounds) >= round_index:
                    raise ValueError(
                        "base failure replay must use only prior development rounds"
                    )
            train_cfg = cfg.get("train", {})
            if not isinstance(train_cfg, dict):
                raise ValueError("train config must be an object")
            wake_policy = train_cfg.get("wake_pressure_balance_policy")
            if wake_policy is not None:
                if str(wake_policy) != WAKE_BALANCE_POLICY:
                    raise ValueError(
                        f"unsupported wake-pressure balance policy: {wake_policy}"
                    )
                training_manifests = [dataset_dir / "train.tsv"]
                focus_rows: dict[pathlib.Path, list[tuple[int, ...]]] = {}
                if isinstance(replay, dict) and int(replay.get("examples", 0)) > 0:
                    replay_manifest = pathlib.Path(str(replay["manifest"]))
                    training_manifests.append(replay_manifest)
                    focus_rows.update(static_replay_focus_rows(replay))
                if (
                    isinstance(base_failure_replay, dict)
                    and int(base_failure_replay.get("examples", 0)) > 0
                ):
                    failure_manifest = pathlib.Path(
                        str(base_failure_replay["manifest"])
                    )
                    training_manifests.append(failure_manifest)
                    focus_rows.update(
                        failure_replay_focus_rows(base_failure_replay)
                    )
                wake_balance = derive_wake_pressure_balance(
                    manifests=training_manifests,
                    tokens=tokens,
                    keywords=keywords,
                    positive_example_weight=float(
                        train_cfg.get(
                            "positive_example_weight",
                            DEFAULT_POSITIVE_EXAMPLE_WEIGHT,
                        )
                    ),
                    focus_rows_by_manifest=focus_rows,
                )
        round_best = None
        for frontend_value in frontends:
            frontend = str(frontend_value)
            candidates = prototype_candidates if backend == "prototype" else [{}]
            for candidate_index, params in enumerate(candidates):
                candidate_dir = work / "candidates" / f"r{round_index:02d}-{frontend}-{candidate_index:02d}"
                candidate_dir.mkdir(parents=True)
                checkpoint = None
                previous_checkpoint = previous_checkpoints.get(frontend)
                if backend == "prototype":
                    model = candidate_dir / "model.kwm"
                    fit_domain_prototype(
                        config=cfg,
                        tokens_path=tokens,
                        carriers_path=dataset_dir / "base" / "token-carriers.json",
                        output=model,
                        training_output=candidate_dir / "fit",
                        feature_dim=int(cfg.get("model", {}).get("feature_dim", 32)),
                        variants_per_token=int(cfg.get("model", {}).get("domain_variants_per_token", 12)),
                        projection_gain=float(params.get("input_scale", 0.010)) * 127.0,
                        output_scale=float(params.get("output_scale", 0.050)),
                        blank_bias=float(params.get("blank_bias", 1.8)),
                        token_bias=float(params.get("token_bias", -1.2)),
                        seed=int(cfg.get("seed", 1337)) + round_index * 1009,
                        frontend=frontend,
                        curriculum_weights=curriculum,
                    )
                    provenance = pathlib.Path(str(model) + ".synthetic-domain-provenance.json")
                else:
                    replay_manifest = (
                        pathlib.Path(str(replay["manifest"]))
                        if isinstance(replay, dict) and int(replay.get("examples", 0)) > 0
                        else None
                    )
                    model, checkpoint = build_torch(
                        cfg=cfg,
                        frontend=frontend,
                        tokens=tokens,
                        keywords=keywords,
                        manifest=dataset_dir / "train.tsv",
                        output=candidate_dir,
                        previous=previous_checkpoint,
                        hard_negative_manifest=replay_manifest,
                        failure_replay_manifest=(
                            pathlib.Path(str(base_failure_replay["manifest"]))
                            if isinstance(base_failure_replay, dict)
                            and int(base_failure_replay.get("examples", 0)) > 0
                            else None
                        ),
                        wake_balance=wake_balance,
                        warm_start_strategy=warm_start_strategy,
                        round_index=round_index,
                    )
                    provenance = pathlib.Path(str(model) + ".provenance.json")
                calibrated, pack, cal_base, cal_domains = calibrate(
                    runner=runner,
                    model=model,
                    tokens=tokens,
                    source_keywords=keywords,
                    references=dataset_dir / "calibration.references.jsonl",
                    output=candidate_dir / "calibration",
                    thresholds=thresholds,
                    rounds=coordinate_rounds,
                    gates=gates,
                    parallel_trials=calibration_parallel_trials,
                    posterior_replay=posterior_replay,
                )
                test_base, test_domains = evaluate(
                    runner=runner,
                    model=model,
                    pack=pack,
                    references=dataset_dir / "test.references.jsonl",
                    output=candidate_dir / "test",
                    posterior_replay=posterior_replay,
                )
                score_value = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)
                record = {
                    "round": round_index,
                    "frontend": frontend,
                    "candidate": candidate_index,
                    "score": score_value,
                    "model": str(model),
                    "model_sha256": sha256_file(model),
                    "provenance": str(provenance),
                    "provenance_sha256": sha256_file(provenance),
                    "keywords": str(calibrated),
                    "pack": str(pack),
                    "calibration": cal_base,
                    "calibration_domains": cal_domains,
                    "test": test_base,
                    "test_domains": test_domains,
                    "calibration_gate": base_gate(cal_base, gates) and domain_gate(cal_domains, gates),
                    "test_gate": base_gate(test_base, gates) and domain_gate(test_domains, gates),
                    "training_acoustic_seed_policy": "train-only-scene-seed-offset-v1",
                    "training_acoustic_seed_offset": acoustic_seed_offset,
                }
                if checkpoint is not None:
                    record["checkpoint"] = str(checkpoint)
                    record["warm_started"] = previous_checkpoint is not None
                    record["warm_start_strategy"] = (
                        warm_start_strategy if previous_checkpoint is not None else "cold-start"
                    )
                    epochs_used, learning_rate_used, training_seed_used = (
                        torch_round_training_values(
                            cfg,
                            round_index=round_index,
                            warm_started=previous_checkpoint is not None,
                        )
                    )
                    record["training_epochs"] = epochs_used
                    record["training_learning_rate"] = learning_rate_used
                    record["training_seed"] = training_seed_used
                    record["hard_negative_replay_examples"] = int(
                        replay.get("examples", 0) if isinstance(replay, dict) else 0
                    )
                    record["hard_negative_replay_manifest_sha256"] = (
                        str(replay.get("manifest_sha256"))
                        if isinstance(replay, dict)
                        else None
                    )
                    record["base_failure_replay_enabled"] = bool(
                        base_failure_replay_enabled
                    )
                    record["base_failure_replay_examples"] = int(
                        base_failure_replay.get("examples", 0)
                        if isinstance(base_failure_replay, dict)
                        else 0
                    )
                    record["base_failure_replay_manifest_sha256"] = (
                        str(base_failure_replay.get("manifest_sha256"))
                        if isinstance(base_failure_replay, dict)
                        else None
                    )
                    record["base_failure_replay_source_rounds"] = sorted(
                        {
                            int(value)
                            for item in (
                                base_failure_replay.get("selected", [])
                                if isinstance(base_failure_replay, dict)
                                else []
                            )
                            if isinstance(item, dict)
                            for value in item.get("source_rounds", [])
                        }
                    )
                    record["wake_balance"] = wake_balance
                records.append(record)
                if round_best is None or score_value < round_best["score"]:
                    round_best = record
                if best is None or score_value < best["score"] - 1.0e-12:
                    best = record
                    stale = 0
                else:
                    stale += 1
                if checkpoint is not None:
                    previous_checkpoints[frontend] = checkpoint
        assert round_best is not None
        completed_round_best_records.append(round_best)
        curriculum_feedback = merge_domain_metrics(
            round_best["calibration_domains"],
            round_best["test_domains"],
        )
        curriculum_result = update_curriculum(
            curriculum_feedback,
            previous=curriculum,
            strength=float(iteration.get("curriculum_strength", 2.0)),
            max_weight=float(iteration.get("max_domain_weight", 6.0)),
        )
        curriculum = curriculum_result
        curriculum_path = work / "curriculum" / f"round-{round_index:02d}.json"
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        curriculum_path.write_text(json.dumps(curriculum_result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        progress_record = build_round_progress(
            round_best,
            curriculum_sha256=sha256_file(curriculum_path),
        )
        append_round_progress(work / "domain-loop-progress.jsonl", progress_record)
        print(
            "domain-round-progress "
            + json.dumps(
                progress_record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        if (
            round_index + 1 >= min_rounds
            and select_strict_candidate(records) is not None
            and bool(iteration.get("stop_on_gate", True))
        ):
            break
        if round_index + 1 >= min_rounds and stale >= patience:
            break

    if best is None:
        raise RuntimeError("domain iteration produced no candidates")
    strict_best = select_strict_candidate(records)
    selected = strict_best or best
    eligible_rounds = sorted(
        {int(record["round"]) for record in records if strict_gate_candidate(record)}
    )
    best_dir = work / "best"
    best_dir.mkdir()
    best_model = best_dir / "model.kwm"
    best_pack = best_dir / "keywords.kwk"
    best_keywords = best_dir / "keywords.tsv"
    best_provenance = best_dir / "model-provenance.json"
    shutil.copy2(selected["model"], best_model)
    shutil.copy2(selected["pack"], best_pack)
    shutil.copy2(selected["keywords"], best_keywords)
    shutil.copy2(selected["provenance"], best_provenance)

    development_qualified = strict_best is not None
    qualification_deferred = bool(args.defer_qualification)
    if qualification_deferred:
        qualification_base: dict = {}
        qualification_domains: dict = {}
        qualification_qualified: bool | None = None
        qualified = False
        evidence_class = "synthetic-domain-development-only"
    else:
        # Standalone iteration keeps the historical qualification behavior.
        # Staged product training defers this work until after refinement so an
        # intermediate candidate is not qualified twice.
        qualification_dataset = work / "qualification-dataset"
        render_domain_dataset(
            config_path,
            qualification_dataset,
            curriculum_weights=None,
            splits=("qualification",),
        )
        qualification_base, qualification_domains = evaluate(
            runner=runner,
            model=best_model,
            pack=best_pack,
            references=qualification_dataset / "qualification.references.jsonl",
            output=best_dir / "qualification",
            posterior_replay=posterior_replay,
        )
        qualification_qualified = base_gate(qualification_base, gates) and domain_gate(
            qualification_domains, gates
        )
        qualified = development_qualified and qualification_qualified
        evidence_class = (
            "synthetic-domain-qualified" if qualified else "synthetic-domain-unqualified"
        )

    manifest = {
        "schema_version": 2,
        "evidence_class": evidence_class,
        "qualified": qualified,
        "development_qualified": development_qualified,
        "qualification_deferred": qualification_deferred,
        "qualification_qualified": qualification_qualified,
        "development_split_roles": {
            "train": "development-training",
            "calibration": "development-calibration",
            "test": "development-feedback",
        },
        "candidate_selection": {
            "policy": "latest-strict-gate-passing-round",
            "eligible_rounds": eligible_rounds,
            "qualification_used_for_selection": False,
            "selected_round": int(strict_best["round"]) if strict_best is not None else None,
            "selected_frontend": (
                str(strict_best["frontend"]) if strict_best is not None else None
            ),
            "selected_score": float(strict_best["score"]) if strict_best is not None else None,
            "objective_fallback_used": strict_best is None,
            "objective_best_round": int(best["round"]),
            "objective_best_frontend": str(best["frontend"]),
            "objective_best_score": float(best["score"]),
        },
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner),
        "best_round": selected["round"],
        "best_frontend": selected["frontend"],
        "best_score": selected["score"],
        "best_model_sha256": sha256_file(best_model),
        "best_pack_sha256": sha256_file(best_pack),
        "warm_start_strategy": warm_start_strategy if backend == "torch_ctc" else None,
        "records": records,
        "final_curriculum": curriculum or {},
        "qualification": qualification_base,
        "qualification_domains": qualification_domains,
        "gates": gates,
        "limitations": [
            "No real human speech is used in this evidence class.",
            "Simulated acoustic scenes and synthetic TTS/tone results are not production far-field qualification.",
            "The command AFE adapter must be used with the shipping audio-pipeline before product claims.",
            "Physical target-board and independent human held-out evidence remain issue #2 gates.",
        ],
    }
    manifest_path = work / "domain-loop-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if args.compact_log:
        print(
            "domain-loop-final "
            + json.dumps(
                {
                    "schema_version": 1,
                    "manifest": str(manifest_path),
                    "record_count": len(records),
                    "development_qualified": development_qualified,
                    "qualification_deferred": qualification_deferred,
                    "selected_round": selected["round"],
                    "selected_frontend": selected["frontend"],
                    "selected_score": selected["score"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    else:
        print(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
    if qualification_deferred:
        return 0 if development_qualified else 1
    return 0 if qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)