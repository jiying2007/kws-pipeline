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
TOOLS = ROOT / "tools"
EVAL = ROOT / "eval"
TRAINING = ROOT / "training"
sys.path.insert(0, str(TOOLS))

from domain_curriculum import update_curriculum  # noqa: E402
from fit_domain_prototype import fit_domain_prototype  # noqa: E402
from frontend_spec import FRONTEND_IDS, FRONTEND_LOGMEL  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
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
) -> tuple[dict, dict]:
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
            str(pack),
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--provenance",
            str(provenance),
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
) -> tuple[pathlib.Path, pathlib.Path, dict, dict]:
    current = keyword_rows(source_keywords)
    for coordinate in range(rounds):
        changed = False
        for index, row in enumerate(current):
            best = None
            for threshold in thresholds:
                trial = [dict(item) for item in current]
                trial[index]["threshold"] = threshold
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
                )
                key = (objective(base, domains, gates), float(base["frr"]), -threshold)
                if best is None or key < best[0]:
                    best = (key, threshold)
            assert best is not None
            if not math.isclose(float(current[index]["threshold"]), best[1]):
                changed = True
            current[index]["threshold"] = best[1]
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
    )
    return tsv, pack, base, domains


def build_torch(
    *,
    cfg: dict,
    frontend: str,
    tokens: pathlib.Path,
    manifest: pathlib.Path,
    output: pathlib.Path,
    previous: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    checkpoint = output / "model.pt"
    model = output / "model.kwm"
    train = cfg.get("train", {})
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(manifest),
        "--tokens",
        str(tokens),
        "--frontend",
        frontend,
        "--feature-dim",
        str(int(cfg.get("model", {}).get("feature_dim", 32))),
        "--hidden-dim",
        str(int(cfg.get("model", {}).get("hidden_dim", 48))),
        "--epochs",
        str(int(train.get("epochs", 10))),
        "--batch-size",
        str(int(train.get("batch_size", 16))),
        "--lr",
        str(float(train.get("lr", 0.001))),
        "--seed",
        str(int(cfg.get("seed", 1337))),
        "--output",
        str(checkpoint),
    ]
    if previous is not None:
        command.extend(["--warm-start", str(previous), "--head-only"])
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
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("runtime runner does not exist")
    work = safe_reset(args.work_dir or pathlib.Path(cfg.get("domain_work_dir", "build/domain-loop")))
    tokens = repo_path(str(cfg["tokens"]))
    keywords = repo_path(str(cfg["keywords"]))
    iteration = cfg.get("domain_iteration", {})
    backend = str(iteration.get("backend", "prototype"))
    if backend not in {"prototype", "torch_ctc"}:
        raise ValueError("domain_iteration.backend must be prototype or torch_ctc")
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
    gates = gate_values(cfg.get("domain_gates", {}))
    prototype_candidates = cfg.get("model", {}).get("prototype_candidates", [])
    if backend == "prototype" and (not isinstance(prototype_candidates, list) or not prototype_candidates):
        raise ValueError("prototype backend needs prototype_candidates")

    best = None
    records: list[dict] = []
    curriculum: dict[str, float] | None = None
    stale = 0
    previous_checkpoints: dict[str, pathlib.Path] = {}
    for round_index in range(max_rounds):
        dataset_dir = work / "datasets" / f"round-{round_index:02d}"
        render_domain_dataset(config_path, dataset_dir, curriculum_weights=curriculum)
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
                "--split",
                f"qualification={dataset_dir / 'qualification.tsv'}",
                "--report",
                str(dataset_dir / "audit.json"),
                "--fail-within-split",
            ]
        )
        round_best = None
        for frontend_value in frontends:
            frontend = str(frontend_value)
            candidates = prototype_candidates if backend == "prototype" else [{}]
            for candidate_index, params in enumerate(candidates):
                candidate_dir = work / "candidates" / f"r{round_index:02d}-{frontend}-{candidate_index:02d}"
                candidate_dir.mkdir(parents=True)
                checkpoint = None
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
                    model, checkpoint = build_torch(
                        cfg=cfg,
                        frontend=frontend,
                        tokens=tokens,
                        manifest=dataset_dir / "train.tsv",
                        output=candidate_dir,
                        previous=previous_checkpoints.get(frontend),
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
                )
                test_base, test_domains = evaluate(
                    runner=runner,
                    model=model,
                    pack=pack,
                    references=dataset_dir / "test.references.jsonl",
                    output=candidate_dir / "test",
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
                }
                if checkpoint is not None:
                    record["checkpoint"] = str(checkpoint)
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
        curriculum_result = update_curriculum(
            round_best["calibration_domains"],
            previous=curriculum,
            strength=float(iteration.get("curriculum_strength", 2.0)),
            max_weight=float(iteration.get("max_domain_weight", 6.0)),
        )
        curriculum = curriculum_result["distance_band_weights"]
        curriculum_path = work / "curriculum" / f"round-{round_index:02d}.json"
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        curriculum_path.write_text(json.dumps(curriculum_result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        if (
            round_index + 1 >= min_rounds
            and best is not None
            and best["calibration_gate"]
            and best["test_gate"]
            and bool(iteration.get("stop_on_gate", True))
        ):
            break
        if round_index + 1 >= min_rounds and stale >= patience:
            break

    if best is None:
        raise RuntimeError("domain iteration produced no candidates")
    best_dir = work / "best"
    best_dir.mkdir()
    best_model = best_dir / "model.kwm"
    best_pack = best_dir / "keywords.kwk"
    best_keywords = best_dir / "keywords.tsv"
    best_provenance = best_dir / "model-provenance.json"
    shutil.copy2(best["model"], best_model)
    shutil.copy2(best["pack"], best_pack)
    shutil.copy2(best["keywords"], best_keywords)
    shutil.copy2(best["provenance"], best_provenance)

    # Qualification is regenerated from the same pinned config but never used by
    # candidate selection or curriculum updates.
    qualification_dataset = work / "qualification-dataset"
    render_domain_dataset(config_path, qualification_dataset, curriculum_weights=None)
    qualification_base, qualification_domains = evaluate(
        runner=runner,
        model=best_model,
        pack=best_pack,
        references=qualification_dataset / "qualification.references.jsonl",
        output=best_dir / "qualification",
    )
    qualified = base_gate(qualification_base, gates) and domain_gate(qualification_domains, gates)
    manifest = {
        "schema_version": 1,
        "evidence_class": "synthetic-domain-qualified",
        "qualified": qualified,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner),
        "best_round": best["round"],
        "best_frontend": best["frontend"],
        "best_score": best["score"],
        "best_model_sha256": sha256_file(best_model),
        "best_pack_sha256": sha256_file(best_pack),
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
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
