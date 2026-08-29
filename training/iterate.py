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
from kws_vocab import load_tokens, vocab_size  # noqa: E402

from prototype_model import build_prototype  # noqa: E402
from synthetic_audio import generate_dataset, load_config, parse_keywords  # noqa: E402


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(argv)}"
        )


def resolve_repo_path(value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def safe_reset_workdir(path: pathlib.Path) -> pathlib.Path:
    work = path.resolve()
    home = pathlib.Path.home().resolve()
    anchor = pathlib.Path(work.anchor).resolve()
    forbidden = {anchor, home, ROOT.resolve(), ROOT.parent.resolve()}
    if work in forbidden or len(work.parts) < 3:
        raise ValueError(f"refusing unsafe synthetic work directory: {work}")
    if work.exists() and not work.is_dir():
        raise ValueError(f"synthetic work path is not a directory: {work}")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    return work


def keyword_rows(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise ValueError(f"{path}:{line_no}: expected 4 TSV columns")
        threshold = float(cols[2])
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError(f"{path}:{line_no}: threshold must be finite and in (0,1)")
        rows.append(
            {
                "id": int(cols[0]),
                "text": cols[1].strip(),
                "threshold": threshold,
                "tokens": cols[3].strip(),
            }
        )
    if not rows:
        raise ValueError(f"{path}: no keyword rows")
    return rows


def write_keyword_tsv(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# id\ttext\tthreshold\texplicit-pinyin-tokens"]
    for row in rows:
        lines.append(
            f"{int(row['id'])}\t{row['text']}\t"
            f"{float(row['threshold']):.6f}\t{row['tokens']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_pack(
    tokens: pathlib.Path, keyword_tsv: pathlib.Path, pack: pathlib.Path
) -> None:
    run(
        [
            sys.executable,
            str(TOOLS / "compile_keywords.py"),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keyword_tsv),
            "--out-pack",
            str(pack),
        ]
    )


def evaluate(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    pack: pathlib.Path,
    references: pathlib.Path,
    audio_root: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    detections = output / "detections.jsonl"
    provenance = output / "detections.provenance.json"
    summary = output / "summary.json"
    false_positives = output / "false-positives.jsonl"
    false_rejects = output / "false-rejects.jsonl"
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
            "--audio-root",
            str(audio_root),
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
    result = json.loads(summary.read_text(encoding="utf-8"))
    result["summary_sha256"] = sha256_file(summary)
    result["false_positives_path"] = str(false_positives)
    result["false_rejects_path"] = str(false_rejects)
    return result


def validate_gates(raw: dict) -> dict[str, float]:
    required = ("max_frr", "max_far_per_hour", "max_p95_latency_ms")
    if not isinstance(raw, dict) or any(key not in raw for key in required):
        raise ValueError("synthetic_gates is incomplete")
    gates = {key: float(raw[key]) for key in required}
    if any(not math.isfinite(value) or value < 0.0 for value in gates.values()):
        raise ValueError("synthetic gate values must be finite and >= 0")
    if gates["max_frr"] > 1.0:
        raise ValueError("synthetic max_frr must be <= 1")
    return gates


def gate_ok(metrics: dict, gates: dict) -> bool:
    return (
        float(metrics["frr"]) <= float(gates["max_frr"])
        and float(metrics["far_per_hour"]) <= float(gates["max_far_per_hour"])
        and float(metrics["p95_post_end_latency_ms"])
        <= float(gates["max_p95_latency_ms"])
    )


def metric_score(metrics: dict, gates: dict) -> float:
    frr = float(metrics["frr"])
    far = float(metrics["far_per_hour"])
    latency = float(metrics["p95_post_end_latency_ms"])
    violation = 0.0
    violation += max(0.0, frr - float(gates["max_frr"])) * 10000.0
    violation += max(0.0, far - float(gates["max_far_per_hour"])) * 100.0
    violation += max(0.0, latency - float(gates["max_p95_latency_ms"])) * 0.1
    return violation + frr * 100.0 + far * 0.1 + latency * 0.001


def candidate_score(calibration: dict, test: dict, gates: dict) -> tuple[float, float, float]:
    calibration_score = metric_score(calibration, gates)
    test_score = metric_score(test, gates)
    return calibration_score + test_score, calibration_score, test_score


def calibrate_thresholds(
    *,
    runner: pathlib.Path,
    model: pathlib.Path,
    tokens: pathlib.Path,
    source_keywords: pathlib.Path,
    references: pathlib.Path,
    audio_root: pathlib.Path,
    output: pathlib.Path,
    thresholds: list[float],
    coordinate_rounds: int,
    gates: dict,
) -> tuple[pathlib.Path, pathlib.Path, dict]:
    rows = keyword_rows(source_keywords)
    current = [dict(row) for row in rows]
    for round_index in range(coordinate_rounds):
        changed = False
        for row_index, row in enumerate(current):
            best: tuple[tuple[float, float, float], float, dict] | None = None
            for threshold in thresholds:
                trial = [dict(item) for item in current]
                trial[row_index]["threshold"] = threshold
                trial_dir = (
                    output
                    / f"coord{round_index}-kw{row['id']}-t{threshold:.3f}"
                )
                tsv = trial_dir / "keywords.tsv"
                pack = trial_dir / "keywords.kwk"
                write_keyword_tsv(trial, tsv)
                compile_pack(tokens, tsv, pack)
                metrics = evaluate(
                    runner=runner,
                    model=model,
                    pack=pack,
                    references=references,
                    audio_root=audio_root,
                    output=trial_dir / "eval",
                )
                key = (
                    metric_score(metrics, gates),
                    float(metrics["frr"]),
                    -threshold,
                )
                if best is None or key < best[0]:
                    best = (key, threshold, metrics)
            assert best is not None
            if not math.isclose(float(current[row_index]["threshold"]), best[1]):
                changed = True
            current[row_index]["threshold"] = best[1]
        if not changed:
            break

    final_tsv = output / "calibrated-keywords.tsv"
    final_pack = output / "calibrated-keywords.kwk"
    write_keyword_tsv(current, final_tsv)
    compile_pack(tokens, final_tsv, final_pack)
    metrics = evaluate(
        runner=runner,
        model=model,
        pack=final_pack,
        references=references,
        audio_root=audio_root,
        output=output / "final-eval",
    )
    return final_tsv, final_pack, metrics


def nonempty(path: pathlib.Path | None) -> bool:
    return path is not None and path.is_file() and path.stat().st_size > 0


def build_torch_candidate(
    *,
    cfg: dict,
    tokens: pathlib.Path,
    train_manifest: pathlib.Path,
    candidate_dir: pathlib.Path,
    previous_checkpoint: pathlib.Path | None,
    hard_negative_manifest: pathlib.Path | None,
    positive_replay_manifest: pathlib.Path | None,
) -> tuple[pathlib.Path, pathlib.Path]:
    checkpoint = candidate_dir / "model.pt"
    model = candidate_dir / "model.kwm"
    train_cfg = cfg.get("train", {})
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(train_manifest),
        "--tokens",
        str(tokens),
        "--feature-dim",
        str(int(cfg.get("model", {}).get("feature_dim", 32))),
        "--hidden-dim",
        str(int(cfg.get("model", {}).get("hidden_dim", 48))),
        "--epochs",
        str(int(train_cfg.get("epochs", 10))),
        "--batch-size",
        str(int(train_cfg.get("batch_size", 16))),
        "--lr",
        str(float(train_cfg.get("lr", 0.001))),
        "--seed",
        str(int(cfg.get("seed", 1337))),
        "--output",
        str(checkpoint),
    ]
    if nonempty(hard_negative_manifest):
        command.extend(["--manifest", str(hard_negative_manifest)])
    if nonempty(positive_replay_manifest):
        command.extend(["--manifest", str(positive_replay_manifest)])
    if previous_checkpoint is not None:
        command.extend(["--warm-start", str(previous_checkpoint), "--head-only"])
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


def model_provenance_path(model: pathlib.Path, backend: str) -> pathlib.Path:
    suffix = ".synthetic-provenance.json" if backend == "prototype" else ".provenance.json"
    provenance = pathlib.Path(str(model) + suffix)
    if not provenance.is_file():
        raise RuntimeError(f"model provenance was not produced: {provenance}")
    return provenance


def mine_hard_negatives(
    false_positives: pathlib.Path,
    audio_root: pathlib.Path,
    output: pathlib.Path,
) -> pathlib.Path:
    manifest = output / "hard-negatives.tsv"
    output_dir = output / "wav"
    run(
        [
            sys.executable,
            str(EVAL / "mine_hard_negatives.py"),
            "--false-positives",
            str(false_positives),
            "--audio-root",
            str(audio_root),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
        ]
    )
    return manifest


def mine_false_rejects(
    false_rejects: pathlib.Path,
    keywords: pathlib.Path,
    tokens: pathlib.Path,
    audio_root: pathlib.Path,
    output: pathlib.Path,
) -> pathlib.Path:
    manifest = output / "missed-positives.tsv"
    output_dir = output / "wav"
    run(
        [
            sys.executable,
            str(EVAL / "mine_false_rejects.py"),
            "--false-rejects",
            str(false_rejects),
            "--keywords",
            str(keywords),
            "--tokens",
            str(tokens),
            "--audio-root",
            str(audio_root),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
        ]
    )
    return manifest


def merge_manifests(destination: pathlib.Path, sources: list[pathlib.Path]) -> None:
    unique: list[str] = []
    seen: set[str] = set()
    for source in sources:
        if not source.is_file():
            continue
        for raw in source.read_text(encoding="utf-8").splitlines():
            line = raw.strip("\n")
            if not line or line in seen:
                continue
            seen.add(line)
            unique.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(unique) + ("\n" if unique else ""), encoding="utf-8"
    )


def copy_best_artifacts(best: dict, best_dir: pathlib.Path) -> dict[str, pathlib.Path | None]:
    best_dir.mkdir()
    artifacts: dict[str, pathlib.Path | None] = {
        "model": best_dir / "model.kwm",
        "pack": best_dir / "keywords.kwk",
        "keywords": best_dir / "keywords.tsv",
        "model_provenance": best_dir / "model-provenance.json",
        "checkpoint": None,
    }
    shutil.copy2(best["model"], artifacts["model"])
    shutil.copy2(best["pack"], artifacts["pack"])
    shutil.copy2(best["keywords"], artifacts["keywords"])
    shutil.copy2(best["model_provenance"], artifacts["model_provenance"])
    if best.get("checkpoint"):
        checkpoint = best_dir / "model.pt"
        shutil.copy2(best["checkpoint"], checkpoint)
        artifacts["checkpoint"] = checkpoint
    return artifacts


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
        raise ValueError(f"runtime runner does not exist: {runner}")
    requested_work = args.work_dir or pathlib.Path(
        cfg.get("work_dir", "build/synthetic-loop")
    )
    work = safe_reset_workdir(requested_work)

    tokens = resolve_repo_path(str(cfg["tokens"]))
    keywords = resolve_repo_path(str(cfg["keywords"]))
    dataset_dir = work / "dataset"
    generate_dataset(config_path, dataset_dir)

    audit_args = [sys.executable, str(TRAINING / "audit_dataset.py")]
    for split in ("train", "calibration", "test", "qualification"):
        audit_args.extend(
            ["--split", f"{split}={dataset_dir / (split + '.tsv')}"]
        )
    audit_report = work / "dataset-audit.json"
    audit_args.extend(["--report", str(audit_report), "--fail-within-split"])
    run(audit_args)

    token_map = load_tokens(tokens)
    parsed_keywords = parse_keywords(keywords, token_map)
    configured_ids = {
        value for item in parsed_keywords for value in item["token_ids"]
    }
    if not configured_ids or max(configured_ids) >= vocab_size(token_map):
        raise ValueError("keyword token IDs are empty or exceed vocabulary")

    iteration_cfg = cfg.get("iteration", {})
    backend = str(iteration_cfg.get("backend", "prototype"))
    if backend not in {"prototype", "torch_ctc"}:
        raise ValueError(f"unsupported iteration backend: {backend}")
    max_rounds = int(iteration_cfg.get("max_rounds", 3))
    min_rounds = int(iteration_cfg.get("min_rounds", 2))
    patience = int(iteration_cfg.get("patience", 2))
    if max_rounds <= 0 or min_rounds <= 0 or min_rounds > max_rounds:
        raise ValueError("iteration rounds must satisfy 0 < min_rounds <= max_rounds")
    if patience < 0:
        raise ValueError("iteration patience must be >= 0")

    thresholds = [
        float(value)
        for value in cfg.get("calibration", {}).get("thresholds", [])
    ]
    if not thresholds or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in thresholds
    ):
        raise ValueError("calibration.thresholds must contain finite values in (0,1)")
    coordinate_rounds = int(
        cfg.get("calibration", {}).get("coordinate_rounds", 1)
    )
    if coordinate_rounds <= 0:
        raise ValueError("calibration.coordinate_rounds must be > 0")
    gates = validate_gates(cfg.get("synthetic_gates", {}))

    prototype_candidates = cfg.get("model", {}).get("prototype_candidates", [])
    if backend == "prototype" and (
        not isinstance(prototype_candidates, list) or not prototype_candidates
    ):
        raise ValueError("prototype backend requires model.prototype_candidates")

    rounds: list[dict] = []
    best: dict | None = None
    stale = 0
    previous_checkpoint: pathlib.Path | None = None
    cumulative_hard_negatives = work / "replay" / "hard-negatives.tsv"
    cumulative_missed_positives = work / "replay" / "missed-positives.tsv"
    hard_negative_sources: list[pathlib.Path] = []
    positive_replay_sources: list[pathlib.Path] = []

    for round_index in range(max_rounds):
        candidate_dir = work / "candidates" / f"round-{round_index:02d}"
        candidate_dir.mkdir(parents=True)
        checkpoint: pathlib.Path | None = None
        if backend == "prototype":
            params = prototype_candidates[
                min(round_index, len(prototype_candidates) - 1)
            ]
            if not isinstance(params, dict):
                raise ValueError("prototype candidate must be an object")
            model = candidate_dir / "model.kwm"
            build_prototype(
                tokens_path=tokens,
                carriers_path=dataset_dir / "token-carriers.json",
                output=model,
                feature_dim=int(cfg.get("model", {}).get("feature_dim", 32)),
                input_scale=float(params.get("input_scale", 0.010)),
                output_scale=float(params.get("output_scale", 0.050)),
                blank_bias=float(params.get("blank_bias", 1.8)),
                token_bias=float(params.get("token_bias", -1.2)),
            )
        else:
            model, checkpoint = build_torch_candidate(
                cfg=cfg,
                tokens=tokens,
                train_manifest=dataset_dir / "train.tsv",
                candidate_dir=candidate_dir,
                previous_checkpoint=previous_checkpoint,
                hard_negative_manifest=cumulative_hard_negatives,
                positive_replay_manifest=cumulative_missed_positives,
            )
        provenance = model_provenance_path(model, backend)

        calibrated_tsv, pack, calibration_metrics = calibrate_thresholds(
            runner=runner,
            model=model,
            tokens=tokens,
            source_keywords=keywords,
            references=dataset_dir / "calibration.references.jsonl",
            audio_root=dataset_dir,
            output=candidate_dir / "calibration",
            thresholds=thresholds,
            coordinate_rounds=coordinate_rounds,
            gates=gates,
        )
        test_metrics = evaluate(
            runner=runner,
            model=model,
            pack=pack,
            references=dataset_dir / "test.references.jsonl",
            audio_root=dataset_dir,
            output=candidate_dir / "test",
        )
        score, calibration_score, test_score = candidate_score(
            calibration_metrics, test_metrics, gates
        )
        record = {
            "round": round_index,
            "model": str(model),
            "model_sha256": sha256_file(model),
            "model_provenance": str(provenance),
            "model_provenance_sha256": sha256_file(provenance),
            "pack": str(pack),
            "pack_sha256": sha256_file(pack),
            "keywords": str(calibrated_tsv),
            "calibration": calibration_metrics,
            "test": test_metrics,
            "score": score,
            "calibration_score": calibration_score,
            "test_score": test_score,
            "calibration_gate": gate_ok(calibration_metrics, gates),
            "test_gate": gate_ok(test_metrics, gates),
        }
        if checkpoint is not None:
            record["checkpoint"] = str(checkpoint)
            record["checkpoint_sha256"] = sha256_file(checkpoint)
        rounds.append(record)

        improved = best is None or score < float(best["score"]) - 1.0e-12
        if improved:
            best = record
            stale = 0
        else:
            stale += 1

        hard_negative_sources.append(
            mine_hard_negatives(
                pathlib.Path(calibration_metrics["false_positives_path"]),
                dataset_dir,
                candidate_dir / "mined-hard-negatives",
            )
        )
        positive_replay_sources.append(
            mine_false_rejects(
                pathlib.Path(calibration_metrics["false_rejects_path"]),
                calibrated_tsv,
                tokens,
                dataset_dir,
                candidate_dir / "mined-missed-positives",
            )
        )
        merge_manifests(cumulative_hard_negatives, hard_negative_sources)
        merge_manifests(cumulative_missed_positives, positive_replay_sources)
        record["cumulative_hard_negatives"] = sum(
            1
            for line in cumulative_hard_negatives.read_text(encoding="utf-8").splitlines()
            if line
        )
        record["cumulative_missed_positives"] = sum(
            1
            for line in cumulative_missed_positives.read_text(encoding="utf-8").splitlines()
            if line
        )
        if checkpoint is not None:
            previous_checkpoint = checkpoint

        if (
            round_index + 1 >= min_rounds
            and best is not None
            and bool(best["calibration_gate"])
            and bool(best["test_gate"])
            and bool(iteration_cfg.get("stop_on_gate", True))
        ):
            break
        if round_index + 1 >= min_rounds and stale >= patience:
            break

    if best is None:
        raise RuntimeError("iteration produced no candidate")

    best_dir = work / "best"
    artifacts = copy_best_artifacts(best, best_dir)
    best_model = artifacts["model"]
    best_pack = artifacts["pack"]
    best_keywords = artifacts["keywords"]
    best_provenance = artifacts["model_provenance"]
    best_checkpoint = artifacts["checkpoint"]
    assert best_model is not None
    assert best_pack is not None
    assert best_keywords is not None
    assert best_provenance is not None

    qualification_metrics = evaluate(
        runner=runner,
        model=best_model,
        pack=best_pack,
        references=dataset_dir / "qualification.references.jsonl",
        audio_root=dataset_dir,
        output=best_dir / "synthetic-qualification",
    )
    qualified = gate_ok(qualification_metrics, gates)
    manifest = {
        "schema_version": 2,
        "evidence_class": "synthetic-only",
        "qualified": qualified,
        "name": str(cfg.get("name", "synthetic-loop")),
        "backend": backend,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner),
        "dataset_summary_sha256": sha256_file(
            dataset_dir / "dataset-summary.json"
        ),
        "dataset_audit_sha256": sha256_file(audit_report),
        "best_round": int(best["round"]),
        "best_score": float(best["score"]),
        "best_model_sha256": sha256_file(best_model),
        "best_model_provenance_sha256": sha256_file(best_provenance),
        "best_pack_sha256": sha256_file(best_pack),
        "best_keywords_sha256": sha256_file(best_keywords),
        "best_checkpoint_sha256": (
            sha256_file(best_checkpoint) if best_checkpoint is not None else None
        ),
        "rounds": rounds,
        "replay": {
            "hard_negative_count": sum(
                1
                for line in cumulative_hard_negatives.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ),
            "missed_positive_count": sum(
                1
                for line in cumulative_missed_positives.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ),
            "hard_negatives_sha256": sha256_file(cumulative_hard_negatives),
            "missed_positives_sha256": sha256_file(cumulative_missed_positives),
        },
        "synthetic_qualification": qualification_metrics,
        "gates": gates,
        "limitations": [
            "No real human voice data is included in this evidence class.",
            "Prototype/tone or synthetic TTS results are not shipping Mandarin acoustic evidence.",
            "Real held-out human speech and target-board qualification remain separate release gates.",
        ],
    }
    manifest_path = work / "synthetic-loop-manifest.json"
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
