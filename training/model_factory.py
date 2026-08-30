#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(ROOT / "tools"))

from domain_curriculum import update_curriculum  # noqa: E402
from iterate import merge_manifests, mine_false_rejects, mine_hard_negatives  # noqa: E402
from iterate_domain import (  # noqa: E402
    base_gate,
    calibrate,
    domain_gate,
    evaluate,
    gate_values,
    keyword_rows,
    objective,
    repo_path,
    safe_reset,
    sha256_file,
)
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402


def run(argv: list[str]) -> None:
    completed = subprocess.run(argv, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(argv)}")


def nonempty(path: pathlib.Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def trim_manifest(path: pathlib.Path, maximum: int) -> None:
    if maximum <= 0 or not path.is_file():
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) > maximum:
        lines = lines[-maximum:]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def line_count(path: pathlib.Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)


def write_replay_keywords(source: pathlib.Path, output: pathlib.Path) -> None:
    rows = keyword_rows(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "# id\ttext\tthreshold\texplicit-pinyin-tokens\n"
        + "".join(
            f"{int(row['id'])}\t{row['text']}\t{float(row['threshold']):.6f}\t{row['tokens']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def training_keyword_rows(path: pathlib.Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise ValueError(f"{path}:{line_no}: expected exactly 4 training keyword columns")
        rows.append((int(cols[0]), cols[3].strip()))
    if not rows:
        raise ValueError("training keyword TSV is empty")
    return rows


def validate_keyword_alignment(training: pathlib.Path, runtime: pathlib.Path) -> None:
    train = training_keyword_rows(training)
    run_rows = [(int(row["id"]), str(row["tokens"])) for row in keyword_rows(runtime)]
    if train != run_rows:
        raise ValueError("training and runtime keyword IDs/token paths must match exactly")


def audit_dataset(dataset_dir: pathlib.Path) -> None:
    command = [sys.executable, str(TRAINING / "audit_dataset.py")]
    for split in ("train", "calibration", "test", "qualification"):
        command.extend(["--split", f"{split}={dataset_dir / (split + '.tsv')}"])
    command.extend([
        "--report",
        str(dataset_dir / "audit.json"),
        "--fail-within-split",
    ])
    run(command)


def build_torch_candidate(
    *,
    cfg: dict,
    frontend: str,
    tokens: pathlib.Path,
    train_manifest: pathlib.Path,
    hard_negatives: pathlib.Path,
    missed_positives: pathlib.Path,
    output: pathlib.Path,
    previous: pathlib.Path | None,
    round_index: int,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.pt"
    model = output / "model.kwm"
    train = cfg.get("train", {})
    initial = previous is None
    epochs = int(train.get("initial_epochs" if initial else "finetune_epochs", 12 if initial else 6))
    lr = float(train.get("initial_lr" if initial else "finetune_lr", 0.001 if initial else 0.0003))
    if epochs <= 0 or not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("training epochs/lr are invalid")
    command = [
        sys.executable,
        str(TRAINING / "train_ctc.py"),
        "--manifest",
        str(train_manifest),
    ]
    for replay in (hard_negatives, missed_positives):
        if nonempty(replay):
            command.extend(["--manifest", str(replay)])
    command.extend([
        "--tokens",
        str(tokens),
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
        str(lr),
        "--seed",
        str(int(cfg.get("seed", 1337)) + round_index * 1009),
        "--require-container-digest",
        "--output",
        str(checkpoint),
    ])
    if previous is not None:
        command.extend(["--warm-start", str(previous)])
        mode = str(cfg.get("model_factory", {}).get("warm_start_mode", "full"))
        if mode == "head_only":
            command.append("--head-only")
        elif mode != "full":
            raise ValueError("model_factory.warm_start_mode must be full or head_only")
    run(command)
    run([
        sys.executable,
        str(TRAINING / "export_model.py"),
        "--checkpoint",
        str(checkpoint),
        "--tokens",
        str(tokens),
        "--output",
        str(model),
    ])
    provenance = pathlib.Path(str(model) + ".provenance.json")
    if not provenance.is_file():
        raise RuntimeError("exported model provenance is missing")
    return model, checkpoint, provenance


def copy_best(best: dict, destination: pathlib.Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    mapping = {
        pathlib.Path(best["model"]): destination / "model.kwm",
        pathlib.Path(best["checkpoint"]): destination / "model.pt",
        pathlib.Path(best["provenance"]): destination / "model-provenance.json",
        pathlib.Path(best["pack"]): destination / "keywords.kwk",
        pathlib.Path(best["keywords"]): destination / "keywords.tsv",
    }
    for source, target in mapping.items():
        shutil.copy2(source, target)
    sums = []
    for path in sorted(destination.iterdir(), key=lambda item: item.name):
        if path.is_file():
            sums.append(f"{sha256_file(path)}  {path.name}")
    (destination / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = load_config(config_path)
    runner = args.runner.resolve()
    if not runner.is_file():
        raise ValueError("runtime runner does not exist")
    image_digest = os.environ.get("KWS_TRAINING_IMAGE_DIGEST", "")
    if not image_digest.startswith("sha256:") or len(image_digest) != 71:
        raise ValueError("model factory requires KWS_TRAINING_IMAGE_DIGEST=sha256:<64 hex>")

    work = safe_reset(args.work_dir)
    tokens = repo_path(str(cfg["tokens"]))
    training_keywords = repo_path(str(cfg["keywords"]))
    runtime_keywords = repo_path(str(cfg.get("runtime_keywords", cfg["keywords"])))
    validate_keyword_alignment(training_keywords, runtime_keywords)

    factory = cfg.get("model_factory", {})
    max_rounds = int(factory.get("max_rounds", 4))
    min_rounds = int(factory.get("min_rounds", 3))
    patience = int(factory.get("patience", 2))
    if not 0 < min_rounds <= max_rounds or patience < 0:
        raise ValueError("invalid model factory round settings")
    frontends = [str(value) for value in cfg.get("model", {}).get("frontends", ["logmel"])]
    if not frontends:
        raise ValueError("model.frontends must be non-empty")
    thresholds = [float(value) for value in cfg.get("calibration", {}).get("thresholds", [])]
    coordinate_rounds = int(cfg.get("calibration", {}).get("coordinate_rounds", 1))
    if not thresholds or coordinate_rounds <= 0:
        raise ValueError("calibration configuration is invalid")
    gates = gate_values(cfg.get("domain_gates", {}))
    replay_cfg = factory.get("replay", {})
    max_hard = int(replay_cfg.get("max_hard_negatives", 256))
    max_missed = int(replay_cfg.get("max_missed_positives", 256))

    cumulative_hard = work / "replay" / "hard-negatives.tsv"
    cumulative_missed = work / "replay" / "missed-positives.tsv"
    hard_sources: list[pathlib.Path] = []
    missed_sources: list[pathlib.Path] = []
    previous_checkpoints: dict[str, pathlib.Path] = {}
    curriculum: dict | None = None
    records: list[dict] = []
    best: dict | None = None
    stale_rounds = 0

    for round_index in range(max_rounds):
        dataset_dir = work / "datasets" / f"round-{round_index:02d}"
        render_domain_dataset(config_path, dataset_dir, curriculum_weights=curriculum)
        audit_dataset(dataset_dir)
        round_records: list[dict] = []
        prior_best_score = float(best["score"]) if best is not None else math.inf

        for frontend in frontends:
            candidate_dir = work / "candidates" / f"r{round_index:02d}-{frontend}"
            model, checkpoint, provenance = build_torch_candidate(
                cfg=cfg,
                frontend=frontend,
                tokens=tokens,
                train_manifest=dataset_dir / "train.tsv",
                hard_negatives=cumulative_hard,
                missed_positives=cumulative_missed,
                output=candidate_dir,
                previous=previous_checkpoints.get(frontend),
                round_index=round_index,
            )
            calibrated, pack, cal_base, cal_domains = calibrate(
                runner=runner,
                model=model,
                tokens=tokens,
                source_keywords=runtime_keywords,
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
            score = objective(cal_base, cal_domains, gates) + objective(test_base, test_domains, gates)
            record = {
                "round": round_index,
                "frontend": frontend,
                "score": score,
                "candidate_dir": str(candidate_dir),
                "model": str(model),
                "model_sha256": sha256_file(model),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "provenance": str(provenance),
                "provenance_sha256": sha256_file(provenance),
                "keywords": str(calibrated),
                "pack": str(pack),
                "pack_sha256": sha256_file(pack),
                "calibration": cal_base,
                "calibration_domains": cal_domains,
                "test": test_base,
                "test_domains": test_domains,
                "calibration_gate": base_gate(cal_base, gates) and domain_gate(cal_domains, gates),
                "test_gate": base_gate(test_base, gates) and domain_gate(test_domains, gates),
                "replay_before": {
                    "hard_negatives": line_count(cumulative_hard),
                    "missed_positives": line_count(cumulative_missed),
                },
            }
            records.append(record)
            round_records.append(record)
            previous_checkpoints[frontend] = checkpoint
            if best is None or score < float(best["score"]) - 1.0e-12:
                best = record

        round_best = min(round_records, key=lambda item: float(item["score"]))
        replay_dir = pathlib.Path(round_best["candidate_dir"]) / "replay-mining"
        hard_sources.append(
            mine_hard_negatives(
                pathlib.Path(round_best["calibration"]["false_positives_path"]),
                dataset_dir / "clips" / "calibration",
                replay_dir / "hard-negatives",
            )
        )
        replay_keywords = replay_dir / "replay-keywords.tsv"
        write_replay_keywords(pathlib.Path(round_best["keywords"]), replay_keywords)
        missed_sources.append(
            mine_false_rejects(
                pathlib.Path(round_best["calibration"]["false_rejects_path"]),
                replay_keywords,
                tokens,
                dataset_dir / "clips" / "calibration",
                replay_dir / "missed-positives",
            )
        )
        merge_manifests(cumulative_hard, hard_sources)
        merge_manifests(cumulative_missed, missed_sources)
        trim_manifest(cumulative_hard, max_hard)
        trim_manifest(cumulative_missed, max_missed)
        round_best["replay_after"] = {
            "hard_negatives": line_count(cumulative_hard),
            "missed_positives": line_count(cumulative_missed),
        }

        curriculum = update_curriculum(
            round_best["calibration_domains"],
            previous=curriculum,
            strength=float(factory.get("curriculum_strength", 3.0)),
            max_weight=float(factory.get("max_domain_weight", 8.0)),
        )
        curriculum_path = work / "curriculum" / f"round-{round_index:02d}.json"
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        curriculum_path.write_text(
            json.dumps(curriculum, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        if best is not None and float(best["score"]) < prior_best_score - 1.0e-12:
            stale_rounds = 0
        else:
            stale_rounds += 1
        if (
            round_index + 1 >= min_rounds
            and best is not None
            and bool(best["calibration_gate"])
            and bool(best["test_gate"])
            and bool(factory.get("stop_on_gate", True))
        ):
            break
        if round_index + 1 >= min_rounds and stale_rounds >= patience:
            break

    if best is None:
        raise RuntimeError("model factory produced no candidate")

    best_dir = work / "best"
    copy_best(best, best_dir)
    qualification_dataset = work / "qualification-dataset"
    render_domain_dataset(config_path, qualification_dataset, curriculum_weights=None)
    audit_dataset(qualification_dataset)
    qualification_base, qualification_domains = evaluate(
        runner=runner,
        model=best_dir / "model.kwm",
        pack=best_dir / "keywords.kwk",
        references=qualification_dataset / "qualification.references.jsonl",
        output=best_dir / "qualification",
    )
    qualified = base_gate(qualification_base, gates) and domain_gate(qualification_domains, gates)

    manifest = {
        "schema_version": 1,
        "evidence_class": "synthetic-domain-model-factory",
        "name": str(cfg.get("name", "model-factory")),
        "qualified": qualified,
        "training_image_digest": image_digest,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(runner),
        "training_keywords_sha256": sha256_file(training_keywords),
        "runtime_keywords_source_sha256": sha256_file(runtime_keywords),
        "best_round": int(best["round"]),
        "best_frontend": str(best["frontend"]),
        "best_score": float(best["score"]),
        "best_model_sha256": sha256_file(best_dir / "model.kwm"),
        "best_checkpoint_sha256": sha256_file(best_dir / "model.pt"),
        "best_provenance_sha256": sha256_file(best_dir / "model-provenance.json"),
        "best_pack_sha256": sha256_file(best_dir / "keywords.kwk"),
        "best_keywords_sha256": sha256_file(best_dir / "keywords.tsv"),
        "rounds": records,
        "final_curriculum": curriculum or {},
        "replay": {
            "hard_negative_count": line_count(cumulative_hard),
            "missed_positive_count": line_count(cumulative_missed),
            "hard_negatives_sha256": sha256_file(cumulative_hard),
            "missed_positives_sha256": sha256_file(cumulative_missed),
        },
        "qualification": qualification_base,
        "qualification_domains": qualification_domains,
        "gates": gates,
        "limitations": [
            "No real human speech is used; tone/synthetic-domain evidence is not shipping Mandarin qualification.",
            "Proxy AFE scenes do not replace the final shipping audio-pipeline, microphone or enclosure.",
            "Physical Cortex-A32 performance and independent real acoustic evidence remain Issue #2 gates.",
        ],
    }
    manifest_path = work / "model-factory-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualified else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
