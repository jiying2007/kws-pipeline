#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
EVAL = ROOT / "eval"
sys.path.insert(0, str(TRAINING))

from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402
from train_ctc import manifest_rows  # noqa: E402

POLICY_ID = "kws-v2-research-reset-v1"
EVIDENCE_CLASS = "kws-v2-research-baseline-scorecard-v1"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def run(command: list[str], log: pathlib.Path) -> None:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}; see {log}"
        )


def safe_reset(path: pathlib.Path) -> pathlib.Path:
    root = path.resolve()
    if root in {ROOT.resolve(), ROOT.parent.resolve(), pathlib.Path.home().resolve()}:
        raise ValueError(f"unsafe work directory: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    return root


def negative_manifest(sources: list[pathlib.Path], output: pathlib.Path) -> int:
    paths: list[pathlib.Path] = []
    seen: set[str] = set()
    for source in sources:
        root = source.resolve().parent
        for row in manifest_rows(source.resolve()):
            if list(row["tokens"]):
                continue
            raw = pathlib.Path(str(row["audio"]))
            path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
            digest = sha256_file(path)
            if digest in seen:
                continue
            seen.add(digest)
            paths.append(path)
    if not paths:
        raise ValueError("negative manifests contain no empty-target examples")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{path}\n" for path in paths), encoding="utf-8")
    return len(paths)


def combined_references(
    sources: list[pathlib.Path],
    output: pathlib.Path,
) -> pathlib.Path:
    rows: list[dict] = []
    seen: set[str] = set()
    for source in sources:
        root = source.resolve().parent
        for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_no}: expected JSON object")
            recording = str(row.get("recording", ""))
            path_value = row.get("audio_path") or row.get("path")
            if not recording or recording in seen or not isinstance(path_value, str):
                raise ValueError(f"{source}:{line_no}: invalid/duplicate reference identity")
            audio = pathlib.Path(path_value)
            if not audio.is_absolute():
                audio = (root / audio).resolve()
            row["path"] = str(audio)
            row.pop("audio_path", None)
            rows.append(row)
            seen.add(recording)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return output


def training_class_balance(manifests: list[pathlib.Path], policy: dict) -> dict:
    positive = 0
    negative = 0
    per_manifest: list[dict] = []
    for manifest in manifests:
        pos = neg = 0
        for row in manifest_rows(manifest.resolve()):
            if list(row["tokens"]):
                pos += 1
            else:
                neg += 1
        positive += pos
        negative += neg
        per_manifest.append(
            {
                "path": str(manifest),
                "positive_examples": pos,
                "negative_examples": neg,
            }
        )
    if positive <= 0 or negative <= 0:
        raise ValueError(
            f"research class balance requires both positive and negative examples: "
            f"positive={positive} negative={negative}"
        )
    raw = float(negative) / float(positive)
    cfg = policy["class_balance"]
    low = float(cfg["minimum_positive_example_weight"])
    high = float(cfg["maximum_positive_example_weight"])
    if not 0.0 < low <= high:
        raise ValueError("research class-balance bounds are invalid")
    weight = min(high, max(low, raw))
    return {
        "policy": str(cfg["policy"]),
        "positive_examples": positive,
        "negative_examples": negative,
        "negative_to_positive_ratio": raw,
        "effective_positive_example_weight": weight,
        "minimum_positive_example_weight": low,
        "maximum_positive_example_weight": high,
        "manifests": per_manifest,
    }


def soft_operating_points(curve: dict, budgets: list[float]) -> dict:
    rows = curve.get("operating_curve")
    if not isinstance(rows, list) or not rows:
        raise ValueError("threshold diagnostic has no operating curve")
    result: dict[str, dict | None] = {}
    for budget in budgets:
        eligible = [
            row
            for row in rows
            if float(row["calibration"]["far_per_hour"]) <= float(budget)
        ]
        if not eligible:
            result[str(budget)] = None
            continue
        selected = min(
            eligible,
            key=lambda row: (
                float(row["calibration"]["frr"]),
                float(row["calibration"]["far_per_hour"]),
                float(row["threshold"]),
            ),
        )
        result[str(budget)] = {
            "threshold": float(selected["threshold"]),
            "calibration": selected["calibration"],
            "test": selected["test"],
        }
    return result


def research_threshold(curve: dict, budgets: list[float]) -> float:
    points = soft_operating_points(curve, budgets)
    for budget in reversed(budgets):
        row = points.get(str(budget))
        if isinstance(row, dict):
            return float(row["threshold"])
    rows = curve["operating_curve"]
    selected = min(
        rows,
        key=lambda row: (
            float(row["calibration"]["frr"])
            + float(row["calibration"]["far_per_hour"]) / max(1.0, max(budgets)),
            float(row["threshold"]),
        ),
    )
    return float(selected["threshold"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast research-only KWS baseline with no replay/controller/curriculum feedback."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument(
        "--policy",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "kws-v2-research-reset-v1.json",
    )
    parser.add_argument("--family", required=True, choices=("rnn", "gru"))
    parser.add_argument("--loss-profile", default="m0-ctc-only")
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument(
        "--negative-sidecar",
        type=pathlib.Path,
        help="optional portable kws-v2-research-negative-sidecar-v1 root",
    )
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    policy_path = args.policy.resolve()
    runner = args.runner.resolve()
    for path, label in ((config_path, "config"), (policy_path, "policy"), (runner, "runner")):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    cfg = load_config(config_path)
    policy = load_object(policy_path)
    if policy.get("policy") != POLICY_ID or policy.get("evidence_scope") != "research-only":
        raise ValueError("research reset policy identity mismatch")
    lane = policy.get("research_lane")
    required_false = (
        "curriculum_feedback",
        "failure_replay",
        "fixed_hard_negative_replay",
        "adaptive_loss_controller",
        "warm_start",
        "hard_gate_required",
        "candidate_freeze_allowed",
        "promotion_allowed",
    )
    if not isinstance(lane, dict) or any(lane.get(key) is not False for key in required_false):
        raise ValueError("research lane must keep adaptive/protected mechanisms disabled")

    profiles = policy.get("loss_ladder")
    if not isinstance(profiles, dict) or args.loss_profile not in profiles:
        raise ValueError(f"unknown loss profile: {args.loss_profile}")
    loss = profiles[args.loss_profile]
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict):
        raise ValueError("config.model must be an object")
    frontends = model_cfg.get("frontends")
    if not isinstance(frontends, list) or len(frontends) != 1:
        raise ValueError("research baseline requires exactly one frontend")
    frontend = str(frontends[0])
    feature_dim = int(model_cfg.get("feature_dim", 32))
    hidden_dim = int(model_cfg.get("hidden_dim", 64))
    tokens = (ROOT / str(cfg["tokens"])).resolve()
    keywords = (ROOT / str(cfg["keywords"])).resolve()

    work = safe_reset(args.work_dir)
    dataset = work / "dataset"
    render_domain_dataset(config_path, dataset, curriculum_weights=None)
    run(
        [
            sys.executable,
            str(TRAINING / "audit_dataset.py"),
            "--split", f"train={dataset / 'train.tsv'}",
            "--split", f"calibration={dataset / 'calibration.tsv'}",
            "--split", f"test={dataset / 'test.tsv'}",
            "--report", str(work / "dataset-audit.json"),
            "--fail-within-split",
        ],
        work / "logs" / "audit.log",
    )

    sidecar_root: pathlib.Path | None = None
    sidecar_receipt: dict | None = None
    sidecar_manifests: dict[str, pathlib.Path] = {}
    sidecar_references: dict[str, pathlib.Path] = {}
    if args.negative_sidecar is not None:
        sidecar_root = args.negative_sidecar.resolve()
        receipt_path = sidecar_root / "sidecar-receipt.json"
        if not receipt_path.is_file():
            raise ValueError(f"negative sidecar receipt missing: {receipt_path}")
        sidecar_receipt = load_object(receipt_path)
        if (
            sidecar_receipt.get("evidence_class") != "kws-v2-research-negative-sidecar-v1"
            or sidecar_receipt.get("evidence_scope") != "research-only"
            or sidecar_receipt.get("target_policy") != "empty-target-nonwake"
            or sidecar_receipt.get("qualification_split_consumed") is not False
            or sidecar_receipt.get("protected_evidence_used") is not False
        ):
            raise ValueError("negative sidecar research/protection contract mismatch")
        for split in ("train", "calibration", "test"):
            item = sidecar_receipt["splits"][split]
            manifest = sidecar_root / str(item["tsv"])
            references = sidecar_root / str(item["references"])
            if not manifest.is_file() or not references.is_file():
                raise ValueError(f"negative sidecar split is incomplete: {split}")
            sidecar_manifests[split] = manifest
            sidecar_references[split] = references

    train_policy = policy["train"]
    train_manifests = [dataset / "train.tsv"]
    if "train" in sidecar_manifests:
        train_manifests.append(sidecar_manifests["train"])
    balance = training_class_balance(train_manifests, policy)
    if loss.get("positive_example_weight_policy") != "class-balance":
        raise ValueError("research loss profiles must use the shared class-balance policy")
    checkpoint = work / "model.pt"
    trainer = TRAINING / ("train_gru_ctc.py" if args.family == "gru" else "train_ctc.py")
    command = [
        sys.executable,
        str(trainer),
        "--manifest", str(dataset / "train.tsv"),
        "--tokens", str(tokens),
        "--keywords", str(keywords),
        "--frontend", frontend,
        "--feature-dim", str(feature_dim),
        "--hidden-dim", str(hidden_dim),
        "--epochs", str(int(train_policy["epochs"])),
        "--batch-size", str(int(train_policy["batch_size"])),
        "--lr", str(float(train_policy["learning_rate"])),
        "--seed", str(int(train_policy["seed"])),
        "--positive-example-weight",
        str(float(balance["effective_positive_example_weight"])),
        "--ordered-token-loss-weight", str(float(loss["ordered_token_loss_weight"])),
        "--keyword-sequence-margin-loss-weight",
        str(float(loss["keyword_sequence_margin_loss_weight"])),
        "--prefix-completion-loss-weight",
        str(float(loss["prefix_completion_loss_weight"])),
        "--recurrent-release-loss-weight",
        str(float(loss["recurrent_release_loss_weight"])),
        "--output", str(checkpoint),
    ]
    for extra_manifest in train_manifests[1:]:
        command.extend(["--manifest", str(extra_manifest)])
    run(command, work / "logs" / "train.log")

    model = work / ("model.kwg" if args.family == "gru" else "model.kwm")
    exporter = TRAINING / ("export_gru_model.py" if args.family == "gru" else "export_model.py")
    run(
        [
            sys.executable,
            str(exporter),
            "--checkpoint", str(checkpoint),
            "--tokens", str(tokens),
            "--output", str(model),
        ],
        work / "logs" / "export.log",
    )

    thresholds = [float(v) for v in policy["threshold_diagnostic"]["common_thresholds"]]
    calibration_references = dataset / "calibration.references.jsonl"
    test_references = dataset / "test.references.jsonl"
    if sidecar_references:
        calibration_references = combined_references(
            [calibration_references, sidecar_references["calibration"]],
            work / "combined-calibration.references.jsonl",
        )
        test_references = combined_references(
            [test_references, sidecar_references["test"]],
            work / "combined-test.references.jsonl",
        )
    curve_path = work / "threshold-operating-curve.json"
    threshold_work = work / "threshold-sweep"
    run(
        [
            sys.executable,
            str(TOOLS / "diagnose_kws_threshold_operating_curve.py"),
            "--runner", str(runner),
            "--model", str(model),
            "--tokens", str(tokens),
            "--keywords", str(keywords),
            "--config", str(config_path),
            "--calibration-references", str(calibration_references),
            "--test-references", str(test_references),
            "--thresholds", *[str(v) for v in thresholds],
            "--diagnostic-round-selection-policy", "research-reset-fixed-checkpoint-v1",
            "--work-dir", str(threshold_work),
            "--output", str(curve_path),
        ],
        work / "logs" / "threshold.log",
    )
    curve = load_object(curve_path)
    budgets = [float(v) for v in policy["threshold_diagnostic"]["far_budgets_per_hour"]]
    operating = soft_operating_points(curve, budgets)
    chosen = research_threshold(curve, budgets)
    pack = threshold_work / f"threshold-{chosen:.3f}" / "keywords.kwk"
    if not pack.is_file():
        raise ValueError(f"diagnostic keyword pack missing for threshold {chosen:.3f}")

    neg_manifest = work / "test-negative-manifest.tsv"
    negative_sources = [dataset / "test.tsv"]
    if "test" in sidecar_manifests:
        negative_sources.append(sidecar_manifests["test"])
    negative_count = negative_manifest(negative_sources, neg_manifest)
    exposure_cfg = policy["negative_exposure"]
    exposure_wav = work / "negative-exposure.wav"
    exposure_refs = work / "negative-exposure.references.jsonl"
    exposure_receipt = work / "negative-exposure.receipt.json"
    run(
        [
            sys.executable,
            str(EVAL / "build_research_negative_exposure.py"),
            "--negative-manifest", str(neg_manifest),
            "--seconds", str(int(exposure_cfg["research_seconds"])),
            "--seed", str(int(train_policy["seed"]) + 9001),
            "--min-injections-per-clip",
            str(int(exposure_cfg["minimum_each_negative_clip_injections"])),
            "--output-wav", str(exposure_wav),
            "--references", str(exposure_refs),
            "--receipt", str(exposure_receipt),
        ],
        work / "logs" / "negative-exposure-build.log",
    )
    run(
        [
            sys.executable,
            str(ROOT / "eval" / "run_corpus.py"),
            "--runner", str(runner),
            "--model", str(model),
            "--keywords", str(pack),
            "--references", str(exposure_refs),
            "--detections", str(work / "negative-exposure.detections.jsonl"),
            "--provenance", str(work / "negative-exposure.provenance.json"),
        ],
        work / "logs" / "negative-exposure-run.log",
    )
    run(
        [
            sys.executable,
            str(ROOT / "eval" / "score_events.py"),
            "--references", str(exposure_refs),
            "--detections", str(work / "negative-exposure.detections.jsonl"),
            "--summary", str(work / "negative-exposure.summary.json"),
            "--false-positives", str(work / "negative-exposure.false-positives.jsonl"),
            "--false-rejects", str(work / "negative-exposure.false-rejects.jsonl"),
        ],
        work / "logs" / "negative-exposure-score.log",
    )

    classifier_cfg = policy["classifier_baseline"]
    classifier_path = work / "classifier-scorecard.json"
    if classifier_cfg.get("enabled") is True:
        run(
            [
                sys.executable,
                str(TRAINING / "research_keyword_classifier.py"),
                "--train-manifest", str(dataset / "train.tsv"),
                "--calibration-manifest", str(dataset / "calibration.tsv"),
                "--test-manifest", str(dataset / "test.tsv"),
                *(
                    [
                        "--train-manifest", str(sidecar_manifests["train"]),
                        "--calibration-manifest", str(sidecar_manifests["calibration"]),
                        "--test-manifest", str(sidecar_manifests["test"]),
                    ]
                    if sidecar_manifests
                    else []
                ),
                "--tokens", str(tokens),
                "--keywords", str(keywords),
                "--frontend", frontend,
                "--feature-dim", str(feature_dim),
                "--hidden-dim", str(int(classifier_cfg["hidden_dim"])),
                "--epochs", str(int(classifier_cfg["epochs"])),
                "--batch-size", str(int(train_policy["batch_size"])),
                "--lr", str(float(classifier_cfg["learning_rate"])),
                "--seed", str(int(train_policy["seed"])),
                "--balance-mode", str(policy["class_balance"]["classifier_balance_mode"]),
                "--output", str(classifier_path),
            ],
            work / "logs" / "classifier.log",
        )

    negative_summary = load_object(work / "negative-exposure.summary.json")
    classifier = load_object(classifier_path) if classifier_path.is_file() else None
    scorecard = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "qualification_split_consumed": False,
        "promotion_allowed": False,
        "family": args.family,
        "frontend": frontend,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "loss_profile": args.loss_profile,
        "loss_weights": loss,
        "class_balance": balance,
        "single_acoustic_render": True,
        "curriculum_feedback": False,
        "replay": False,
        "adaptive_controller": False,
        "warm_start": False,
        "hard_gate_required": False,
        "model": str(model),
        "model_sha256": sha256_file(model),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config_path),
        "policy_sha256": sha256_file(policy_path),
        "data_limitations": policy["data_limitations"],
        "threshold_curve_sha256": sha256_file(curve_path),
        "pareto_thresholds": curve["pareto_thresholds"],
        "soft_operating_points_by_far_budget": operating,
        "research_negative_threshold": chosen,
        "research_negative_unique_clips": negative_count,
        "research_negative_exposure": {
            "seconds": float(negative_summary["audio_hours"]) * 3600.0,
            "false_accepts": negative_summary["false_accepts"],
            "far_per_hour": negative_summary["far_per_hour"],
            "shipping_far_claim_allowed": False,
        },
        "classifier_baseline": classifier,
        "ordinary_speech_negative_sidecar": (
            {
                "enabled": True,
                "receipt_sha256": sha256_file(sidecar_root / "sidecar-receipt.json"),
                "total_recordings": int(sidecar_receipt["total_recordings"]),
                "train_recordings": int(sidecar_receipt["splits"]["train"]["recordings"]),
                "calibration_recordings": int(sidecar_receipt["splits"]["calibration"]["recordings"]),
                "test_recordings": int(sidecar_receipt["splits"]["test"]["recordings"]),
                "target_policy": "empty-target-nonwake",
            }
            if sidecar_receipt is not None and sidecar_root is not None
            else {"enabled": False}
        ),
    }
    target = work / "research-scorecard.json"
    target.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"kws-research-baseline: family={args.family} profile={args.loss_profile} "
        f"pareto={scorecard['pareto_thresholds']} "
        f"negative-far/h={scorecard['research_negative_exposure']['far_per_hour']:.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
