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
sys.path.insert(0, str(TOOLS))

from kws_vocab import load_tokens  # noqa: E402
from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import load_config  # noqa: E402
from train_ctc import load_keyword_operating_points, manifest_rows  # noqa: E402

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


def hydrate_rendered_dataset_cache(
    cache_root: pathlib.Path,
    dataset: pathlib.Path,
    config_path: pathlib.Path,
) -> dict:
    cache = cache_root.resolve()
    receipt_path = cache / "cache-receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"rendered dataset cache receipt missing: {receipt_path}")
    receipt = load_object(receipt_path)
    if (
        receipt.get("evidence_class") != "kws-v2-research-rendered-dataset-cache-v1"
        or receipt.get("evidence_scope") != "research-only"
        or receipt.get("portable_paths_only") is not True
        or receipt.get("qualification_split_consumed") is not False
        or receipt.get("protected_evidence_used") is not False
    ):
        raise ValueError("rendered dataset cache evidence contract mismatch")
    config_sha = sha256_file(config_path)
    if str(receipt.get("source_config_sha256", "")) != config_sha:
        raise ValueError("rendered dataset cache config SHA does not match candidate config")
    splits = receipt.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "calibration", "test"}:
        raise ValueError("rendered dataset cache split contract mismatch")

    clips = cache / "clips"
    if not clips.is_dir():
        raise ValueError("rendered dataset cache is missing clips/")
    dataset.mkdir(parents=True, exist_ok=False)
    shutil.copytree(clips, dataset / "clips")

    hydrated: dict[str, dict] = {}
    for split in ("train", "calibration", "test"):
        item = splits[split]
        if not isinstance(item, dict):
            raise ValueError(f"rendered dataset cache split is invalid: {split}")
        manifest_name = str(item.get("manifest", ""))
        source_manifest = cache / manifest_name
        if (
            not manifest_name
            or not source_manifest.is_file()
            or sha256_file(source_manifest) != str(item.get("manifest_sha256", ""))
        ):
            raise ValueError(f"rendered dataset cache manifest hash mismatch: {split}")
        rows: list[str] = []
        for line_no, raw in enumerate(
            source_manifest.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if "\t" not in raw:
                raise ValueError(f"{source_manifest}:{line_no}: expected WAV<TAB>targets")
            audio_text, targets = raw.split("\t", 1)
            relative = pathlib.Path(audio_text)
            if relative.is_absolute() or not relative.parts or relative.parts[0] != "clips":
                raise ValueError(f"{source_manifest}:{line_no}: cache audio path is not portable")
            target_audio = (dataset / relative).resolve()
            try:
                target_audio.relative_to(dataset.resolve())
            except ValueError as exc:
                raise ValueError("rendered dataset cache audio escapes hydrated root") from exc
            if not target_audio.is_file():
                raise ValueError(f"hydrated cache WAV missing: {target_audio}")
            rows.append(f"{target_audio}\t{targets}\n")
        target_manifest = dataset / f"{split}.tsv"
        target_manifest.write_text("".join(rows), encoding="utf-8")
        evidence = {
            "recordings": int(item["recordings"]),
            "portable_manifest_sha256": str(item["manifest_sha256"]),
            "hydrated_manifest_sha256": sha256_file(target_manifest),
        }

        refs_name = item.get("references")
        if refs_name is not None:
            source_refs = cache / str(refs_name)
            if (
                not source_refs.is_file()
                or sha256_file(source_refs) != str(item.get("references_sha256", ""))
            ):
                raise ValueError(f"rendered dataset cache references hash mismatch: {split}")
            refs: list[dict] = []
            for line_no, raw in enumerate(
                source_refs.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError(f"{source_refs}:{line_no}: expected JSON object")
                path_text = row.get("path")
                if not isinstance(path_text, str):
                    raise ValueError(f"{source_refs}:{line_no}: missing portable path")
                relative = pathlib.Path(path_text)
                if relative.is_absolute() or not relative.parts or relative.parts[0] != "clips":
                    raise ValueError(f"{source_refs}:{line_no}: reference path is not portable")
                target_audio = (dataset / relative).resolve()
                try:
                    target_audio.relative_to(dataset.resolve())
                except ValueError as exc:
                    raise ValueError("rendered reference escapes hydrated root") from exc
                if not target_audio.is_file():
                    raise ValueError(f"hydrated reference WAV missing: {target_audio}")
                value = dict(row)
                value["audio_path"] = str(target_audio)
                value["path"] = target_audio.name
                refs.append(value)
            target_refs = dataset / f"{split}.references.jsonl"
            target_refs.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    for row in refs
                ),
                encoding="utf-8",
            )
            evidence["portable_references_sha256"] = str(item["references_sha256"])
            evidence["hydrated_references_sha256"] = sha256_file(target_refs)
            evidence["reference_recordings"] = len(refs)
        hydrated[split] = evidence

    return {
        "enabled": True,
        "evidence_class": str(receipt["evidence_class"]),
        "receipt_sha256": sha256_file(receipt_path),
        "source_config_sha256": str(receipt["source_config_sha256"]),
        "source_domain_index_sha256": str(receipt["source_domain_index_sha256"]),
        "source_domain_summary_sha256": str(receipt["source_domain_summary_sha256"]),
        "unique_wav_files": int(receipt["unique_wav_files"]),
        "unique_wav_sha256": int(receipt["unique_wav_sha256"]),
        "hydrated_splits": hydrated,
    }


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


def training_class_balance(
    manifests: list[pathlib.Path],
    policy: dict,
    keyword_sequences: list[list[int]],
) -> dict:
    keyword_targets = {
        tuple(int(value) for value in sequence) for sequence in keyword_sequences
    }
    wake = 0
    tokenized_nonwake = 0
    empty = 0
    per_manifest: list[dict] = []
    for manifest in manifests:
        m_wake = m_tokenized = m_empty = 0
        for row in manifest_rows(manifest.resolve()):
            target = tuple(int(value) for value in row["tokens"])
            if not target:
                m_empty += 1
            elif target in keyword_targets:
                m_wake += 1
            else:
                m_tokenized += 1
        wake += m_wake
        tokenized_nonwake += m_tokenized
        empty += m_empty
        per_manifest.append(
            {
                "path": str(manifest),
                "wake_examples": m_wake,
                "tokenized_nonwake_examples": m_tokenized,
                "empty_target_examples": m_empty,
            }
        )
    target_bearing = wake + tokenized_nonwake
    if wake <= 0 or tokenized_nonwake <= 0 or empty <= 0:
        raise ValueError(
            "research class balance requires wake, tokenized-nonwake, and empty-target "
            f"coverage: wake={wake} tokenized_nonwake={tokenized_nonwake} empty={empty}"
        )
    cfg = policy["class_balance"]
    target_raw = float(empty) / float(target_bearing)
    target_low = float(cfg["minimum_target_bearing_weight"])
    target_high = float(cfg["maximum_target_bearing_weight"])
    wake_raw = float(tokenized_nonwake) / float(wake)
    wake_low = float(cfg["minimum_wake_weight"])
    wake_high = float(cfg["maximum_wake_weight"])
    if not 0.0 < target_low <= target_high or not 0.0 < wake_low <= wake_high:
        raise ValueError("research class-balance bounds are invalid")
    return {
        "policy": str(cfg["policy"]),
        "wake_examples": wake,
        "tokenized_nonwake_examples": tokenized_nonwake,
        "empty_target_examples": empty,
        "target_bearing_examples": target_bearing,
        "empty_to_target_bearing_ratio": target_raw,
        "tokenized_nonwake_to_wake_ratio": wake_raw,
        "effective_target_bearing_weight": min(target_high, max(target_low, target_raw)),
        "effective_wake_example_weight": min(wake_high, max(wake_low, wake_raw)),
        "target_bearing_weight_bounds": [target_low, target_high],
        "wake_weight_bounds": [wake_low, wake_high],
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
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--negative-exposure-seconds", type=int)
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    parser.add_argument(
        "--negative-sidecar",
        type=pathlib.Path,
        help="optional portable kws-v2-research-negative-sidecar-v1 root",
    )
    parser.add_argument(
        "--prepared-dataset-cache",
        type=pathlib.Path,
        help="optional portable kws-v2-research-rendered-dataset-cache-v1 root",
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

    ladder = policy.get("loss_ladder")
    tuning = policy.get("tuning_profiles", {})
    if not isinstance(ladder, dict) or not isinstance(tuning, dict):
        raise ValueError("research loss profiles must be objects")
    if args.loss_profile in ladder:
        loss = ladder[args.loss_profile]
        loss_profile_origin = "loss-ladder"
        loss_profile_base = args.loss_profile
    elif args.loss_profile in tuning:
        loss = tuning[args.loss_profile]
        loss_profile_origin = "single-variable-tuning"
        loss_profile_base = str(loss.get("base_profile", ""))
        if loss_profile_base not in ladder:
            raise ValueError("tuning profile base_profile is not in loss ladder")
    else:
        raise ValueError(f"unknown loss profile: {args.loss_profile}")
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
    token_map = load_tokens(tokens)
    keyword_sequences, _, _ = load_keyword_operating_points(keywords, token_map)

    work = safe_reset(args.work_dir)
    dataset = work / "dataset"
    if args.prepared_dataset_cache is not None:
        rendered_dataset_cache = hydrate_rendered_dataset_cache(
            args.prepared_dataset_cache,
            dataset,
            config_path,
        )
    else:
        render_domain_dataset(config_path, dataset, curriculum_weights=None)
        rendered_dataset_cache = {"enabled": False}
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
    training_epochs = (
        int(args.epochs) if args.epochs is not None else int(train_policy["epochs"])
    )
    training_seed = (
        int(args.training_seed)
        if args.training_seed is not None
        else int(train_policy["seed"])
    )
    if training_epochs <= 0:
        raise ValueError("research training epochs must be positive")
    if training_seed < 0:
        raise ValueError("research training seed must be non-negative")
    train_manifests = [dataset / "train.tsv"]
    if "train" in sidecar_manifests:
        train_manifests.append(sidecar_manifests["train"])
    balance = training_class_balance(train_manifests, policy, keyword_sequences)
    if (
        loss.get("target_bearing_weight_policy") != "class-balance"
        or loss.get("wake_example_weight_policy") != "class-balance"
    ):
        raise ValueError("research loss profiles must use shared target/wake balance policies")
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
        "--epochs", str(training_epochs),
        "--batch-size", str(int(train_policy["batch_size"])),
        "--lr", str(float(train_policy["learning_rate"])),
        "--seed", str(training_seed),
        "--positive-example-weight",
        str(float(balance["effective_target_bearing_weight"])),
        "--wake-example-weight",
        str(float(balance["effective_wake_example_weight"])),
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

    thresholds = [float(v) for v in policy["threshold_diagnostic"]["common_thresholds"]]
    float_diagnostic = work / "float-ctc-confidence.json"
    float_command = [
        sys.executable,
        str(TRAINING / "diagnose_float_ctc_confidence.py"),
        "--family", args.family,
        "--checkpoint", str(checkpoint),
        "--tokens", str(tokens),
        "--keywords", str(keywords),
        "--split-manifest", f"calibration={dataset / 'calibration.tsv'}",
        "--split-manifest", f"test={dataset / 'test.tsv'}",
        "--thresholds", *[str(v) for v in thresholds],
        "--batch-size", str(int(train_policy["batch_size"])),
        "--output", str(float_diagnostic),
    ]
    if sidecar_manifests:
        float_command.extend(
            [
                "--split-manifest", f"calibration={sidecar_manifests['calibration']}",
                "--split-manifest", f"test={sidecar_manifests['test']}",
            ]
        )
    run(float_command, work / "logs" / "float-ctc-confidence.log")

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
    exposure_seconds = (
        int(args.negative_exposure_seconds)
        if args.negative_exposure_seconds is not None
        else int(exposure_cfg["research_seconds"])
    )
    if exposure_seconds <= 0:
        raise ValueError("negative exposure seconds must be positive")
    exposure_wav = work / "negative-exposure.wav"
    exposure_refs = work / "negative-exposure.references.jsonl"
    exposure_receipt = work / "negative-exposure.receipt.json"
    run(
        [
            sys.executable,
            str(EVAL / "build_research_negative_exposure.py"),
            "--negative-manifest", str(neg_manifest),
            "--seconds", str(exposure_seconds),
            "--seed", str(training_seed + 9001),
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
                "--seed", str(training_seed),
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
        "training_epochs": training_epochs,
        "training_seed": training_seed,
        "frontend": frontend,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "loss_profile": args.loss_profile,
        "loss_profile_origin": loss_profile_origin,
        "loss_profile_base": loss_profile_base,
        "loss_weights": loss,
        "class_balance": balance,
        "single_acoustic_render": True,
        "rendered_dataset_cache": rendered_dataset_cache,
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
        "float_ctc_confidence": load_object(float_diagnostic),
        "float_ctc_confidence_sha256": sha256_file(float_diagnostic),
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
