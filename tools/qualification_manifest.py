#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import wave

from model_provenance import validate_model_provenance
from qualification_common import (
    FRAME_HOP_SAMPLES,
    FRAME_LENGTH_SAMPLES,
    MODEL_VERSION,
    PACK_VERSION,
    SAMPLE_RATE_HZ,
    SOURCE_SHA_RE,
    close_enough,
    finite,
    json_int,
    load_json,
    read_model,
    read_pack,
    read_vocabulary,
    required_text,
    sha256_file,
    sha256_value,
    validate_runtime_config,
)
from qualification_metrics import validate_board, validate_evidence, validate_eval


def artifact(path: pathlib.Path, digest: str) -> dict:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{path}: release artifact must be non-empty")
    return {"name": path.name, "sha256": digest, "bytes": size}


def validate_attestation_verification(value: dict, expected: dict[str, str]) -> dict:
    if json_int(value.get("schema_version"), "attestation.schema_version") != 1:
        raise ValueError("attestation verification schema_version must be 1")
    if value.get("verified") is not True:
        raise ValueError("attestation verification must report verified=true")
    if value.get("subject_kind") != "kws-target-evidence":
        raise ValueError("attestation subject_kind must be kws-target-evidence")
    result = {
        "schema_version": 1,
        "verified": True,
        "subject_kind": "kws-target-evidence",
        "issuer": required_text(value, "issuer", "attestation"),
        "trust_policy": required_text(value, "trust_policy", "attestation"),
        "verified_at_utc": required_text(value, "verified_at_utc", "attestation"),
    }
    if not result["verified_at_utc"].endswith("Z"):
        raise ValueError("attestation verified_at_utc must be UTC")
    for key, expected_hash in expected.items():
        measured = sha256_value(value.get(key), f"attestation.{key}")
        if measured != expected_hash:
            raise ValueError(f"attestation {key} does not match selected artifact")
        result[key] = measured
    return result


def load_jsonl(path: pathlib.Path, label: str) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{label}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def reference_stats(path: pathlib.Path) -> tuple[dict[str, float], int, float]:
    recordings: dict[str, float] = {}
    expected_total = 0
    total_seconds = 0.0
    for index, row in enumerate(load_jsonl(path, "references")):
        name = row.get("recording")
        if not isinstance(name, str) or not name or name in recordings:
            raise ValueError(
                f"references[{index}]: recording must be non-empty and unique"
            )
        duration = finite(
            row.get("duration_s"), f"references[{index}].duration_s", 0.0
        )
        if duration <= 0.0:
            raise ValueError(f"references[{index}].duration_s must be > 0")
        expected = row.get("expected")
        if not isinstance(expected, list):
            raise ValueError(f"references[{index}].expected must be a list")
        for event_index, event in enumerate(expected):
            if not isinstance(event, dict):
                raise ValueError(
                    f"references[{index}].expected[{event_index}] must be an object"
                )
            json_int(event.get("keyword_id"), "reference keyword_id", 0)
            start_s = finite(event.get("start_s"), "reference start_s", 0.0)
            end_s = finite(event.get("end_s"), "reference end_s", 0.0)
            if end_s < start_s or end_s > duration:
                raise ValueError("reference expected window is outside recording")
        recordings[name] = duration
        expected_total += len(expected)
        total_seconds += duration
    if not recordings:
        raise ValueError("reference corpus is empty")
    return recordings, expected_total, total_seconds / 3600.0


def detection_count(path: pathlib.Path, recordings: dict[str, float]) -> int:
    rows = load_jsonl(path, "detections")
    for index, row in enumerate(rows):
        name = row.get("recording")
        if not isinstance(name, str) or name not in recordings:
            raise ValueError(f"detections[{index}] references unknown recording")
        json_int(row.get("keyword_id"), "detection keyword_id", 0)
        time_s = finite(row.get("time_s"), "detection time_s", 0.0)
        confidence = finite(row.get("confidence"), "detection confidence", 0.0)
        if time_s > recordings[name] or confidence > 1.0:
            raise ValueError(f"detections[{index}] contains out-of-range values")
    return len(rows)


def board_wav_stats(path: pathlib.Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError("board audio must be mono 16-kHz PCM16 WAV")
        frames = reader.getnframes()
    if frames <= 0:
        raise ValueError("board audio must be non-empty")
    seconds = frames / float(SAMPLE_RATE_HZ)
    blocks = (frames + FRAME_HOP_SAMPLES - 1) // FRAME_HOP_SAMPLES
    return seconds, blocks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--model-provenance", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--training-tokens", required=True, type=pathlib.Path)
    parser.add_argument(
        "--training-manifest",
        required=True,
        action="append",
        type=pathlib.Path,
    )
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--eval-runner", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path)
    parser.add_argument("--detections", required=True, type=pathlib.Path)
    parser.add_argument("--eval-summary", required=True, type=pathlib.Path)
    parser.add_argument("--eval-provenance", required=True, type=pathlib.Path)
    parser.add_argument("--board-summary", required=True, type=pathlib.Path)
    parser.add_argument("--board-runner", required=True, type=pathlib.Path)
    parser.add_argument("--board-audio", required=True, type=pathlib.Path)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    parser.add_argument("--evidence-raw", required=True, type=pathlib.Path)
    parser.add_argument("--collector", required=True, type=pathlib.Path)
    parser.add_argument("--attestation-verification", required=True, type=pathlib.Path)
    parser.add_argument("--sku", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source_sha = args.source_sha.strip().lower()
    if SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("--source-sha must be a 40- or 64-character hex Git object id")
    corpus_id = args.corpus_id.strip()
    sku = args.sku.strip()
    if not corpus_id:
        raise ValueError("--corpus-id must be non-empty")
    if not sku:
        raise ValueError("--sku must be non-empty")

    model = read_model(args.model)
    pack = read_pack(args.keywords)
    vocabulary = read_vocabulary(args.tokens)
    training_vocabulary = read_vocabulary(args.training_tokens)
    if not (
        model["vocab_size"] == pack["vocab_size"] == vocabulary["size"]
        and model["vocab_fingerprint"]
        == pack["vocab_fingerprint"]
        == vocabulary["fingerprint"]
    ):
        raise ValueError("model, keyword pack, and token vocabulary identity differ")
    if (
        training_vocabulary["size"] != vocabulary["size"]
        or training_vocabulary["fingerprint"] != vocabulary["fingerprint"]
    ):
        raise ValueError(
            "training token vocabulary mapping differs from release token vocabulary"
        )
    runtime = validate_runtime_config(args.config, model)

    paths = {
        "model": args.model,
        "model_provenance": args.model_provenance,
        "model_checkpoint": args.checkpoint,
        "training_tokens": args.training_tokens,
        "keyword_pack": args.keywords,
        "tokens": args.tokens,
        "config": args.config,
        "eval_runner": args.eval_runner,
        "references": args.references,
        "detections": args.detections,
        "eval_summary": args.eval_summary,
        "eval_provenance": args.eval_provenance,
        "board_runner": args.board_runner,
        "board_audio": args.board_audio,
        "board_summary": args.board_summary,
        "evidence": args.evidence,
        "evidence_raw": args.evidence_raw,
        "collector": args.collector,
        "attestation_verification": args.attestation_verification,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    training_manifest_hashes = [sha256_file(path) for path in args.training_manifest]

    model_lineage = validate_model_provenance(
        args.model_provenance,
        model_sha256=hashes["model"],
        model_bytes=model["bytes"],
        feature_dim=model["feature_dim"],
        hidden_dim=model["hidden_dim"],
        vocab_size=model["vocab_size"],
        vocab_fingerprint=model["vocab_fingerprint"],
        tokens_sha256=hashes["tokens"],
        checkpoint_sha256=hashes["model_checkpoint"],
        training_tokens_sha256=hashes["training_tokens"],
        training_manifest_sha256s=training_manifest_hashes,
    )

    recordings, expected_count, audio_hours = reference_stats(args.references)
    detections_count = detection_count(args.detections, recordings)
    eval_summary = load_json(args.eval_summary)
    eval_provenance = load_json(args.eval_provenance)
    if (
        json_int(eval_summary.get("recordings"), "evaluation.recordings", 1)
        != len(recordings)
    ):
        raise ValueError("evaluation recording count does not match reference file")
    if (
        json_int(eval_summary.get("expected"), "evaluation.expected", 0)
        != expected_count
    ):
        raise ValueError("evaluation expected count does not match reference file")
    close_enough(
        finite(eval_summary.get("audio_hours"), "evaluation.audio_hours", 0.0),
        audio_hours,
        "evaluation.audio_hours",
        1e-12,
        1e-12,
    )
    if (
        json_int(eval_provenance.get("recordings"), "provenance.recordings", 1)
        != len(recordings)
        or json_int(eval_provenance.get("detections"), "provenance.detections", 0)
        != detections_count
    ):
        raise ValueError("evaluation provenance counts do not match selected files")
    evaluation = validate_eval(
        eval_summary,
        eval_provenance,
        {
            "runner_sha256": hashes["eval_runner"],
            "model_sha256": hashes["model"],
            "keyword_pack_sha256": hashes["keyword_pack"],
            "references_sha256": hashes["references"],
            "detections_sha256": hashes["detections"],
        },
    )

    board_summary = load_json(args.board_summary)
    board = validate_board(
        board_summary,
        model["bytes"],
        pack["bytes"],
        source_sha,
        {
            "runner_sha256": hashes["board_runner"],
            "model_sha256": hashes["model"],
            "keyword_pack_sha256": hashes["keyword_pack"],
            "audio_sha256": hashes["board_audio"],
        },
    )
    board_seconds, board_blocks = board_wav_stats(args.board_audio)
    close_enough(
        board["audio_seconds"], board_seconds, "board.audio_seconds", 1e-9, 1e-6
    )
    if board["blocks"] != board_blocks * board["repeats"]:
        raise ValueError("board block count does not match board WAV and repeats")
    evidence = validate_evidence(
        load_json(args.evidence),
        sku=sku,
        source_sha=source_sha,
        actual_hashes={
            "collector_sha256": hashes["collector"],
            "raw_evidence_sha256": hashes["evidence_raw"],
            "attestation_verification_sha256": hashes["attestation_verification"],
            "board_runner_sha256": hashes["board_runner"],
            "model_sha256": hashes["model"],
            "keyword_pack_sha256": hashes["keyword_pack"],
            "board_audio_sha256": hashes["board_audio"],
        },
    )
    attestation = validate_attestation_verification(
        load_json(args.attestation_verification),
        {
            "subject_sha256": hashes["evidence_raw"],
            "collector_sha256": hashes["collector"],
            "board_runner_sha256": hashes["board_runner"],
            "model_sha256": hashes["model"],
            "keyword_pack_sha256": hashes["keyword_pack"],
        },
    )

    training_manifest_artifacts = [
        artifact(path, digest)
        for path, digest in zip(args.training_manifest, training_manifest_hashes)
    ]
    artifacts = {
        "model": {
            **artifact(args.model, hashes["model"]),
            "feature_dim": model["feature_dim"],
            "hidden_dim": model["hidden_dim"],
        },
        "model_provenance": artifact(
            args.model_provenance, hashes["model_provenance"]
        ),
        "model_checkpoint": artifact(
            args.checkpoint, hashes["model_checkpoint"]
        ),
        "training_tokens": artifact(
            args.training_tokens, hashes["training_tokens"]
        ),
        "training_manifests": training_manifest_artifacts,
        "keyword_pack": {
            **artifact(args.keywords, hashes["keyword_pack"]),
            "keyword_count": pack["keyword_count"],
        },
        "tokens": artifact(args.tokens, hashes["tokens"]),
        "config": artifact(args.config, hashes["config"]),
        "eval_runner": artifact(args.eval_runner, hashes["eval_runner"]),
        "references": artifact(args.references, hashes["references"]),
        "detections": artifact(args.detections, hashes["detections"]),
        "board_runner": artifact(args.board_runner, hashes["board_runner"]),
        "board_audio": artifact(args.board_audio, hashes["board_audio"]),
        "collector": artifact(args.collector, hashes["collector"]),
        "evidence_raw": artifact(args.evidence_raw, hashes["evidence_raw"]),
        "attestation_verification": artifact(
            args.attestation_verification, hashes["attestation_verification"]
        ),
        "evidence": artifact(args.evidence, hashes["evidence"]),
    }

    manifest = {
        "schema_version": 1,
        "sku": sku,
        "source_sha": source_sha,
        "corpus_id": corpus_id,
        "runtime": {
            "model_abi": MODEL_VERSION,
            "keyword_pack_abi": PACK_VERSION,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "frame_length_samples": FRAME_LENGTH_SAMPLES,
            "frame_hop_samples": FRAME_HOP_SAMPLES,
            **runtime,
        },
        "vocabulary": {
            "size": vocabulary["size"],
            "fingerprint": f"0x{vocabulary['fingerprint']:016x}",
            "sha256": vocabulary["sha256"],
        },
        "artifacts": artifacts,
        "model_lineage": {
            "provenance_sha256": hashes["model_provenance"],
            **model_lineage,
        },
        "evaluation": {
            "summary_sha256": hashes["eval_summary"],
            "provenance_sha256": hashes["eval_provenance"],
            **evaluation,
        },
        "board": {"summary_sha256": hashes["board_summary"], **board},
        "evidence": {
            "sha256": hashes["evidence"],
            "attestation": attestation,
            **evidence,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
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
    print(f"wrote qualification manifest: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        wave.Error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
