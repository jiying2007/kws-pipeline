#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from qualification_common import (
    FRAME_HOP_SAMPLES,
    FRAME_LENGTH_SAMPLES,
    MODEL_VERSION,
    PACK_VERSION,
    SAMPLE_RATE_HZ,
    SOURCE_SHA_RE,
    load_json,
    read_model,
    read_pack,
    read_vocabulary,
    sha256_file,
    validate_runtime_config,
)
from qualification_metrics import validate_board, validate_evidence, validate_eval


def artifact(path: pathlib.Path, digest: str) -> dict:
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"{path}: release artifact must be non-empty")
    return {"name": path.name, "sha256": digest, "bytes": size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=pathlib.Path)
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
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source_sha = args.source_sha.strip().lower()
    if SOURCE_SHA_RE.fullmatch(source_sha) is None:
        raise ValueError("--source-sha must be a 40- or 64-character hex Git object id")
    corpus_id = args.corpus_id.strip()
    if not corpus_id:
        raise ValueError("--corpus-id must be non-empty")

    model = read_model(args.model)
    pack = read_pack(args.keywords)
    vocabulary = read_vocabulary(args.tokens)
    if not (
        model["vocab_size"] == pack["vocab_size"] == vocabulary["size"]
        and model["vocab_fingerprint"]
        == pack["vocab_fingerprint"]
        == vocabulary["fingerprint"]
    ):
        raise ValueError("model, keyword pack, and token vocabulary identity differ")
    runtime = validate_runtime_config(args.config, model)

    paths = {
        "model": args.model,
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
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}

    evaluation = validate_eval(
        load_json(args.eval_summary),
        load_json(args.eval_provenance),
        {
            "runner_sha256": hashes["eval_runner"],
            "model_sha256": hashes["model"],
            "keyword_pack_sha256": hashes["keyword_pack"],
            "references_sha256": hashes["references"],
            "detections_sha256": hashes["detections"],
        },
    )
    board = validate_board(
        load_json(args.board_summary),
        model["bytes"],
        pack["bytes"],
        {
            "runner_sha256": hashes["board_runner"],
            "model_sha256": hashes["model"],
            "keyword_pack_sha256": hashes["keyword_pack"],
            "audio_sha256": hashes["board_audio"],
        },
    )
    evidence = validate_evidence(load_json(args.evidence))

    artifacts = {
        "model": {
            **artifact(args.model, hashes["model"]),
            "feature_dim": model["feature_dim"],
            "hidden_dim": model["hidden_dim"],
        },
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
    }

    manifest = {
        "schema_version": 1,
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
        "evaluation": {
            "summary_sha256": hashes["eval_summary"],
            "provenance_sha256": hashes["eval_provenance"],
            **evaluation,
        },
        "board": {"summary_sha256": hashes["board_summary"], **board},
        "evidence": {"sha256": hashes["evidence"], **evidence},
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
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
