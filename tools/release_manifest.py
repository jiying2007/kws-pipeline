#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

MODEL_HEADER = struct.Struct("<4sHHHHHHIIIfffQIIIIII")
PACK_HEADER = struct.Struct("<4sHHHHIQ")
MODEL_VERSION = 2
PACK_VERSION = 2
MODEL_HEADER_BYTES = 72
PACK_HEADER_BYTES = 24
SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value, label: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def positive_int(value, label: str, allow_zero: bool = False) -> int:
    result = int(value)
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def required_text(obj: dict, key: str, label: str) -> str:
    value = str(obj.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label}.{key} must be non-empty")
    return value


def read_model(path: pathlib.Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < MODEL_HEADER_BYTES:
        raise ValueError(f"{path}: model is shorter than ABI-v2 header")
    fields = MODEL_HEADER.unpack(blob[:MODEL_HEADER_BYTES])
    (
        magic,
        version,
        header_bytes,
        feature_dim,
        hidden_dim,
        model_vocab_size,
        reserved,
        sample_rate,
        frame_length,
        frame_hop,
        wx_scale,
        wh_scale,
        wo_scale,
        fingerprint,
        wx_off,
        wh_off,
        bh_off,
        wo_off,
        bo_off,
        total_bytes,
    ) = fields
    if magic != b"KWSP" or version != MODEL_VERSION or header_bytes != MODEL_HEADER_BYTES:
        raise ValueError(f"{path}: expected KWSP ABI v{MODEL_VERSION}")
    if reserved != 0 or total_bytes != len(blob):
        raise ValueError(f"{path}: non-canonical model header")
    if sample_rate != SAMPLE_RATE_HZ or frame_length != FRAME_LENGTH_SAMPLES or frame_hop != FRAME_HOP_SAMPLES:
        raise ValueError(f"{path}: model geometry is outside ABI-v2 contract")
    if feature_dim <= 0 or hidden_dim <= 0 or model_vocab_size <= 1 or fingerprint == 0:
        raise ValueError(f"{path}: invalid model dimensions or vocabulary identity")
    for label, value in (("wx_scale", wx_scale), ("wh_scale", wh_scale), ("wo_scale", wo_scale)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{path}: {label} must be finite and positive")
    if not (MODEL_HEADER_BYTES <= wx_off <= wh_off <= bh_off <= wo_off <= bo_off < total_bytes):
        raise ValueError(f"{path}: invalid model tensor offsets")
    return {
        "bytes": len(blob),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "vocab_size": model_vocab_size,
        "vocab_fingerprint": fingerprint,
    }


def read_pack(path: pathlib.Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < PACK_HEADER_BYTES:
        raise ValueError(f"{path}: keyword pack is shorter than ABI-v2 header")
    magic, version, header_bytes, count, pack_vocab_size, total_bytes, fingerprint = PACK_HEADER.unpack(
        blob[:PACK_HEADER_BYTES]
    )
    if magic != b"KWKP" or version != PACK_VERSION or header_bytes != PACK_HEADER_BYTES:
        raise ValueError(f"{path}: expected KWKP ABI v{PACK_VERSION}")
    if count <= 0 or total_bytes != len(blob) or fingerprint == 0:
        raise ValueError(f"{path}: invalid keyword-pack header")
    return {
        "bytes": len(blob),
        "keyword_count": count,
        "vocab_size": pack_vocab_size,
        "vocab_fingerprint": fingerprint,
    }


def validate_eval(summary: dict, provenance: dict) -> dict:
    if int(provenance.get("schema_version", 0)) != 1:
        raise ValueError("evaluation provenance schema_version must be 1")
    for key in (
        "runner_sha256",
        "model_sha256",
        "keyword_pack_sha256",
        "references_sha256",
        "detections_sha256",
    ):
        value = str(provenance.get(key, ""))
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"evaluation provenance {key} must be SHA256 hex")
    if summary.get("references_sha256") != provenance["references_sha256"] or summary.get(
        "detections_sha256"
    ) != provenance["detections_sha256"]:
        raise ValueError("evaluation summary does not match provenance inputs")

    result = {
        "audio_hours": finite(summary["audio_hours"], "evaluation.audio_hours", 0.0),
        "expected": positive_int(summary["expected"], "evaluation.expected", allow_zero=True),
        "matched": positive_int(summary["matched"], "evaluation.matched", allow_zero=True),
        "false_rejects": positive_int(
            summary["false_rejects"], "evaluation.false_rejects", allow_zero=True
        ),
        "false_accepts": positive_int(
            summary["false_accepts"], "evaluation.false_accepts", allow_zero=True
        ),
        "frr": finite(summary["frr"], "evaluation.frr", 0.0),
        "far_per_hour": finite(
            summary["far_per_hour"], "evaluation.far_per_hour", 0.0
        ),
        "p50_post_end_latency_ms": finite(
            summary["p50_post_end_latency_ms"],
            "evaluation.p50_post_end_latency_ms",
            0.0,
        ),
        "p95_post_end_latency_ms": finite(
            summary["p95_post_end_latency_ms"],
            "evaluation.p95_post_end_latency_ms",
            0.0,
        ),
        "references_sha256": provenance["references_sha256"],
        "detections_sha256": provenance["detections_sha256"],
        "runner_sha256": provenance["runner_sha256"],
    }
    if result["frr"] > 1.0 or result["matched"] > result["expected"]:
        raise ValueError("evaluation summary contains impossible counts/rates")
    return result


def validate_board(summary: dict, model_bytes: int, pack_bytes: int) -> dict:
    if int(summary.get("schema_version", 0)) != 1:
        raise ValueError("board benchmark schema_version must be 1")
    if int(summary.get("block_samples", 0)) != FRAME_HOP_SAMPLES:
        raise ValueError("board benchmark must use one 20-ms KWS hop per block")
    if int(summary.get("model_bytes", -1)) != model_bytes or int(
        summary.get("keyword_pack_bytes", -1)
    ) != pack_bytes:
        raise ValueError("board benchmark artifact sizes do not match release artifacts")
    result = {
        "audio_seconds": finite(summary["audio_seconds"], "board.audio_seconds", 0.0),
        "repeats": positive_int(summary["repeats"], "board.repeats"),
        "blocks": positive_int(summary["blocks"], "board.blocks"),
        "arena_bytes": positive_int(summary["arena_bytes"], "board.arena_bytes"),
        "block_deadline_us": finite(
            summary["block_deadline_us"], "board.block_deadline_us", 0.0
        ),
        "mean_process_us": finite(
            summary["mean_process_us"], "board.mean_process_us", 0.0
        ),
        "p50_process_us": finite(
            summary["p50_process_us"], "board.p50_process_us", 0.0
        ),
        "p95_process_us": finite(
            summary["p95_process_us"], "board.p95_process_us", 0.0
        ),
        "p99_process_us": finite(
            summary["p99_process_us"], "board.p99_process_us", 0.0
        ),
        "max_process_us": finite(
            summary["max_process_us"], "board.max_process_us", 0.0
        ),
        "rtf": finite(summary["rtf"], "board.rtf", 0.0),
        "p99_headroom": finite(summary["p99_headroom"], "board.p99_headroom", 0.0),
    }
    if result["block_deadline_us"] != 20000.0:
        raise ValueError("board benchmark deadline must be 20000 us for ABI-v2 hop")
    if not (
        result["p50_process_us"] <= result["p95_process_us"]
        <= result["p99_process_us"]
        <= result["max_process_us"]
    ):
        raise ValueError("board benchmark percentiles are not monotonic")
    return result


def validate_evidence(evidence: dict) -> dict:
    result = {
        "target": required_text(evidence, "target", "evidence"),
        "board_revision": required_text(evidence, "board_revision", "evidence"),
        "soc": required_text(evidence, "soc", "evidence"),
        "toolchain": required_text(evidence, "toolchain", "evidence"),
        "compiler_flags": required_text(evidence, "compiler_flags", "evidence"),
        "governor": required_text(evidence, "governor", "evidence"),
        "audio_frontend": required_text(evidence, "audio_frontend", "evidence"),
        "soak_hours": finite(evidence["soak_hours"], "evidence.soak_hours", 0.0),
        "cpu_percent": finite(evidence["cpu_percent"], "evidence.cpu_percent", 0.0),
        "rss_kib": finite(evidence["rss_kib"], "evidence.rss_kib", 0.0),
        "stack_high_water_bytes": finite(
            evidence["stack_high_water_bytes"],
            "evidence.stack_high_water_bytes",
            0.0,
        ),
        "max_temp_c": finite(evidence["max_temp_c"], "evidence.max_temp_c"),
        "average_power_mw": finite(
            evidence["average_power_mw"], "evidence.average_power_mw", 0.0
        ),
    }
    if result["cpu_percent"] > 100.0:
        raise ValueError("evidence.cpu_percent must be <= 100")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--eval-summary", required=True, type=pathlib.Path)
    parser.add_argument("--eval-provenance", required=True, type=pathlib.Path)
    parser.add_argument("--board-summary", required=True, type=pathlib.Path)
    parser.add_argument("--evidence", required=True, type=pathlib.Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    source_sha = args.source_sha.strip().lower()
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_sha) is None:
        raise ValueError("--source-sha must be a 40- or 64-character hex Git object id")
    corpus_id = args.corpus_id.strip()
    if not corpus_id:
        raise ValueError("--corpus-id must be non-empty")

    model = read_model(args.model)
    pack = read_pack(args.keywords)
    token_map = load_tokens(args.tokens)
    token_size = vocab_size(token_map)
    token_fingerprint = vocab_fingerprint(token_map)
    if not (
        model["vocab_size"] == pack["vocab_size"] == token_size
        and model["vocab_fingerprint"]
        == pack["vocab_fingerprint"]
        == token_fingerprint
    ):
        raise ValueError("model, keyword pack, and token vocabulary identity differ")

    model_hash = sha256_file(args.model)
    pack_hash = sha256_file(args.keywords)
    eval_summary = load_json(args.eval_summary)
    eval_provenance = load_json(args.eval_provenance)
    if eval_provenance.get("model_sha256") != model_hash or eval_provenance.get(
        "keyword_pack_sha256"
    ) != pack_hash:
        raise ValueError("evaluation provenance references different model/keyword pack")

    board_summary = load_json(args.board_summary)
    evidence_raw = load_json(args.evidence)
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
        },
        "vocabulary": {
            "size": token_size,
            "fingerprint": f"0x{token_fingerprint:016x}",
            "sha256": sha256_file(args.tokens),
        },
        "artifacts": {
            "model": {
                "name": args.model.name,
                "sha256": model_hash,
                "bytes": model["bytes"],
                "feature_dim": model["feature_dim"],
                "hidden_dim": model["hidden_dim"],
            },
            "keyword_pack": {
                "name": args.keywords.name,
                "sha256": pack_hash,
                "bytes": pack["bytes"],
                "keyword_count": pack["keyword_count"],
            },
            "config": {
                "name": args.config.name,
                "sha256": sha256_file(args.config),
                "bytes": args.config.stat().st_size,
            },
        },
        "evaluation": {
            "summary_sha256": sha256_file(args.eval_summary),
            "provenance_sha256": sha256_file(args.eval_provenance),
            **validate_eval(eval_summary, eval_provenance),
        },
        "board": {
            "summary_sha256": sha256_file(args.board_summary),
            **validate_board(board_summary, model["bytes"], pack["bytes"]),
        },
        "evidence": {
            "sha256": sha256_file(args.evidence),
            **validate_evidence(evidence_raw),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
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
