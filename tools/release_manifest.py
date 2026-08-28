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
PACK_RECORD = struct.Struct("<IfHH16H")
MODEL_VERSION = 2
PACK_VERSION = 2
MODEL_HEADER_BYTES = 72
PACK_HEADER_BYTES = 24
SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512
MAX_KEYWORDS = 16
MAX_TOKENS_PER_KEYWORD = 16
SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def sha256_value(value, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA256 hex")
    return value


def json_int(value, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f"..{maximum}" if maximum is not None else " or greater"
        raise ValueError(f"{label} must be in {minimum}{suffix}")
    return value


def finite(value, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return result


def required_text(obj: dict, key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be non-empty text")
    result = value.strip()
    if result.upper() == "REPLACE_ME":
        raise ValueError(f"{label}.{key} still contains the template placeholder")
    return result


def align4(value: int) -> int:
    return (value + 3) & ~3


def all_finite_f32(blob: bytes, offset: int, count: int) -> bool:
    return all(
        math.isfinite(struct.unpack_from("<f", blob, offset + index * 4)[0])
        for index in range(count)
    )


def read_model(path: pathlib.Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < MODEL_HEADER_BYTES:
        raise ValueError(f"{path}: model is shorter than ABI-v2 header")
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
    ) = MODEL_HEADER.unpack(blob[:MODEL_HEADER_BYTES])

    if magic != b"KWSP" or version != MODEL_VERSION or header_bytes != MODEL_HEADER_BYTES:
        raise ValueError(f"{path}: expected canonical KWSP ABI v{MODEL_VERSION}")
    if reserved != 0 or total_bytes != len(blob):
        raise ValueError(f"{path}: non-canonical model header")
    if (
        sample_rate != SAMPLE_RATE_HZ
        or frame_length != FRAME_LENGTH_SAMPLES
        or frame_hop != FRAME_HOP_SAMPLES
    ):
        raise ValueError(f"{path}: model geometry is outside ABI-v2 contract")
    if not (1 <= feature_dim <= MAX_FEATURE_DIM):
        raise ValueError(f"{path}: feature_dim out of range")
    if not (1 <= hidden_dim <= MAX_HIDDEN_DIM):
        raise ValueError(f"{path}: hidden_dim out of range")
    if not (2 <= model_vocab_size <= MAX_VOCAB_SIZE) or fingerprint == 0:
        raise ValueError(f"{path}: invalid model vocabulary identity")
    for label, value in (
        ("wx_scale", wx_scale),
        ("wh_scale", wh_scale),
        ("wo_scale", wo_scale),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{path}: {label} must be finite and positive")

    wx_bytes = hidden_dim * feature_dim
    wh_bytes = hidden_dim * hidden_dim
    bh_bytes = hidden_dim * 4
    wo_bytes = model_vocab_size * hidden_dim
    bo_bytes = model_vocab_size * 4
    expected_wx = align4(MODEL_HEADER_BYTES)
    expected_wh = align4(expected_wx + wx_bytes)
    expected_bh = align4(expected_wh + wh_bytes)
    expected_wo = align4(expected_bh + bh_bytes)
    expected_bo = align4(expected_wo + wo_bytes)
    expected_total = expected_bo + bo_bytes
    if (
        wx_off != expected_wx
        or wh_off != expected_wh
        or bh_off != expected_bh
        or wo_off != expected_wo
        or bo_off != expected_bo
        or total_bytes != expected_total
    ):
        raise ValueError(f"{path}: non-canonical model tensor layout")
    if not all_finite_f32(blob, bh_off, hidden_dim) or not all_finite_f32(
        blob, bo_off, model_vocab_size
    ):
        raise ValueError(f"{path}: model bias contains NaN/Inf")

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
    magic, version, header_bytes, count, pack_vocab_size, total_bytes, fingerprint = (
        PACK_HEADER.unpack(blob[:PACK_HEADER_BYTES])
    )
    if magic != b"KWKP" or version != PACK_VERSION or header_bytes != PACK_HEADER_BYTES:
        raise ValueError(f"{path}: expected canonical KWKP ABI v{PACK_VERSION}")
    if not (1 <= count <= MAX_KEYWORDS):
        raise ValueError(f"{path}: keyword count out of range")
    if not (2 <= pack_vocab_size <= MAX_VOCAB_SIZE) or fingerprint == 0:
        raise ValueError(f"{path}: invalid keyword-pack vocabulary identity")
    if total_bytes != len(blob) or total_bytes != PACK_HEADER_BYTES + count * PACK_RECORD.size:
        raise ValueError(f"{path}: non-canonical keyword-pack length")

    seen_ids: set[int] = set()
    seen_paths: set[tuple[int, ...]] = set()
    for index in range(count):
        offset = PACK_HEADER_BYTES + index * PACK_RECORD.size
        keyword_id, threshold, num_tokens, reserved, *tokens = PACK_RECORD.unpack_from(
            blob, offset
        )
        if not math.isfinite(threshold) or not (0.0 < threshold < 1.0):
            raise ValueError(f"{path}: keyword[{index}] threshold is invalid")
        if not (1 <= num_tokens <= MAX_TOKENS_PER_KEYWORD) or reserved != 0:
            raise ValueError(f"{path}: keyword[{index}] record is non-canonical")
        active = tuple(tokens[:num_tokens])
        padding = tokens[num_tokens:]
        if any(token <= 0 or token >= pack_vocab_size for token in active):
            raise ValueError(f"{path}: keyword[{index}] token out of range")
        if any(token != 0 for token in padding):
            raise ValueError(f"{path}: keyword[{index}] padding must be zero")
        if keyword_id in seen_ids or active in seen_paths:
            raise ValueError(f"{path}: duplicate keyword id or token path")
        seen_ids.add(keyword_id)
        seen_paths.add(active)

    return {
        "bytes": len(blob),
        "keyword_count": count,
        "vocab_size": pack_vocab_size,
        "vocab_fingerprint": fingerprint,
    }


def close_enough(actual: float, expected: float, label: str, *, rel: float, abs_: float) -> None:
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_):
        raise ValueError(f"{label} is internally inconsistent: {actual} vs {expected}")


def validate_eval(summary: dict, provenance: dict) -> dict:
    if json_int(provenance.get("schema_version"), "evaluation provenance schema_version") != 1:
        raise ValueError("evaluation provenance schema_version must be 1")
    hashes = {
        key: sha256_value(provenance.get(key), f"evaluation provenance {key}")
        for key in (
            "runner_sha256",
            "model_sha256",
            "keyword_pack_sha256",
            "references_sha256",
            "detections_sha256",
        )
    }
    recordings = json_int(provenance.get("recordings"), "evaluation provenance recordings", 1)
    detections = json_int(
        provenance.get("detections"), "evaluation provenance detections", 0
    )
    if summary.get("references_sha256") != hashes["references_sha256"] or summary.get(
        "detections_sha256"
    ) != hashes["detections_sha256"]:
        raise ValueError("evaluation summary does not match provenance inputs")

    result = {
        "recordings": json_int(summary["recordings"], "evaluation.recordings", 1),
        "audio_hours": finite(summary["audio_hours"], "evaluation.audio_hours", 0.0),
        "expected": json_int(summary["expected"], "evaluation.expected", 0),
        "matched": json_int(summary["matched"], "evaluation.matched", 0),
        "false_rejects": json_int(
            summary["false_rejects"], "evaluation.false_rejects", 0
        ),
        "false_accepts": json_int(
            summary["false_accepts"], "evaluation.false_accepts", 0
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
        "references_sha256": hashes["references_sha256"],
        "detections_sha256": hashes["detections_sha256"],
        "runner_sha256": hashes["runner_sha256"],
    }
    if result["audio_hours"] <= 0.0:
        raise ValueError("evaluation.audio_hours must be > 0")
    if result["recordings"] != recordings:
        raise ValueError("evaluation recording count does not match provenance")
    if result["matched"] + result["false_rejects"] != result["expected"]:
        raise ValueError("evaluation expected/matched/false-reject counts disagree")
    if result["matched"] + result["false_accepts"] != detections:
        raise ValueError("evaluation matched/false-accept counts disagree with detections")
    expected_frr = (
        result["false_rejects"] / result["expected"] if result["expected"] else 0.0
    )
    expected_far = result["false_accepts"] / result["audio_hours"]
    close_enough(result["frr"], expected_frr, "evaluation.frr", rel=1e-9, abs_=1e-12)
    close_enough(
        result["far_per_hour"], expected_far, "evaluation.far_per_hour", rel=1e-9, abs_=1e-12
    )
    if result["frr"] > 1.0 or result["p50_post_end_latency_ms"] > result[
        "p95_post_end_latency_ms"
    ]:
        raise ValueError("evaluation summary contains impossible values")
    return result


def validate_board(
    summary: dict,
    *,
    model_bytes: int,
    pack_bytes: int,
    model_sha256: str,
    pack_sha256: str,
    runner_sha256: str,
    audio_sha256: str,
) -> dict:
    if json_int(summary.get("schema_version"), "board.schema_version") != 1:
        raise ValueError("board benchmark schema_version must be 1")
    for key, expected in (
        ("runner_sha256", runner_sha256),
        ("model_sha256", model_sha256),
        ("keyword_pack_sha256", pack_sha256),
        ("audio_sha256", audio_sha256),
    ):
        actual = sha256_value(summary.get(key), f"board.{key}")
        if actual != expected:
            raise ValueError(f"board benchmark {key} does not match release artifact")
    if json_int(summary.get("block_samples"), "board.block_samples") != FRAME_HOP_SAMPLES:
        raise ValueError("board benchmark must use one 20-ms KWS hop per block")
    if json_int(summary.get("model_bytes"), "board.model_bytes", 1) != model_bytes or json_int(
        summary.get("keyword_pack_bytes"), "board.keyword_pack_bytes", 1
    ) != pack_bytes:
        raise ValueError("board benchmark artifact sizes do not match release artifacts")

    result = {
        "runner_sha256": runner_sha256,
        "model_sha256": model_sha256,
        "keyword_pack_sha256": pack_sha256,
        "audio_sha256": audio_sha256,
        "audio_seconds": finite(summary["audio_seconds"], "board.audio_seconds", 0.0),
        "repeats": json_int(summary["repeats"], "board.repeats", 1),
        "blocks": json_int(summary["blocks"], "board.blocks", 1),
        "arena_bytes": json_int(summary["arena_bytes"], "board.arena_bytes", 1),
        "block_deadline_us": finite(
            summary["block_deadline_us"], "board.block_deadline_us", 0.0
        ),
        "total_process_us": finite(
            summary["total_process_us"], "board.total_process_us", 0.0
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
    if result["audio_seconds"] <= 0.0 or result["block_deadline_us"] != 20000.0:
        raise ValueError("board benchmark audio/deadline is invalid")
    if not (
        result["p50_process_us"] <= result["p95_process_us"]
        <= result["p99_process_us"]
        <= result["max_process_us"]
    ):
        raise ValueError("board benchmark percentiles are not monotonic")
    close_enough(
        result["mean_process_us"],
        result["total_process_us"] / result["blocks"],
        "board.mean_process_us",
        rel=1e-6,
        abs_=0.01,
    )
    close_enough(
        result["rtf"],
        result["total_process_us"]
        / (result["audio_seconds"] * result["repeats"] * 1_000_000.0),
        "board.rtf",
        rel=2e-5,
        abs_=1e-8,
    )
    expected_headroom = (
        result["block_deadline_us"] / result["p99_process_us"]
        if result["p99_process_us"] > 0.0
        else 0.0
    )
    close_enough(
        result["p99_headroom"],
        expected_headroom,
        "board.p99_headroom",
        rel=2e-5,
        abs_=1e-4,
    )
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
    parser.add_argument("--board-runner", required=True, type=pathlib.Path)
    parser.add_argument("--board-audio", required=True, type=pathlib.Path)
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
    board_runner_hash = sha256_file(args.board_runner)
    board_audio_hash = sha256_file(args.board_audio)
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
            "board_runner": {
                "name": args.board_runner.name,
                "sha256": board_runner_hash,
                "bytes": args.board_runner.stat().st_size,
            },
            "board_audio": {
                "name": args.board_audio.name,
                "sha256": board_audio_hash,
                "bytes": args.board_audio.stat().st_size,
            },
        },
        "evaluation": {
            "summary_sha256": sha256_file(args.eval_summary),
            "provenance_sha256": sha256_file(args.eval_provenance),
            **validate_eval(eval_summary, eval_provenance),
        },
        "board": {
            "summary_sha256": sha256_file(args.board_summary),
            **validate_board(
                board_summary,
                model_bytes=model["bytes"],
                pack_bytes=pack["bytes"],
                model_sha256=model_hash,
                pack_sha256=pack_hash,
                runner_sha256=board_runner_hash,
                audio_sha256=board_audio_hash,
            ),
        },
        "evidence": {
            "sha256": sha256_file(args.evidence),
            **validate_evidence(evidence_raw),
        },
    }

    if manifest["artifacts"]["config"]["bytes"] <= 0:
        raise ValueError("runtime config must be non-empty")

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
