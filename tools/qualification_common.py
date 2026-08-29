from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import struct

from kws_vocab import load_tokens, vocab_fingerprint, vocab_size

MODEL_HEADER = struct.Struct("<4sHHHHHHIIIfffQIIIIII")
PACK_HEADER = struct.Struct("<4sHHHHIQ")
PACK_RECORD = struct.Struct("<IfHBBBBH16H")
MODEL_VERSION = 2
PACK_VERSION = 3
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
FRONTEND_KINDS = {0: "logmel", 1: "pcen-lite"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_SHA_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
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
        raise ValueError(f"{label} is out of range")
    return value


def finite(value, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} is out of range or non-finite")
    return result


def required_text(obj: dict, key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be non-empty text")
    result = value.strip()
    if result.upper() == "REPLACE_ME":
        raise ValueError(f"{label}.{key} still contains the template placeholder")
    return result


def close_enough(actual: float, expected: float, label: str, rel: float, abs_: float) -> None:
    if not math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_):
        raise ValueError(f"{label} is internally inconsistent: {actual} vs {expected}")


def align4(value: int) -> int:
    return (value + 3) & ~3


def read_model(path: pathlib.Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < MODEL_HEADER_BYTES:
        raise ValueError(f"{path}: model is shorter than ABI-v2 header")
    fields = MODEL_HEADER.unpack(blob[:MODEL_HEADER_BYTES])
    (magic, version, header_bytes, feature_dim, hidden_dim, model_vocab_size,
     frontend_kind, sample_rate, frame_length, frame_hop, wx_scale, wh_scale,
     wo_scale, fingerprint, wx_off, wh_off, bh_off, wo_off, bo_off,
     total_bytes) = fields
    if magic != b"KWSP" or version != MODEL_VERSION or header_bytes != MODEL_HEADER_BYTES:
        raise ValueError(f"{path}: expected canonical KWSP ABI v{MODEL_VERSION}")
    if frontend_kind not in FRONTEND_KINDS or total_bytes != len(blob):
        raise ValueError(f"{path}: non-canonical model header")
    if (sample_rate, frame_length, frame_hop) != (
        SAMPLE_RATE_HZ, FRAME_LENGTH_SAMPLES, FRAME_HOP_SAMPLES
    ):
        raise ValueError(f"{path}: model geometry is outside ABI-v2 contract")
    if not (1 <= feature_dim <= MAX_FEATURE_DIM and 1 <= hidden_dim <= MAX_HIDDEN_DIM):
        raise ValueError(f"{path}: model dimensions are out of range")
    if not (2 <= model_vocab_size <= MAX_VOCAB_SIZE) or fingerprint == 0:
        raise ValueError(f"{path}: invalid model vocabulary identity")
    if any(not math.isfinite(scale) or scale <= 0.0 for scale in (wx_scale, wh_scale, wo_scale)):
        raise ValueError(f"{path}: model scale must be finite and positive")
    expected_wx = align4(MODEL_HEADER_BYTES)
    expected_wh = align4(expected_wx + hidden_dim * feature_dim)
    expected_bh = align4(expected_wh + hidden_dim * hidden_dim)
    expected_wo = align4(expected_bh + hidden_dim * 4)
    expected_bo = align4(expected_wo + model_vocab_size * hidden_dim)
    expected_total = expected_bo + model_vocab_size * 4
    if (wx_off, wh_off, bh_off, wo_off, bo_off, total_bytes) != (
        expected_wx, expected_wh, expected_bh, expected_wo, expected_bo, expected_total
    ):
        raise ValueError(f"{path}: non-canonical model tensor layout")
    for offset, count in ((bh_off, hidden_dim), (bo_off, model_vocab_size)):
        if any(not math.isfinite(struct.unpack_from("<f", blob, offset + i * 4)[0]) for i in range(count)):
            raise ValueError(f"{path}: model bias contains NaN/Inf")
    return {
        "bytes": len(blob), "feature_dim": feature_dim, "hidden_dim": hidden_dim,
        "vocab_size": model_vocab_size, "vocab_fingerprint": fingerprint,
        "frontend_kind": frontend_kind, "frontend_name": FRONTEND_KINDS[frontend_kind],
    }


def read_pack(path: pathlib.Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < PACK_HEADER_BYTES:
        raise ValueError(f"{path}: keyword pack is shorter than ABI-v3 header")
    magic, version, header_bytes, count, pack_vocab_size, total_bytes, fingerprint = PACK_HEADER.unpack(blob[:PACK_HEADER_BYTES])
    if magic != b"KWKP" or version != PACK_VERSION or header_bytes != PACK_HEADER_BYTES:
        raise ValueError(f"{path}: expected canonical KWKP ABI v{PACK_VERSION}")
    if not (1 <= count <= MAX_KEYWORDS and 2 <= pack_vocab_size <= MAX_VOCAB_SIZE) or fingerprint == 0:
        raise ValueError(f"{path}: invalid keyword-pack header")
    if total_bytes != len(blob) or total_bytes != PACK_HEADER_BYTES + count * PACK_RECORD.size:
        raise ValueError(f"{path}: non-canonical keyword-pack length")
    seen_ids: set[int] = set()
    seen_paths: set[tuple[int, ...]] = set()
    for index in range(count):
        fields = PACK_RECORD.unpack_from(blob, PACK_HEADER_BYTES + index * PACK_RECORD.size)
        keyword_id, threshold, num_tokens, min_blanks, priority, policy, grace_frames, reserved, *tokens = fields
        if not math.isfinite(threshold) or not (0.0 < threshold < 1.0):
            raise ValueError(f"{path}: keyword[{index}] threshold is invalid")
        if not (1 <= num_tokens <= MAX_TOKENS_PER_KEYWORD) or reserved != 0 or policy > 2:
            raise ValueError(f"{path}: keyword[{index}] record is non-canonical")
        if policy == 1 and min_blanks == 0:
            raise ValueError(f"{path}: longest keyword policy requires trailing blank")
        if policy == 2 and grace_frames == 0:
            raise ValueError(f"{path}: grace keyword policy requires grace frames")
        active = tuple(tokens[:num_tokens])
        if any(token <= 0 or token >= pack_vocab_size for token in active) or any(tokens[num_tokens:]):
            raise ValueError(f"{path}: keyword[{index}] token/padding is invalid")
        if keyword_id in seen_ids or active in seen_paths:
            raise ValueError(f"{path}: duplicate keyword id or token path")
        seen_ids.add(keyword_id)
        seen_paths.add(active)
    return {
        "bytes": len(blob), "keyword_count": count, "vocab_size": pack_vocab_size,
        "vocab_fingerprint": fingerprint,
    }


def read_vocabulary(path: pathlib.Path) -> dict:
    token_map = load_tokens(path)
    return {
        "size": vocab_size(token_map),
        "fingerprint": vocab_fingerprint(token_map),
        "sha256": sha256_file(path),
    }


def validate_runtime_config(path: pathlib.Path, model: dict) -> dict:
    config = load_json(path)
    if json_int(config.get("sample_rate_hz"), "config.sample_rate_hz") != SAMPLE_RATE_HZ:
        raise ValueError("runtime config sample rate does not match ABI-v2")
    if json_int(config.get("frame_length_samples"), "config.frame_length_samples") != FRAME_LENGTH_SAMPLES or json_int(config.get("frame_hop_samples"), "config.frame_hop_samples") != FRAME_HOP_SAMPLES:
        raise ValueError("runtime config frame geometry does not match ABI-v2")
    if json_int(config.get("feature_dim"), "config.feature_dim", 1) != model["feature_dim"] or json_int(config.get("hidden_dim"), "config.hidden_dim", 1) != model["hidden_dim"]:
        raise ValueError("runtime config dimensions do not match model")
    config_frontend = config.get("frontend")
    if config_frontend is not None and config_frontend != model["frontend_name"]:
        raise ValueError("runtime config frontend does not match model")
    required_text(config, "vocab_target", "config")
    runtime = config.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("config.runtime must be an object")
    result = {
        "min_speech_dbfs": finite(runtime.get("min_speech_dbfs"), "config.runtime.min_speech_dbfs"),
        "token_boost": finite(runtime.get("token_boost"), "config.runtime.token_boost", 0.0),
        "state_retention": finite(runtime.get("state_retention"), "config.runtime.state_retention"),
        "refractory_ms": json_int(runtime.get("refractory_ms"), "config.runtime.refractory_ms", 0, 10000),
    }
    if not 0.0 < result["state_retention"] < 1.0:
        raise ValueError("config.runtime.state_retention must be in (0,1)")
    return result
