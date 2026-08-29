from __future__ import annotations

import hashlib
import json
import math
import pathlib
import random
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

from frontend_spec import features_pcm16
from synthetic_audio import augment, render_tone_tokens

MODEL_VERSION = 2
HEADER_BYTES = 72
SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def align4(buffer: bytearray) -> None:
    while len(buffer) % 4:
        buffer.append(0)


def float_bytes(values: list[float]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def mean_vectors(vectors: list[list[float]], feature_dim: int) -> list[float]:
    if not vectors:
        raise ValueError("cannot fit prototype from zero feature vectors")
    sums = [0.0] * feature_dim
    for vector in vectors:
        if len(vector) != feature_dim:
            raise ValueError("prototype training feature dimension mismatch")
        for index, value in enumerate(vector):
            sums[index] += value
    scale = 1.0 / len(vectors)
    return [value * scale for value in sums]


def energetic_frames(frames: list[list[float]]) -> list[list[float]]:
    if not frames:
        return []
    peaks = [max(frame) - min(frame) for frame in frames]
    peak = max(peaks)
    if peak <= 1.0e-9:
        return frames
    threshold = peak * 0.45
    selected = [frame for frame, value in zip(frames, peaks) if value >= threshold]
    return selected or [frames[peaks.index(peak)]]


def q8_vector(vector: list[float]) -> tuple[list[int], float]:
    maximum = max(max(abs(value) for value in vector), 1.0e-8)
    quantized = [
        max(-127, min(127, int(round(value * 127.0 / maximum)))) for value in vector
    ]
    return quantized, maximum


def fit_prototype(
    *,
    config: dict,
    tokens_path: pathlib.Path,
    carriers_path: pathlib.Path,
    output: pathlib.Path,
    training_output: pathlib.Path,
    feature_dim: int,
    variants_per_token: int,
    projection_gain: float,
    output_scale: float,
    blank_bias: float,
    token_bias: float,
    seed: int,
) -> dict:
    if variants_per_token <= 0:
        raise ValueError("variants_per_token must be > 0")
    if not math.isfinite(projection_gain) or projection_gain <= 0.0:
        raise ValueError("projection_gain must be finite and > 0")
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output_scale must be finite and > 0")
    if not all(math.isfinite(value) for value in (blank_bias, token_bias)):
        raise ValueError("prototype biases must be finite")

    token_map = load_tokens(tokens_path)
    size = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    carriers = json.loads(carriers_path.read_text(encoding="utf-8"))
    if not isinstance(carriers, dict) or not carriers:
        raise ValueError("token carrier map must be a non-empty object")
    active = sorted(carriers, key=lambda token: token_map[token])
    hidden_dim = len(active)
    if hidden_dim > 64:
        raise ValueError("prototype hidden dimension exceeds runtime bound")

    generator_cfg = config.get("generator", {})
    tts_cfg = dict(generator_cfg.get("tts", {}))
    augment_cfg = generator_cfg.get("augment", {})
    # Isolated fitting samples contain only the target token. Silence around the
    # token is deliberately retained so the energetic-frame selector sees the
    # same frame boundaries as the runtime frontend.
    tts_cfg["lead_ms"] = max(80.0, float(tts_cfg.get("lead_ms", 160.0)))
    tts_cfg["tail_ms"] = max(80.0, float(tts_cfg.get("tail_ms", 180.0)))

    training_output.mkdir(parents=True, exist_ok=True)
    token_frames: dict[str, list[list[float]]] = {token: [] for token in active}
    sample_rows: list[dict] = []
    for token_index, token in enumerate(active):
        for variant in range(variants_per_token):
            sample_seed = seed + token_index * 100_003 + variant * 1009
            rng = random.Random(sample_seed)
            samples = render_tone_tokens([token], carriers, rng, tts_cfg)
            samples = augment(samples, rng, augment_cfg)
            frames = energetic_frames(features_pcm16(samples, feature_dim=feature_dim))
            token_frames[token].extend(frames)
            # The fitting evidence records generated PCM by canonical little-endian
            # sample bytes without needing to retain every WAV in CI.
            pcm = b"".join(struct.pack("<h", value) for value in samples)
            sample_rows.append(
                {
                    "token": token,
                    "token_id": token_map[token],
                    "variant": variant,
                    "seed": sample_seed,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "frames_used": len(frames),
                }
            )

    means = {
        token: mean_vectors(token_frames[token], feature_dim) for token in active
    }
    fitted: dict[str, dict] = {}
    wx_rows: list[list[int]] = []
    raw_maxima: list[float] = []
    for token in active:
        others = [means[other] for other in active if other != token]
        background = (
            mean_vectors(others, feature_dim) if others else [0.0] * feature_dim
        )
        delta = [value - base for value, base in zip(means[token], background)]
        q, maximum = q8_vector(delta)
        wx_rows.append(q)
        raw_maxima.append(maximum)
        fitted[token] = {
            "token_id": token_map[token],
            "frames": len(token_frames[token]),
            "delta_max_abs": maximum,
            "top_features": sorted(
                range(feature_dim), key=lambda index: abs(delta[index]), reverse=True
            )[:6],
        }

    # One global matrix scale is required by ABI v2. Normalize each row to its
    # own direction, then use projection_gain/127 so hidden activation depends
    # on the learned discriminant direction rather than raw token energy.
    wx_scale = projection_gain / 127.0
    wx = bytearray()
    for row in wx_rows:
        wx.extend((value & 0xFF) for value in row)
    wh = bytearray(hidden_dim * hidden_dim)
    wo = bytearray(size * hidden_dim)
    bh = [0.0] * hidden_dim
    bo = [-6.0] * size
    bo[0] = blank_bias
    for hidden_index, token in enumerate(active):
        token_id = token_map[token]
        wo[token_id * hidden_dim + hidden_index] = 127
        bo[token_id] = token_bias

    buffer = bytearray(b"\x00" * HEADER_BYTES)
    offsets: list[int] = []
    blocks = (bytes(wx), bytes(wh), float_bytes(bh), bytes(wo), float_bytes(bo))
    for block in blocks:
        align4(buffer)
        offsets.append(len(buffer))
        buffer.extend(block)
    total = len(buffer)
    header = struct.pack(
        "<4sHHHHHHIIIfffQIIIIII",
        b"KWSP",
        MODEL_VERSION,
        HEADER_BYTES,
        feature_dim,
        hidden_dim,
        size,
        0,
        SAMPLE_RATE_HZ,
        FRAME_LENGTH_SAMPLES,
        FRAME_HOP_SAMPLES,
        wx_scale,
        1.0e-4,
        output_scale,
        fingerprint,
        *offsets,
        total,
    )
    if len(header) != HEADER_BYTES:
        raise RuntimeError("internal ABI-v2 header size mismatch")
    buffer[:HEADER_BYTES] = header
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(buffer)

    samples_path = training_output / "token-fit-samples.jsonl"
    samples_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in sample_rows) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "evidence_class": "synthetic-fitted-prototype",
        "model_sha256": sha256_file(output),
        "model_bytes": total,
        "tokens_sha256": sha256_file(tokens_path),
        "carrier_map_sha256": sha256_file(carriers_path),
        "fit_samples_sha256": sha256_file(samples_path),
        "vocab_fingerprint": f"0x{fingerprint:016x}",
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "variants_per_token": variants_per_token,
        "projection_gain": projection_gain,
        "wx_scale": wx_scale,
        "output_scale": output_scale,
        "blank_bias": blank_bias,
        "token_bias": token_bias,
        "tokens": fitted,
        "note": "Weights were fitted only from deterministic synthetic training-token samples; no calibration/test/qualification audio was used.",
    }
    provenance_path = pathlib.Path(str(output) + ".synthetic-provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance
