from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

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


def build_prototype(
    *,
    tokens_path: pathlib.Path,
    carriers_path: pathlib.Path,
    output: pathlib.Path,
    feature_dim: int,
    input_scale: float,
    output_scale: float,
    blank_bias: float,
    token_bias: float,
) -> dict:
    token_map = load_tokens(tokens_path)
    size = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    carriers = json.loads(carriers_path.read_text(encoding="utf-8"))
    if not isinstance(carriers, dict) or not carriers:
        raise ValueError("token carrier map must be a non-empty object")
    active = sorted(carriers, key=lambda token: token_map[token])
    if len(active) > 64:
        raise ValueError("prototype hidden dimension exceeds runtime bound")
    hidden_dim = len(active)
    if not (0.0 < input_scale < 1.0 and 0.0 < output_scale < 1.0):
        raise ValueError("prototype scales must be in (0,1)")
    if not all(math.isfinite(value) for value in (blank_bias, token_bias)):
        raise ValueError("prototype biases must be finite")

    wx = bytearray(hidden_dim * feature_dim)
    wh = bytearray(hidden_dim * hidden_dim)
    wo = bytearray(size * hidden_dim)
    bh = [0.0] * hidden_dim
    bo = [-6.0] * size
    bo[0] = blank_bias

    for hidden_index, token in enumerate(active):
        feature_index = int(carriers[token]["feature_index"])
        if feature_index < 0 or feature_index >= feature_dim:
            raise ValueError(f"carrier feature index for {token} is out of range")
        token_id = token_map[token]
        wx[hidden_index * feature_dim + feature_index] = 127
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
        input_scale,
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

    provenance = {
        "schema_version": 1,
        "evidence_class": "synthetic-prototype",
        "model_sha256": sha256_file(output),
        "model_bytes": total,
        "tokens_sha256": sha256_file(tokens_path),
        "vocab_fingerprint": f"0x{fingerprint:016x}",
        "carrier_map_sha256": sha256_file(carriers_path),
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "active_tokens": active,
        "input_scale": input_scale,
        "output_scale": output_scale,
        "blank_bias": blank_bias,
        "token_bias": token_bias,
        "note": "Synthetic-only prototype weights; not a Mandarin acoustic quality claim.",
    }
    provenance_path = pathlib.Path(str(output) + ".synthetic-provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--carriers", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--input-scale", type=float, default=0.010)
    parser.add_argument("--output-scale", type=float, default=0.050)
    parser.add_argument("--blank-bias", type=float, default=1.8)
    parser.add_argument("--token-bias", type=float, default=-1.2)
    args = parser.parse_args()
    result = build_prototype(
        tokens_path=args.tokens.resolve(),
        carriers_path=args.carriers.resolve(),
        output=args.output.resolve(),
        feature_dim=args.feature_dim,
        input_scale=args.input_scale,
        output_scale=args.output_scale,
        blank_bias=args.blank_bias,
        token_bias=args.token_bias,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
