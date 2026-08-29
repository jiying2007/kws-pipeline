from __future__ import annotations

import hashlib
import json
import pathlib
import random
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

from acoustic_scene import render_scene
from fit_prototype import (
    align4,
    energetic_partition,
    evaluate_confusion,
    float_bytes,
    quantize_softmax,
    runtime_hidden,
    train_softmax,
)
from frontend_spec import FRONTEND_LOGMEL, features_pcm16, frontend_id
from render_domains import sample_scene, validate_domains
from synthetic_audio import render_tone_tokens

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


def fit_domain_prototype(
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
    frontend: str = FRONTEND_LOGMEL,
    curriculum_weights: dict[str, float] | None = None,
    epochs: int = 360,
    learning_rate: float = 0.32,
    l2: float = 2.0e-4,
) -> dict:
    if variants_per_token < 8:
        raise ValueError("domain prototype variants_per_token must be >= 8")
    domains = validate_domains(config)
    input_scale = projection_gain / 127.0
    token_map = load_tokens(tokens_path)
    size = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    carriers = json.loads(carriers_path.read_text(encoding="utf-8"))
    if not isinstance(carriers, dict) or not carriers:
        raise ValueError("token carrier map must be non-empty")
    active = sorted(carriers, key=lambda token: token_map[token])
    classes = [0] + [token_map[token] for token in active]
    hidden_dim = feature_dim
    generator_cfg = config.get("generator", {})
    tts_cfg = dict(generator_cfg.get("tts", {}))
    tts_cfg["lead_ms"] = max(120.0, float(tts_cfg.get("lead_ms", 160.0)))
    tts_cfg["tail_ms"] = max(120.0, float(tts_cfg.get("tail_ms", 180.0)))

    training_output.mkdir(parents=True, exist_ok=True)
    fit_examples: list[tuple[list[float], int]] = []
    validation_examples: list[tuple[list[float], int]] = []
    rows: list[dict] = []
    histogram = {"near": 0, "mid": 0, "far": 0}
    fit_variants = max(6, (variants_per_token * 3) // 4)

    for token_index, token in enumerate(active):
        token_id = token_map[token]
        for variant in range(variants_per_token):
            sample_seed = seed + token_index * 1_000_003 + variant * 65_537
            rng = random.Random(sample_seed)
            clean = render_tone_tokens([token], carriers, rng, tts_cfg)
            scene = sample_scene(
                domains,
                rng,
                curriculum_weights=curriculum_weights if variant < fit_variants else None,
            )
            mono, scene_meta = render_scene(
                clean,
                scene,
                seed=sample_seed + 17,
                afe=domains["afe"],
            )
            histogram[scene_meta["distance_band"]] += 1
            frames = features_pcm16(mono, feature_dim=feature_dim, frontend=frontend)
            energetic, background = energetic_partition(frames)
            destination = fit_examples if variant < fit_variants else validation_examples
            for frame in energetic:
                destination.append((runtime_hidden(frame, input_scale), token_id))
            if variant < fit_variants:
                for frame in background:
                    fit_examples.append((runtime_hidden(frame, input_scale), 0))
            pcm = b"".join(struct.pack("<h", value) for value in mono)
            rows.append(
                {
                    "token": token,
                    "token_id": token_id,
                    "variant": variant,
                    "partition": "fit" if variant < fit_variants else "validation",
                    "seed": sample_seed,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "domain": scene_meta,
                    "energetic_frames": len(energetic),
                    "background_frames": len(background),
                }
            )

    # Explicit pure-background domain scenes teach blank across near/mid/far
    # spectral conditions. The clean carrier is silence, while the renderer still
    # contributes room/noise/playback and AFE transformations.
    for variant in range(max(12, variants_per_token)):
        sample_seed = seed + 50_000_003 + variant * 4099
        rng = random.Random(sample_seed)
        scene = sample_scene(domains, rng, curriculum_weights=curriculum_weights)
        silence = [0] * (FRAME_LENGTH_SAMPLES + 4 * FRAME_HOP_SAMPLES)
        mono, scene_meta = render_scene(silence, scene, seed=sample_seed + 29, afe=domains["afe"])
        frames = features_pcm16(mono, feature_dim=feature_dim, frontend=frontend)
        for frame in frames:
            fit_examples.append((runtime_hidden(frame, input_scale), 0))
        histogram[scene_meta["distance_band"]] += 1
        rows.append(
            {
                "token": "<blk>",
                "token_id": 0,
                "variant": variant,
                "partition": "fit-background",
                "seed": sample_seed,
                "domain": scene_meta,
                "energetic_frames": 0,
                "background_frames": len(frames),
            }
        )

    weights, biases, losses = train_softmax(
        fit_examples,
        classes,
        hidden_dim,
        blank_bias=blank_bias,
        token_bias=token_bias,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    quantized, actual_output_scale, maximum = quantize_softmax(weights, output_scale)
    train_confusion = evaluate_confusion(fit_examples, classes, quantized, biases, actual_output_scale)
    validation_confusion = evaluate_confusion(validation_examples, classes, quantized, biases, actual_output_scale)
    if validation_confusion["accuracy"] < 0.985:
        raise ValueError(
            "quantized domain prototype token validation accuracy is below 98.5%; "
            f"got {validation_confusion['accuracy']:.6f}"
        )

    wx = bytearray(hidden_dim * feature_dim)
    for index in range(min(hidden_dim, feature_dim)):
        wx[index * feature_dim + index] = 127
    wh = bytearray(hidden_dim * hidden_dim)
    bh = [0.0] * hidden_dim
    wo = bytearray(size * hidden_dim)
    bo = [-8.0] * size
    for class_id in classes:
        bo[class_id] = biases[class_id]
        base = class_id * hidden_dim
        for index, value in enumerate(quantized[class_id]):
            wo[base + index] = value & 0xFF

    buffer = bytearray(b"\x00" * HEADER_BYTES)
    offsets: list[int] = []
    for block in (bytes(wx), bytes(wh), float_bytes(bh), bytes(wo), float_bytes(bo)):
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
        frontend_id(frontend),
        SAMPLE_RATE_HZ,
        FRAME_LENGTH_SAMPLES,
        FRAME_HOP_SAMPLES,
        input_scale,
        1.0e-4,
        actual_output_scale,
        fingerprint,
        *offsets,
        total,
    )
    buffer[:HEADER_BYTES] = header
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(buffer)

    sample_path = training_output / "domain-fit-samples.jsonl"
    sample_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, allow_nan=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    diagnostics = {
        "schema_version": 1,
        "frontend": frontend,
        "distance_histogram": histogram,
        "curriculum_weights": curriculum_weights or {},
        "optimizer": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
        "train_confusion": train_confusion,
        "validation_confusion": validation_confusion,
    }
    diagnostics_path = training_output / "domain-softmax-diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    provenance = {
        "schema_version": 3,
        "evidence_class": "synthetic-domain-trained-softmax-prototype",
        "model_sha256": sha256_file(output),
        "model_bytes": total,
        "tokens_sha256": sha256_file(tokens_path),
        "carrier_map_sha256": sha256_file(carriers_path),
        "fit_samples_sha256": sha256_file(sample_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "vocab_fingerprint": f"0x{fingerprint:016x}",
        "frontend_name": frontend,
        "frontend_kind": frontend_id(frontend),
        "distance_histogram": histogram,
        "curriculum_weights": curriculum_weights or {},
        "actual_output_scale": actual_output_scale,
        "float_output_weight_max_abs": maximum,
        "validation_confusion": validation_confusion,
        "note": "All optimizer samples are synthetic acoustic-domain scenes; evaluation and qualification splits are never used for fitting.",
    }
    provenance_path = pathlib.Path(str(output) + ".synthetic-domain-provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return provenance
