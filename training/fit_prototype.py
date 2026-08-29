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
from synthetic_audio import (
    augment,
    clamp16,
    noise_profile,
    render_tone_tokens,
)

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


def energetic_partition(
    frames: list[list[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    if not frames:
        return [], []
    peaks = [max(frame) - min(frame) for frame in frames]
    peak = max(peaks)
    if peak <= 1.0e-9:
        return [], frames
    threshold = peak * 0.45
    energetic = [frame for frame, value in zip(frames, peaks) if value >= threshold]
    background = [frame for frame, value in zip(frames, peaks) if value < threshold]
    if not energetic:
        energetic = [frames[peaks.index(peak)]]
    return energetic, background


def runtime_hidden(frame: list[float], input_scale: float) -> list[float]:
    # The emitted Wx matrix is a 127-valued identity matrix, so this is exactly
    # the runtime hidden transform when Wh/Bh are zero.
    gain = 127.0 * input_scale
    return [math.tanh(gain * value) for value in frame]


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exps = [math.exp(max(-60.0, min(0.0, value - maximum))) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def train_softmax(
    examples: list[tuple[list[float], int]],
    classes: list[int],
    hidden_dim: int,
    *,
    blank_bias: float,
    token_bias: float,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[dict[int, list[float]], dict[int, float], list[float]]:
    if epochs <= 0 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("invalid prototype softmax optimizer settings")
    if not examples:
        raise ValueError("cannot train prototype from zero examples")

    class_to_index = {class_id: index for index, class_id in enumerate(classes)}
    counts = {class_id: 0 for class_id in classes}
    for features, class_id in examples:
        if len(features) != hidden_dim or class_id not in class_to_index:
            raise ValueError("prototype training example shape/class mismatch")
        counts[class_id] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError("prototype training requires examples for every class")

    weights = {class_id: [0.0] * hidden_dim for class_id in classes}
    biases = {
        class_id: blank_bias if class_id == 0 else token_bias for class_id in classes
    }
    losses: list[float] = []
    class_count = float(len(classes))

    for epoch in range(epochs):
        grad_w = {class_id: [0.0] * hidden_dim for class_id in classes}
        grad_b = {class_id: 0.0 for class_id in classes}
        loss = 0.0
        for features, target in examples:
            logits = [
                biases[class_id]
                + sum(
                    weight * value
                    for weight, value in zip(weights[class_id], features)
                )
                for class_id in classes
            ]
            probabilities = softmax(logits)
            sample_weight = 1.0 / (class_count * counts[target])
            target_index = class_to_index[target]
            loss -= sample_weight * math.log(max(probabilities[target_index], 1.0e-12))
            for index, class_id in enumerate(classes):
                error = probabilities[index] - (1.0 if class_id == target else 0.0)
                scale = sample_weight * error
                grad_b[class_id] += scale
                row = grad_w[class_id]
                for feature_index, value in enumerate(features):
                    row[feature_index] += scale * value

        regularization = 0.0
        for class_id in classes:
            for feature_index in range(hidden_dim):
                weight = weights[class_id][feature_index]
                regularization += 0.5 * l2 * weight * weight
                grad_w[class_id][feature_index] += l2 * weight
        loss += regularization
        losses.append(loss)

        rate = learning_rate / math.sqrt(1.0 + epoch / 80.0)
        for class_id in classes:
            biases[class_id] -= rate * grad_b[class_id]
            for feature_index in range(hidden_dim):
                weights[class_id][feature_index] -= rate * grad_w[class_id][feature_index]

    return weights, biases, losses


def quantize_softmax(
    weights: dict[int, list[float]],
    requested_scale: float,
) -> tuple[dict[int, list[int]], float, float]:
    maximum = max(
        max(abs(value) for value in row) if row else 0.0 for row in weights.values()
    )
    actual_scale = max(requested_scale, maximum / 127.0, 1.0e-8)
    quantized = {
        class_id: [
            max(-127, min(127, int(round(value / actual_scale)))) for value in row
        ]
        for class_id, row in weights.items()
    }
    return quantized, actual_scale, maximum


def classify_quantized(
    features: list[float],
    classes: list[int],
    quantized: dict[int, list[int]],
    biases: dict[int, float],
    scale: float,
) -> tuple[int, float, float]:
    logits = [
        biases[class_id]
        + scale
        * sum(weight * value for weight, value in zip(quantized[class_id], features))
        for class_id in classes
    ]
    ranked = sorted(range(len(classes)), key=lambda index: logits[index], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else ranked[0]
    probabilities = softmax(logits)
    return classes[best], probabilities[best], logits[best] - logits[second]


def evaluate_confusion(
    examples: list[tuple[list[float], int]],
    classes: list[int],
    quantized: dict[int, list[int]],
    biases: dict[int, float],
    scale: float,
) -> dict:
    matrix = {
        str(target): {str(predicted): 0 for predicted in classes} for target in classes
    }
    correct = 0
    min_margin = float("inf")
    min_confidence = 1.0
    for features, target in examples:
        predicted, confidence, margin = classify_quantized(
            features, classes, quantized, biases, scale
        )
        matrix[str(target)][str(predicted)] += 1
        correct += int(predicted == target)
        min_margin = min(min_margin, margin)
        min_confidence = min(min_confidence, confidence)
    return {
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / len(examples) if examples else 0.0,
        "min_top1_margin": min_margin if examples else 0.0,
        "min_top1_confidence": min_confidence if examples else 0.0,
        "matrix": matrix,
    }


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
    epochs: int = 320,
    learning_rate: float = 0.35,
    l2: float = 2.0e-4,
) -> dict:
    if variants_per_token < 4:
        raise ValueError("variants_per_token must be >= 4")
    if feature_dim <= 0 or feature_dim > 40 or feature_dim > 64:
        raise ValueError("prototype feature dimension is outside runtime bounds")
    if not math.isfinite(projection_gain) or projection_gain <= 0.0:
        raise ValueError("projection_gain must be finite and > 0")
    if not math.isfinite(output_scale) or output_scale <= 0.0:
        raise ValueError("output_scale must be finite and > 0")
    if not all(math.isfinite(value) for value in (blank_bias, token_bias)):
        raise ValueError("prototype biases must be finite")

    input_scale = projection_gain / 127.0
    token_map = load_tokens(tokens_path)
    size = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    carriers = json.loads(carriers_path.read_text(encoding="utf-8"))
    if not isinstance(carriers, dict) or not carriers:
        raise ValueError("token carrier map must be a non-empty object")
    active = sorted(carriers, key=lambda token: token_map[token])
    active_ids = [token_map[token] for token in active]
    classes = [0] + active_ids
    hidden_dim = feature_dim

    generator_cfg = config.get("generator", {})
    tts_cfg = dict(generator_cfg.get("tts", {}))
    augment_cfg = generator_cfg.get("augment", {})
    tts_cfg["lead_ms"] = max(100.0, float(tts_cfg.get("lead_ms", 160.0)))
    tts_cfg["tail_ms"] = max(100.0, float(tts_cfg.get("tail_ms", 180.0)))

    training_output.mkdir(parents=True, exist_ok=True)
    fit_examples: list[tuple[list[float], int]] = []
    validation_examples: list[tuple[list[float], int]] = []
    sample_rows: list[dict] = []
    class_frame_counts = {class_id: 0 for class_id in classes}

    # Use deterministic token-only synthetic samples for frame supervision. The
    # last quarter of variants is held out from optimizer updates and is used
    # only for an internal train-domain confusion check.
    fit_variants = max(3, (variants_per_token * 3) // 4)
    for token_index, token in enumerate(active):
        token_id = token_map[token]
        for variant in range(variants_per_token):
            sample_seed = seed + token_index * 100_003 + variant * 1009
            rng = random.Random(sample_seed)
            samples = render_tone_tokens([token], carriers, rng, tts_cfg)
            samples = augment(samples, rng, augment_cfg)
            all_frames = features_pcm16(samples, feature_dim=feature_dim)
            energetic, background = energetic_partition(all_frames)
            destination = fit_examples if variant < fit_variants else validation_examples
            for frame in energetic:
                destination.append((runtime_hidden(frame, input_scale), token_id))
                class_frame_counts[token_id] += 1
            # Lead/tail frames teach the explicit CTC blank class. Only use them
            # for fitting variants so validation remains token-focused.
            if variant < fit_variants:
                for frame in background:
                    fit_examples.append((runtime_hidden(frame, input_scale), 0))
                    class_frame_counts[0] += 1
            pcm = b"".join(struct.pack("<h", value) for value in samples)
            sample_rows.append(
                {
                    "token": token,
                    "token_id": token_id,
                    "variant": variant,
                    "partition": "fit" if variant < fit_variants else "validation",
                    "seed": sample_seed,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                    "energetic_frames": len(energetic),
                    "background_frames": len(background),
                }
            )

    # Explicit background/noise frames prevent the learned head from assigning
    # token probability to fan/motor/media spectra merely because no token is
    # strongly dominant. These are generated independently from all eval splits.
    profiles = augment_cfg.get("noise_profiles", ["white", "fan", "motor", "media"])
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("prototype fitting requires non-empty noise profiles")
    for profile_index, profile_value in enumerate(profiles):
        profile = str(profile_value)
        for variant in range(max(6, variants_per_token // 2)):
            sample_seed = seed + 7_000_001 + profile_index * 100_003 + variant * 1237
            rng = random.Random(sample_seed)
            count = FRAME_LENGTH_SAMPLES + 3 * FRAME_HOP_SAMPLES
            noise = noise_profile(profile, count, rng)
            amplitude = rng.uniform(700.0, 5200.0)
            samples = [clamp16(value * amplitude) for value in noise]
            frames = features_pcm16(samples, feature_dim=feature_dim)
            for frame in frames:
                fit_examples.append((runtime_hidden(frame, input_scale), 0))
                class_frame_counts[0] += 1
            pcm = b"".join(struct.pack("<h", value) for value in samples)
            sample_rows.append(
                {
                    "token": "<blk>",
                    "token_id": 0,
                    "variant": variant,
                    "partition": "fit-background",
                    "profile": profile,
                    "seed": sample_seed,
                    "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
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
    quantized, actual_output_scale, float_weight_max = quantize_softmax(
        weights, output_scale
    )
    train_confusion = evaluate_confusion(
        fit_examples, classes, quantized, biases, actual_output_scale
    )
    validation_confusion = evaluate_confusion(
        validation_examples, classes, quantized, biases, actual_output_scale
    )
    if validation_confusion["accuracy"] < 0.995:
        raise ValueError(
            "quantized prototype token validation accuracy is below 99.5%; "
            f"got {validation_confusion['accuracy']:.6f}"
        )

    # Identity feature projection plus learned competitive output head. Wh stays
    # zero so this CI backend remains a deterministic frame classifier while the
    # production torch_ctc backend keeps the full recurrent architecture.
    wx = bytearray(hidden_dim * feature_dim)
    for index in range(min(hidden_dim, feature_dim)):
        wx[index * feature_dim + index] = 127
    wh = bytearray(hidden_dim * hidden_dim)
    bh = [0.0] * hidden_dim
    wo = bytearray(size * hidden_dim)
    bo = [-8.0] * size
    for class_id in classes:
        bo[class_id] = biases[class_id]
        row = quantized[class_id]
        base = class_id * hidden_dim
        for index, value in enumerate(row):
            wo[base + index] = value & 0xFF

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
        actual_output_scale,
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
    diagnostics = {
        "schema_version": 1,
        "classes": classes,
        "class_frame_counts": {str(key): value for key, value in class_frame_counts.items()},
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
    diagnostics_path = training_output / "softmax-diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    provenance = {
        "schema_version": 2,
        "evidence_class": "synthetic-trained-softmax-prototype",
        "model_sha256": sha256_file(output),
        "model_bytes": total,
        "tokens_sha256": sha256_file(tokens_path),
        "carrier_map_sha256": sha256_file(carriers_path),
        "fit_samples_sha256": sha256_file(samples_path),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "vocab_fingerprint": f"0x{fingerprint:016x}",
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "variants_per_token": variants_per_token,
        "fit_variants_per_token": fit_variants,
        "input_scale": input_scale,
        "requested_output_scale": output_scale,
        "actual_output_scale": actual_output_scale,
        "float_output_weight_max_abs": float_weight_max,
        "blank_bias_initial": blank_bias,
        "token_bias_initial": token_bias,
        "optimizer": diagnostics["optimizer"],
        "train_confusion": train_confusion,
        "validation_confusion": validation_confusion,
        "note": "Competitive softmax weights were trained only from deterministic synthetic token/background fitting samples; calibration/test/qualification audio was never used for optimizer updates.",
    }
    provenance_path = pathlib.Path(str(output) + ".synthetic-provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return provenance
