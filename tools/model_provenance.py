from __future__ import annotations

from collections import Counter
import pathlib

from qualification_common import finite, json_int, load_json, sha256_value

FRONTEND_SPEC_VERSION = 1


def text(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def boolean(value, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def validate_model_provenance(
    path: pathlib.Path,
    *,
    model_sha256: str,
    model_bytes: int,
    feature_dim: int,
    hidden_dim: int,
    vocab_size: int,
    vocab_fingerprint: int,
    tokens_sha256: str,
    checkpoint_sha256: str,
    training_tokens_sha256: str,
    training_manifest_sha256s: list[str],
) -> dict:
    provenance = load_json(path)
    if json_int(provenance.get("schema_version"), "model provenance schema_version") != 1:
        raise ValueError("model provenance schema_version must be 1")

    model = provenance.get("model")
    checkpoint = provenance.get("checkpoint")
    tokens = provenance.get("tokens")
    training = provenance.get("training")
    quantization = provenance.get("quantization")
    if not all(
        isinstance(item, dict)
        for item in (model, checkpoint, tokens, training, quantization)
    ):
        raise ValueError("model provenance is missing required object sections")

    if (
        sha256_value(model.get("sha256"), "model provenance model.sha256")
        != model_sha256
    ):
        raise ValueError("model provenance references a different .kwm")
    if json_int(model.get("bytes"), "model provenance model.bytes", 1) != model_bytes:
        raise ValueError("model provenance model size does not match .kwm")
    if json_int(model.get("abi"), "model provenance model.abi") != 2:
        raise ValueError("model provenance requires model ABI v2")
    if (
        json_int(model.get("feature_dim"), "model provenance model.feature_dim", 1)
        != feature_dim
    ):
        raise ValueError("model provenance feature_dim does not match .kwm")
    if (
        json_int(model.get("hidden_dim"), "model provenance model.hidden_dim", 1)
        != hidden_dim
    ):
        raise ValueError("model provenance hidden_dim does not match .kwm")
    if (
        json_int(model.get("vocab_size"), "model provenance model.vocab_size", 2)
        != vocab_size
    ):
        raise ValueError("model provenance vocab_size does not match .kwm")
    expected_fingerprint = f"0x{vocab_fingerprint:016x}"
    if model.get("vocab_fingerprint") != expected_fingerprint:
        raise ValueError("model provenance vocabulary fingerprint does not match .kwm")
    if (
        json_int(
            model.get("frontend_spec_version"),
            "model provenance model.frontend_spec_version",
        )
        != FRONTEND_SPEC_VERSION
    ):
        raise ValueError("model provenance frontend spec is unsupported")

    checkpoint_name = text(checkpoint.get("name"), "model provenance checkpoint.name")
    checkpoint_hash = sha256_value(
        checkpoint.get("sha256"), "model provenance checkpoint.sha256"
    )
    if checkpoint_hash != checkpoint_sha256:
        raise ValueError("model provenance checkpoint hash does not match selected checkpoint")

    if (
        sha256_value(tokens.get("sha256"), "model provenance tokens.sha256")
        != tokens_sha256
    ):
        raise ValueError("model provenance export token file differs from release vocabulary")
    training_tokens_hash = sha256_value(
        tokens.get("checkpoint_sha256"),
        "model provenance tokens.checkpoint_sha256",
    )
    if training_tokens_hash != training_tokens_sha256:
        raise ValueError(
            "model provenance training-token hash does not match selected training vocabulary"
        )
    byte_identical = boolean(
        tokens.get("byte_identical_to_training"),
        "model provenance tokens.byte_identical_to_training",
    )
    if byte_identical != (tokens_sha256 == training_tokens_hash):
        raise ValueError("model provenance token-byte identity flag is inconsistent")

    manifests = training.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise ValueError("model provenance training.manifests must be non-empty")
    normalized_manifests: list[dict] = []
    for index, item in enumerate(manifests):
        if not isinstance(item, dict):
            raise ValueError(
                f"model provenance training.manifests[{index}] must be an object"
            )
        normalized_manifests.append(
            {
                "name": text(
                    item.get("name"),
                    f"model provenance training.manifests[{index}].name",
                ),
                "sha256": sha256_value(
                    item.get("sha256"),
                    f"model provenance training.manifests[{index}].sha256",
                ),
            }
        )
    recorded_manifest_hashes = [item["sha256"] for item in normalized_manifests]
    if Counter(recorded_manifest_hashes) != Counter(training_manifest_sha256s):
        raise ValueError(
            "model provenance training manifests do not match selected manifest files"
        )

    normalized_training = {
        "manifests": normalized_manifests,
        "examples": json_int(
            training.get("examples"), "model provenance training.examples", 1
        ),
        "seed": json_int(training.get("seed"), "model provenance training.seed", 0),
        "epochs": json_int(
            training.get("epochs"), "model provenance training.epochs", 1
        ),
        "batch_size": json_int(
            training.get("batch_size"), "model provenance training.batch_size", 1
        ),
        "learning_rate": finite(
            training.get("learning_rate"),
            "model provenance training.learning_rate",
            0.0,
        ),
        "optimizer": text(
            training.get("optimizer"), "model provenance training.optimizer"
        ),
        "weight_decay": finite(
            training.get("weight_decay"),
            "model provenance training.weight_decay",
            0.0,
        ),
        "grad_clip_norm": finite(
            training.get("grad_clip_norm"),
            "model provenance training.grad_clip_norm",
            0.0,
        ),
    }
    if (
        normalized_training["learning_rate"] <= 0.0
        or normalized_training["grad_clip_norm"] <= 0.0
    ):
        raise ValueError("model provenance training learning-rate/grad-clip must be > 0")

    if quantization.get("scheme") != "symmetric-int8-per-matrix":
        raise ValueError("model provenance quantization scheme is unsupported")
    normalized_quantization: dict[str, dict | str] = {
        "scheme": "symmetric-int8-per-matrix"
    }
    for matrix in ("in_proj", "rec_proj", "out_proj"):
        stats = quantization.get(matrix)
        if not isinstance(stats, dict):
            raise ValueError(f"model provenance quantization.{matrix} is missing")
        normalized_quantization[matrix] = {
            "scale": finite(
                stats.get("scale"),
                f"model provenance quantization.{matrix}.scale",
                0.0,
            ),
            "max_abs_error": finite(
                stats.get("max_abs_error"),
                f"model provenance quantization.{matrix}.max_abs_error",
                0.0,
            ),
            "rmse": finite(
                stats.get("rmse"),
                f"model provenance quantization.{matrix}.rmse",
                0.0,
            ),
            "signal_rms": finite(
                stats.get("signal_rms"),
                f"model provenance quantization.{matrix}.signal_rms",
                0.0,
            ),
            "snr_db": finite(
                stats.get("snr_db"), f"model provenance quantization.{matrix}.snr_db"
            ),
        }
        matrix_stats = normalized_quantization[matrix]
        if not isinstance(matrix_stats, dict) or matrix_stats["scale"] <= 0.0:
            raise ValueError(f"model provenance quantization.{matrix}.scale must be > 0")

    return {
        "model_sha256": model_sha256,
        "tokens_sha256": tokens_sha256,
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": checkpoint_hash,
        "training_tokens_sha256": training_tokens_hash,
        "token_bytes_identical_to_training": byte_identical,
        "frontend_spec_version": FRONTEND_SPEC_VERSION,
        "training": normalized_training,
        "quantization": normalized_quantization,
    }
