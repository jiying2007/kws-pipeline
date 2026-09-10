from __future__ import annotations

import math
import pathlib
import wave

import torch

from frontend import features
from model import TinyStreamingRNN
from sequence_margin import _decoder_sequence_log_confidence

FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320


def load_checkpoint_model(path: pathlib.Path) -> tuple[TinyStreamingRNN, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyStreamingRNN(
        int(checkpoint["feature_dim"]),
        int(checkpoint["hidden_dim"]),
        int(checkpoint["vocab_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def acoustic_from_samples(
    samples: list[int], *, feature_dim: int, frontend: str
) -> torch.Tensor:
    if not samples:
        raise ValueError("cannot score empty audio")
    pcm = torch.tensor(samples, dtype=torch.float32) / 32768.0
    return features(
        pcm,
        feature_dim,
        frame_len=FRAME_LENGTH_SAMPLES,
        hop=FRAME_HOP_SAMPLES,
        frontend=frontend,
    )


def read_pcm16(path: pathlib.Path) -> list[int]:
    import struct

    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
        raw = reader.readframes(reader.getnframes())
    if not raw:
        raise ValueError(f"{path}: WAV is empty")
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def keyword_confidences(
    model: TinyStreamingRNN,
    samples: list[int],
    *,
    feature_dim: int,
    frontend: str,
    keyword_sequences: list[list[int]],
) -> list[float]:
    acoustic = acoustic_from_samples(samples, feature_dim=feature_dim, frontend=frontend)
    with torch.no_grad():
        log_probs = model(acoustic.unsqueeze(0)).log_softmax(dim=2)[:, 0, :]
        values = [
            _decoder_sequence_log_confidence(log_probs, tuple(int(v) for v in sequence))
            for sequence in keyword_sequences
        ]
    return [0.0 if not torch.isfinite(value) else math.exp(float(value)) for value in values]


def score_wav(
    model: TinyStreamingRNN,
    path: pathlib.Path,
    *,
    feature_dim: int,
    frontend: str,
    keyword_sequences: list[list[int]],
) -> list[float]:
    return keyword_confidences(
        model,
        read_pcm16(path),
        feature_dim=feature_dim,
        frontend=frontend,
        keyword_sequences=keyword_sequences,
    )
