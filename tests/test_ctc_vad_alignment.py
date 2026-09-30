#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import tempfile
import wave
from types import SimpleNamespace

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from train_ctc import (  # noqa: E402
    Manifest,
    auxiliary_vad_log_probs,
    collate,
    ordered_token_loss,
    pcm_vad_mask,
    validate_warm_start,
    vad_aligned_ctc_log_probs,
)
from verify_model_promotion_bundle import require_promotable_training  # noqa: E402


def main() -> int:
    pcm = torch.cat((torch.zeros(400), torch.full((320,), 0.5)))
    assert torch.equal(pcm_vad_mask(pcm, -55.0), torch.tensor([False, True]))

    logits = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[2.0, 3.0, 1.0], [1.0, 3.0, 2.0]],
        ],
        requires_grad=True,
    )
    vad = torch.tensor([[False, True], [False, False]])
    aligned = vad_aligned_ctc_log_probs(logits, vad, torch.tensor([1, 0]))
    lengths = torch.tensor([2, 2])
    unmodified, unmodified_lengths = auxiliary_vad_log_probs(
        logits, vad, lengths, torch.tensor([1, 0]), enabled=False)
    assert unmodified is logits and unmodified_lengths is lengths
    compressed, compressed_lengths = auxiliary_vad_log_probs(
        logits, vad, lengths, torch.tensor([1, 0]), enabled=True)
    assert torch.equal(compressed_lengths, torch.tensor([1, 2]))
    assert torch.equal(compressed[0, 0], logits[1, 0])
    assert torch.equal(compressed[:, 1], logits[:, 1])
    try:
        auxiliary_vad_log_probs(logits, None, lengths, torch.tensor([1, 0]), enabled=True)
    except ValueError as exc:
        assert "requires a VAD mask" in str(exc)
    else:
        raise AssertionError("auxiliary VAD alignment accepted no mask")

    # A target whose first chronological quarter is silent must still have
    # learnable active frames after time compression.
    silence_first = torch.zeros((8, 1, 3), requires_grad=True)
    active_only, active_lengths = auxiliary_vad_log_probs(
        silence_first.log_softmax(dim=2),
        torch.tensor([[False, False, False, False, True, True, True, True]]),
        torch.tensor([8]), torch.tensor([2]), enabled=True,
    )
    ordered, _, _ = ordered_token_loss(
        active_only, torch.tensor([1, 2]), active_lengths, torch.tensor([2]),
    )
    ordered.backward()
    assert silence_first.grad[:4].abs().sum() == 0
    assert silence_first.grad[4:].abs().sum() > 0
    assert torch.equal(aligned[0, 0], torch.tensor([0.0, -30.0, -30.0]))
    assert torch.equal(aligned[1, 0], logits[1, 0])
    assert torch.equal(aligned[:, 1], logits[:, 1])
    aligned[1, 0, 1].backward()
    assert logits.grad[1, 0, 1] == 1.0
    assert logits.grad[0, 0, 1] == 0.0

    acoustic = torch.zeros((2, 3))
    target = torch.tensor([1], dtype=torch.long)
    assert len(collate([(acoustic, target)])) == 4
    padded = collate([(acoustic, target, torch.tensor([False, True]))])
    assert len(padded) == 5
    assert torch.equal(padded[4][0, :2], torch.tensor([False, True]))
    with tempfile.TemporaryDirectory(prefix="ctc-vad-alignment-") as temporary:
        root = pathlib.Path(temporary)
        wav = root / "silence.wav"
        with wave.open(str(wav), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x00" * 400)
        manifest = root / "train.tsv"
        manifest.write_text("silence.wav\t1\n", encoding="utf-8")
        dataset = Manifest([manifest], 32, 5, "logmel", ctc_vad_threshold_dbfs=-55.0)
        try:
            dataset[0]
        except ValueError as exc:
            assert "too few speech-active frames" in str(exc)
        else:
            raise AssertionError("unlearnable CTC/VAD row was accepted")
    assert require_promotable_training({"sample_weighting": {}}) == {"sample_weighting": {}}
    for training in (
        {"development_only": True},
        {"ctc_vad_alignment": {"development_only": True}},
    ):
        try:
            require_promotable_training(training)
        except ValueError as exc:
            assert "cannot promote" in str(exc)
        else:
            raise AssertionError("development VAD model could enter product promotion")
    development_checkpoint = {
        "feature_dim": 32,
        "hidden_dim": 64,
        "vocab_size": 5,
        "frame_length_samples": 400,
        "frame_hop_samples": 320,
        "frontend_spec_version": 2,
        "vocab_fingerprint": 123,
        "frontend_kind": 0,
        "development_recipe": "development-pcm-dbfs-gated-ctc-v1",
    }
    try:
        validate_warm_start(
            development_checkpoint,
            SimpleNamespace(feature_dim=32, hidden_dim=64, frontend="logmel"),
            5,
            123,
            "0" * 64,
        )
    except ValueError as exc:
        assert "development-only checkpoint" in str(exc)
    else:
        raise AssertionError("development VAD checkpoint could warm-start formal training")
    print("test_ctc_vad_alignment: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
