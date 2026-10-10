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

from frontend import features  # noqa: E402
from frontend_spec import features_pcm16  # noqa: E402

from train_ctc import (  # noqa: E402
    Manifest,
    collate,
    pcm_vad_mask,
    validate_warm_start,
    vad_aligned_ctc_log_probs,
)
from verify_model_promotion_bundle import require_promotable_training  # noqa: E402


def main() -> int:
    # Frontend/VAD fixtures only: no model or optimizer is instantiated here.
    for frontend in ("logmel", "pcen-lite"):
        for count, steps in ((0, 0), (1, 0), (399, 0), (400, 1), (719, 1), (720, 2)):
            samples = [(index % 37) * 311 - 5500 for index in range(count)]
            pcm = torch.tensor(samples, dtype=torch.float32) / 32768.0
            acoustic = features(pcm, frontend=frontend)
            if tuple(acoustic.shape) != (steps, 32):
                raise AssertionError((frontend, count, acoustic.shape, steps))
            if steps:
                expected = torch.tensor(features_pcm16(samples, frontend=frontend))
                torch.testing.assert_close(acoustic, expected, atol=5.0e-4, rtol=1.0e-4)
            if count and tuple(pcm_vad_mask(pcm, -55.0).shape) != (steps,):
                raise AssertionError(("VAD frame count", count, steps))
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
