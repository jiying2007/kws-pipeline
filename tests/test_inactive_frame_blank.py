#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from train_ctc import inactive_frame_blank_loss  # noqa: E402


def main() -> int:
    logits = torch.tensor(
        [[[0.0, 4.0, 0.0]], [[0.0, 4.0, 0.0]],
         [[0.0, 0.0, 4.0]], [[0.0, 4.0, 0.0]], [[0.0, 4.0, 0.0]]],
        requires_grad=True,
    )
    log_probs = logits.log_softmax(dim=2)
    vad = torch.tensor([[False, True, True, False, False]], dtype=torch.bool)
    lengths = torch.tensor([4])
    targets = torch.tensor([2])
    loss = inactive_frame_blank_loss(log_probs, vad, lengths, targets)
    assert loss.shape == (1,) and float(loss[0].detach()) > 0.0
    loss.sum().backward()
    gradient = logits.grad.detach().abs().sum(dim=2).squeeze(1)
    assert float(gradient[0]) > 0.0 and float(gradient[3]) > 0.0
    assert all(float(gradient[index]) == 0.0 for index in (1, 2, 4))

    blank_favored = logits.detach().clone()
    blank_favored[[0, 3], 0, 0] = 8.0
    improved = inactive_frame_blank_loss(
        blank_favored.log_softmax(dim=2), vad, lengths, targets,
    )
    assert float(improved[0]) < float(loss[0].detach())

    empty_target = inactive_frame_blank_loss(
        log_probs.detach(), vad, lengths, torch.tensor([0]),
    )
    assert float(empty_target[0]) == 0.0
    try:
        inactive_frame_blank_loss(log_probs.detach(), vad[:, :-1], lengths, targets)
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("accepted a misaligned VAD mask")
    print("inactive frame blank objective: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
