from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from objective_config import (  # noqa: E402
    auxiliary_loss_weights,
    optional_objective_cli_args,
    verify_auxiliary_loss_readback,
)
from suffix_root_loss import (  # noqa: E402
    SUFFIX_ROOT_BLANK_MARGIN,
    SUFFIX_ROOT_POLICY,
    suffix_root_suppression_loss,
)


def main() -> int:
    assert SUFFIX_ROOT_POLICY == "vad-inactive-one-token-prefix-omission-hinge-v1"
    assert SUFFIX_ROOT_BLANK_MARGIN == 0.5

    logits = torch.full((4, 3, 5), -4.0, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        logits[:, :, 0] = -0.2
        logits[0, 0, 0] = -2.0
        logits[0, 0, 3] = -0.1
        logits[1, 0, 3] = -0.01
    targets = torch.tensor([4, 3, 4, 3, 4, 3, 4, 1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([3, 4, 3], dtype=torch.long)
    input_lengths = torch.tensor([4, 4, 4], dtype=torch.long)
    vad_mask = torch.tensor(
        [
            [False, True, True, True],
            [False, True, True, True],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )
    keyword_sequences = [[3, 4, 3, 4], [1, 2, 3, 4]]

    loss = suffix_root_suppression_loss(
        log_probs=logits,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        vad_mask=vad_mask,
        keyword_sequences=keyword_sequences,
    )
    assert loss.shape == (3,)
    assert float(loss[0].detach()) > 2.0
    assert float(loss[1].detach()) == 0.0
    assert float(loss[2].detach()) == 0.0
    loss.sum().backward()
    assert float(logits.grad[0, 0, 3]) > 0.0
    assert float(logits.grad[0, 0, 0]) < 0.0
    assert float(logits.grad[1, 0, 3]) == 0.0

    all_active = suffix_root_suppression_loss(
        log_probs=logits.detach(),
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        vad_mask=torch.ones_like(vad_mask),
        keyword_sequences=keyword_sequences,
    )
    assert torch.equal(all_active, torch.zeros_like(all_active))

    weights = auxiliary_loss_weights({"suffix_root_suppression_loss_weight": 0.1})
    assert weights == {"suffix_root_suppression_loss_weight": 0.1}
    legacy = {
        "ordered_token_loss_weight": 0.35,
        "keyword_sequence_margin_loss_weight": 0.10,
        "prefix_completion_loss_weight": 0.10,
        "recurrent_release_loss_weight": 0.05,
    }
    normalized = verify_auxiliary_loss_readback({}, legacy)
    assert normalized["suffix_root_suppression_loss_weight"] == 0.0
    args = optional_objective_cli_args(
        {
            "suffix_root_suppression_loss_weight": 0.1,
            "ctc_vad_align": True,
        }
    )
    assert "--suffix-root-suppression-loss-weight" in args
    assert "0.1" in args
    assert "--ctc-vad-align" in args
    assert "--ctc-vad-align" not in optional_objective_cli_args({"ctc_vad_align": False})

    trainer = (ROOT / "training" / "train_ctc.py").read_text(encoding="utf-8")
    assert "--suffix-root-suppression-loss-weight requires --ctc-vad-align" in trainer
    assert "suffix_root_policy" in trainer
    assert "suffix_root_suppression_loss_weight" in trainer

    print("test_suffix_root_objective: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
