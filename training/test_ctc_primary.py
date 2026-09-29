#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

import torch
from torch import nn
from torch.utils.data import Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from ctc_primary import (  # noqa: E402
    LABEL_PRIOR_ALPHA,
    build_label_prior_contract,
    estimate_training_label_priors,
    label_prior_ctc_loss,
    normalize_label_prior_contract,
    primary_ctc_per_sample,
)
from objective_config import (  # noqa: E402
    ctc_primary_policy_setting,
    optional_objective_cli_args,
)
from objective_contract import (  # noqa: E402
    CTC_PRIMARY_POLICY_LABEL_PRIOR,
    CTC_PRIMARY_POLICY_STANDARD,
)


def simple_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    logits = torch.tensor(
        [
            [[3.0, -1.0, -2.0], [3.0, -1.0, -2.0]],
            [[-1.0, 4.0, -2.0], [3.0, -1.0, -2.0]],
            [[3.0, -1.0, -2.0], [3.0, -1.0, -2.0]],
            [[-1.0, -2.0, 4.0], [3.0, -1.0, -2.0]],
            [[3.0, -1.0, -2.0], [3.0, -1.0, -2.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([1, 2], dtype=torch.long)
    input_lengths = torch.tensor([5, 5], dtype=torch.long)
    target_lengths = torch.tensor([2, 0], dtype=torch.long)
    return logits.log_softmax(dim=2), targets, input_lengths, target_lengths


class PriorDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            (
                torch.tensor(
                    [[1.0], [2.0], [3.0]],
                    dtype=torch.float32,
                ),
                torch.tensor([1], dtype=torch.long),
            ),
            (
                torch.tensor(
                    [[-1.0], [0.0]],
                    dtype=torch.float32,
                ),
                torch.empty(0, dtype=torch.long),
            ),
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class PriorModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(1, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor([[0.25], [0.75], [-0.50]], dtype=torch.float32)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def prior_collate(batch):
    xs, ys = zip(*batch)
    xlen = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    ylen = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    max_t = int(xlen.max())
    padded = torch.zeros((max_t, len(xs), 1), dtype=torch.float32)
    for index, x in enumerate(xs):
        padded[: x.shape[0], index] = x
    targets = torch.cat(ys) if any(int(y.numel()) for y in ys) else torch.empty(
        0, dtype=torch.long
    )
    return padded, targets, xlen, ylen


def main() -> int:
    policy, configured = ctc_primary_policy_setting({})
    assert policy == CTC_PRIMARY_POLICY_STANDARD
    assert configured is False
    assert optional_objective_cli_args({}) == []

    policy, configured = ctc_primary_policy_setting(
        {"ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR}
    )
    assert policy == CTC_PRIMARY_POLICY_LABEL_PRIOR
    assert configured is True
    assert optional_objective_cli_args(
        {"ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR}
    ) == ["--ctc-primary-policy", CTC_PRIMARY_POLICY_LABEL_PRIOR]
    try:
        ctc_primary_policy_setting({"ctc_primary_policy": "unsupported"})
    except ValueError as exc:
        assert "ctc_primary_policy" in str(exc)
    else:
        raise AssertionError("unsupported primary CTC policy was accepted")

    log_probs, targets, input_lengths, target_lengths = simple_batch()
    standard_loss = nn.CTCLoss(
        blank=0,
        zero_infinity=True,
        reduction="none",
    )
    expected = standard_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
    )
    actual = primary_ctc_per_sample(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        policy=CTC_PRIMARY_POLICY_STANDARD,
        standard_loss=standard_loss,
        label_prior_contract=None,
    )
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=0.0)

    uniform = torch.full((3,), 1.0 / 3.0, dtype=torch.float32)
    zero_alpha = label_prior_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        uniform,
        alpha=0.0,
    )
    assert torch.allclose(zero_alpha, expected, atol=1.0e-5, rtol=1.0e-5)

    dataset = PriorDataset()
    model = PriorModel()
    priors_a, frames_a = estimate_training_label_priors(
        model,
        dataset,
        batch_size=2,
        collate_fn=prior_collate,
        vocab_size=3,
    )
    priors_b, frames_b = estimate_training_label_priors(
        model,
        dataset,
        batch_size=1,
        collate_fn=prior_collate,
        vocab_size=3,
    )
    assert frames_a == frames_b == 5
    assert torch.allclose(priors_a, priors_b, atol=1.0e-7, rtol=0.0)
    assert abs(float(priors_a.sum()) - 1.0) < 1.0e-7

    contract = build_label_prior_contract(
        priors_a,
        frame_count=frames_a,
        training_corpus_sha256="a" * 64,
        source_float_state_sha256="b" * 64,
    )
    normalized = normalize_label_prior_contract(contract)
    assert normalized == contract
    assert normalized["alpha"] == LABEL_PRIOR_ALPHA
    assert normalized["frame_count"] == 5

    label_loss = primary_ctc_per_sample(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        policy=CTC_PRIMARY_POLICY_LABEL_PRIOR,
        standard_loss=standard_loss,
        label_prior_contract={
            **contract,
            "values": [float(value) for value in priors_a.tolist()],
            "values_sha256": contract["values_sha256"],
        },
    )
    assert label_loss.shape == expected.shape
    assert torch.isfinite(label_loss).all()
    assert not torch.allclose(label_loss, expected)

    try:
        primary_ctc_per_sample(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            policy=CTC_PRIMARY_POLICY_STANDARD,
            standard_loss=standard_loss,
            label_prior_contract=contract,
        )
    except ValueError as exc:
        assert "standard CTC" in str(exc)
    else:
        raise AssertionError("standard CTC accepted label-prior metadata")

    print("test_ctc_primary: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
