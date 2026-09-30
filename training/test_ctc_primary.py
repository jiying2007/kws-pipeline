#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

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
from training_state import state_identity  # noqa: E402
from verify_training_readback import sha256_file, verify_candidate  # noqa: E402


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
        label_prior_contract=contract,
    )
    reference_label_loss = label_prior_ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        priors_a,
        alpha=LABEL_PRIOR_ALPHA,
    )
    assert label_loss.shape == expected.shape
    assert torch.isfinite(label_loss).all()
    assert torch.allclose(
        label_loss,
        reference_label_loss,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
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

    # Readback must bind config -> checkpoint -> provenance -> prior contract.
    with tempfile.TemporaryDirectory(prefix="ctc-primary-readback-") as tmp:
        candidate = pathlib.Path(tmp) / "candidate"
        candidate.mkdir()
        checkpoint_path = candidate / "model.pt"
        model_path = candidate / "model.kwm"
        provenance_path = candidate / "model.kwm.provenance.json"

        state = {"weight": torch.tensor([1.0, -2.0], dtype=torch.float32)}
        identity = state_identity(state)
        corpus_identity = {
            "schema_version": 1,
            "corpus_sha256": "c" * 64,
        }
        readback_prior = build_label_prior_contract(
            torch.tensor([0.8, 0.1, 0.1], dtype=torch.float32),
            frame_count=123,
            training_corpus_sha256=corpus_identity["corpus_sha256"],
            source_float_state_sha256=identity["sha256"],
        )
        auxiliary = {
            "ordered_token_loss_weight": 0.35,
            "keyword_sequence_margin_loss_weight": 0.10,
            "prefix_completion_loss_weight": 0.10,
            "recurrent_release_loss_weight": 0.05,
            "suffix_root_suppression_loss_weight": 0.0,
        }
        runtime = {
            "cpu_runtime": {"identity": "test-cpu"},
            "torch_runtime": {"identity": "test-torch"},
        }
        payload = {
            "state_dict": state,
            "float_state_identity": identity,
            "auxiliary_loss_weights": auxiliary,
            "ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR,
            "ctc_primary_policy_scope": "primary-loss-path-v1",
            "ctc_label_prior": readback_prior,
            "training_environment": runtime,
            "training_corpus_identity": corpus_identity,
        }
        torch.save(payload, checkpoint_path)
        model_path.write_bytes(b"KWM-test-model")
        provenance = {
            "checkpoint": {"sha256": sha256_file(checkpoint_path)},
            "model": {"sha256": sha256_file(model_path)},
            "training": {
                "float_state_identity": identity,
                "auxiliary_loss_weights": auxiliary,
                "ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR,
                "ctc_label_prior": readback_prior,
                "environment": runtime,
                "corpus_identity": corpus_identity,
            },
        }
        provenance_path.write_text(
            json.dumps(provenance, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        verified = verify_candidate(
            {"ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR},
            checkpoint_path,
        )
        assert verified["ctc_primary_policy"] == CTC_PRIMARY_POLICY_LABEL_PRIOR
        assert verified["ctc_label_prior"] == readback_prior
        assert verified["training_corpus_sha256"] == "c" * 64

        bad = json.loads(json.dumps(provenance))
        bad["training"]["ctc_label_prior"]["values_sha256"] = "0" * 64
        provenance_path.write_text(
            json.dumps(bad, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            verify_candidate(
                {"ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR},
                checkpoint_path,
            )
        except ValueError as exc:
            assert "label-prior" in str(exc) or "values SHA" in str(exc)
        else:
            raise AssertionError("tampered label-prior readback was accepted")

        bad = json.loads(json.dumps(provenance))
        bad["training"]["corpus_identity"]["corpus_sha256"] = "d" * 64
        provenance_path.write_text(
            json.dumps(bad, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        try:
            verify_candidate(
                {"ctc_primary_policy": CTC_PRIMARY_POLICY_LABEL_PRIOR},
                checkpoint_path,
            )
        except ValueError as exc:
            assert "training corpus identity" in str(exc)
        else:
            raise AssertionError("mismatched label-prior corpus was accepted")

    print("test_ctc_primary: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
