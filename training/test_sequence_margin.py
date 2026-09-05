#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn.functional as F

from sequence_margin import keyword_sequence_margin_loss


def make_logits(tokens: list[int], *, steps: int = 12, vocab: int = 5) -> torch.Tensor:
    logits = torch.full((steps, 1, vocab), -6.0, dtype=torch.float32)
    logits[:, :, 0] = 4.0
    count = len(tokens)
    positions = []
    for occurrence in range(count):
        start = (occurrence * steps) // count
        stop = max(start + 1, ((occurrence + 1) * steps) // count)
        positions.append((start + stop - 1) // 2)
    for position, token in zip(positions, tokens):
        logits[position, 0, 0] = -6.0
        logits[position, 0, token] = 8.0
    return logits.requires_grad_().log_softmax(dim=2)


def true_nll(log_probs: torch.Tensor, target: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    targets = torch.tensor(target, dtype=torch.long)
    input_lengths = torch.tensor([log_probs.shape[0]], dtype=torch.long)
    target_lengths = torch.tensor([len(target)], dtype=torch.long)
    raw = F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    return targets, input_lengths, target_lengths, raw


def margin(
    log_probs: torch.Tensor,
    target: list[int],
    keyword_sequences: list[list[int]] | None = None,
) -> torch.Tensor:
    targets, input_lengths, target_lengths, raw = true_nll(log_probs, target)
    return keyword_sequence_margin_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=input_lengths,
        target_lengths=target_lengths,
        true_ctc_nll=raw,
        keyword_sequences=keyword_sequences
        or [[1, 2, 3, 4], [3, 4, 3, 4]],
        blank=0,
        margin=0.05,
    )


def main() -> int:
    # A hard-negative kw2 prefix becomes unsafe when the missing terminal wo1 is
    # acoustically present: the full wake path outranks the configured negative
    # target and must receive a positive discriminative penalty.
    unsafe = make_logits([3, 4, 3, 4])
    unsafe_loss = margin(unsafe, [3, 4, 3])
    assert float(unsafe_loss.item()) > 0.05
    unsafe_loss.mean().backward()
    assert unsafe.grad_fn is not None

    # Adding unrelated wake words must not dilute the one dangerous path. Product
    # semantics are wake-on-any-keyword, so the per-sample objective is max hinge.
    unsafe_more_keywords = make_logits([3, 4, 3, 4])
    expanded_loss = margin(
        unsafe_more_keywords,
        [3, 4, 3],
        [[1, 2, 3, 4], [3, 4, 3, 4], [2, 3, 2, 3]],
    )
    assert abs(float(expanded_loss.item()) - float(unsafe_loss.item())) < 1.0e-5

    # When the incomplete prefix really terminates before the final wake token,
    # the true negative path already has adequate margin and needs no penalty.
    safe = make_logits([3, 4, 3])
    safe_loss = margin(safe, [3, 4, 3])
    assert float(safe_loss.item()) < 1.0e-6

    # A clean genuine keyword with every target strongly owning its chronological
    # completion region must already beat blank and other wake paths.
    positive = make_logits([1, 2, 3, 4])
    positive_loss = margin(positive, [1, 2, 3, 4])
    assert float(positive_loss.item()) < 1.0e-6

    # Streaming-completion contract: the whole wake path can still beat the
    # all-blank path because three tokens are very strong, while the terminal
    # token is locally weaker than blank. The temporal completion hinge must catch
    # that weakest occurrence instead of averaging it away.
    terminal_logits = torch.full((12, 1, 5), -6.0, dtype=torch.float32)
    terminal_logits[:, :, 0] = 4.0
    for position, token in zip([1, 4, 7], [1, 2, 3]):
        terminal_logits[position, 0, 0] = -6.0
        terminal_logits[position, 0, token] = 8.0
    terminal_logits[10, 0, 0] = 4.0
    terminal_logits[10, 0, 4] = 3.0
    terminal_log_probs = terminal_logits.requires_grad_().log_softmax(dim=2)
    terminal_loss = margin(terminal_log_probs, [1, 2, 3, 4])
    assert float(terminal_loss.item()) > 0.50
    terminal_loss.mean().backward()
    assert terminal_logits.grad is not None
    assert float(terminal_logits.grad[10, 0, 4].abs()) > 0.0

    # Recall-side global contract: all four target tokens are acoustically present,
    # but blank remains stronger at every completion frame. The true wake loses
    # both globally and temporally and therefore must receive a positive penalty.
    weak_logits = torch.full((12, 1, 5), -6.0, dtype=torch.float32)
    weak_logits[:, :, 0] = 6.0
    for position, token in zip([1, 4, 7, 10], [1, 2, 3, 4]):
        weak_logits[position, 0, token] = 4.0
    weak_log_probs = weak_logits.requires_grad_().log_softmax(dim=2)
    weak_loss = margin(weak_log_probs, [1, 2, 3, 4])
    assert float(weak_loss.item()) > 0.50
    weak_loss.mean().backward()
    assert weak_logits.grad is not None

    # If an input is too short to realize any wake sequence, each impossible wake sequence
    # has +inf CTC NLL. It is safely separated and must not be converted to zero
    # NLL, which would create a false margin penalty.
    short_logits = torch.full((2, 1, 5), -6.0, dtype=torch.float32)
    short_logits[:, :, 0] = 6.0
    short_log_probs = short_logits.log_softmax(dim=2)
    empty_targets = torch.empty(0, dtype=torch.long)
    short_lengths = torch.tensor([2], dtype=torch.long)
    empty_lengths = torch.tensor([0], dtype=torch.long)
    blank_nll = F.ctc_loss(
        short_log_probs,
        empty_targets,
        short_lengths,
        empty_lengths,
        blank=0,
        reduction="none",
        zero_infinity=False,
    )
    impossible_loss = keyword_sequence_margin_loss(
        log_probs=short_log_probs,
        targets=empty_targets,
        input_lengths=short_lengths,
        target_lengths=empty_lengths,
        true_ctc_nll=blank_nll,
        keyword_sequences=[[1, 2, 3, 4], [3, 4, 3, 4]],
        blank=0,
        margin=0.05,
    )
    assert torch.isfinite(impossible_loss).all()
    assert float(impossible_loss.item()) == 0.0

    print("test_sequence_margin: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
