#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TRAINING))
sys.path.insert(0, str(TOOLS))

from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402
from model import TinyStreamingRNN  # noqa: E402
from train_ctc import (  # noqa: E402
    FRAME_HOP_SAMPLES,
    FRAME_LENGTH_SAMPLES,
    FRONTEND_SPEC_VERSION,
    GRAD_CLIP_NORM,
    KEYWORD_SEQUENCE_MARGIN,
    KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT,
    MAX_FEATURE_DIM,
    MAX_HIDDEN_DIM,
    MAX_VOCAB_SIZE,
    ORDERED_TOKEN_LOSS_WEIGHT,
    POSITIVE_EXAMPLE_WEIGHT,
    PREFIX_COMPLETION_LOSS_WEIGHT,
    PREFIX_COMPLETION_TAIL_STEPS,
    RECURRENT_RELEASE_CONTEXT_STEPS,
    RECURRENT_RELEASE_LOSS_WEIGHT,
    RECURRENT_RELEASE_TAIL_STEPS,
    RECURRENT_RELEASE_WARMUP_STEPS,
    WEIGHT_DECAY,
    Manifest,
    collate,
    frontend_id,
    keyword_sequence_margin_loss,
    load_keyword_operating_points,
    optional_sha256,
    ordered_token_loss,
    recurrent_release_loss,
    sha256_file,
    strict_prefix_completion_loss,
    training_environment,
    validate_warm_start,
)

VARIANTS = {"rnn-standard", "rnn-multiseed", "rnn-terminal"}
POSITIVE_TERMINAL_MARGIN_LOG = 0.15
POSITIVE_TERMINAL_LOSS_WEIGHT = 0.10


def positive_terminal_completion_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    keyword_sequences: list[list[int]],
) -> torch.Tensor:
    """Require the final wake token to beat blank/competitors in its suffix region.

    This is the positive counterpart to strict-prefix completion loss. It does not
    use shadow/formal evidence and applies symmetrically to every configured wake.
    """
    known = {tuple(int(v) for v in sequence) for sequence in keyword_sequences}
    losses: list[torch.Tensor] = []
    offset = 0
    for batch_index, raw_length in enumerate(target_lengths.tolist()):
        length = int(raw_length)
        sequence = tuple(int(v) for v in targets[offset : offset + length].tolist())
        offset += length
        if not sequence or sequence not in known:
            continue
        steps = int(input_lengths[batch_index])
        if steps <= 0:
            raise ValueError("positive terminal loss received empty acoustic sequence")
        terminal = int(sequence[-1])
        # Match ordered-token segmentation: the last target owns the last 1/N of
        # the valid acoustic region. Include two preceding frames for jitter.
        start = max(0, ((length - 1) * steps) // length - 2)
        region = log_probs[start:steps, batch_index, :]
        if region.numel() == 0:
            raise ValueError("positive terminal loss suffix region is empty")
        terminal_best = region[:, terminal].max()
        left = region[:, :terminal]
        right = region[:, terminal + 1 :]
        competitors = [part.reshape(-1) for part in (left, right) if part.numel()]
        if not competitors:
            continue
        competitor_best = torch.cat(competitors).max()
        losses.append(torch.relu(log_probs.new_tensor(POSITIVE_TERMINAL_MARGIN_LOG) - (terminal_best - competitor_best)))
    if offset != int(targets.numel()):
        raise ValueError("flattened targets do not match target lengths")
    if not losses:
        return log_probs.sum() * 0.0
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--warm-start", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--frontend", default="logmel")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    token_map = load_tokens(args.tokens)
    keyword_sequences, keyword_operating_points, margin_profile_path = load_keyword_operating_points(
        args.keywords, token_map
    )
    vocab_size_value = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    if args.variant not in VARIANTS:
        parser.error("unsupported RNN experiment variant")
    if not 1 <= args.feature_dim <= MAX_FEATURE_DIM:
        parser.error("feature dim out of range")
    if not 1 <= args.hidden_dim <= MAX_HIDDEN_DIM:
        parser.error("hidden dim out of range")
    if not 2 <= vocab_size_value <= MAX_VOCAB_SIZE:
        parser.error("vocabulary size out of range")
    if args.epochs <= 0 or args.batch_size <= 0 or not math.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("invalid optimizer budget")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)

    dataset = Manifest(args.manifest, args.feature_dim, vocab_size_value, args.frontend)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=shuffle_generator,
    )
    model = TinyStreamingRNN(args.feature_dim, args.hidden_dim, vocab_size_value)
    checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
    validate_warm_start(checkpoint, args, vocab_size_value, fingerprint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    terminal_enabled = args.variant == "rnn-terminal"
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        total_terminal = 0.0
        for x, y, xlen, ylen in loader:
            log_probs = model(x).log_softmax(dim=2)
            raw_ctc = loss_fn(log_probs, y, xlen, ylen)
            sample_weights = torch.where(
                ylen > 0,
                torch.full_like(ylen, POSITIVE_EXAMPLE_WEIGHT, dtype=torch.float32),
                torch.ones_like(ylen, dtype=torch.float32),
            )
            normalized_ctc = raw_ctc / xlen.to(dtype=raw_ctc.dtype).clamp_min(1.0)
            ctc_loss = (normalized_ctc * sample_weights).sum() / sample_weights.sum()
            ordered_loss, _, _ = ordered_token_loss(log_probs, y, xlen, ylen)
            margin_per_sample = keyword_sequence_margin_loss(
                log_probs=log_probs,
                targets=y,
                input_lengths=xlen,
                target_lengths=ylen,
                true_ctc_nll=raw_ctc,
                keyword_sequences=keyword_sequences,
                blank=0,
                margin=KEYWORD_SEQUENCE_MARGIN,
                keyword_operating_points=keyword_operating_points,
            )
            margin_loss = (margin_per_sample * sample_weights).sum() / sample_weights.sum()
            completion_per_sample = strict_prefix_completion_loss(
                log_probs=log_probs,
                targets=y,
                input_lengths=xlen,
                target_lengths=ylen,
                keyword_sequences=keyword_sequences,
                keyword_operating_points=keyword_operating_points,
            )
            completion_loss = (completion_per_sample * sample_weights).sum() / sample_weights.sum()
            release_loss = recurrent_release_loss(log_probs, xlen)
            terminal_loss = (
                positive_terminal_completion_loss(
                    log_probs=log_probs,
                    targets=y,
                    input_lengths=xlen,
                    target_lengths=ylen,
                    keyword_sequences=keyword_sequences,
                )
                if terminal_enabled
                else log_probs.sum() * 0.0
            )
            loss = (
                ctc_loss
                + ORDERED_TOKEN_LOSS_WEIGHT * ordered_loss
                + KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT * margin_loss
                + PREFIX_COMPLETION_LOSS_WEIGHT * completion_loss
                + RECURRENT_RELEASE_LOSS_WEIGHT * release_loss
                + (POSITIVE_TERMINAL_LOSS_WEIGHT * terminal_loss if terminal_enabled else 0.0)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach())
            total_terminal += float(terminal_loss.detach())
        print(
            f"variant={args.variant} epoch={epoch + 1} loss={total / max(1, len(loader)):.6f} "
            f"terminal={total_terminal / max(1, len(loader)):.6f}"
        )

    environment = training_environment()
    environment["training_code_sha256"]["experiments/model_family/train_rnn_variant.py"] = sha256_file(
        pathlib.Path(__file__).resolve()
    )
    environment["training_code_sha256"] = dict(sorted(environment["training_code_sha256"].items()))
    metadata = [{"name": path.name, "sha256": sha256_file(path)} for path in args.manifest]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "vocab_size": vocab_size_value,
            "vocab_fingerprint": fingerprint,
            "tokens_sha256": sha256_file(args.tokens),
            "keywords_sha256": sha256_file(args.keywords),
            "keyword_sequences": keyword_sequences,
            "keyword_operating_points": keyword_operating_points,
            "keyword_margin_profile_sha256": optional_sha256(margin_profile_path),
            "frame_length_samples": FRAME_LENGTH_SAMPLES,
            "frame_hop_samples": FRAME_HOP_SAMPLES,
            "frontend_spec_version": FRONTEND_SPEC_VERSION,
            "frontend_name": args.frontend,
            "frontend_kind": frontend_id(args.frontend),
            "training_examples": len(dataset),
            "training_manifests": metadata,
            "training_corpus_identity": dataset.corpus_identity,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "ctc_reduction": "per-frame-weighted",
            "positive_example_weight": POSITIVE_EXAMPLE_WEIGHT,
            "ordered_token_loss_weight": ORDERED_TOKEN_LOSS_WEIGHT,
            "keyword_sequence_margin": KEYWORD_SEQUENCE_MARGIN,
            "keyword_sequence_margin_loss_weight": KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT,
            "prefix_completion_loss_weight": PREFIX_COMPLETION_LOSS_WEIGHT,
            "prefix_completion_tail_steps": PREFIX_COMPLETION_TAIL_STEPS,
            "recurrent_release_tail_steps": RECURRENT_RELEASE_TAIL_STEPS,
            "recurrent_release_warmup_steps": RECURRENT_RELEASE_WARMUP_STEPS,
            "recurrent_release_context_steps": RECURRENT_RELEASE_CONTEXT_STEPS,
            "recurrent_release_loss_weight": RECURRENT_RELEASE_LOSS_WEIGHT,
            "experimental_variant": args.variant,
            "positive_terminal_completion_loss_weight": (
                POSITIVE_TERMINAL_LOSS_WEIGHT if terminal_enabled else 0.0
            ),
            "positive_terminal_margin_log": POSITIVE_TERMINAL_MARGIN_LOG if terminal_enabled else 0.0,
            "formal_qualification_used": False,
            "training_environment": environment,
        },
        args.output,
    )
    print(json.dumps({"variant": args.variant, "examples": len(dataset), "output": str(args.output)}))


if __name__ == "__main__":
    main()
