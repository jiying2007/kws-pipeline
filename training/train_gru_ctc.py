#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import random

import torch
from torch import nn
from torch.utils.data import DataLoader

from gru_model import ARCHITECTURE, TinyStreamingGRU
from train_ctc import (
    FRAME_HOP_SAMPLES,
    FRAME_LENGTH_SAMPLES,
    FRONTEND_IDS,
    FRONTEND_LOGMEL,
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
)
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size

ROOT = pathlib.Path(__file__).resolve().parents[1]


def validate_warm_start(
    checkpoint: dict,
    args: argparse.Namespace,
    vocab_size_value: int,
    fingerprint: int,
) -> None:
    if str(checkpoint.get("architecture", "")) != ARCHITECTURE:
        raise ValueError("GRU warm-start checkpoint architecture mismatch")
    expected = {
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "vocab_size": vocab_size_value,
        "frame_length_samples": FRAME_LENGTH_SAMPLES,
        "frame_hop_samples": FRAME_HOP_SAMPLES,
        "frontend_spec_version": FRONTEND_SPEC_VERSION,
        "vocab_fingerprint": fingerprint,
        "frontend_kind": frontend_id(args.frontend),
    }
    for key, value in expected.items():
        if int(checkpoint.get(key, -1)) != value:
            raise ValueError(
                f"GRU warm-start {key}={checkpoint.get(key)!r} does not match {value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only Tiny-GRU CTC trainer with the shipping loss contract."
    )
    parser.add_argument("--manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--frontend", choices=sorted(FRONTEND_IDS), default=FRONTEND_LOGMEL)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warm-start", type=pathlib.Path)
    parser.add_argument("--positive-example-weight", type=float, default=POSITIVE_EXAMPLE_WEIGHT)
    parser.add_argument("--ordered-token-loss-weight", type=float, default=ORDERED_TOKEN_LOSS_WEIGHT)
    args = parser.parse_args()

    token_map = load_tokens(args.tokens)
    keyword_sequences, keyword_operating_points, margin_profile_path = (
        load_keyword_operating_points(args.keywords, token_map)
    )
    vocab_size_value = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    if not 2 <= vocab_size_value <= MAX_VOCAB_SIZE:
        parser.error(f"token vocabulary must contain 2..{MAX_VOCAB_SIZE} entries")
    if not 1 <= args.feature_dim <= MAX_FEATURE_DIM:
        parser.error(f"--feature-dim must be 1..{MAX_FEATURE_DIM}")
    if not 1 <= args.hidden_dim <= MAX_HIDDEN_DIM:
        parser.error(f"--hidden-dim must be 1..{MAX_HIDDEN_DIM}")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("epochs and batch size must be > 0")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("--lr must be finite and > 0")
    if not math.isfinite(args.positive_example_weight) or args.positive_example_weight <= 0.0:
        parser.error("--positive-example-weight must be finite and > 0")
    if not math.isfinite(args.ordered_token_loss_weight) or args.ordered_token_loss_weight < 0.0:
        parser.error("--ordered-token-loss-weight must be finite and >= 0")

    environment = training_environment()
    environment["training_code_sha256"]["training/gru_model.py"] = sha256_file(
        ROOT / "training" / "gru_model.py"
    )
    environment["training_code_sha256"]["training/train_gru_ctc.py"] = sha256_file(
        pathlib.Path(__file__).resolve()
    )
    environment["training_code_sha256"] = dict(
        sorted(environment["training_code_sha256"].items())
    )

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
    model = TinyStreamingGRU(args.feature_dim, args.hidden_dim, vocab_size_value)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        validate_warm_start(checkpoint, args, vocab_size_value, fingerprint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")
    model.train()
    for epoch in range(args.epochs):
        total = total_ctc = total_ordered = total_margin = total_completion = total_release = 0.0
        ordered_correct = ordered_total = 0
        for x, y, xlen, ylen in loader:
            log_probs = model(x).log_softmax(dim=2)
            raw_ctc = loss_fn(log_probs, y, xlen, ylen)
            sample_weights = torch.where(
                ylen > 0,
                torch.full_like(ylen, args.positive_example_weight, dtype=torch.float32),
                torch.ones_like(ylen, dtype=torch.float32),
            )
            normalized_ctc = raw_ctc / xlen.to(dtype=raw_ctc.dtype).clamp_min(1.0)
            ctc_loss = (normalized_ctc * sample_weights).sum() / sample_weights.sum()
            ordered_loss, batch_correct, batch_total = ordered_token_loss(
                log_probs, y, xlen, ylen
            )
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
            completion_loss = (
                completion_per_sample * sample_weights
            ).sum() / sample_weights.sum()
            release_loss = recurrent_release_loss(log_probs, xlen)
            loss = (
                ctc_loss
                + args.ordered_token_loss_weight * ordered_loss
                + KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT * margin_loss
                + PREFIX_COMPLETION_LOSS_WEIGHT * completion_loss
                + RECURRENT_RELEASE_LOSS_WEIGHT * release_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach())
            total_ctc += float(ctc_loss.detach())
            total_ordered += float(ordered_loss.detach())
            total_margin += float(margin_loss.detach())
            total_completion += float(completion_loss.detach())
            total_release += float(release_loss.detach())
            ordered_correct += batch_correct
            ordered_total += batch_total
        batches = max(1, len(loader))
        print(
            f"epoch={epoch + 1} loss={total / batches:.6f} "
            f"ctc={total_ctc / batches:.6f} ordered={total_ordered / batches:.6f} "
            f"margin={total_margin / batches:.6f} completion={total_completion / batches:.6f} "
            f"release={total_release / batches:.6f} "
            f"ordered_token_acc={ordered_correct / max(1, ordered_total):.6f}"
        )

    manifest_metadata = [
        {"name": path.name, "sha256": sha256_file(path)} for path in args.manifest
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": ARCHITECTURE,
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
            "training_manifests": manifest_metadata,
            "training_corpus_identity": dataset.corpus_identity,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "ctc_reduction": "per-frame-weighted",
            "positive_example_weight": args.positive_example_weight,
            "ordered_token_loss_weight": args.ordered_token_loss_weight,
            "keyword_sequence_margin": KEYWORD_SEQUENCE_MARGIN,
            "keyword_sequence_margin_loss_weight": KEYWORD_SEQUENCE_MARGIN_LOSS_WEIGHT,
            "prefix_completion_loss_weight": PREFIX_COMPLETION_LOSS_WEIGHT,
            "prefix_completion_tail_steps": PREFIX_COMPLETION_TAIL_STEPS,
            "prefix_completion_policy": "strict-prefix-terminal-hinge-v1",
            "recurrent_release_tail_steps": RECURRENT_RELEASE_TAIL_STEPS,
            "recurrent_release_warmup_steps": RECURRENT_RELEASE_WARMUP_STEPS,
            "recurrent_release_context_steps": RECURRENT_RELEASE_CONTEXT_STEPS,
            "recurrent_release_tail_mode": "terminal-context-repeat",
            "recurrent_release_loss_weight": RECURRENT_RELEASE_LOSS_WEIGHT,
            "hard_negative_capable": True,
            "experimental": True,
            "training_environment": environment,
        },
        args.output,
    )
    print(
        f"saved experimental {ARCHITECTURE} checkpoint {args.output}: "
        f"examples={len(dataset)} vocab={vocab_size_value} frontend={args.frontend}"
    )


if __name__ == "__main__":
    main()
