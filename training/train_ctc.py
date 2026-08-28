#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import random
import wave

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from frontend import features
from model import TinyStreamingRNN

MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320


class Manifest(Dataset):
    def __init__(
        self,
        paths: list[pathlib.Path],
        feature_dim: int,
        vocab_size: int,
    ):
        self.rows: list[tuple[pathlib.Path, list[int]]] = []
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        for path in paths:
            root = path.parent
            for line_no, raw in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not raw.strip() or raw.lstrip().startswith("#"):
                    continue
                if "\t" not in raw:
                    raise ValueError(f"{path}:{line_no}: expected WAV<TAB>token_ids")
                wav_path, token_text = raw.split("\t", 1)
                wav_path = wav_path.strip()
                if not wav_path:
                    raise ValueError(f"{path}:{line_no}: empty WAV path")
                wav = pathlib.Path(wav_path)
                wav = wav if wav.is_absolute() else root / wav
                tokens = [int(value) for value in token_text.split()]
                if any(token <= 0 or token >= vocab_size for token in tokens):
                    raise ValueError(
                        f"{path}:{line_no}: targets must be in 1..{vocab_size - 1}"
                    )
                self.rows.append((wav, tokens))
        if not self.rows:
            raise ValueError("training manifests contain no examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        path, tokens = self.rows[idx]
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != 1
                or wf.getframerate() != 16000
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
            raw = wf.readframes(wf.getnframes())
        pcm = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
        acoustic = features(
            pcm,
            self.feature_dim,
            frame_len=FRAME_LENGTH_SAMPLES,
            hop=FRAME_HOP_SAMPLES,
        )
        repeated_neighbors = sum(
            1 for left, right in zip(tokens, tokens[1:]) if left == right
        )
        minimum_ctc_steps = len(tokens) + repeated_neighbors
        if acoustic.shape[0] < minimum_ctc_steps:
            raise ValueError(
                f"{path}: {acoustic.shape[0]} acoustic step(s) cannot align "
                f"CTC target requiring at least {minimum_ctc_steps} step(s)"
            )
        return acoustic, torch.tensor(tokens, dtype=torch.long)


def collate(batch):
    xs, ys = zip(*batch)
    xlen = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    ylen = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    max_t = int(xlen.max())
    feature_dim = xs[0].shape[1]
    padded = torch.zeros((len(xs), max_t, feature_dim))
    for index, x in enumerate(xs):
        padded[index, : x.shape[0]] = x
    targets = (
        torch.cat(ys)
        if any(y.numel() for y in ys)
        else torch.empty(0, dtype=torch.long)
    )
    return padded, targets, xlen, ylen


def validate_warm_start(checkpoint: dict, args: argparse.Namespace) -> None:
    expected = {
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "vocab_size": args.vocab_size,
        "frame_length_samples": FRAME_LENGTH_SAMPLES,
        "frame_hop_samples": FRAME_HOP_SAMPLES,
    }
    for key, value in expected.items():
        if int(checkpoint.get(key, -1)) != value:
            raise ValueError(
                f"warm-start {key}={checkpoint.get(key)!r} does not match {value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warm-start", type=pathlib.Path)
    parser.add_argument("--head-only", action="store_true")
    args = parser.parse_args()

    if not 2 <= args.vocab_size <= MAX_VOCAB_SIZE:
        parser.error(f"--vocab-size must be 2..{MAX_VOCAB_SIZE}")
    if not 1 <= args.feature_dim <= MAX_FEATURE_DIM:
        parser.error(f"--feature-dim must be 1..{MAX_FEATURE_DIM}")
    if not 1 <= args.hidden_dim <= MAX_HIDDEN_DIM:
        parser.error(f"--hidden-dim must be 1..{MAX_HIDDEN_DIM}")
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        parser.error("--lr must be finite and > 0")
    if args.head_only and not args.warm_start:
        parser.error("--head-only requires --warm-start")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = Manifest(args.manifest, args.feature_dim, args.vocab_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    model = TinyStreamingRNN(args.feature_dim, args.hidden_dim, args.vocab_size)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        validate_warm_start(checkpoint, args)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.head_only:
        for parameter in model.in_proj.parameters():
            parameter.requires_grad = False
        for parameter in model.rec_proj.parameters():
            parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1.0e-4)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y, xlen, ylen in loader:
            logits = model(x).log_softmax(dim=2)
            loss = loss_fn(logits, y, xlen, ylen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "vocab_size": args.vocab_size,
            "frame_length_samples": FRAME_LENGTH_SAMPLES,
            "frame_hop_samples": FRAME_HOP_SAMPLES,
            "training_examples": len(dataset),
            "hard_negative_capable": True,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
