#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import random
import sys
import wave

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint, vocab_size  # noqa: E402

from frontend import features
from model import TinyStreamingRNN

MAX_FEATURE_DIM = 40
MAX_HIDDEN_DIM = 64
MAX_VOCAB_SIZE = 512
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
FRONTEND_SPEC_VERSION = 1
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP_NORM = 5.0


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Manifest(Dataset):
    def __init__(
        self,
        paths: list[pathlib.Path],
        feature_dim: int,
        vocab_size_value: int,
    ):
        self.rows: list[tuple[pathlib.Path, list[int]]] = []
        self.feature_dim = feature_dim
        self.vocab_size = vocab_size_value
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
                if any(token <= 0 or token >= vocab_size_value for token in tokens):
                    raise ValueError(
                        f"{path}:{line_no}: targets must be in 1..{vocab_size_value - 1}"
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


def validate_warm_start(
    checkpoint: dict,
    args: argparse.Namespace,
    vocab_size_value: int,
    fingerprint: int,
) -> None:
    expected = {
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "vocab_size": vocab_size_value,
        "frame_length_samples": FRAME_LENGTH_SAMPLES,
        "frame_hop_samples": FRAME_HOP_SAMPLES,
        "frontend_spec_version": FRONTEND_SPEC_VERSION,
        "vocab_fingerprint": fingerprint,
    }
    for key, value in expected.items():
        if int(checkpoint.get(key, -1)) != value:
            raise ValueError(
                f"warm-start {key}={checkpoint.get(key)!r} does not match {value}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
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

    token_map = load_tokens(args.tokens)
    vocab_size_value = vocab_size(token_map)
    fingerprint = vocab_fingerprint(token_map)
    if not 2 <= vocab_size_value <= MAX_VOCAB_SIZE:
        parser.error(f"token vocabulary must contain 2..{MAX_VOCAB_SIZE} entries")
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
    torch.use_deterministic_algorithms(True)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(args.seed)

    dataset = Manifest(args.manifest, args.feature_dim, vocab_size_value)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=shuffle_generator,
    )
    model = TinyStreamingRNN(args.feature_dim, args.hidden_dim, vocab_size_value)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        validate_warm_start(checkpoint, args, vocab_size_value, fingerprint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    if args.head_only:
        for parameter in model.in_proj.parameters():
            parameter.requires_grad = False
        for parameter in model.rec_proj.parameters():
            parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x, y, xlen, ylen in loader:
            logits = model(x).log_softmax(dim=2)
            loss = loss_fn(logits, y, xlen, ylen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            optimizer.step()
            total += float(loss.detach())
        print(f"epoch={epoch + 1} loss={total / max(1, len(loader)):.6f}")

    manifest_metadata = [
        {
            "name": path.name,
            "sha256": sha256_file(path),
        }
        for path in args.manifest
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": args.feature_dim,
            "hidden_dim": args.hidden_dim,
            "vocab_size": vocab_size_value,
            "vocab_fingerprint": fingerprint,
            "tokens_sha256": sha256_file(args.tokens),
            "frame_length_samples": FRAME_LENGTH_SAMPLES,
            "frame_hop_samples": FRAME_HOP_SAMPLES,
            "frontend_spec_version": FRONTEND_SPEC_VERSION,
            "training_examples": len(dataset),
            "training_manifests": manifest_metadata,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "grad_clip_norm": GRAD_CLIP_NORM,
            "hard_negative_capable": True,
        },
        args.output,
    )
    print(
        f"saved {args.output}: examples={len(dataset)} vocab={vocab_size_value} "
        f"fingerprint=0x{fingerprint:016x}"
    )


if __name__ == "__main__":
    main()
