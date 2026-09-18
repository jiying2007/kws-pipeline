#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import wave

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from frontend import features  # noqa: E402
from frontend_spec import FRONTEND_IDS  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from train_ctc import load_keyword_operating_points, manifest_rows  # noqa: E402

EVIDENCE_CLASS = "kws-v2-research-keyword-classifier-v1"


def read_pcm(path: pathlib.Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16000
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"expected mono 16-kHz PCM16 WAV: {path}")
        raw = reader.readframes(reader.getnframes())
    if not raw:
        raise ValueError(f"empty WAV: {path}")
    return torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


class ClipDataset(Dataset):
    def __init__(
        self,
        manifest: pathlib.Path,
        keyword_sequences: list[list[int]],
        feature_dim: int,
        frontend: str,
    ):
        self.items: list[tuple[torch.Tensor, int]] = []
        roots = manifest.resolve().parent
        lookup = {tuple(seq): index + 1 for index, seq in enumerate(keyword_sequences)}
        for row in manifest_rows(manifest.resolve()):
            raw = pathlib.Path(str(row["audio"]))
            path = raw.resolve() if raw.is_absolute() else (roots / raw).resolve()
            target = tuple(int(v) for v in row["tokens"])
            label = int(lookup.get(target, 0))
            feat = features(read_pcm(path), feature_dim=feature_dim, frontend=frontend)
            self.items.append((feat, label))
        if not self.items:
            raise ValueError(f"empty classifier manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def collate(batch):
    feats, labels = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in feats], dtype=torch.long)
    max_t = int(lengths.max())
    dim = int(feats[0].shape[1])
    x = torch.zeros((len(feats), max_t, dim), dtype=torch.float32)
    for index, feat in enumerate(feats):
        x[index, : feat.shape[0]] = feat
    return x, lengths, torch.tensor(labels, dtype=torch.long)


class ClipGRU(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])


def evaluate(model: nn.Module, loader: DataLoader, classes: int) -> dict:
    model.eval()
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    correct = total = 0
    positive_total = positive_correct = 0
    negative_total = negative_false_positive = 0
    with torch.no_grad():
        for x, lengths, labels in loader:
            probs = model(x, lengths).softmax(dim=1)
            pred = probs.argmax(dim=1)
            keyword_score = probs[:, 1:].amax(dim=1)
            for truth, guess, score in zip(labels.tolist(), pred.tolist(), keyword_score.tolist()):
                confusion[int(truth)][int(guess)] += 1
                total += 1
                correct += int(truth == guess)
                if truth == 0:
                    negative_total += 1
                    negative_false_positive += int(guess != 0)
                    negative_scores.append(float(score))
                else:
                    positive_total += 1
                    positive_correct += int(truth == guess)
                    positive_scores.append(float(score))
    return {
        "examples": total,
        "accuracy": correct / max(1, total),
        "positive_examples": positive_total,
        "positive_recall": positive_correct / max(1, positive_total),
        "negative_examples": negative_total,
        "negative_false_positive_clip_rate": negative_false_positive / max(1, negative_total),
        "confusion": confusion,
        "score_distribution": {
            "positive_keyword_probability": {
                "p10": quantile(positive_scores, 0.10),
                "p50": quantile(positive_scores, 0.50),
                "p90": quantile(positive_scores, 0.90),
            },
            "negative_max_keyword_probability": {
                "p50": quantile(negative_scores, 0.50),
                "p90": quantile(negative_scores, 0.90),
                "p95": quantile(negative_scores, 0.95),
                "p99": quantile(negative_scores, 0.99),
            },
            "separation_gap_p10_positive_minus_p99_negative": (
                None
                if not positive_scores or not negative_scores
                else float(quantile(positive_scores, 0.10) - quantile(negative_scores, 0.99))
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only clip classifier separability baseline.")
    parser.add_argument("--train-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--calibration-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--test-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--frontend", choices=sorted(FRONTEND_IDS), required=True)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.feature_dim <= 0 or args.hidden_dim <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("feature/hidden dims, epochs and batch size must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    token_map = load_tokens(args.tokens)
    keyword_sequences, _, _ = load_keyword_operating_points(args.keywords, token_map)
    classes = len(keyword_sequences) + 1

    train = ClipDataset(args.train_manifest, keyword_sequences, args.feature_dim, args.frontend)
    cal = ClipDataset(args.calibration_manifest, keyword_sequences, args.feature_dim, args.frontend)
    test = ClipDataset(args.test_manifest, keyword_sequences, args.feature_dim, args.frontend)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=generator
    )
    cal_loader = DataLoader(cal, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = ClipGRU(args.feature_dim, args.hidden_dim, classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict] = []
    for epoch in range(args.epochs):
        model.train()
        losses: list[float] = []
        for x, lengths, labels in train_loader:
            logits = model(x, lengths)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "loss": sum(losses) / max(1, len(losses))})

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "architecture": "clip-gru-v1",
        "frontend": args.frontend,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "classes": classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "history": history,
        "calibration": evaluate(model, cal_loader, classes),
        "test": evaluate(model, test_loader, classes),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "research-keyword-classifier: "
        f"test-acc={result['test']['accuracy']:.4f} "
        f"test-neg-fp={result['test']['negative_false_positive_clip_rate']:.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, TypeError, ValueError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
