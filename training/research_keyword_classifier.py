#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import wave

# The research classifier must be reproducible across heterogeneous hosted
# runners. Apply a conservative cross-CPU contract before importing torch.
_RESEARCH_CPU_ENV = {
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "MKL_CBWR": "COMPATIBLE",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ATEN_CPU_CAPABILITY": "default",
}
for _name, _value in _RESEARCH_CPU_ENV.items():
    os.environ[_name] = _value

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from frontend import features  # noqa: E402
from gru_model import TinyStreamingGRU  # noqa: E402
from frontend_spec import FRONTEND_IDS  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from train_ctc import (  # noqa: E402
    load_keyword_operating_points,
    manifest_rows,
    sha256_file,
    training_environment,
)

# train_ctc imports model.py, whose production contract uses AVX2/two threads.
# The research classifier deliberately uses the stronger compatible/single-thread
# contract above. Re-assert the environment and thread pool after imports.
for _name, _value in _RESEARCH_CPU_ENV.items():
    os.environ[_name] = _value
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

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


class ExactBalancedEpochSampler(Sampler[int]):
    """Exact per-class epoch counts with deterministic within-class cycling."""

    def __init__(self, labels: list[int], classes: int, seed: int):
        if classes <= 0:
            raise ValueError("classes must be positive")
        self.indices_by_class = [
            [index for index, label in enumerate(labels) if int(label) == class_id]
            for class_id in range(classes)
        ]
        if any(not indices for indices in self.indices_by_class):
            raise ValueError("exact-balanced sampler requires every class")
        self.samples_per_class = len(labels) // classes
        if self.samples_per_class <= 0:
            raise ValueError("not enough samples for exact-balanced sampler")
        self.num_samples = self.samples_per_class * classes
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(
            self.seed + 1_000_003 * self.epoch
        )
        chosen: list[int] = []
        for indices in self.indices_by_class:
            remaining = self.samples_per_class
            while remaining > 0:
                order = torch.randperm(len(indices), generator=generator).tolist()
                take = min(remaining, len(order))
                chosen.extend(indices[position] for position in order[:take])
                remaining -= take
        mixing = torch.randperm(len(chosen), generator=generator).tolist()
        return iter(chosen[position] for position in mixing)

    def __len__(self) -> int:
        return self.num_samples


class ClipDataset(Dataset):
    def __init__(
        self,
        manifests: list[pathlib.Path],
        keyword_sequences: list[list[int]],
        feature_dim: int,
        frontend: str,
    ):
        self.items: list[tuple[torch.Tensor, int]] = []
        self.labels: list[int] = []
        lookup = {tuple(seq): index + 1 for index, seq in enumerate(keyword_sequences)}
        for manifest in manifests:
            root = manifest.resolve().parent
            for row in manifest_rows(manifest.resolve()):
                raw = pathlib.Path(str(row["audio"]))
                path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
                target = tuple(int(v) for v in row["tokens"])
                label = int(lookup.get(target, 0))
                feat = features(read_pcm(path), feature_dim=feature_dim, frontend=frontend)
                self.items.append((feat, label))
                self.labels.append(label)
        if not self.items:
            raise ValueError("classifier manifests are empty")

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
    """Clip head over the exact TinyStreamingGRU recurrent cell."""

    def __init__(self, feature_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.encoder = TinyStreamingGRU(feature_dim, hidden_dim, classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = x.new_zeros((x.shape[0], self.encoder.gru.hidden_size))
        for frame_index in range(x.shape[1]):
            next_hidden = self.encoder.gru(x[:, frame_index, :], hidden)
            active = (lengths > frame_index).unsqueeze(1)
            hidden = torch.where(active, next_hidden, hidden)
        return self.encoder.out_proj(hidden)


class StackedClipGRU(nn.Module):
    """Two causal GRUCell layers with clip classification from the final state."""

    def __init__(self, feature_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.gru1 = nn.GRUCell(feature_dim, hidden_dim)
        self.gru2 = nn.GRUCell(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, classes)
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden1 = x.new_zeros((x.shape[0], self.hidden_dim))
        hidden2 = x.new_zeros((x.shape[0], self.hidden_dim))
        for frame_index in range(x.shape[1]):
            next1 = self.gru1(x[:, frame_index, :], hidden1)
            next2 = self.gru2(next1, hidden2)
            active = (lengths > frame_index).unsqueeze(1)
            hidden1 = torch.where(active, next1, hidden1)
            hidden2 = torch.where(active, next2, hidden2)
        return self.head(hidden2)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    classes: int,
    thresholds: list[float],
) -> dict:
    model.eval()
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    correct = total = 0
    positive_total = positive_correct = 0
    negative_total = negative_false_positive = 0
    threshold_rows = {
        threshold: {
            "wake_examples": 0,
            "wake_correct": 0,
            "negative_examples": 0,
            "false_accepts": 0,
        }
        for threshold in thresholds
    }
    with torch.no_grad():
        for x, lengths, labels in loader:
            probs = model(x, lengths).softmax(dim=1)
            pred = probs.argmax(dim=1)
            keyword_score = probs[:, 1:].amax(dim=1)
            keyword_pred = probs[:, 1:].argmax(dim=1) + 1
            true_class_score = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            for truth, guess, keyword_guess, score, true_score in zip(
                labels.tolist(),
                pred.tolist(),
                keyword_pred.tolist(),
                keyword_score.tolist(),
                true_class_score.tolist(),
            ):
                confusion[int(truth)][int(guess)] += 1
                for threshold, row in threshold_rows.items():
                    if truth == 0:
                        row["negative_examples"] += 1
                        row["false_accepts"] += int(score >= threshold)
                    else:
                        row["wake_examples"] += 1
                        row["wake_correct"] += int(
                            score >= threshold and int(keyword_guess) == int(truth)
                        )
                total += 1
                correct += int(truth == guess)
                if truth == 0:
                    negative_total += 1
                    negative_false_positive += int(guess != 0)
                    negative_scores.append(float(score))
                else:
                    positive_total += 1
                    positive_correct += int(truth == guess)
                    positive_scores.append(float(true_score))
    return {
        "examples": total,
        "accuracy": correct / max(1, total),
        "positive_examples": positive_total,
        "positive_recall": positive_correct / max(1, positive_total),
        "negative_examples": negative_total,
        "negative_false_positive_clip_rate": negative_false_positive / max(1, negative_total),
        "confusion": confusion,
        "operating_curve": [
            {
                "threshold": float(threshold),
                "wake_examples": int(row["wake_examples"]),
                "wake_exact_recall": (
                    row["wake_correct"] / max(1, row["wake_examples"])
                ),
                "negative_examples": int(row["negative_examples"]),
                "false_accepts": int(row["false_accepts"]),
                "negative_false_positive_rate": (
                    row["false_accepts"] / max(1, row["negative_examples"])
                ),
            }
            for threshold, row in sorted(threshold_rows.items())
        ],
        "score_distribution": {
            "positive_score_semantics": "true-keyword-probability",
            "negative_score_semantics": "max-keyword-probability",
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


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _initialize_gru_cell(cell: nn.GRUCell) -> None:
    for gate in cell.weight_ih.chunk(3, dim=0):
        nn.init.xavier_uniform_(gate)
    for gate in cell.weight_hh.chunk(3, dim=0):
        nn.init.orthogonal_(gate)
    nn.init.zeros_(cell.bias_ih)
    nn.init.zeros_(cell.bias_hh)


def initialize_clip_gru(model: nn.Module, mode: str, seed: int) -> None:
    if mode == "default-pytorch-v1":
        return
    if mode != "gru-orthogonal-xavier-v1":
        raise ValueError(f"unsupported init mode: {mode}")
    torch.manual_seed(seed)
    with torch.no_grad():
        if isinstance(model, ClipGRU):
            _initialize_gru_cell(model.encoder.gru)
            nn.init.xavier_uniform_(model.encoder.out_proj.weight)
            nn.init.zeros_(model.encoder.out_proj.bias)
        elif isinstance(model, StackedClipGRU):
            _initialize_gru_cell(model.gru1)
            _initialize_gru_cell(model.gru2)
            nn.init.xavier_uniform_(model.head.weight)
            nn.init.zeros_(model.head.bias)
        else:
            raise ValueError(f"unsupported model type for init mode: {type(model).__name__}")


def learning_rate_for_epoch(
    base_lr: float,
    epoch_index: int,
    total_epochs: int,
    mode: str,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> float:
    if mode == "fixed-v1":
        return float(base_lr)
    if mode != "warmup-cosine-v1":
        raise ValueError(f"unsupported LR schedule: {mode}")
    if warmup_epochs <= 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup epochs must be in 1..epochs-1")
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min LR ratio must be in (0,1]")
    if epoch_index < warmup_epochs:
        return float(base_lr) * float(epoch_index + 1) / float(warmup_epochs)
    tail_epochs = total_epochs - warmup_epochs
    if tail_epochs <= 1:
        return float(base_lr) * float(min_lr_ratio)
    progress = float(epoch_index - warmup_epochs) / float(tail_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (
        float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only clip classifier separability baseline.")
    parser.add_argument("--train-manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--calibration-manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--test-manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--frontend", choices=sorted(FRONTEND_IDS), required=True)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--encoder-architecture",
        choices=("single-gru-v1", "stacked-gru-v2"),
        default="single-gru-v1",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-schedule",
        choices=("fixed-v1", "warmup-cosine-v1"),
        default="fixed-v1",
    )
    parser.add_argument("--warmup-epochs", type=int, default=4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--sampler-seed", type=int)
    parser.add_argument(
        "--init-mode",
        choices=("default-pytorch-v1", "gru-orthogonal-xavier-v1"),
        default="default-pytorch-v1",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        ],
    )
    parser.add_argument(
        "--balance-mode",
        choices=("none", "equal-class-sampler-v1", "exact-balanced-epoch-v2"),
        default="equal-class-sampler-v1",
    )
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.feature_dim <= 0 or args.hidden_dim <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error("feature/hidden dims, epochs and batch size must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and > 0")
    if args.warmup_epochs <= 0:
        parser.error("--warmup-epochs must be > 0")
    if not math.isfinite(args.min_lr_ratio) or not 0.0 < args.min_lr_ratio <= 1.0:
        parser.error("--min-lr-ratio must be finite and in (0,1]")
    if args.lr_schedule == "warmup-cosine-v1" and args.warmup_epochs >= args.epochs:
        parser.error("--warmup-epochs must be < --epochs for warmup-cosine-v1")
    thresholds = sorted(set(float(value) for value in args.thresholds))
    if not thresholds or any(
        not math.isfinite(value) or not 0.0 < value < 1.0
        for value in thresholds
    ):
        parser.error("--thresholds must contain unique finite values in (0,1)")

    model_seed = args.seed if args.model_seed is None else int(args.model_seed)
    sampler_seed = args.seed if args.sampler_seed is None else int(args.sampler_seed)
    random.seed(model_seed)
    torch.manual_seed(model_seed)
    torch.use_deterministic_algorithms(True)
    token_map = load_tokens(args.tokens)
    keyword_sequences, _, _ = load_keyword_operating_points(args.keywords, token_map)
    classes = len(keyword_sequences) + 1

    train = ClipDataset(args.train_manifest, keyword_sequences, args.feature_dim, args.frontend)
    cal = ClipDataset(args.calibration_manifest, keyword_sequences, args.feature_dim, args.frontend)
    test = ClipDataset(args.test_manifest, keyword_sequences, args.feature_dim, args.frontend)

    generator = torch.Generator().manual_seed(sampler_seed)
    class_counts = [train.labels.count(index) for index in range(classes)]
    if any(count <= 0 for count in class_counts):
        raise ValueError(f"classifier train split is missing class coverage: {class_counts}")
    sampler = None
    shuffle = True
    if args.balance_mode == "equal-class-sampler-v1":
        sample_weights = [
            1.0 / float(class_counts[label])
            for label in train.labels
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    elif args.balance_mode == "exact-balanced-epoch-v2":
        sampler = ExactBalancedEpochSampler(
            train.labels,
            classes=classes,
            seed=sampler_seed,
        )
        shuffle = False
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate,
        generator=generator if sampler is None else None,
    )
    cal_loader = DataLoader(cal, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    if args.encoder_architecture == "single-gru-v1":
        model = ClipGRU(args.feature_dim, args.hidden_dim, classes)
        model_architecture = "clip-gru-v1"
        recurrent_impl = "tiny-streaming-gru-cell-v1"
        encoder_layers = 1
    else:
        model = StackedClipGRU(args.feature_dim, args.hidden_dim, classes)
        model_architecture = "clip-stacked-gru-v2"
        recurrent_impl = "two-layer-gru-cell-v2"
        encoder_layers = 2
    initialize_clip_gru(model, args.init_mode, model_seed)
    initial_model_state_sha256 = model_state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict] = []
    for epoch in range(args.epochs):
        if isinstance(sampler, ExactBalancedEpochSampler):
            sampler.set_epoch(epoch)
        epoch_lr = learning_rate_for_epoch(
            args.lr,
            epoch,
            args.epochs,
            args.lr_schedule,
            args.warmup_epochs,
            args.min_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
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
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": epoch_lr,
                "loss": sum(losses) / max(1, len(losses)),
            }
        )

    environment = training_environment()
    environment["training_code_sha256"]["training/gru_model.py"] = sha256_file(
        ROOT / "training" / "gru_model.py"
    )
    environment["training_code_sha256"]["training/research_keyword_classifier.py"] = sha256_file(
        pathlib.Path(__file__).resolve()
    )
    environment["training_code_sha256"] = dict(
        sorted(environment["training_code_sha256"].items())
    )

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "architecture": model_architecture,
        "encoder_architecture": args.encoder_architecture,
        "recurrent_impl": recurrent_impl,
        "encoder_layers": encoder_layers,
        "frontend": args.frontend,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "classes": classes,
        "seed": args.seed,
        "model_seed": model_seed,
        "sampler_seed": sampler_seed,
        "seed_policy": "independent-model-sampler-v1",
        "init_mode": args.init_mode,
        "optimizer": "AdamW",
        "base_learning_rate": args.lr,
        "lr_schedule": args.lr_schedule,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_ratio": args.min_lr_ratio,
        "epochs": args.epochs,
        "balance_mode": args.balance_mode,
        "training_class_counts": class_counts,
        "sampler_samples_per_epoch": (
            int(len(sampler)) if sampler is not None else int(len(train))
        ),
        "sampler_exact_class_count_per_epoch": (
            int(sampler.samples_per_class)
            if isinstance(sampler, ExactBalancedEpochSampler)
            else None
        ),
        "initial_model_state_sha256": initial_model_state_sha256,
        "model_state_sha256": model_state_sha256(model),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_environment": environment,
        "research_cpu_contract": {
            "policy": "cross-cpu-compatible-single-thread-v1",
            "environment": dict(sorted(_RESEARCH_CPU_ENV.items())),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        },
        "thresholds": thresholds,
        "history": history,
        "calibration": evaluate(model, cal_loader, classes, thresholds),
        "test": evaluate(model, test_loader, classes, thresholds),
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
