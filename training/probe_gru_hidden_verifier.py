#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from gru_model import ARCHITECTURE, TinyStreamingGRU  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from train_ctc import Manifest, load_keyword_operating_points, vocab_size  # noqa: E402

EVIDENCE_CLASS = "kws-v2-frozen-encoder-verifier-probe-v1"
POOLING_MODES = ("last", "mean", "max")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p01": quantile(values, 0.01),
        "p10": quantile(values, 0.10),
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def class_label(target: tuple[int, ...], keyword_lookup: dict[tuple[int, ...], int]) -> int:
    value = keyword_lookup.get(target)
    return int(value + 1) if value is not None else 0


def extract_representations(
    *,
    model: TinyStreamingGRU,
    manifests: list[tuple[str, pathlib.Path]],
    feature_dim: int,
    vocab: int,
    frontend: str,
    keyword_lookup: dict[tuple[int, ...], int],
) -> dict:
    reps: dict[str, list[torch.Tensor]] = {mode: [] for mode in POOLING_MODES}
    labels: list[int] = []
    sources: list[str] = []
    target_kinds: list[str] = []
    examples_by_manifest: list[dict] = []

    model.eval()
    with torch.no_grad():
        for source_name, manifest in manifests:
            dataset = Manifest([manifest.resolve()], feature_dim, vocab, frontend)
            examples_by_manifest.append(
                {
                    "source": source_name,
                    "path": str(manifest.resolve()),
                    "sha256": sha256_file(manifest.resolve()),
                    "examples": len(dataset),
                }
            )
            for index in range(len(dataset)):
                acoustic, target_tensor = dataset[index]
                target = tuple(int(value) for value in target_tensor.tolist())
                label = class_label(target, keyword_lookup)
                hidden = acoustic.new_zeros((1, model.gru.hidden_size))
                states: list[torch.Tensor] = []
                for frame in acoustic:
                    hidden = model.gru(frame.unsqueeze(0), hidden)
                    states.append(hidden.squeeze(0).clone())
                if not states:
                    raise ValueError(f"{manifest}: empty acoustic representation")
                stack = torch.stack(states, dim=0)
                reps["last"].append(stack[-1])
                reps["mean"].append(stack.mean(dim=0))
                reps["max"].append(stack.amax(dim=0))
                labels.append(label)
                sources.append(source_name)
                if label > 0:
                    target_kinds.append("wake")
                elif target:
                    target_kinds.append("tokenized-nonwake")
                else:
                    target_kinds.append("empty-target-nonwake")

    if not labels:
        raise ValueError("probe extraction produced no examples")
    return {
        "representations": {
            mode: torch.stack(values, dim=0) for mode, values in reps.items()
        },
        "labels": torch.tensor(labels, dtype=torch.long),
        "sources": sources,
        "target_kinds": target_kinds,
        "manifests": examples_by_manifest,
    }


def standardize(
    train: torch.Tensor,
    *others: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor], dict]:
    mean = train.mean(dim=0)
    std = train.std(dim=0, unbiased=False).clamp_min(1.0e-5)
    return (
        (train - mean) / std,
        [(value - mean) / std for value in others],
        {
            "feature_count": int(train.shape[1]),
            "mean_abs": float(mean.abs().mean()),
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        },
    )


def train_linear_probe(
    *,
    x: torch.Tensor,
    labels: torch.Tensor,
    classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[nn.Linear, list[dict], list[int]]:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    counts = [int((labels == index).sum()) for index in range(classes)]
    if any(count <= 0 for count in counts):
        raise ValueError(f"probe train split is missing class coverage: {counts}")
    weights = torch.tensor(
        [1.0 / float(counts[int(label)]) for label in labels.tolist()],
        dtype=torch.double,
    )
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        TensorDataset(x, labels),
        batch_size=batch_size,
        sampler=sampler,
    )
    head = nn.Linear(int(x.shape[1]), classes)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    for epoch in range(epochs):
        head.train()
        losses: list[float] = []
        for batch_x, batch_y in loader:
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(
            {
                "epoch": epoch + 1,
                "loss": sum(losses) / max(1, len(losses)),
            }
        )
    return head, history, counts


def evaluate(
    *,
    head: nn.Linear,
    x: torch.Tensor,
    labels: torch.Tensor,
    sources: list[str],
    target_kinds: list[str],
    class_names: list[str],
) -> dict:
    head.eval()
    with torch.no_grad():
        probs = head(x).softmax(dim=1)
    pred = probs.argmax(dim=1)
    truth = labels.tolist()
    guesses = pred.tolist()
    true_keyword_scores: list[float] = []
    nonwake_scores: list[float] = []
    tokenized_nonwake_scores: list[float] = []
    empty_nonwake_scores: list[float] = []
    correct = 0
    wake_count = 0
    wake_exact_correct = 0
    wake_triggered = 0
    negative_count = 0
    negative_fp = 0
    source_negative: dict[str, dict[str, int]] = {}
    per_class: dict[str, dict[str, int]] = {
        class_names[index]: {"examples": 0, "correct": 0}
        for index in range(len(class_names))
    }

    for index, (label, guess) in enumerate(zip(truth, guesses)):
        correct += int(label == guess)
        name = class_names[label]
        per_class[name]["examples"] += 1
        per_class[name]["correct"] += int(label == guess)
        keyword_score = float(probs[index, 1:].amax())
        if label == 0:
            negative_count += 1
            is_fp = int(guess != 0)
            negative_fp += is_fp
            nonwake_scores.append(keyword_score)
            if target_kinds[index] == "tokenized-nonwake":
                tokenized_nonwake_scores.append(keyword_score)
            elif target_kinds[index] == "empty-target-nonwake":
                empty_nonwake_scores.append(keyword_score)
            bucket = source_negative.setdefault(
                sources[index], {"examples": 0, "false_positives": 0}
            )
            bucket["examples"] += 1
            bucket["false_positives"] += is_fp
        else:
            wake_count += 1
            wake_exact_correct += int(label == guess)
            wake_triggered += int(guess != 0)
            true_keyword_scores.append(float(probs[index, label]))

    source_metrics = {
        source: {
            **values,
            "false_positive_rate": values["false_positives"] / max(1, values["examples"]),
        }
        for source, values in sorted(source_negative.items())
    }
    class_metrics = {
        name: {
            **values,
            "recall": values["correct"] / max(1, values["examples"]),
        }
        for name, values in per_class.items()
    }
    wake_p10 = quantile(true_keyword_scores, 0.10)
    nonwake_p99 = quantile(nonwake_scores, 0.99)
    return {
        "examples": len(truth),
        "accuracy": correct / max(1, len(truth)),
        "wake_examples": wake_count,
        "wake_exact_recall": wake_exact_correct / max(1, wake_count),
        "wake_trigger_recall": wake_triggered / max(1, wake_count),
        "negative_examples": negative_count,
        "negative_false_positive_rate": negative_fp / max(1, negative_count),
        "per_class": class_metrics,
        "negative_by_source": source_metrics,
        "true_keyword_probability": distribution(true_keyword_scores),
        "nonwake_max_keyword_probability": distribution(nonwake_scores),
        "tokenized_nonwake_max_keyword_probability": distribution(
            tokenized_nonwake_scores
        ),
        "empty_target_nonwake_max_keyword_probability": distribution(
            empty_nonwake_scores
        ),
        "separation_gap_p10_wake_minus_p99_nonwake": (
            None
            if wake_p10 is None or nonwake_p99 is None
            else wake_p10 - nonwake_p99
        ),
    }


def calibration_key(row: dict) -> tuple:
    gap = row["calibration"]["separation_gap_p10_wake_minus_p99_nonwake"]
    gap_value = float(gap) if gap is not None else -1.0e9
    return (
        -gap_value,
        -float(row["calibration"]["wake_exact_recall"]),
        float(row["calibration"]["negative_false_positive_rate"]),
        row["pooling"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe frozen GRU hidden states with tiny linear keyword-verifier heads."
    )
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--train-manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument(
        "--calibration-manifest", required=True, action="append", type=pathlib.Path
    )
    parser.add_argument("--test-manifest", required=True, action="append", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config = load_object(args.config.resolve())
    if config.get("policy") != "kws-v2-architecture-probe-v1":
        raise ValueError("architecture probe policy identity mismatch")
    if config.get("evidence_scope") != "research-only":
        raise ValueError("architecture probe must remain research-only")
    authority = config.get("authority")
    if not isinstance(authority, dict) or authority.get("promotion_allowed") is not False:
        raise ValueError("architecture probe cannot grant promotion authority")

    probe = config["probe"]
    modes = tuple(str(value) for value in probe["pooling_modes"])
    if tuple(modes) != POOLING_MODES:
        raise ValueError("probe pooling modes drifted from contracted last/mean/max set")
    classes = [str(value) for value in probe["classes"]]
    if classes != ["nonwake", "keyword-1", "keyword-2"]:
        raise ValueError("probe class contract drifted")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if str(checkpoint.get("architecture")) != ARCHITECTURE:
        raise ValueError("probe checkpoint is not tiny-gru-v1")
    frontend = str(checkpoint["frontend_name"])
    feature_dim = int(checkpoint["feature_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])

    token_map = load_tokens(args.tokens.resolve())
    vocab = vocab_size(token_map)
    if vocab != int(checkpoint["vocab_size"]):
        raise ValueError("checkpoint/token vocabulary size mismatch")
    keyword_sequences, _, _ = load_keyword_operating_points(
        args.keywords.resolve(), token_map
    )
    if len(keyword_sequences) != 2:
        raise ValueError("probe currently contracts exactly two wake words")
    keyword_lookup = {
        tuple(int(value) for value in sequence): index
        for index, sequence in enumerate(keyword_sequences)
    }

    model = TinyStreamingGRU(feature_dim, hidden_dim, vocab)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    def named(paths: list[pathlib.Path]) -> list[tuple[str, pathlib.Path]]:
        if len(paths) != 2:
            raise ValueError("each probe split requires canonical + ordinary sidecar manifests")
        return [("canonical", paths[0].resolve()), ("ordinary-sidecar", paths[1].resolve())]

    train = extract_representations(
        model=model,
        manifests=named(args.train_manifest),
        feature_dim=feature_dim,
        vocab=vocab,
        frontend=frontend,
        keyword_lookup=keyword_lookup,
    )
    calibration = extract_representations(
        model=model,
        manifests=named(args.calibration_manifest),
        feature_dim=feature_dim,
        vocab=vocab,
        frontend=frontend,
        keyword_lookup=keyword_lookup,
    )
    test = extract_representations(
        model=model,
        manifests=named(args.test_manifest),
        feature_dim=feature_dim,
        vocab=vocab,
        frontend=frontend,
        keyword_lookup=keyword_lookup,
    )

    results: list[dict] = []
    for mode_index, mode in enumerate(modes):
        train_x, [cal_x, test_x], standardization = standardize(
            train["representations"][mode],
            calibration["representations"][mode],
            test["representations"][mode],
        )
        head, history, counts = train_linear_probe(
            x=train_x,
            labels=train["labels"],
            classes=len(classes),
            epochs=int(probe["epochs"]),
            batch_size=int(probe["batch_size"]),
            learning_rate=float(probe["learning_rate"]),
            weight_decay=float(probe["weight_decay"]),
            seed=int(probe["seed"]) + mode_index * 1009,
        )
        results.append(
            {
                "pooling": mode,
                "training_class_counts": counts,
                "standardization": standardization,
                "history": history,
                "calibration": evaluate(
                    head=head,
                    x=cal_x,
                    labels=calibration["labels"],
                    sources=calibration["sources"],
                    target_kinds=calibration["target_kinds"],
                    class_names=classes,
                ),
                "test": evaluate(
                    head=head,
                    x=test_x,
                    labels=test["labels"],
                    sources=test["sources"],
                    target_kinds=test["target_kinds"],
                    class_names=classes,
                ),
            }
        )

    selected = min(results, key=calibration_key)
    rules = config["decision_rules"]
    strong = (
        float(selected["test"]["wake_exact_recall"])
        >= float(rules["probe_strong_if_test_wake_recall_gte"])
        and float(selected["test"]["negative_false_positive_rate"])
        <= float(rules["probe_strong_if_test_negative_fp_lte"])
        and float(selected["test"]["separation_gap_p10_wake_minus_p99_nonwake"])
        > float(rules["probe_strong_if_test_gap_gt"])
    )
    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "promotion_allowed": False,
        "shipping_metric": False,
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "architecture": ARCHITECTURE,
            "frontend": frontend,
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "vocab_size": vocab,
        },
        "config_sha256": sha256_file(args.config.resolve()),
        "tokens_sha256": sha256_file(args.tokens.resolve()),
        "keywords_sha256": sha256_file(args.keywords.resolve()),
        "train_manifests": train["manifests"],
        "calibration_manifests": calibration["manifests"],
        "test_manifests": test["manifests"],
        "probe_architecture": str(probe["architecture"]),
        "encoder_trainable": False,
        "results": results,
        "calibration_selected_pooling": selected["pooling"],
        "selected_test": selected["test"],
        "strong_probe": strong,
        "next_architecture_action": (
            str(rules["strong_probe_next"])
            if strong
            else str(rules["weak_probe_next"])
        ),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"frozen-encoder-verifier-probe: selected={selected['pooling']} "
        f"wake-recall={selected['test']['wake_exact_recall']:.4f} "
        f"negative-fp={selected['test']['negative_false_positive_rate']:.4f} "
        f"gap={selected['test']['separation_gap_p10_wake_minus_p99_nonwake']:.4f} "
        f"strong={strong}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
