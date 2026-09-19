#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tools"))

from gru_model import TinyStreamingGRU  # noqa: E402
from kws_vocab import load_tokens  # noqa: E402
from model import TinyStreamingRNN  # noqa: E402
from sequence_margin import _decoder_sequence_log_confidence, _target_rows  # noqa: E402
from train_ctc import (  # noqa: E402
    Manifest,
    collate,
    load_keyword_operating_points,
    vocab_size,
)

EVIDENCE_CLASS = "kws-v2-research-float-ctc-confidence-v1"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * q))
    return float(ordered[index])


def parse_manifest(raw: str) -> tuple[str, pathlib.Path]:
    name, sep, value = raw.partition("=")
    if not sep or not name.strip() or not value.strip():
        raise ValueError("--split-manifest must be NAME=PATH")
    return name.strip(), pathlib.Path(value).resolve()


def compact_distribution(values: list[float]) -> dict:
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


def build_model(family: str, checkpoint: dict) -> torch.nn.Module:
    feature_dim = int(checkpoint["feature_dim"])
    hidden_dim = int(checkpoint["hidden_dim"])
    vocab = int(checkpoint["vocab_size"])
    if family == "gru":
        model: torch.nn.Module = TinyStreamingGRU(feature_dim, hidden_dim, vocab)
    else:
        model = TinyStreamingRNN(feature_dim, hidden_dim, vocab)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def evaluate_split(
    *,
    model: torch.nn.Module,
    manifests: list[pathlib.Path],
    feature_dim: int,
    vocab: int,
    frontend: str,
    keyword_sequences: list[list[int]],
    batch_size: int,
    thresholds: list[float],
) -> dict:
    dataset = Manifest(manifests, feature_dim, vocab, frontend)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    keyword_tuples = [tuple(int(v) for v in row) for row in keyword_sequences]
    keyword_lookup = {row: index for index, row in enumerate(keyword_tuples)}
    positive_scores: list[float] = []
    nonwake_scores: list[float] = []
    tokenized_nonwake_scores: list[float] = []
    empty_target_nonwake_scores: list[float] = []
    wake_frames = 0
    wake_blank_top1_frames = 0
    wake_keyword_root_top1_frames = 0
    wake_any_keyword_token_top1_frames = 0
    keyword_roots = {sequence[0] for sequence in keyword_tuples}
    keyword_tokens = {token for sequence in keyword_tuples for token in sequence}
    keyword_positive: dict[str, list[float]] = {
        str(index + 1): [] for index in range(len(keyword_tuples))
    }

    with torch.no_grad():
        for x, y, xlen, ylen in loader:
            log_probs = model(x).log_softmax(dim=2)
            targets = _target_rows(y, ylen)
            for batch_index, true_row in enumerate(targets):
                steps = int(xlen[batch_index])
                sample = log_probs[:steps, batch_index, :]
                confidences = [
                    float(torch.exp(_decoder_sequence_log_confidence(sample, sequence)))
                    for sequence in keyword_tuples
                ]
                wake_index = keyword_lookup.get(true_row)
                if wake_index is None:
                    score = max(confidences)
                    nonwake_scores.append(score)
                    if true_row:
                        tokenized_nonwake_scores.append(score)
                    else:
                        empty_target_nonwake_scores.append(score)
                else:
                    score = confidences[wake_index]
                    positive_scores.append(score)
                    keyword_positive[str(wake_index + 1)].append(score)
                    top1 = sample.argmax(dim=1)
                    wake_frames += int(top1.numel())
                    wake_blank_top1_frames += int((top1 == 0).sum())
                    wake_keyword_root_top1_frames += int(
                        sum(int(token in keyword_roots) for token in top1.tolist())
                    )
                    wake_any_keyword_token_top1_frames += int(
                        sum(int(token in keyword_tokens) for token in top1.tolist())
                    )

    operating_curve: list[dict] = []
    for threshold in thresholds:
        false_rejects = sum(score < threshold for score in positive_scores)
        false_accepts = sum(score >= threshold for score in nonwake_scores)
        operating_curve.append(
            {
                "threshold": threshold,
                "positive_examples": len(positive_scores),
                "false_rejects": false_rejects,
                "frr": false_rejects / max(1, len(positive_scores)),
                "nonwake_examples": len(nonwake_scores),
                "false_accepts": false_accepts,
                "nonwake_false_positive_clip_rate": false_accepts / max(1, len(nonwake_scores)),
            }
        )

    positive_p10 = quantile(positive_scores, 0.10)
    nonwake_p99 = quantile(nonwake_scores, 0.99)
    return {
        "examples": len(dataset),
        "wake_examples": len(positive_scores),
        "nonwake_examples": len(nonwake_scores),
        "positive_true_keyword_confidence": compact_distribution(positive_scores),
        "nonwake_max_keyword_confidence": compact_distribution(nonwake_scores),
        "tokenized_nonwake_max_keyword_confidence": compact_distribution(
            tokenized_nonwake_scores
        ),
        "empty_target_nonwake_max_keyword_confidence": compact_distribution(
            empty_target_nonwake_scores
        ),
        "wake_top1_frame_diagnostics": {
            "frames": wake_frames,
            "blank_top1_frames": wake_blank_top1_frames,
            "blank_top1_fraction": (
                wake_blank_top1_frames / wake_frames if wake_frames else None
            ),
            "keyword_root_top1_frames": wake_keyword_root_top1_frames,
            "keyword_root_top1_fraction": (
                wake_keyword_root_top1_frames / wake_frames if wake_frames else None
            ),
            "any_keyword_token_top1_frames": wake_any_keyword_token_top1_frames,
            "any_keyword_token_top1_fraction": (
                wake_any_keyword_token_top1_frames / wake_frames if wake_frames else None
            ),
        },
        "per_keyword_positive_confidence": {
            key: compact_distribution(value) for key, value in keyword_positive.items()
        },
        "separation_gap_p10_wake_minus_p99_nonwake": (
            None
            if positive_p10 is None or nonwake_p99 is None
            else positive_p10 - nonwake_p99
        ),
        "operating_curve": operating_curve,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Research-only float-checkpoint CTC keyword-confidence diagnostic."
    )
    parser.add_argument("--family", required=True, choices=("rnn", "gru"))
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--split-manifest", required=True, action="append")
    parser.add_argument("--thresholds", nargs="+", required=True, type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    thresholds = sorted(set(float(value) for value in args.thresholds))
    if not thresholds or any(not 0.0 < value < 1.0 for value in thresholds):
        parser.error("--thresholds must contain values in (0,1)")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(args.family, checkpoint)
    token_map = load_tokens(args.tokens.resolve())
    keyword_sequences, _, _ = load_keyword_operating_points(
        args.keywords.resolve(), token_map
    )
    vocab = vocab_size(token_map)
    if vocab != int(checkpoint["vocab_size"]):
        raise ValueError("checkpoint/token vocabulary size mismatch")
    frontend = str(checkpoint["frontend_name"])
    feature_dim = int(checkpoint["feature_dim"])

    by_split: dict[str, list[pathlib.Path]] = {}
    for raw in args.split_manifest:
        name, path = parse_manifest(raw)
        if not path.is_file():
            raise ValueError(f"split manifest missing: {path}")
        by_split.setdefault(name, []).append(path)

    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "evidence_scope": "research-only",
        "diagnostic_only": True,
        "protected_evidence_used": False,
        "shipping_metric": False,
        "family": args.family,
        "checkpoint": str(checkpoint_path),
        "frontend": frontend,
        "feature_dim": feature_dim,
        "thresholds": thresholds,
        "splits": {
            name: evaluate_split(
                model=model,
                manifests=manifests,
                feature_dim=feature_dim,
                vocab=vocab,
                frontend=frontend,
                keyword_sequences=keyword_sequences,
                batch_size=args.batch_size,
                thresholds=thresholds,
            )
            for name, manifests in sorted(by_split.items())
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "float-ctc-confidence: "
        + " ".join(
            f"{name}:wake={row['wake_examples']}:nonwake={row['nonwake_examples']}:"
            f"gap={row['separation_gap_p10_wake_minus_p99_nonwake']}"
            for name, row in result["splits"].items()
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
