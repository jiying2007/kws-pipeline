#!/usr/bin/env python3
"""Measure train-split boundary separability before changing decoder rules.

This diagnostic is development-only and feedback-allowed. It consumes only the
governed train external-base split and a frozen caller-supplied model. It does
not read calibration/test/qualification labels, retrain, change decoder policy,
or produce release authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import struct
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
TOOLS = ROOT / "tools"
sys.path[:0] = [str(EVAL), str(TOOLS)]

from build_decoder_boundary_references import (  # noqa: E402
    FRAME_HOP_SAMPLES,
    SAMPLE_RATE_HZ,
    internal_split_from_alignment,
    paused_samples,
    read_pcm16,
    resolve_audio,
    safe_name,
    select_boundary_sources,
    stitched_samples,
    write_wav,
)
from diagnose_sequence_margin_runtime_gap import (  # noqa: E402
    TRACE_HEADER_BYTES,
    TRACE_MAGIC,
    decoder_sequence_log_confidence,
    log_softmax,
)
from run_corpus import ensure_cached_trace  # noqa: E402

POLICY = "decoder-boundary-train-separability-v1"
EVIDENCE_CLASS = "decoder-boundary-separability-development-v1"
FEATURES = (
    "left_sequence_log_confidence",
    "left_last_token_margin_max",
    "gap_blank_top1_ratio",
    "gap_blank_margin_mean",
    "gap_blank_longest_run_frames",
    "gap_speech_active_ratio",
    "right_first_token_margin_max",
    "right_sequence_log_confidence",
    "split_sequence_log_confidence",
    "full_sequence_log_confidence",
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def assert_train_split(rows: list[dict]) -> None:
    seen = 0
    for row in rows:
        provenance = row.get("speech_like_provenance")
        if not isinstance(provenance, dict):
            continue
        source_id = str(provenance.get("source_id", ""))
        if source_id:
            seen += 1
            if not source_id.startswith("speech-like:train:"):
                raise ValueError(
                    "boundary separability accepts only speech-like:train source rows"
                )
    if seen == 0:
        raise ValueError("train split contains no speech-like provenance")


def read_trace_frames(path: pathlib.Path) -> list[dict]:
    data = path.read_bytes()
    if len(data) < TRACE_HEADER_BYTES or data[:8] != TRACE_MAGIC:
        raise ValueError(f"invalid posterior trace header: {path}")
    vocab_size = struct.unpack_from("<H", data, 12)[0]
    frame_count = struct.unpack_from("<Q", data, 40)[0]
    if vocab_size < 2 or frame_count <= 0:
        raise ValueError(f"invalid posterior trace dimensions: {path}")
    record_bytes = 16 + 4 * vocab_size
    if len(data) != TRACE_HEADER_BYTES + frame_count * record_bytes:
        raise ValueError(f"posterior trace size mismatch: {path}")
    frames: list[dict] = []
    offset = TRACE_HEADER_BYTES
    last_end = 0
    for _ in range(frame_count):
        end_sample = struct.unpack_from("<Q", data, offset)[0]
        flags = data[offset + 8 : offset + 16]
        if end_sample <= last_end or flags[0] not in (0, 1) or any(flags[1:]):
            raise ValueError(f"invalid posterior trace frame metadata: {path}")
        logits = list(
            struct.unpack_from("<" + "f" * vocab_size, data, offset + 16)
        )
        if any(not math.isfinite(value) for value in logits):
            raise ValueError(f"non-finite posterior trace logits: {path}")
        frames.append(
            {
                "end_sample": end_sample,
                "speech_active": int(flags[0]),
                "logits": logits,
            }
        )
        last_end = end_sample
        offset += record_bytes
    return frames


def longest_true_run(values: list[bool]) -> int:
    best = 0
    current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def token_margin(row: list[float], token: int) -> float:
    if not 0 < token < len(row):
        raise ValueError("token margin requires a nonblank in-vocabulary token")
    competitor = max(value for index, value in enumerate(row) if index != token)
    return row[token] - competitor


def finite_sequence_score(log_probs: list[list[float]], sequence: tuple[int, ...]) -> float:
    value = decoder_sequence_log_confidence(log_probs, sequence)
    if not math.isfinite(value):
        raise ValueError("boundary sequence score is not finite")
    return value


def extract_features(
    frames: list[dict],
    *,
    gap_start_sample: int,
    gap_end_sample: int,
    left_target: tuple[int, ...],
    right_target: tuple[int, ...],
    full_target: tuple[int, ...],
) -> dict[str, float]:
    if not 0 < gap_start_sample < gap_end_sample:
        raise ValueError("invalid gap geometry")
    pre = [frame for frame in frames if frame["end_sample"] <= gap_start_sample]
    gap = [
        frame
        for frame in frames
        if gap_start_sample < frame["end_sample"] <= gap_end_sample
    ]
    post = [frame for frame in frames if frame["end_sample"] > gap_end_sample]
    if not pre or not gap or not post:
        raise ValueError("posterior trace does not cover pre/gap/post windows")

    pre_lp = log_softmax([frame["logits"] for frame in pre])
    post_lp = log_softmax([frame["logits"] for frame in post])
    all_lp = log_softmax([frame["logits"] for frame in frames])
    left_score = finite_sequence_score(pre_lp, left_target)
    right_score = finite_sequence_score(post_lp, right_target)
    split_score = (
        left_score * len(left_target) + right_score * len(right_target)
    ) / len(full_target)

    blank_top1 = [
        max(range(len(frame["logits"])), key=frame["logits"].__getitem__) == 0
        for frame in gap
    ]
    blank_margins = [
        frame["logits"][0] - max(frame["logits"][1:]) for frame in gap
    ]
    lookback = pre[-15:]
    lookahead = post[:15]

    return {
        "left_sequence_log_confidence": left_score,
        "left_last_token_margin_max": max(
            token_margin(frame["logits"], left_target[-1]) for frame in lookback
        ),
        "gap_blank_top1_ratio": sum(blank_top1) / len(blank_top1),
        "gap_blank_margin_mean": sum(blank_margins) / len(blank_margins),
        "gap_blank_longest_run_frames": float(longest_true_run(blank_top1)),
        "gap_speech_active_ratio": sum(
            int(frame["speech_active"]) for frame in gap
        ) / len(gap),
        "right_first_token_margin_max": max(
            token_margin(frame["logits"], right_target[0]) for frame in lookahead
        ),
        "right_sequence_log_confidence": right_score,
        "split_sequence_log_confidence": split_score,
        "full_sequence_log_confidence": finite_sequence_score(all_lp, full_target),
    }


def candidate_thresholds(values: list[float]) -> list[float]:
    ordered = sorted(set(values))
    if not ordered:
        raise ValueError("threshold search requires values")
    span = max(1.0, ordered[-1] - ordered[0])
    result = [ordered[0] - span]
    result.extend((a + b) / 2.0 for a, b in zip(ordered, ordered[1:]))
    result.append(ordered[-1] + span)
    return result


def classify(value: float, direction: str, threshold: float) -> bool:
    if direction == "positive_higher":
        return value >= threshold
    if direction == "positive_lower":
        return value <= threshold
    raise ValueError("invalid threshold direction")


def fit_scalar_threshold(samples: list[tuple[float, bool]]) -> dict:
    if not samples or {label for _, label in samples} != {False, True}:
        raise ValueError("threshold fit requires both classes")
    thresholds = candidate_thresholds([value for value, _ in samples])
    choices: list[tuple[int, str, float]] = []
    for direction in ("positive_higher", "positive_lower"):
        for threshold in thresholds:
            correct = sum(
                classify(value, direction, threshold) is label
                for value, label in samples
            )
            choices.append((correct, direction, threshold))
    correct, direction, threshold = max(
        choices, key=lambda item: (item[0], item[1] == "positive_higher", -abs(item[2]))
    )
    return {
        "direction": direction,
        "threshold": threshold,
        "correct": correct,
        "total": len(samples),
        "accuracy": correct / len(samples),
    }


def summarize_feature(records: list[dict], feature: str) -> dict:
    positives = [float(row["positive"]["features"][feature]) for row in records]
    negatives = [float(row["negative"]["features"][feature]) for row in records]
    fitted = fit_scalar_threshold(
        [(value, True) for value in positives]
        + [(value, False) for value in negatives]
    )
    voices = sorted({str(row["voice_id"]) for row in records})
    loov_correct = 0
    loov_total = 0
    folds: list[dict] = []
    if len(voices) >= 2:
        for voice in voices:
            train: list[tuple[float, bool]] = []
            held: list[tuple[float, bool]] = []
            for row in records:
                target = held if row["voice_id"] == voice else train
                target.append((float(row["positive"]["features"][feature]), True))
                target.append((float(row["negative"]["features"][feature]), False))
            if not train or not held:
                continue
            rule = fit_scalar_threshold(train)
            correct = sum(
                classify(value, rule["direction"], float(rule["threshold"])) is label
                for value, label in held
            )
            loov_correct += correct
            loov_total += len(held)
            folds.append(
                {
                    "held_out_voice": voice,
                    "direction": rule["direction"],
                    "threshold": rule["threshold"],
                    "correct": correct,
                    "total": len(held),
                }
            )

    positive_min, positive_max = min(positives), max(positives)
    negative_min, negative_max = min(negatives), max(negatives)
    range_direction = None
    range_margin = 0.0
    if positive_min > negative_max:
        range_direction = "positive_higher"
        range_margin = positive_min - negative_max
    elif positive_max < negative_min:
        range_direction = "positive_lower"
        range_margin = negative_min - positive_max

    deltas = [p - n for p, n in zip(positives, negatives)]
    paired_direction = None
    if all(value > 0.0 for value in deltas):
        paired_direction = "positive_higher"
    elif all(value < 0.0 for value in deltas):
        paired_direction = "positive_lower"

    return {
        "positive": {
            "min": positive_min,
            "max": positive_max,
            "mean": sum(positives) / len(positives),
        },
        "negative": {
            "min": negative_min,
            "max": negative_max,
            "mean": sum(negatives) / len(negatives),
        },
        "range_disjoint": range_direction is not None,
        "range_direction": range_direction,
        "range_margin": range_margin,
        "paired_direction_consistent": paired_direction is not None,
        "paired_direction": paired_direction,
        "paired_deltas": deltas,
        "best_train_scalar_rule": fitted,
        "leave_one_voice_out": {
            "folds": folds,
            "correct": loov_correct,
            "total": loov_total,
            "accuracy": loov_correct / loov_total if loov_total else None,
        },
    }


def build(args: argparse.Namespace) -> dict:
    os.chdir(ROOT)
    index = args.dataset_index.resolve()
    model = args.model.resolve()
    posterior_dump = args.posterior_dump.resolve()
    cache = args.posterior_cache.resolve()
    output = args.output_dir.resolve()
    for path, label in (
        (index, "dataset index"),
        (model, "model"),
        (posterior_dump, "posterior dump"),
    ):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    rows = load_jsonl(index)
    assert_train_split(rows)
    selected = select_boundary_sources(rows)
    model_sha = sha256_file(model)
    posterior_dump_sha = sha256_file(posterior_dump)

    gap_samples = round(args.gap_ms * SAMPLE_RATE_HZ / 1000.0)
    lead_samples = round(args.lead_ms * SAMPLE_RATE_HZ / 1000.0)
    tail_samples = round(args.tail_ms * SAMPLE_RATE_HZ / 1000.0)
    if (
        gap_samples <= 0
        or any(
            value < 0 or value % FRAME_HOP_SAMPLES
            for value in (gap_samples, lead_samples, tail_samples)
        )
    ):
        raise ValueError("gap/lead/tail durations must be hop-aligned")

    audio_dir = output / "audio"
    alignment_dir = output / "alignment"
    cache.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for item in selected:
        voice = str(item["voice_id"])
        keyword_id = int(item["keyword_id"])
        target = tuple(int(value) for value in item["target_ids"])
        middle = len(target) // 2
        left_target = target[:middle]
        right_target = target[middle:]
        positive_row = item["positive"]
        left_row = item["left"]
        right_row = item["right"]

        positive_audio = resolve_audio(index, positive_row)
        raw = read_pcm16(positive_audio)
        start = positive_row.get("event_start_frame")
        end = positive_row.get("event_end_frame")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(raw)
        ):
            raise ValueError("exact positive row has invalid activity bounds")

        stem = f"{safe_name(voice)}-kw{keyword_id}"
        alignment_audio = alignment_dir / f"{stem}-fullword.wav"
        write_wav(alignment_audio, [0] * lead_samples + raw + [0] * tail_samples)
        alignment_sha = sha256_file(alignment_audio)
        alignment_trace, alignment_trace_summary, _ = ensure_cached_trace(
            posterior_dump=posterior_dump,
            cache_root=cache,
            model=model,
            audio=alignment_audio,
            model_sha256=model_sha,
            audio_sha256=alignment_sha,
            posterior_dump_sha256=posterior_dump_sha,
        )
        alignment = internal_split_from_alignment(
            [frame["logits"] for frame in read_trace_frames(alignment_trace)],
            target,
            event_start_sample=start,
            event_end_sample=end,
            lead_samples=lead_samples,
        )
        split = int(alignment["split_samples"])

        positive_samples = paused_samples(
            raw, split, gap_samples, lead_samples, tail_samples
        )
        positive_out = audio_dir / f"{stem}-within-word-pause.wav"
        write_wav(positive_out, positive_samples)
        positive_sha = sha256_file(positive_out)
        positive_trace, positive_trace_summary, _ = ensure_cached_trace(
            posterior_dump=posterior_dump,
            cache_root=cache,
            model=model,
            audio=positive_out,
            model_sha256=model_sha,
            audio_sha256=positive_sha,
            posterior_dump_sha256=posterior_dump_sha,
        )
        positive_features = extract_features(
            read_trace_frames(positive_trace),
            gap_start_sample=lead_samples + split,
            gap_end_sample=lead_samples + split + gap_samples,
            left_target=left_target,
            right_target=right_target,
            full_target=target,
        )

        left_audio = resolve_audio(index, left_row)
        right_audio = resolve_audio(index, right_row)
        left_samples = read_pcm16(left_audio)
        right_samples = read_pcm16(right_audio)
        negative_samples, effective_gap = stitched_samples(
            left_samples, right_samples, gap_samples, lead_samples, tail_samples
        )
        negative_out = audio_dir / f"{stem}-cross-boundary.wav"
        write_wav(negative_out, negative_samples)
        negative_sha = sha256_file(negative_out)
        negative_trace, negative_trace_summary, _ = ensure_cached_trace(
            posterior_dump=posterior_dump,
            cache_root=cache,
            model=model,
            audio=negative_out,
            model_sha256=model_sha,
            audio_sha256=negative_sha,
            posterior_dump_sha256=posterior_dump_sha,
        )
        negative_gap_start = lead_samples + len(left_samples)
        negative_features = extract_features(
            read_trace_frames(negative_trace),
            gap_start_sample=negative_gap_start,
            gap_end_sample=negative_gap_start + effective_gap,
            left_target=left_target,
            right_target=right_target,
            full_target=target,
        )

        records.append(
            {
                "voice_id": voice,
                "keyword_id": keyword_id,
                "target_ids": list(target),
                "positive_source_id": positive_row["speech_like_provenance"]["source_id"],
                "left_source_id": left_row["speech_like_provenance"]["source_id"],
                "right_source_id": right_row["speech_like_provenance"]["source_id"],
                "alignment": {
                    "audio_sha256": alignment_sha,
                    "trace_sha256": alignment_trace_summary["trace_sha256"],
                    "posterior_dump_sha256": alignment_trace_summary["posterior_dump_sha256"],
                },
                "positive": {
                    "audio_sha256": positive_sha,
                    "trace_sha256": positive_trace_summary["trace_sha256"],
                    "posterior_dump_sha256": positive_trace_summary["posterior_dump_sha256"],
                    "gap_start_sample": lead_samples + split,
                    "gap_end_sample": lead_samples + split + gap_samples,
                    "features": positive_features,
                },
                "negative": {
                    "audio_sha256": negative_sha,
                    "trace_sha256": negative_trace_summary["trace_sha256"],
                    "posterior_dump_sha256": negative_trace_summary["posterior_dump_sha256"],
                    "gap_start_sample": negative_gap_start,
                    "gap_end_sample": negative_gap_start + effective_gap,
                    "features": negative_features,
                },
                "paired_delta_positive_minus_negative": {
                    feature: positive_features[feature] - negative_features[feature]
                    for feature in FEATURES
                },
            }
        )

    feature_summary = {
        feature: summarize_feature(records, feature) for feature in FEATURES
    }
    stable = [
        feature
        for feature, summary in feature_summary.items()
        if summary["best_train_scalar_rule"]["accuracy"] == 1.0
        and summary["leave_one_voice_out"]["accuracy"] == 1.0
    ]
    result = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": True,
        "protected_evidence_used": False,
        "release_authority": False,
        "source_split": "train",
        "dataset_index_sha256": sha256_file(index),
        "model_sha256": model_sha,
        "posterior_dump_sha256": posterior_dump_sha,
        "gap_ms": args.gap_ms,
        "lead_ms": args.lead_ms,
        "tail_ms": args.tail_ms,
        "pairs": len(records),
        "voices": len({row["voice_id"] for row in records}),
        "features": feature_summary,
        "stable_scalar_features": stable,
        "simple_scalar_separability_observed": bool(stable),
        "records": records,
    }
    write_json(output / "separability.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure train-only decoder-boundary feature separability."
    )
    parser.add_argument("--dataset-index", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gap-ms", type=int, default=400)
    parser.add_argument("--lead-ms", type=int, default=1000)
    parser.add_argument("--tail-ms", type=int, default=1000)
    args = parser.parse_args()
    result = build(args)
    print(
        json.dumps(
            {
                "pairs": result["pairs"],
                "voices": result["voices"],
                "stable_scalar_features": result["stable_scalar_features"],
                "simple_scalar_separability_observed": result[
                    "simple_scalar_separability_observed"
                ],
                "selection_feedback_allowed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
