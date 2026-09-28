#!/usr/bin/env python3
"""Build product-realistic decoder-boundary references.

Positive authority is an existing full-keyword *natural pause* recording from
the governed speech-like base. Negative authority is a same-voice pair of
standalone half-keyword utterances separated by a long digital-silence boundary.

This intentionally does NOT create an inserted 400-ms pause inside a full word:
that construction is an ambiguity stress, not a product acceptance label.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from build_decoder_boundary_references import (  # noqa: E402
    FRAME_HOP_SAMPLES,
    SAMPLE_RATE_HZ,
    load_jsonl,
    read_pcm16,
    repo_relative_path,
    resolve_audio,
    safe_name,
    sha256_file,
    source_utterance_id,
    stitched_samples,
    voice_id,
    write_jsonl,
    write_wav,
)

POLICY = "product-natural-pause-boundary-reference-v1"
EVIDENCE_CLASS = "decoder-product-boundary-reference-builder-v1"
POSITIVE_ROLE = "natural-full-phrase-pause-v1"
NEGATIVE_ROLE = "cross-utterance-long-gap-v1"
DEFAULT_GAP_MS = 400
DEFAULT_LEAD_MS = 1000
DEFAULT_TAIL_MS = 1000


def select_product_boundary_sources(rows: list[dict]) -> list[dict]:
    pauses: dict[tuple[str, int], dict] = {}
    negatives: dict[tuple[str, tuple[int, ...]], dict] = {}

    for row in rows:
        voice = voice_id(row)
        target = tuple(int(value) for value in row.get("target_ids", ()))
        kind = str(row.get("kind", ""))
        if kind == "positive" and target:
            keyword_id = row.get("keyword_id")
            if (
                isinstance(keyword_id, bool)
                or not isinstance(keyword_id, int)
                or keyword_id <= 0
            ):
                continue
            if source_utterance_id(row).endswith("-pause"):
                key = (voice, keyword_id)
                if key in pauses:
                    raise ValueError(f"duplicate pause positive for voice/keyword: {key}")
                pauses[key] = row
        elif kind == "negative" and target:
            key = (voice, target)
            if key in negatives:
                raise ValueError(f"duplicate negative target for voice: {key}")
            negatives[key] = row

    selected: list[dict] = []
    for (voice, keyword_id), positive in sorted(pauses.items()):
        target = tuple(int(value) for value in positive.get("target_ids", ()))
        if len(target) < 2 or len(target) % 2:
            raise ValueError(
                f"keyword {keyword_id} target length must be even for boundary construction"
            )
        middle = len(target) // 2
        left_target = target[:middle]
        right_target = target[middle:]
        left = negatives.get((voice, left_target))
        right = negatives.get((voice, right_target))
        if left is None or right is None:
            raise ValueError(
                f"voice {voice} keyword {keyword_id} lacks standalone half-word negatives"
            )
        selected.append(
            {
                "voice_id": voice,
                "keyword_id": keyword_id,
                "target_ids": target,
                "positive": positive,
                "left": left,
                "right": right,
            }
        )
    if not selected:
        raise ValueError("speech-like index contains no natural full-keyword pause sources")
    return selected


def positive_event(row: dict, *, keyword_id: int, sample_count: int) -> dict:
    start = row.get("event_start_frame")
    end = row.get("event_end_frame")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or not 0 <= start < end <= sample_count
    ):
        raise ValueError("natural pause positive has invalid activity bounds")
    return {
        "keyword_id": keyword_id,
        "start_s": start / SAMPLE_RATE_HZ,
        "end_s": end / SAMPLE_RATE_HZ,
    }


def build(args: argparse.Namespace) -> dict:
    index = args.dataset_index.resolve()
    output = args.output_dir.resolve()
    rows = load_jsonl(index)
    selected = select_product_boundary_sources(rows)

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
        raise ValueError("gap/lead/tail durations must be positive hop-aligned values")

    audio_dir = output / "audio"
    positive_refs: list[dict] = []
    negative_refs: list[dict] = []
    evidence: list[dict] = []

    for item in selected:
        voice = str(item["voice_id"])
        keyword_id = int(item["keyword_id"])
        positive_row = item["positive"]
        left_row = item["left"]
        right_row = item["right"]
        stem = f"{safe_name(voice)}-kw{keyword_id}"

        positive_source = resolve_audio(index, positive_row)
        positive_samples = read_pcm16(positive_source)
        positive_out = audio_dir / f"{stem}-natural-pause.wav"
        positive_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(positive_source, positive_out)
        event = positive_event(
            positive_row,
            keyword_id=keyword_id,
            sample_count=len(positive_samples),
        )
        positive_refs.append(
            {
                "recording": f"{stem}-natural-pause",
                "path": repo_relative_path(positive_out),
                "duration_s": len(positive_samples) / SAMPLE_RATE_HZ,
                "boundary_role": POSITIVE_ROLE,
                "expected": [event],
            }
        )

        left_audio = resolve_audio(index, left_row)
        right_audio = resolve_audio(index, right_row)
        left_samples = read_pcm16(left_audio)
        right_samples = read_pcm16(right_audio)
        negative_samples, effective_gap = stitched_samples(
            left_samples, right_samples, gap_samples, lead_samples, tail_samples
        )
        negative_out = audio_dir / f"{stem}-cross-utterance.wav"
        write_wav(negative_out, negative_samples)
        negative_refs.append(
            {
                "recording": f"{stem}-cross-utterance",
                "path": repo_relative_path(negative_out),
                "duration_s": len(negative_samples) / SAMPLE_RATE_HZ,
                "boundary_role": NEGATIVE_ROLE,
                "expected": [],
            }
        )

        evidence.append(
            {
                "voice_id": voice,
                "keyword_id": keyword_id,
                "target_ids": list(item["target_ids"]),
                "positive_source_id": positive_row["speech_like_provenance"]["source_id"],
                "positive_wav_sha256": str(positive_row["wav_sha256"]),
                "positive_output_sha256": sha256_file(positive_out),
                "left_source_id": left_row["speech_like_provenance"]["source_id"],
                "right_source_id": right_row["speech_like_provenance"]["source_id"],
                "requested_gap_samples": gap_samples,
                "cross_boundary_effective_silence_samples": effective_gap,
                "cross_boundary_wav_sha256": sha256_file(negative_out),
            }
        )

    positive_path = output / "within-word-natural-pause.references.jsonl"
    negative_path = output / "cross-utterance-long-gap.references.jsonl"
    write_jsonl(positive_path, positive_refs)
    write_jsonl(negative_path, negative_refs)
    summary = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "protected_evidence_used": False,
        "release_authority": False,
        "ambiguity_stress": False,
        "dataset_index": str(index),
        "dataset_index_sha256": sha256_file(index),
        "gap_ms": args.gap_ms,
        "positive_role": POSITIVE_ROLE,
        "negative_role": NEGATIVE_ROLE,
        "positive_references": positive_path.name,
        "positive_references_sha256": sha256_file(positive_path),
        "negative_references": negative_path.name,
        "negative_references_sha256": sha256_file(negative_path),
        "recordings_per_role": len(evidence),
        "records": evidence,
    }
    (output / "boundary-reference-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build natural-pause positives and long-gap cross-utterance negatives."
    )
    parser.add_argument("--dataset-index", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS)
    parser.add_argument("--lead-ms", type=int, default=DEFAULT_LEAD_MS)
    parser.add_argument("--tail-ms", type=int, default=DEFAULT_TAIL_MS)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps({
        "recordings_per_role": result["recordings_per_role"],
        "positive_role": result["positive_role"],
        "negative_role": result["negative_role"],
        "selection_feedback_allowed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
