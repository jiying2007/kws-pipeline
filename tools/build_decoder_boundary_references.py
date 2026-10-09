#!/usr/bin/env python3
"""Build decoder-boundary *ambiguity stress* corpora from speech-like base audio.

This is not product acceptance authority. The positive corpus inserts silence
inside one original full-keyword recording; the negative corpus joins standalone
halves. At long equal gaps these labels can be acoustically ambiguous. Canonical
product boundary acceptance is owned by build_decoder_boundary_product_references.py.

The positive corpus inserts silence inside one original full-keyword recording at
an internal CTC-aligned token boundary. The negative corpus joins standalone
half-keyword utterances from the same voice with a silence boundary and carries
no expected wake label.

This is development-only evidence construction. It does not change model,
decoder, threshold, training, qualification, or shipping authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import struct
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
TOOLS = ROOT / "tools"
sys.path[:0] = [str(EVAL), str(TOOLS)]

from diagnose_sequence_margin_runtime_gap import log_softmax, read_trace_logits  # noqa: E402
from run_corpus import ensure_cached_trace  # noqa: E402

POLICY = "boundary-reference-builder-v2"
ALIGNMENT_POLICY = "ctc-viterbi-event-window-token-midpoint-v1"
SAMPLE_RATE_HZ = 16000
FRAME_LENGTH_SAMPLES = 400
FRAME_HOP_SAMPLES = 320
DEFAULT_GAP_MS = 400
DEFAULT_LEAD_MS = 1000
DEFAULT_TAIL_MS = 1000


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def read_pcm16(path: pathlib.Path) -> list[int]:
    if not path.is_file():
        raise ValueError(f"audio file missing: {path}")
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != SAMPLE_RATE_HZ
        ):
            raise ValueError(f"{path}: expected mono PCM16 16-kHz WAV")
        raw = reader.readframes(reader.getnframes())
    if not raw or len(raw) % 2:
        raise ValueError(f"{path}: empty/invalid PCM16 WAV")
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    if not samples:
        raise ValueError("cannot write an empty boundary WAV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.writeframes(b"".join(struct.pack("<h", int(sample)) for sample in samples))


def repo_relative_path(path: pathlib.Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("boundary reference output must stay inside repository root") from exc


def source_utterance_id(row: dict) -> str:
    provenance = row.get("speech_like_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("speech-like row is missing provenance")
    source_id = str(provenance.get("source_id", "")).strip()
    if not source_id or ":" not in source_id:
        raise ValueError("speech-like source_id is invalid")
    return source_id.rsplit(":", 1)[-1]


def voice_id(row: dict) -> str:
    provenance = row.get("speech_like_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("speech-like row is missing provenance")
    value = str(provenance.get("voice_id", "")).strip()
    if not value:
        raise ValueError("speech-like voice_id is missing")
    return value


def resolve_audio(index: pathlib.Path, row: dict) -> pathlib.Path:
    raw = row.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("speech-like row path is missing")
    path = pathlib.Path(raw.strip())
    path = path.resolve() if path.is_absolute() else (index.parent / path).resolve()
    expected = str(row.get("wav_sha256", ""))
    if not path.is_file() or len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"speech-like audio/hash mismatch: {path}")
    return path


def select_boundary_sources(rows: list[dict]) -> list[dict]:
    """Return exact positives and same-voice half-word sources per keyword."""
    positives: dict[tuple[str, int], dict] = {}
    negatives: dict[tuple[str, tuple[int, ...]], dict] = {}

    for row in rows:
        voice = voice_id(row)
        target = tuple(int(value) for value in row.get("target_ids", ()))
        kind = str(row.get("kind", ""))
        if kind == "positive" and source_utterance_id(row).endswith("-exact"):
            keyword_id = row.get("keyword_id")
            if isinstance(keyword_id, bool) or not isinstance(keyword_id, int) or keyword_id <= 0:
                raise ValueError("exact positive row has invalid keyword_id")
            key = (voice, keyword_id)
            if key in positives:
                raise ValueError(f"duplicate exact positive for voice/keyword: {key}")
            positives[key] = row
        elif kind == "negative" and target:
            key = (voice, target)
            if key in negatives:
                raise ValueError(f"duplicate negative target for voice: {key}")
            negatives[key] = row

    selected: list[dict] = []
    for (voice, keyword_id), positive in sorted(positives.items()):
        target = tuple(int(value) for value in positive.get("target_ids", ()))
        if len(target) < 2 or len(target) % 2:
            raise ValueError(
                f"keyword {keyword_id} target length must be even for boundary construction"
            )
        middle = len(target) // 2
        left_target, right_target = target[:middle], target[middle:]
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
        raise ValueError("speech-like index contains no exact boundary source sets")
    return selected


def _finite_rows(logits: list[list[float]]) -> None:
    if not logits or not all(isinstance(row, list) and len(row) >= 2 for row in logits):
        raise ValueError("forced alignment requires nonempty logits with vocab >=2")
    width = len(logits[0])
    if any(
        len(row) != width or any(not math.isfinite(value) for value in row)
        for row in logits
    ):
        raise ValueError("forced alignment logits must be rectangular and finite")


def ctc_viterbi_alignment(logits: list[list[float]], target: tuple[int, ...]) -> list[int]:
    _finite_rows(logits)
    if not target or any(
        type(token) is not int or token <= 0 or token >= len(logits[0])
        for token in target
    ):
        raise ValueError("forced alignment target contains invalid token id")

    labels: list[int] = [0]
    for token in target:
        labels.extend((token, 0))
    states = len(labels)
    neg = float("-inf")
    probs = log_softmax(logits)
    previous = [neg] * states
    previous[0] = probs[0][0]
    if states > 1:
        previous[1] = probs[0][labels[1]]
    parents: list[list[int]] = [[-1] * states]

    for frame in range(1, len(probs)):
        current = [neg] * states
        row_parent = [-1] * states
        for state, label in enumerate(labels):
            choices = [(previous[state], state)]
            if state > 0:
                choices.append((previous[state - 1], state - 1))
            if state > 1 and label != 0 and label != labels[state - 2]:
                choices.append((previous[state - 2], state - 2))
            best_score, best_parent = max(choices, key=lambda item: (item[0], -item[1]))
            if math.isfinite(best_score):
                current[state] = best_score + probs[frame][label]
                row_parent[state] = best_parent
        previous = current
        parents.append(row_parent)

    finals = [states - 1]
    if states > 1:
        finals.append(states - 2)
    final = max(finals, key=lambda state: (previous[state], -state))
    if not math.isfinite(previous[final]):
        raise ValueError("no finite CTC alignment for target")

    path = [final]
    for frame in range(len(probs) - 1, 0, -1):
        parent = parents[frame][path[-1]]
        if parent < 0:
            raise ValueError("incomplete CTC alignment backtrace")
        path.append(parent)
    path.reverse()
    for index in range(len(target)):
        if (2 * index + 1) not in path:
            raise ValueError("CTC alignment did not consume every target token")
    return path


def internal_split_from_alignment(
    logits: list[list[float]],
    target: tuple[int, ...],
    *,
    event_start_sample: int,
    event_end_sample: int,
    lead_samples: int,
) -> dict:
    if len(target) < 2 or len(target) % 2:
        raise ValueError("boundary continuity requires an even-length wake target")
    if (
        type(event_start_sample) is not int
        or type(event_end_sample) is not int
        or not 0 <= event_start_sample < event_end_sample
    ):
        raise ValueError("invalid source activity window")
    if type(lead_samples) is not int or lead_samples < 0 or lead_samples % FRAME_HOP_SAMPLES:
        raise ValueError("alignment lead must be nonnegative and hop aligned")

    start = event_start_sample + lead_samples
    end = event_end_sample + lead_samples
    eligible = [
        frame
        for frame in range(len(logits))
        if start <= frame * FRAME_HOP_SAMPLES + FRAME_LENGTH_SAMPLES <= end
    ]
    if not eligible:
        raise ValueError("wake event contains no posterior frames")

    path = ctc_viterbi_alignment([logits[frame] for frame in eligible], target)
    middle = len(target) // 2
    left_state = 2 * (middle - 1) + 1
    right_state = 2 * middle + 1
    left = [eligible[i] for i, state in enumerate(path) if state == left_state]
    right = [eligible[i] for i, state in enumerate(path) if state == right_state]
    if not left or not right or max(left) >= min(right):
        raise ValueError("CTC alignment has no ordered internal token boundary")

    left_end = max(left) * FRAME_HOP_SAMPLES + FRAME_LENGTH_SAMPLES
    right_start = min(right) * FRAME_HOP_SAMPLES
    absolute_split = (
        ((left_end + right_start) // 2 // FRAME_HOP_SAMPLES) * FRAME_HOP_SAMPLES
    )
    split = absolute_split - lead_samples
    if (
        split % FRAME_HOP_SAMPLES
        or not event_start_sample < split < event_end_sample
    ):
        raise ValueError("aligned internal split lies outside source activity window")
    return {
        "split_samples": split,
        "left_token_last_frame": max(left),
        "right_token_first_frame": min(right),
        "eligible_first_frame": eligible[0],
        "eligible_last_frame": eligible[-1],
        "alignment_policy": ALIGNMENT_POLICY,
    }


def within_word_expected_event(
    *,
    keyword_id: int,
    event_start_sample: int,
    event_end_sample: int,
    split_sample: int,
    gap_samples: int,
    lead_samples: int,
) -> dict:
    values = (
        keyword_id,
        event_start_sample,
        event_end_sample,
        split_sample,
        gap_samples,
        lead_samples,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("within-word event geometry must use integers")
    if (
        keyword_id <= 0
        or not 0 <= event_start_sample < split_sample < event_end_sample
        or gap_samples <= 0
        or lead_samples < 0
    ):
        raise ValueError("invalid within-word event geometry")
    start_s = (lead_samples + event_start_sample) / SAMPLE_RATE_HZ
    end_s = (lead_samples + event_end_sample + gap_samples) / SAMPLE_RATE_HZ
    # This boundary corpus proves continuity only when the detector completes
    # after the entire resumed right half has been heard. Merely firing at gap
    # exit can still be a delayed emission of a terminal path completed early.
    match_not_before_s = end_s
    return {
        "keyword_id": keyword_id,
        "start_s": start_s,
        "end_s": end_s,
        "match_not_before_s": match_not_before_s,
    }


def paused_samples(
    raw: list[int], split_samples: int, gap_samples: int, lead_samples: int, tail_samples: int
) -> list[int]:
    if (
        not raw
        or type(split_samples) is not int
        or not 0 < split_samples < len(raw)
        or split_samples % FRAME_HOP_SAMPLES
        or gap_samples <= 0
        or gap_samples % FRAME_HOP_SAMPLES
    ):
        raise ValueError("invalid within-word pause geometry")
    return (
        [0] * lead_samples
        + raw[:split_samples]
        + [0] * gap_samples
        + raw[split_samples:]
        + [0] * tail_samples
    )


def stitched_samples(
    left: list[int], right: list[int], gap_samples: int, lead_samples: int, tail_samples: int
) -> tuple[list[int], int]:
    if not left or not right or gap_samples <= 0 or gap_samples % FRAME_HOP_SAMPLES:
        raise ValueError("invalid cross-boundary geometry")
    alignment_pad = (-len(left)) % FRAME_HOP_SAMPLES
    silence = alignment_pad + gap_samples
    return (
        [0] * lead_samples + left + [0] * silence + right + [0] * tail_samples,
        silence,
    )


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return result or "voice"


def build(args: argparse.Namespace) -> dict:
    index = args.dataset_index.resolve()
    rows = load_jsonl(index)
    selected = select_boundary_sources(rows)

    model = args.model.resolve()
    posterior_dump = args.posterior_dump.resolve()
    cache = args.posterior_cache.resolve()
    for path, label in ((model, "model"), (posterior_dump, "posterior dump")):
        if not path.is_file():
            raise ValueError(f"{label} is missing: {path}")

    output = args.output_dir.resolve()
    audio_dir = output / "audio"
    alignment_dir = output / "alignment"
    output.mkdir(parents=True, exist_ok=True)

    gap_samples = round(args.gap_ms * SAMPLE_RATE_HZ / 1000.0)
    lead_samples = round(args.lead_ms * SAMPLE_RATE_HZ / 1000.0)
    tail_samples = round(args.tail_ms * SAMPLE_RATE_HZ / 1000.0)
    for value, label in (
        (gap_samples, "gap"),
        (lead_samples, "lead"),
        (tail_samples, "tail"),
    ):
        if value < 0 or value % FRAME_HOP_SAMPLES:
            raise ValueError(f"{label} duration must be nonnegative and hop aligned")
    if gap_samples <= 0:
        raise ValueError("gap duration must be > 0")

    model_sha = sha256_file(model)
    posterior_dump_sha = sha256_file(posterior_dump)
    positive_refs: list[dict] = []
    negative_refs: list[dict] = []
    evidence: list[dict] = []

    for item in selected:
        voice = str(item["voice_id"])
        keyword_id = int(item["keyword_id"])
        target = tuple(int(value) for value in item["target_ids"])
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
        trace, trace_summary, cache_hit = ensure_cached_trace(
            posterior_dump=posterior_dump,
            cache_root=cache,
            model=model,
            audio=alignment_audio,
            model_sha256=model_sha,
            audio_sha256=alignment_sha,
            posterior_dump_sha256=posterior_dump_sha,
        )
        alignment = internal_split_from_alignment(
            read_trace_logits(trace),
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
        positive_event = within_word_expected_event(
            keyword_id=keyword_id,
            event_start_sample=start,
            event_end_sample=end,
            split_sample=split,
            gap_samples=gap_samples,
            lead_samples=lead_samples,
        )
        positive_refs.append(
            {
                "recording": f"{stem}-within-word-pause",
                "path": repo_relative_path(positive_out),
                "duration_s": len(positive_samples) / SAMPLE_RATE_HZ,
                "boundary_role": "inserted-silence-ambiguity-positive-v1",
                "expected": [positive_event],
            }
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
        negative_refs.append(
            {
                "recording": f"{stem}-cross-boundary",
                "path": repo_relative_path(negative_out),
                "duration_s": len(negative_samples) / SAMPLE_RATE_HZ,
                "boundary_role": "stitched-half-ambiguity-negative-v1",
                "expected": [],
            }
        )

        evidence.append(
            {
                "voice_id": voice,
                "keyword_id": keyword_id,
                "target_ids": list(target),
                "positive_source_id": positive_row["speech_like_provenance"]["source_id"],
                "positive_wav_sha256": str(positive_row["wav_sha256"]),
                "left_source_id": left_row["speech_like_provenance"]["source_id"],
                "left_wav_sha256": str(left_row["wav_sha256"]),
                "right_source_id": right_row["speech_like_provenance"]["source_id"],
                "right_wav_sha256": str(right_row["wav_sha256"]),
                "right_source_reused": str(left_row["wav_sha256"]) == str(right_row["wav_sha256"]),
                "split_alignment": alignment,
                "requested_gap_samples": gap_samples,
                "cross_boundary_effective_silence_samples": effective_gap,
                "alignment_audio_sha256": alignment_sha,
                "posterior_trace_sha256": str(trace_summary["trace_sha256"]),
                "posterior_dump_sha256": trace_summary["posterior_dump_sha256"],
                "posterior_cache_hit": cache_hit,
                "within_word_wav_sha256": sha256_file(positive_out),
                "cross_boundary_wav_sha256": sha256_file(negative_out),
            }
        )

    positive_path = output / "within-word-pause.references.jsonl"
    negative_path = output / "cross-boundary-negative.references.jsonl"
    write_jsonl(positive_path, positive_refs)
    write_jsonl(negative_path, negative_refs)

    summary = {
        "schema_version": 1,
        "evidence_class": "decoder-boundary-reference-builder-v2",
        "policy": POLICY,
        "development_only": True,
        "selection_feedback_allowed": False,
        "release_authority": False,
        "acceptance_authority": False,
        "ambiguity_stress": True,
        "dataset_index": str(index),
        "dataset_index_sha256": sha256_file(index),
        "model_sha256": model_sha,
        "posterior_dump_sha256": posterior_dump_sha,
        "gap_ms": args.gap_ms,
        "lead_ms": args.lead_ms,
        "tail_ms": args.tail_ms,
        "within_word_references": positive_path.name,
        "within_word_references_sha256": sha256_file(positive_path),
        "cross_boundary_references": negative_path.name,
        "cross_boundary_references_sha256": sha256_file(negative_path),
        "recordings_per_role": len(evidence),
        "records": evidence,
    }
    summary_path = output / "boundary-reference-summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build corrected within-word/cross-boundary decoder references."
    )
    parser.add_argument("--dataset-index", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-dump", required=True, type=pathlib.Path)
    parser.add_argument("--posterior-cache", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--gap-ms", type=int, default=DEFAULT_GAP_MS)
    parser.add_argument("--lead-ms", type=int, default=DEFAULT_LEAD_MS)
    parser.add_argument("--tail-ms", type=int, default=DEFAULT_TAIL_MS)
    args = parser.parse_args()
    summary = build(args)
    print(
        json.dumps(
            {
                "recordings_per_role": summary["recordings_per_role"],
                "policy": summary["policy"],
                "development_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError, wave.Error) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
