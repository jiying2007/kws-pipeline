#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import AUDIO_PATH_FIELDS, resolve_audio_path  # noqa: E402

if __package__:
    from .score_events import load_jsonl, uint32_value, validate_recordings
else:
    from score_events import load_jsonl, uint32_value, validate_recordings

SAMPLE_RATE_HZ = 16000


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_references(path: pathlib.Path, audio_root: pathlib.Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    # Keep the scorer's event schema, but never coerce malformed labels or
    # accept point annotations as proof that surrounding PCM is negative.
    for row in rows:
        name = row.get("recording")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("reference recording must be non-empty text")
        finite(row.get("duration_s"), f"{name}: duration_s")
        if row.get("annotation_status", "complete") != "complete":
            raise ValueError(f"{name}: reference annotations must be complete")
        if "expected_keywords" in row:
            raise ValueError(f"{name}: event annotations required, not clip-presence labels")
        if not isinstance(row.get("expected"), list):
            raise ValueError(f"{name}: expected annotation list is required (use [] for negatives)")
        for event in row["expected"]:
            if not isinstance(event, dict):
                raise ValueError(f"{name}: expected event must be an object")
            start = finite(event.get("start_s"), f"{name}: start_s")
            end = finite(event.get("end_s"), f"{name}: end_s")
            if end <= start:
                raise ValueError(f"{name}: expected event must have positive duration")
    references = validate_recordings(rows)
    sources: dict[tuple[int, int], str] = {}
    for row in rows:
        name = row["recording"]
        source = resolve_audio_path(row, audio_root, f"reference {name}")
        stat = source.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in sources:
            raise ValueError(f"ambiguous reference source: {name} and {sources[identity]}")
        sources[identity] = name
        references[name]["source"] = source
    return references


def wav_frames(reader: wave.Wave_read, source: pathlib.Path) -> int:
    if (
        reader.getnchannels() != 1
        or reader.getframerate() != SAMPLE_RATE_HZ
        or reader.getsampwidth() != 2
        or reader.getcomptype() != "NONE"
    ):
        raise ValueError(f"{source}: expected mono 16-kHz PCM16 WAV")
    return reader.getnframes()


def clip_bounds(total: int, center_s: float, pre_s: float, post_s: float) -> tuple[int, int]:
    start = max(0, int(round((center_s - pre_s) * SAMPLE_RATE_HZ)))
    end = min(total, int(round((center_s + post_s) * SAMPLE_RATE_HZ)))
    if end <= start:
        raise ValueError(f"empty hard-negative clip around {center_s}s")
    return start, end


def overlaps_reference(start: int, end: int, events: list[dict]) -> bool:
    # Check the frames that will actually be exported, against every keyword.
    # Round labels outward so a sub-sample positive cannot disappear; adjacent
    # half-open intervals at exact sample boundaries remain non-overlapping.
    return any(
        start < math.ceil(event["end_s"] * SAMPLE_RATE_HZ)
        and math.floor(event["start_s"] * SAMPLE_RATE_HZ) < end
        for event in events
    )


def validate_outputs(outputs: list[pathlib.Path], inputs: list[pathlib.Path]) -> None:
    # Stable-filesystem preflight, not a concurrent filesystem snapshot guarantee.
    # Protect the original reference/FP evidence and every referenced source WAV.
    def resolve(path: pathlib.Path) -> pathlib.Path:
        try:
            return path.resolve()
        except RuntimeError as exc:
            raise ValueError(f"cannot resolve output/input path {path}: {exc}") from exc

    paths = {resolve(path) for path in inputs}
    identities = {(stat.st_dev, stat.st_ino) for path in inputs for stat in [path.stat()]}
    for output in outputs:
        resolved = resolve(output)
        identity = None
        try:
            stat = output.stat()
            identity = (stat.st_dev, stat.st_ino)
        except FileNotFoundError:
            pass
        if resolved in paths or identity in identities:
            raise ValueError(f"output aliases an input or another output: {output}")
        paths.add(resolved)
        if identity is not None:
            identities.add(identity)


def extract_clip(source: pathlib.Path, target: pathlib.Path, start: int, end: int) -> None:
    with wave.open(str(source), "rb") as reader:
        if end > wav_frames(reader, source):
            raise ValueError(f"{source}: source shortened after clip validation")
        reader.setpos(start)
        frames = reader.readframes(end - start)
        if len(frames) != (end - start) * 2:
            raise ValueError(f"{source}: truncated hard-negative clip")
        params = reader.getparams()
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--false-positives", required=True, type=pathlib.Path)
    parser.add_argument("--references", required=True, type=pathlib.Path,
                        help="original complete event annotations used for scoring, not predictions")
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--pre-s", type=float, default=1.5)
    parser.add_argument("--post-s", type=float, default=1.0)
    args = parser.parse_args()
    pre_s = finite(args.pre_s, "--pre-s")
    post_s = finite(args.post_s, "--post-s")
    if pre_s < 0.0 or post_s <= 0.0:
        parser.error("--pre-s must be >= 0 and --post-s must be > 0")

    references = load_references(args.references, args.audio_root)
    rows = load_jsonl(args.false_positives)
    planned: list[tuple[pathlib.Path, pathlib.Path, int, int]] = []
    skipped = 0
    # Validate all candidate metadata before creating any output. FP membership identifies a
    # candidate only; the original reference is the sole label authority.
    for index, row in enumerate(rows):
        name = row.get("recording")
        if not isinstance(name, str) or name not in references:
            raise ValueError(f"false-positive[{index}]: unknown reference recording")
        reference = references[name]
        source = reference["source"]
        if any(field in row for field in AUDIO_PATH_FIELDS):
            fp_source = resolve_audio_path(row, args.audio_root, f"false-positive[{index}]")
            if not source.samefile(fp_source):
                raise ValueError(f"{name}: false-positive path conflicts with reference source")
        if "duration_s" in row and finite(row["duration_s"], f"{name}: duration_s") != reference["duration_s"]:
            raise ValueError(f"{name}: false-positive duration_s conflicts with reference")
        center_s = finite(row.get("time_s"), f"false-positive[{index}].time_s")
        keyword_id = uint32_value(row.get("keyword_id", 0), f"false-positive[{index}].keyword_id")
        with wave.open(str(source), "rb") as reader:
            total = wav_frames(reader, source)
        duration_s = total / SAMPLE_RATE_HZ
        if not math.isclose(reference["duration_s"], duration_s,
                            rel_tol=0.0, abs_tol=1.0 / SAMPLE_RATE_HZ):
            raise ValueError(f"{name}: reference duration_s does not match WAV duration")
        if not 0.0 <= center_s <= duration_s:
            raise ValueError(f"{name}: false-positive time_s is outside source duration")
        start, end = clip_bounds(total, center_s, pre_s, post_s)
        if overlaps_reference(start, end, reference["expected"]):
            skipped += 1
            continue
        target = args.output_dir / f"hardneg_{index:06d}_kw{keyword_id}.wav"
        planned.append((source, target, start, end))

    validate_outputs(
        [args.manifest, *(target for _, target, _, _ in planned)],
        [args.references, args.false_positives, *(ref["source"] for ref in references.values())],
    )
    manifest_lines: list[str] = []
    for source, target, start, end in planned:
        extract_clip(source, target, start, end)
        manifest_lines.append(f"{target.resolve()}\t")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    print(f"mined {len(manifest_lines)} hard-negative clip(s); skipped {skipped} reference-overlapping candidate(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, OverflowError, TypeError, ValueError, wave.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
