#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens  # noqa: E402

SAMPLE_RATE_HZ = 16000
UINT32_MAX = 0xFFFFFFFF


def load_jsonl(path: pathlib.Path) -> list[dict]:
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def uint32_value(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < 0 or result > UINT32_MAX:
        raise ValueError(f"{label} must fit uint32")
    return result


def parse_keyword_targets(
    keywords: pathlib.Path, tokens: pathlib.Path
) -> dict[int, list[int]]:
    token_map = load_tokens(tokens)
    result: dict[int, list[int]] = {}
    for line_no, raw in enumerate(keywords.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        cols = raw.split("\t")
        if len(cols) != 4:
            raise ValueError(f"{keywords}:{line_no}: expected 4 TSV columns")
        keyword_id = uint32_value(int(cols[0]), f"{keywords}:{line_no}: keyword_id")
        token_names = cols[3].split()
        if keyword_id in result or not token_names:
            raise ValueError(f"{keywords}:{line_no}: duplicate/empty keyword")
        missing = [token for token in token_names if token not in token_map]
        if missing:
            raise ValueError(
                f"{keywords}:{line_no}: unknown tokens: {', '.join(missing)}"
            )
        target = [token_map[token] for token in token_names]
        if any(token_id == 0 for token_id in target):
            raise ValueError("blank token cannot be a replay target")
        result[keyword_id] = target
    if not result:
        raise ValueError("keyword TSV contains no targets")
    return result


def resolve_source(row: dict, audio_root: pathlib.Path) -> pathlib.Path:
    value = row.get("path") or row.get("recording")
    if not value:
        raise ValueError("false-reject row must contain path or recording")
    path = pathlib.Path(str(value))
    return path if path.is_absolute() else audio_root / path


def finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def extract_event(
    source: pathlib.Path,
    target: pathlib.Path,
    start_s: float,
    end_s: float,
    context_s: float,
) -> None:
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != SAMPLE_RATE_HZ
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{source}: expected mono 16-kHz PCM16 WAV")
        total = reader.getnframes()
        duration_s = total / SAMPLE_RATE_HZ
        if start_s > duration_s or end_s > duration_s:
            raise ValueError(
                f"{source}: false-reject event window exceeds source duration"
            )
        start = max(0, int(round((start_s - context_s) * SAMPLE_RATE_HZ)))
        end = min(total, int(round((end_s + context_s) * SAMPLE_RATE_HZ)))
        if end <= start:
            raise ValueError(f"{source}: empty false-reject replay clip")
        reader.setpos(start)
        frames = reader.readframes(end - start)
        params = reader.getparams()

    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--false-rejects", required=True, type=pathlib.Path)
    parser.add_argument("--keywords", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--context-s", type=float, default=0.20)
    args = parser.parse_args()
    context_s = finite(args.context_s, "context-s")
    if context_s < 0.0:
        parser.error("--context-s must be >= 0")

    targets = parse_keyword_targets(args.keywords, args.tokens)
    rows = load_jsonl(args.false_rejects)
    manifest_lines: list[str] = []
    for index, row in enumerate(rows):
        keyword_id = uint32_value(
            row.get("keyword_id"), f"false-reject[{index}].keyword_id"
        )
        if keyword_id not in targets:
            raise ValueError(f"false-reject keyword_id {keyword_id} is not configured")
        start_s = finite(row.get("start_s"), f"false-reject[{index}].start_s")
        end_s = finite(row.get("end_s"), f"false-reject[{index}].end_s")
        if start_s < 0.0 or end_s < start_s:
            raise ValueError(f"false-reject[{index}] has invalid event window")
        source = resolve_source(row, args.audio_root)
        clip_name = f"missed_{index:06d}_kw{keyword_id}.wav"
        target = args.output_dir / clip_name
        extract_event(source, target, start_s, end_s, context_s)
        token_text = " ".join(str(value) for value in targets[keyword_id])
        manifest_lines.append(f"{target.resolve()}\t{token_text}")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(manifest_lines) + ("\n" if manifest_lines else ""),
        encoding="utf-8",
    )
    print(f"mined {len(manifest_lines)} false-reject positive replay clip(s)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        wave.Error,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
