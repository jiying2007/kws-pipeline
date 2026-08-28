#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import wave


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


def resolve_source(row: dict, audio_root: pathlib.Path) -> pathlib.Path:
    value = row.get("path") or row.get("recording")
    if not value:
        raise ValueError("false-positive row must contain path or recording")
    path = pathlib.Path(str(value))
    return path if path.is_absolute() else audio_root / path


def extract_clip(
    source: pathlib.Path,
    target: pathlib.Path,
    center_s: float,
    pre_s: float,
    post_s: float,
) -> None:
    with wave.open(str(source), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getframerate() != 16000
            or reader.getsampwidth() != 2
            or reader.getcomptype() != "NONE"
        ):
            raise ValueError(f"{source}: expected mono 16-kHz PCM16 WAV")
        total = reader.getnframes()
        start = max(0, int(round((center_s - pre_s) * 16000.0)))
        end = min(total, int(round((center_s + post_s) * 16000.0)))
        if end <= start:
            raise ValueError(f"{source}: empty hard-negative clip around {center_s}s")
        reader.setpos(start)
        frames = reader.readframes(end - start)
        params = reader.getparams()

    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--false-positives", required=True, type=pathlib.Path)
    parser.add_argument("--audio-root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--manifest", required=True, type=pathlib.Path)
    parser.add_argument("--pre-s", type=float, default=1.5)
    parser.add_argument("--post-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.pre_s < 0.0 or args.post_s <= 0.0:
        parser.error("--pre-s must be >= 0 and --post-s must be > 0")

    rows = load_jsonl(args.false_positives)
    manifest_lines: list[str] = []
    for index, row in enumerate(rows):
        source = resolve_source(row, args.audio_root)
        center_s = float(row["time_s"])
        keyword_id = int(row.get("keyword_id", 0))
        clip_name = f"hardneg_{index:06d}_kw{keyword_id}.wav"
        target = args.output_dir / clip_name
        extract_clip(source, target, center_s, args.pre_s, args.post_s)
        manifest_lines.append(f"{target.resolve()}\t")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    print(f"mined {len(manifest_lines)} hard-negative clip(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
