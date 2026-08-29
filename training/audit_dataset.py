#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import wave
from collections import defaultdict

SAMPLE_RATE_HZ = 16000


def split_assignment(text: str, label: str) -> tuple[str, pathlib.Path]:
    if "=" not in text:
        raise ValueError(f"{label} must use NAME=PATH")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name or not raw_path.strip():
        raise ValueError(f"{label} must use non-empty NAME=PATH")
    return name, pathlib.Path(raw_path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: pathlib.Path) -> tuple[int, float, str]:
    try:
        with wave.open(str(path), "rb") as wf:
            if (
                wf.getnchannels() != 1
                or wf.getframerate() != SAMPLE_RATE_HZ
                or wf.getsampwidth() != 2
                or wf.getcomptype() != "NONE"
            ):
                raise ValueError(f"{path}: expected mono 16-kHz PCM16 WAV")
            frames = wf.getnframes()
            pcm = wf.readframes(frames)
            if len(pcm) != frames * 2:
                raise ValueError(f"{path}: truncated PCM payload")
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"{path}: invalid WAV: {exc}") from exc
    return frames, frames / SAMPLE_RATE_HZ, hashlib.sha256(pcm).hexdigest()


def parse_tsv(path: pathlib.Path) -> list[str]:
    rows: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" not in raw:
            raise ValueError(f"{path}:{line_no}: expected WAV<TAB>...")
        wav_path = raw.split("\t", 1)[0].strip()
        if not wav_path:
            raise ValueError(f"{path}:{line_no}: empty WAV path")
        rows.append(wav_path)
    return rows


def parse_jsonl(path: pathlib.Path) -> list[str]:
    rows: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError(f"{path}:{line_no}: expected object with string path")
        wav_path = row["path"].strip()
        if not wav_path:
            raise ValueError(f"{path}:{line_no}: empty path")
        rows.append(wav_path)
    return rows


def manifest_paths(path: pathlib.Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        return parse_jsonl(path)
    return parse_tsv(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect decoded-PCM leakage across training/calibration/evaluation splits. "
            "WAV container SHA256 is retained as provenance, but split identity uses "
            "the canonical PCM16 payload so metadata/repackaging cannot hide leakage."
        )
    )
    parser.add_argument(
        "--split",
        required=True,
        action="append",
        help="split manifest as NAME=PATH; TSV uses first column, JSONL uses path field",
    )
    parser.add_argument(
        "--audio-root",
        action="append",
        default=[],
        help="optional split-specific audio root as NAME=DIR",
    )
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--fail-within-split", action="store_true")
    args = parser.parse_args()

    split_specs: list[tuple[str, pathlib.Path]] = []
    seen_names: set[str] = set()
    for text in args.split:
        name, manifest = split_assignment(text, "--split")
        if name in seen_names:
            raise ValueError(f"duplicate split name: {name}")
        seen_names.add(name)
        split_specs.append((name, manifest))

    roots: dict[str, pathlib.Path] = {}
    for text in args.audio_root:
        name, root = split_assignment(text, "--audio-root")
        if name in roots:
            raise ValueError(f"duplicate audio root for split: {name}")
        roots[name] = root
    unknown_roots = sorted(set(roots) - seen_names)
    if unknown_roots:
        raise ValueError(f"audio roots reference unknown split(s): {', '.join(unknown_roots)}")

    by_pcm_hash: dict[str, list[dict]] = defaultdict(list)
    split_summaries: dict[str, dict] = {}
    within_duplicates: list[dict] = []

    for name, manifest in split_specs:
        if not manifest.is_file():
            raise ValueError(f"{manifest}: split manifest does not exist")
        root = roots.get(name, manifest.parent)
        rows = manifest_paths(manifest)
        if not rows:
            raise ValueError(f"{manifest}: split contains no audio rows")
        local_pcm_hashes: dict[str, list[str]] = defaultdict(list)
        total_frames = 0
        resolved_rows: list[dict] = []

        for raw_path in rows:
            wav_path = pathlib.Path(raw_path)
            if not wav_path.is_absolute():
                wav_path = root / wav_path
            try:
                resolved = wav_path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(f"{wav_path}: audio file does not exist") from exc
            frames, duration_s, pcm_sha256 = inspect_wav(resolved)
            file_sha256 = sha256_file(resolved)
            total_frames += frames
            entry = {
                "split": name,
                "path": str(resolved),
                "pcm_sha256": pcm_sha256,
                "file_sha256": file_sha256,
                "frames": frames,
                "duration_s": duration_s,
            }
            resolved_rows.append(entry)
            local_pcm_hashes[pcm_sha256].append(str(resolved))
            by_pcm_hash[pcm_sha256].append(entry)

        for digest, paths in sorted(local_pcm_hashes.items()):
            if len(paths) > 1:
                within_duplicates.append(
                    {"split": name, "pcm_sha256": digest, "paths": sorted(paths)}
                )
        split_summaries[name] = {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "examples": len(resolved_rows),
            "unique_pcm": len(local_pcm_hashes),
            "audio_hours": total_frames / SAMPLE_RATE_HZ / 3600.0,
        }

    cross_split_leaks: list[dict] = []
    for digest, entries in sorted(by_pcm_hash.items()):
        splits = sorted({entry["split"] for entry in entries})
        if len(splits) > 1:
            cross_split_leaks.append(
                {
                    "pcm_sha256": digest,
                    "splits": splits,
                    "paths": sorted({entry["path"] for entry in entries}),
                    "file_sha256": sorted({entry["file_sha256"] for entry in entries}),
                }
            )

    report = {
        "schema_version": 2,
        "identity": "decoded-mono-16khz-pcm16-sha256",
        "splits": split_summaries,
        "cross_split_leaks": cross_split_leaks,
        "within_split_duplicates": within_duplicates,
        "clean": not cross_split_leaks
        and (not args.fail_within_split or not within_duplicates),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")

    if cross_split_leaks:
        print(
            f"dataset audit failed: {len(cross_split_leaks)} cross-split duplicate PCM hash(es)",
            file=sys.stderr,
        )
        return 1
    if args.fail_within_split and within_duplicates:
        print(
            f"dataset audit failed: {len(within_duplicates)} within-split duplicate PCM hash(es)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
