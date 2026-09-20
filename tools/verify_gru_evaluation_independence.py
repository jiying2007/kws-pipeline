#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib

SHA256_HEX = set("0123456789abcdef")
FRESH_SPLITS = ("calibration", "test", "qualification")


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in SHA256_HEX for ch in value)


def index_rows(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"domain index is missing: {path}")
    rows: list[dict] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_no}: expected JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"domain index is empty: {path}")
    return rows


def split_hashes(path: pathlib.Path, split: str) -> set[str]:
    result: set[str] = set()
    count = 0
    for row in index_rows(path):
        if str(row.get("split")) != split:
            continue
        count += 1
        digest = str(row.get("wav_sha256") or "")
        if not valid_sha256(digest):
            raise ValueError(f"{path}: {split} row is missing a valid WAV SHA256")
        if digest in result:
            raise ValueError(f"{path}: {split} contains duplicate WAV SHA256 evidence")
        result.add(digest)
    if count == 0:
        raise ValueError(f"{path}: split has no WAV identity evidence: {split}")
    return result


def verify_fresh(index: pathlib.Path) -> dict:
    by_split = {split: split_hashes(index, split) for split in FRESH_SPLITS}
    seen: set[str] = set()
    for split in FRESH_SPLITS:
        overlap = seen & by_split[split]
        if overlap:
            raise ValueError(
                f"fresh validation split {split} overlaps {len(overlap)} WAV SHA256 identity(ies) with an earlier fresh split"
            )
        seen.update(by_split[split])
    return {
        "schema_version": 1,
        "policy": "frozen-gru-fresh-internal-sha-independence-v1",
        "qualified": True,
        "all_fresh_splits_sha_disjoint": True,
        "splits": {split: len(by_split[split]) for split in FRESH_SPLITS},
        "wav_sha256_count": len(seen),
    }


def verify_shadow(root: pathlib.Path, summary_path: pathlib.Path) -> dict:
    summary = load_object(summary_path)
    if summary.get("evidence_class") != "development-only-shadow-qualification":
        raise ValueError("shadow summary evidence class mismatch")
    # The receipt below asserts qualified: True, so it must not describe a
    # shadow run that did not qualify. Boolean True only: 1 is not evidence.
    if summary.get("qualified") is not True:
        raise ValueError("shadow summary is not qualified")
    seeds_raw = summary.get("seeds")
    if not isinstance(seeds_raw, list) or not 8 <= len(seeds_raw) <= 16:
        raise ValueError("shadow summary must contain 8..16 seeds")
    seeds = [int(value) for value in seeds_raw]
    if len(set(seeds)) != len(seeds):
        raise ValueError("shadow summary contains duplicate seeds")

    seen: set[str] = set()
    per_seed: dict[str, int] = {}
    for seed in seeds:
        index = root / f"seed-{seed}" / "dataset" / "domain-index.jsonl"
        values = split_hashes(index, "qualification")
        overlap = seen & values
        if overlap:
            raise ValueError(
                f"shadow seed {seed} overlaps {len(overlap)} WAV SHA256 identity(ies) with an earlier shadow seed"
            )
        seen.update(values)
        per_seed[str(seed)] = len(values)

    return {
        "schema_version": 1,
        "policy": "frozen-gru-shadow-seed-sha-independence-v1",
        "qualified": True,
        "all_shadow_seeds_sha_disjoint": True,
        "seeds": seeds,
        "per_seed_wav_sha256_count": per_seed,
        "wav_sha256_count": len(seen),
        "formal_qualification_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("fresh", "shadow"))
    parser.add_argument("--fresh-index", type=pathlib.Path)
    parser.add_argument("--shadow-root", type=pathlib.Path)
    parser.add_argument("--shadow-summary", type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if args.mode == "fresh":
        if args.fresh_index is None or args.shadow_root is not None or args.shadow_summary is not None:
            raise ValueError("fresh mode requires only --fresh-index")
        result = verify_fresh(args.fresh_index.resolve())
    else:
        if args.shadow_root is None or args.shadow_summary is None or args.fresh_index is not None:
            raise ValueError("shadow mode requires only --shadow-root and --shadow-summary")
        result = verify_shadow(args.shadow_root.resolve(), args.shadow_summary.resolve())

    write_object(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
