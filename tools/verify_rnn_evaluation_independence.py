#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib

SHA256_HEX = set("0123456789abcdef")
FRESH_SPLITS = ("calibration", "test", "qualification")


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"expected JSON object: {path}")
    return value


def write_object(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def split_hashes(path: pathlib.Path, split: str) -> set[str]:
    if not path.is_file(): raise ValueError(f"domain index missing: {path}")
    result: set[str] = set(); count = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip(): continue
        row = json.loads(raw)
        if not isinstance(row, dict): raise ValueError(f"{path}:{line_no}: expected object")
        if str(row.get("split")) != split: continue
        count += 1; digest = str(row.get("wav_sha256") or "")
        if len(digest) != 64 or any(ch not in SHA256_HEX for ch in digest): raise ValueError(f"{path}: invalid {split} WAV SHA")
        if digest in result: raise ValueError(f"{path}: duplicate {split} WAV SHA")
        result.add(digest)
    if count == 0: raise ValueError(f"{path}: split has no WAV identity evidence: {split}")
    return result


def verify_fresh(index: pathlib.Path) -> dict:
    by_split = {split: split_hashes(index, split) for split in FRESH_SPLITS}; seen: set[str] = set()
    for split in FRESH_SPLITS:
        if seen & by_split[split]: raise ValueError(f"RNN fresh split {split} overlaps earlier split")
        seen.update(by_split[split])
    return {"schema_version": 1, "policy": "frozen-rnn-fresh-internal-sha-independence-v1", "model_family": "rnn", "qualified": True, "all_fresh_splits_sha_disjoint": True, "splits": {s: len(v) for s, v in by_split.items()}, "wav_sha256_count": len(seen)}


def verify_shadow(root: pathlib.Path, summary_path: pathlib.Path) -> dict:
    summary = load_object(summary_path)
    if summary.get("evidence_class") != "development-only-shadow-qualification" or summary.get("qualified") is not True:
        raise ValueError("RNN shadow summary evidence mismatch")
    seeds = [int(value) for value in summary.get("seeds", [])]
    if not 8 <= len(seeds) <= 16 or len(set(seeds)) != len(seeds): raise ValueError("RNN shadow seeds invalid")
    seen: set[str] = set(); per_seed: dict[str, int] = {}
    for seed in seeds:
        values = split_hashes(root / f"seed-{seed}" / "dataset" / "domain-index.jsonl", "qualification")
        if seen & values: raise ValueError(f"RNN shadow seed {seed} overlaps earlier shadow seed")
        seen.update(values); per_seed[str(seed)] = len(values)
    return {"schema_version": 1, "policy": "frozen-rnn-shadow-seed-sha-independence-v1", "model_family": "rnn", "qualified": True, "all_shadow_seeds_sha_disjoint": True, "seeds": seeds, "per_seed_wav_sha256_count": per_seed, "wav_sha256_count": len(seen), "formal_qualification_used": False}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", required=True, choices=("fresh", "shadow")); parser.add_argument("--fresh-index", type=pathlib.Path); parser.add_argument("--shadow-root", type=pathlib.Path); parser.add_argument("--shadow-summary", type=pathlib.Path); parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    if args.mode == "fresh":
        if args.fresh_index is None: raise ValueError("fresh mode requires --fresh-index")
        result = verify_fresh(args.fresh_index.resolve())
    else:
        if args.shadow_root is None or args.shadow_summary is None: raise ValueError("shadow mode requires root and summary")
        result = verify_shadow(args.shadow_root.resolve(), args.shadow_summary.resolve())
    write_object(args.output.resolve(), result); print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}"); raise SystemExit(2)
