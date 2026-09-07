#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import platform
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
sys.path.insert(0, str(TRAINING))

from synthetic_audio import generate_dataset, sha256_file  # noqa: E402


def digest_text(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def cpu_model() -> str:
    path = pathlib.Path("/proc/cpuinfo")
    if path.is_file():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.lower().startswith("model name") and ":" in raw:
                return raw.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def recording_key(row: dict) -> str:
    return (
        f"{row['split']}|{row['kind']}|{row['family_id']}|"
        f"variant={int(row.get('variant', 0))}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOT / "configs" / "training" / "xiaowo.torch-domain.json",
    )
    parser.add_argument("--work-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    config = args.config.resolve(strict=True)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    summary = generate_dataset(config, work)

    index_path = work / "dataset-index.jsonl"
    rows = [
        json.loads(raw)
        for raw in index_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    if not rows:
        raise SystemExit("synthetic generator produced no index rows")

    semantic_by_key: dict[str, dict] = {}
    pcm_by_key: dict[str, str] = {}
    ordered_keys: list[str] = []
    category_rows: dict[str, list[str]] = defaultdict(list)
    category_semantic_rows: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit("dataset index row is not an object")
        key = recording_key(row)
        if key in semantic_by_key:
            raise SystemExit(f"duplicate semantic recording key: {key}")
        ordered_keys.append(key)

        semantic = {
            field: value
            for field, value in row.items()
            if field not in {"path", "wav_sha256"}
        }
        semantic_by_key[key] = semantic

        path = pathlib.Path(str(row["path"]))
        actual_sha = sha256_file(path)
        declared_sha = str(row["wav_sha256"])
        if actual_sha != declared_sha:
            raise SystemExit(
                f"index/file SHA mismatch for {key}: {declared_sha} != {actual_sha}"
            )
        pcm_by_key[key] = actual_sha
        category = f"{row['split']}|{row['kind']}"
        category_rows[category].append(f"{key}\0{actual_sha}")
        category_semantic_rows[category].append(f"{key}\0{canonical(semantic)}")

    sorted_keys = sorted(semantic_by_key)
    semantic_digest = digest_text(
        [f"{key}\0{canonical(semantic_by_key[key])}" for key in sorted_keys]
    )
    order_digest = digest_text(ordered_keys)
    pcm_tree_digest = digest_text(
        [f"{key}\0{pcm_by_key[key]}" for key in sorted_keys]
    )

    category_digests = {
        category: {
            "examples": len(category_rows[category]),
            "semantic_sha256": digest_text(sorted(category_semantic_rows[category])),
            "pcm_tree_sha256": digest_text(sorted(category_rows[category])),
        }
        for category in sorted(category_rows)
    }

    libc_name, libc_version = platform.libc_ver()
    report = {
        "schema_version": 1,
        "probe": "production-synthetic-base-pcm-determinism",
        "environment": {
            "cpu_model": cpu_model(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "libc": {"name": libc_name, "version": libc_version},
        },
        "config": {
            "path": config.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config),
            "training_seed": int(summary["seed"]),
        },
        "examples": len(rows),
        "semantic_sha256": semantic_digest,
        "index_order_sha256": order_digest,
        "pcm_tree_sha256": pcm_tree_digest,
        "mapping_sha256": str(summary["mapping_sha256"]),
        "category_digests": category_digests,
        "recordings": {
            key: {
                "semantic": semantic_by_key[key],
                "wav_sha256": pcm_by_key[key],
            }
            for key in sorted_keys
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cpu_model": report["environment"]["cpu_model"],
                "training_seed": report["config"]["training_seed"],
                "examples": report["examples"],
                "semantic_sha256": semantic_digest,
                "index_order_sha256": order_digest,
                "pcm_tree_sha256": pcm_tree_digest,
                "mapping_sha256": report["mapping_sha256"],
                "category_digests": category_digests,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
