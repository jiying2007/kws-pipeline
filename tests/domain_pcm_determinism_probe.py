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

from render_domains import render_domain_dataset  # noqa: E402
from synthetic_audio import sha256_file  # noqa: E402


def canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def digest_lines(lines: list[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def cpu_model() -> str:
    path = pathlib.Path("/proc/cpuinfo")
    if path.is_file():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if raw.lower().startswith("model name") and ":" in raw:
                return raw.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


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
    summary = render_domain_dataset(config, work, curriculum_weights=None)

    index_path = work / "domain-index.jsonl"
    rows = [
        json.loads(raw)
        for raw in index_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    if not rows:
        raise SystemExit("domain renderer produced no rows")

    semantic: dict[str, dict] = {}
    source_pcm: dict[str, str] = {}
    rendered_pcm: dict[str, str] = {}
    ordered_keys: list[str] = []
    categories: dict[str, list[str]] = defaultdict(list)
    category_source: dict[str, list[str]] = defaultdict(list)
    category_rendered: dict[str, list[str]] = defaultdict(list)

    for row in rows:
        target = pathlib.Path(str(row["path"]))
        source = pathlib.Path(str(row["source_path"]))
        key = f"{row['split']}|{target.name}"
        if key in semantic:
            raise SystemExit(f"duplicate domain recording key: {key}")
        ordered_keys.append(key)

        target_sha = sha256_file(target)
        declared_target_sha = str(row["wav_sha256"])
        if target_sha != declared_target_sha:
            raise SystemExit(f"rendered WAV SHA mismatch for {key}")
        source_sha = sha256_file(source)
        declared_source_sha = str(row["source_wav_sha256"])
        if source_sha != declared_source_sha:
            raise SystemExit(f"source WAV SHA mismatch for {key}")

        normalized = {
            field: value
            for field, value in row.items()
            if field
            not in {"path", "source_path", "wav_sha256", "source_wav_sha256"}
        }
        semantic[key] = normalized
        source_pcm[key] = source_sha
        rendered_pcm[key] = target_sha
        category = f"{row['split']}|{row['kind']}"
        categories[category].append(f"{key}\0{canonical(normalized)}")
        category_source[category].append(f"{key}\0{source_sha}")
        category_rendered[category].append(f"{key}\0{target_sha}")

    keys = sorted(semantic)
    report = {
        "schema_version": 1,
        "probe": "production-round-zero-domain-pcm-determinism",
        "environment": {
            "cpu_model": cpu_model(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "libc": dict(zip(("name", "version"), platform.libc_ver())),
        },
        "config": {
            "path": config.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(config),
            "training_seed": 1337,
            "curriculum": None,
        },
        "examples": len(rows),
        "semantic_sha256": digest_lines(
            [f"{key}\0{canonical(semantic[key])}" for key in keys]
        ),
        "index_order_sha256": digest_lines(ordered_keys),
        "source_pcm_tree_sha256": digest_lines(
            [f"{key}\0{source_pcm[key]}" for key in keys]
        ),
        "rendered_pcm_tree_sha256": digest_lines(
            [f"{key}\0{rendered_pcm[key]}" for key in keys]
        ),
        "category_digests": {
            category: {
                "examples": len(categories[category]),
                "semantic_sha256": digest_lines(sorted(categories[category])),
                "source_pcm_tree_sha256": digest_lines(sorted(category_source[category])),
                "rendered_pcm_tree_sha256": digest_lines(sorted(category_rendered[category])),
            }
            for category in sorted(categories)
        },
        "summary": {
            "evidence_class": summary["evidence_class"],
            "evaluation_sampling": summary["evaluation_sampling"],
            "distance_histogram": summary["distance_histogram"],
            "distance_histogram_by_split": summary["distance_histogram_by_split"],
            "curriculum": summary["curriculum"],
            "afe": summary["afe"],
        },
        "recordings": {
            key: {
                "semantic": semantic[key],
                "source_wav_sha256": source_pcm[key],
                "rendered_wav_sha256": rendered_pcm[key],
            }
            for key in keys
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
                "examples": report["examples"],
                "semantic_sha256": report["semantic_sha256"],
                "index_order_sha256": report["index_order_sha256"],
                "source_pcm_tree_sha256": report["source_pcm_tree_sha256"],
                "rendered_pcm_tree_sha256": report["rendered_pcm_tree_sha256"],
                "category_digests": report["category_digests"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
