#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_model_release import verify as verify_pinned_release  # noqa: E402

MAX_ENTRY_BYTES = 5_000_000
FORBIDDEN_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".zip", ".tar", ".gz", ".tgz"}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-dir", required=True, type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--max-entry-bytes", type=int, default=MAX_ENTRY_BYTES)
    args = parser.parse_args()

    root = args.registry_dir.resolve()
    registry = load_json(root / "registry.json")
    if registry.get("policy") != "model-registry-entry-v1":
        raise ValueError("registry policy mismatch")
    tag = str(registry.get("release_tag") or "")
    if root.name != tag:
        raise ValueError("registry directory must equal release tag")
    release = registry.get("source_release")
    training = registry.get("training")
    storage = registry.get("storage")
    if not isinstance(release, dict) or release.get("immutable") is not True:
        raise ValueError("registry does not bind an immutable release")
    if not isinstance(training, dict) or not isinstance(storage, dict):
        raise ValueError("registry training/storage metadata missing")
    if str(release.get("target_commitish") or "") != str(training.get("head_sha") or ""):
        raise ValueError("release target/training head mismatch")
    if storage.get("mode") != "git-direct-no-lfs-v1":
        raise ValueError("unexpected registry storage mode")

    asset_rows = registry.get("assets")
    if not isinstance(asset_rows, list) or not asset_rows:
        raise ValueError("registry asset list missing")
    expected: dict[str, str] = {}
    total = 0
    for row in asset_rows:
        if not isinstance(row, dict):
            raise ValueError("registry asset row must be object")
        name = str(row.get("name") or "")
        if not name or "/" in name or "\\" in name or name in expected:
            raise ValueError(f"invalid/duplicate registry asset name: {name!r}")
        path = root / name
        if not path.is_file():
            raise ValueError(f"registry asset missing: {name}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"raw/archive payload forbidden in model registry: {name}")
        digest = sha256(path)
        if digest != str(row.get("sha256") or ""):
            raise ValueError(f"registry SHA mismatch: {name}")
        size = path.stat().st_size
        if size != int(row.get("size_bytes", -1)):
            raise ValueError(f"registry size mismatch: {name}")
        expected[name] = digest
        total += size

    actual = {p.name for p in root.iterdir() if p.is_file() and p.name != "registry.json"}
    if actual != set(expected):
        raise ValueError(
            f"registry file-set mismatch: actual={sorted(actual)} expected={sorted(expected)}"
        )
    product_lineage = {
        "xiaowo-effective-training-config.json",
        "xiaowo-product-speech-like-base-contract.json",
        "training-invocation.json",
    }
    present_lineage = product_lineage & actual
    if present_lineage and present_lineage != product_lineage:
        missing_lineage = sorted(product_lineage - present_lineage)
        raise ValueError(
            f"speech-like product registry lineage is incomplete: missing={missing_lineage}"
        )
    if total != int(storage.get("release_assets_bytes", -1)):
        raise ValueError("registry total byte count drift")
    if total > args.max_entry_bytes or total > int(storage.get("max_entry_bytes", 0)):
        raise ValueError("registry entry exceeds direct-Git size policy")

    sums: dict[str, str] = {}
    sums_path = root / "MODEL_SHA256SUMS"
    for number, raw in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"MODEL_SHA256SUMS:{number}: malformed")
        sums[parts[1].strip().lstrip("*").strip()] = parts[0].lower()
    for name, digest in sums.items():
        if name not in expected or expected[name] != digest:
            raise ValueError(f"MODEL_SHA256SUMS mismatch: {name}")

    if args.config:
        report = verify_pinned_release(root, args.config, tag)
        config = load_json(args.config)
        model = config["model"]
        if int(training.get("run_id", 0)) != int(model.get("training_run_id", 0)):
            raise ValueError("registry/config training run mismatch")
        if str(training.get("head_sha") or "") != str(model.get("trained_head_sha") or ""):
            raise ValueError("registry/config trained head mismatch")
        if bool(registry.get("shipping_approved")) != bool(config.get("shipping_approved")):
            raise ValueError("registry/config shipping_approved mismatch")
        pending = list(config.get("shipping_evidence_boundary", {}).get("pending") or [])
        if list(registry.get("shipping_blockers") or []) != pending:
            raise ValueError("registry/config shipping blockers mismatch")
    else:
        report = {"manifest_entries": len(sums)}

    print(json.dumps({
        "schema_version": 1,
        "policy": "model-registry-verification-v1",
        "release_tag": tag,
        "assets": len(expected),
        "bytes": total,
        "shipping_approved": bool(registry.get("shipping_approved")),
        "manifest_entries": report["manifest_entries"],
        "verified": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
