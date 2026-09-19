#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import shutil

MAX_ENTRY_BYTES = 5_000_000
TAG_RE = re.compile(r"^model-[A-Za-z0-9._-]+$")
REQUIRED_ASSETS = {
    "continuous-far-stream-contract.json",
    "continuous-far-summary.json",
    "model-promotion-manifest.json",
    "MODEL_SHA256SUMS",
    "qualification-cohort.json",
    "qualification-summary.json",
    "robustness-summary.json",
    "training-run-summary.json",
    "xiaowo-keywords.kwk",
    "xiaowo-keywords.tsv",
    "xiaowo-model-provenance.json",
    "xiaowo-model.kwm",
    "xiaowo-model.pt",
}


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


def parse_sums(path: pathlib.Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{number}: malformed checksum line")
        digest = parts[0].lower()
        name = parts[1].strip().lstrip("*").strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{path}:{number}: invalid SHA256")
        if not name or name in rows:
            raise ValueError(f"{path}:{number}: invalid/duplicate asset name")
        rows[name] = digest
    if not rows:
        raise ValueError("MODEL_SHA256SUMS is empty")
    return rows


def verify_assets(root: pathlib.Path) -> tuple[list[dict], int]:
    missing = sorted(name for name in REQUIRED_ASSETS if not (root / name).is_file())
    if missing:
        raise ValueError(f"release assets missing: {missing}")
    sums = parse_sums(root / "MODEL_SHA256SUMS")
    for name, digest in sums.items():
        path = root / name
        if not path.is_file():
            raise ValueError(f"checksum manifest references missing asset: {name}")
        if sha256(path) != digest:
            raise ValueError(f"release asset checksum mismatch: {name}")
    actual = {path.name for path in root.iterdir() if path.is_file()}
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
    unexpected = sorted(actual - (set(sums) | {"MODEL_SHA256SUMS"}))
    if unexpected:
        raise ValueError(f"release directory has assets outside checksum manifest: {unexpected}")
    rows: list[dict] = []
    total = 0
    for path in sorted((p for p in root.iterdir() if p.is_file()), key=lambda p: p.name):
        size = path.stat().st_size
        total += size
        rows.append({"name": path.name, "sha256": sha256(path), "size_bytes": size})
    return rows, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--release-json", required=True, type=pathlib.Path)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--max-entry-bytes", type=int, default=MAX_ENTRY_BYTES)
    args = parser.parse_args()

    assets = args.assets_dir.resolve()
    if not assets.is_dir():
        raise ValueError(f"assets directory missing: {assets}")
    release = load_json(args.release_json)
    tag = str(release.get("tag_name") or "")
    if not TAG_RE.fullmatch(tag):
        raise ValueError(f"invalid model release tag: {tag!r}")
    if bool(release.get("draft")) or bool(release.get("prerelease")):
        raise ValueError("model release must be final")
    if release.get("immutable") is not True:
        raise ValueError("model release must be immutable before Git mirroring")

    promotion = load_json(assets / "model-promotion-manifest.json")
    source = promotion.get("source")
    if not isinstance(source, dict):
        raise ValueError("promotion manifest source missing")
    trained_head = str(source.get("head_sha") or "")
    training_run_id = int(source.get("training_run_id", 0))
    qualification_seed = int(source.get("qualification_seed", 0))
    if str(release.get("target_commitish") or "") != trained_head:
        raise ValueError("release target differs from promotion trained head")

    asset_rows, total = verify_assets(assets)
    if total > args.max_entry_bytes:
        raise ValueError(
            f"registry entry {total} bytes exceeds direct-Git bound {args.max_entry_bytes}; "
            "introduce an explicit LFS/storage-policy change first"
        )

    shipping_approved = False
    evidence_status = "synthetic-qualified"
    blockers: list[str] = []
    config_sha = None
    if args.config:
        config = load_json(args.config)
        model = config.get("model")
        if not isinstance(model, dict):
            raise ValueError("shipping config model object missing")
        if str(model.get("release_tag") or "") != tag:
            raise ValueError("shipping config pins a different model release")
        if int(model.get("training_run_id", 0)) != training_run_id:
            raise ValueError("shipping config training run differs from promoted model")
        if str(model.get("trained_head_sha") or "") != trained_head:
            raise ValueError("shipping config trained head differs from promoted model")
        if int(model.get("qualification_seed", 0)) != qualification_seed:
            raise ValueError("shipping config qualification seed differs from promoted model")
        pins = {
            "xiaowo-model.kwm": str(model.get("model_sha256") or ""),
            "xiaowo-model.pt": str(model.get("checkpoint_sha256") or ""),
            "xiaowo-keywords.kwk": str(model.get("keyword_pack_sha256") or ""),
            "xiaowo-keywords.tsv": str(model.get("keyword_tsv_sha256") or ""),
        }
        for name, digest in pins.items():
            if sha256(assets / name) != digest:
                raise ValueError(f"shipping config digest mismatch: {name}")
        shipping_approved = bool(config.get("shipping_approved", False))
        evidence_status = str(config.get("evidence_status") or "synthetic-qualified")
        boundary = config.get("shipping_evidence_boundary", {})
        if isinstance(boundary, dict):
            blockers = list(boundary.get("pending") or [])
        config_sha = sha256(args.config)

    target = args.output_root.resolve() / tag
    if target.exists():
        raise ValueError(f"registry entry already exists and is immutable: {target}")
    target.mkdir(parents=True)
    for row in asset_rows:
        shutil.copy2(assets / row["name"], target / row["name"])

    registry = {
        "schema_version": 1,
        "policy": "model-registry-entry-v1",
        "release_tag": tag,
        "evidence_status": evidence_status,
        "shipping_approved": shipping_approved,
        "shipping_blockers": blockers,
        "source_release": {
            "id": int(release.get("id", 0)),
            "tag_name": tag,
            "target_commitish": trained_head,
            "immutable": True,
            "published_at": release.get("published_at"),
        },
        "training": {
            "run_id": training_run_id,
            "head_sha": trained_head,
            "qualification_seed": qualification_seed,
        },
        "storage": {
            "mode": "git-direct-no-lfs-v1",
            "max_entry_bytes": int(args.max_entry_bytes),
            "release_assets_bytes": total,
        },
        "shipping_config_sha256": config_sha,
        "assets": asset_rows,
    }
    (target / "registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(registry, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
