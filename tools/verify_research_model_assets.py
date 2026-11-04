#!/usr/bin/env python3
"""Verify locally available frozen research model assets against pinned SHA256.

This offline tool checks the four immutable model/keyword SHA256 pins and the
MODEL_SHA256SUMS manifest independently. It neither fetches nor publishes a
GitHub Release, and reports no acoustic, human, rights or physical-board PASS.
Historical asset naming remains only as byte-format provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

HEX = set("0123456789abcdef")
SUMS_NAME = "MODEL_SHA256SUMS"

# Contract key -> frozen asset name. The contract is the authority for the
# frozen comparator source; file names are inherited from immutable Git history
# produces.
PINNED_ASSETS = {
    "model_sha256": "xiaowo-model.kwm",
    "checkpoint_sha256": "xiaowo-model.pt",
    "keyword_pack_sha256": "xiaowo-keywords.kwk",
    "keyword_tsv_sha256": "xiaowo-keywords.tsv",
}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha(value: object, label: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(ch not in HEX for ch in text):
        raise ValueError(f"{label} is missing/invalid")
    return text


def parse_sums(path: pathlib.Path) -> dict[str, str]:
    """Parse sha256sum output into {file name: digest}."""
    entries: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{SUMS_NAME}:{number}: expected '<digest>  <name>', got {raw!r}")
        digest = parts[0].lower()
        name = parts[1].strip().lstrip("*").strip()
        if len(digest) != 64 or any(ch not in HEX for ch in digest):
            raise ValueError(f"{SUMS_NAME}:{number}: not a SHA256 digest: {digest!r}")
        if not name:
            raise ValueError(f"{SUMS_NAME}:{number}: empty file name")
        if name in entries:
            raise ValueError(f"{SUMS_NAME}:{number}: duplicate entry for {name}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"{SUMS_NAME}: no entries")
    return entries


def verify(assets: pathlib.Path, config: pathlib.Path, registry_tag: str) -> dict[str, object]:
    if not assets.is_dir():
        raise ValueError(f"assets directory does not exist: {assets}")
    contract = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    model = contract.get("model")
    if not isinstance(model, dict):
        raise ValueError("contract has no 'model' object")

    pinned_tag = str(model.get("registry_tag") or "")
    if not pinned_tag:
        raise ValueError("contract model.registry_tag is missing")
    if registry_tag and registry_tag != pinned_tag:
        raise ValueError(f"--registry-tag {registry_tag!r} does not match contract {pinned_tag!r}")

    sums_path = assets / SUMS_NAME
    if not sums_path.is_file():
        raise ValueError(f"missing {SUMS_NAME} in {assets}")
    sums = parse_sums(sums_path)

    # 1. Every manifest entry must be present and match its recorded digest.
    for name in sorted(sums):
        path = assets / name
        if not path.is_file():
            raise ValueError(f"{SUMS_NAME} lists {name}, which is absent from {assets}")
        actual = sha256(path)
        if actual != sums[name]:
            raise ValueError(f"{name}: digest {actual} does not match {SUMS_NAME} {sums[name]}")

    # 2. Every contract pin must be present, match, and be covered by the
    #    manifest -- a pinned asset the manifest omits would be skipped by a
    #    consumer who only walks MODEL_SHA256SUMS.
    pins: dict[str, str] = {}
    for key, name in sorted(PINNED_ASSETS.items()):
        pin = require_sha(model.get(key), f"model.{key}")
        path = assets / name
        if not path.is_file():
            raise ValueError(f"contract pins {name} (model.{key}) but it is absent from {assets}")
        actual = sha256(path)
        if actual != pin:
            raise ValueError(f"{name}: digest {actual} does not match contract model.{key} {pin}")
        if name not in sums:
            raise ValueError(f"{name} is pinned by the contract but absent from {SUMS_NAME}")
        pins[name] = pin

    return {"registry_tag": pinned_tag, "manifest_entries": len(sums), "pinned_assets": pins}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify frozen research model assets against pinned source digests."
    )
    parser.add_argument(
        "--assets-dir",
        required=True,
        type=pathlib.Path,
        help="directory holding the downloaded frozen assets",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "configs" / "research" / "two-keyword-synthetic-comparator.json",
        help="research comparator contract that pins the archived model SHA256 pins",
    )
    parser.add_argument(
        "--registry-tag",
        default="",
        help="assert that the contract pins this exact registry tag",
    )
    args = parser.parse_args()
    report = verify(args.assets_dir, args.config, args.registry_tag)
    print(
        "verify_research_model_assets: ok "
        f"registry_tag={report['registry_tag']} "
        f"manifest_entries={report['manifest_entries']} "
        f"pinned_assets={len(report['pinned_assets'])}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
