#!/usr/bin/env python3
"""Verify a downloaded immutable model release against the product contract.

The README states that the qualified model is bound to four SHA256 digests, and
those digests live in ``configs/shipping.xiaowo.json``. Until now nothing
checked a download against them, so a consumer who fetched the release assets
had no fail-closed way to confirm they received the qualified bytes.

Fetching is deliberately out of scope -- ``gh release download`` or plain HTTPS
is the downloader's job. Keeping the network out makes this check deterministic
and runnable in CI.

The two checks are not redundant. ``MODEL_SHA256SUMS`` is shipped *inside* the
release, so on its own it only proves the download is internally consistent: an
attacker who replaces an asset can regenerate the manifest to match. The
contract pins live in the repository, so they are what actually binds the
download to the qualified tuple.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

HEX = set("0123456789abcdef")
SUMS_NAME = "MODEL_SHA256SUMS"

# Contract key -> release asset name. The contract is the authority for the
# qualified tuple; the release asset names are what `gh release download`
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


def verify(assets: pathlib.Path, config: pathlib.Path, release_tag: str) -> dict[str, object]:
    if not assets.is_dir():
        raise ValueError(f"assets directory does not exist: {assets}")
    contract = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("contract must be a JSON object")
    model = contract.get("model")
    if not isinstance(model, dict):
        raise ValueError("contract has no 'model' object")

    pinned_tag = str(model.get("release_tag") or "")
    if not pinned_tag:
        raise ValueError("contract model.release_tag is missing")
    if release_tag and release_tag != pinned_tag:
        raise ValueError(f"--release-tag {release_tag!r} does not match contract {pinned_tag!r}")

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

    return {"release_tag": pinned_tag, "manifest_entries": len(sums), "pinned_assets": pins}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a downloaded model release against the product contract."
    )
    parser.add_argument(
        "--assets-dir",
        required=True,
        type=pathlib.Path,
        help="directory holding the downloaded release assets",
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "configs" / "shipping.xiaowo.json",
        help="product contract that pins the qualified digests",
    )
    parser.add_argument(
        "--release-tag",
        default="",
        help="assert that the contract pins this exact release tag",
    )
    args = parser.parse_args()
    report = verify(args.assets_dir, args.config, args.release_tag)
    print(
        "verify_model_release: ok "
        f"release_tag={report['release_tag']} "
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
