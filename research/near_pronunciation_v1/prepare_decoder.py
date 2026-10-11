#!/usr/bin/env python3
"""Materialize a pinned offline decoder overlay without editing shipping source.

Standard library only. This prepares C source; it never compiles, downloads,
opens model/corpus data, invokes acoustic inference, or runs a campaign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

ALTERED_FILES = frozenset({
    "src/decoder.c", "src/decoder.h", "src/kws.c", "src/kws_debug.h",
    "tools/kws_decoder_replay.c",
})


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def materialize(source_root: Path, output_dir: Path) -> dict:
    source_root = source_root.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output directory already exists; choose a fresh build directory")
    overlay_path = Path(__file__).with_name("decoder_overlay.json")
    raw_overlay = overlay_path.read_bytes()
    overlay = json.loads(raw_overlay)
    if (overlay.get("schema_version") != 1 or
            overlay.get("research_guard") != "KWS_RESEARCH_CHILD_ADMISSION" or
            set(overlay.get("files", {})) != ALTERED_FILES or
            not ALTERED_FILES.issubset(overlay.get("source_sha256", {}))):
        raise ValueError("invalid decoder overlay contract")
    originals = {}
    for name, expected in overlay["source_sha256"].items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != name:
            raise ValueError("noncanonical source path")
        path = (source_root / name).resolve(strict=True)
        if not path.is_relative_to(source_root):
            raise ValueError("source path escapes source root")
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"pinned source hash mismatch: {name}")
        originals[name] = data
    outputs = {}
    for name, patch in overlay["files"].items():
        text = originals[name].decode("utf-8")
        for replacement in patch["replacements"]:
            old, new = replacement["old"], replacement["new"]
            if not old or text.count(old) != 1:
                raise ValueError(f"non-unique patch context: {name}")
            text = text.replace(old, new, 1)
        data = text.encode("utf-8")
        if digest(data) != patch["derived_sha256"]:
            raise ValueError(f"derived source hash mismatch: {name}")
        outputs[name] = data
    receipt = {
        "schema_version": 1,
        "kind": "offline_research_decoder_source_overlay",
        "research_guard": overlay["research_guard"],
        "base_tree": overlay["base_tree"],
        "overlay_sha256": digest(raw_overlay),
        "preparer_sha256": digest(Path(__file__).read_bytes()),
        "original_source_sha256": overlay["source_sha256"],
        "derived_source_sha256": {name: digest(data) for name, data in outputs.items()},
        "shipping_source_modified": False,
        "acoustic_inference_run": False,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".decoder-overlay-", dir=output_dir.parent))
    try:
        for name, data in outputs.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (staging / "source-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Refuse seeded output even if it appeared while validation ran.
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("output directory appeared during preparation")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = materialize(args.source_root, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"decoder overlay refused: {exc}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
