from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "verify_model_release.py"

# Mirrors tools/verify_model_release.py: contract key -> release asset name.
PINNED = {
    "model_sha256": "xiaowo-model.kwm",
    "checkpoint_sha256": "xiaowo-model.pt",
    "keyword_pack_sha256": "xiaowo-keywords.kwk",
    "keyword_tsv_sha256": "xiaowo-keywords.tsv",
}
TAG = "model-abcdef012345"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sums(assets: pathlib.Path) -> None:
    """Regenerate MODEL_SHA256SUMS from whatever is currently on disk."""
    entries = sorted(
        path for path in assets.iterdir() if path.is_file() and path.name != "MODEL_SHA256SUMS"
    )
    (assets / "MODEL_SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in entries), encoding="utf-8"
    )


def build(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Create a self-consistent release directory and a contract that pins it."""
    assets = root / "assets"
    assets.mkdir()
    for index, name in enumerate(sorted(PINNED.values())):
        (assets / name).write_bytes(bytes([index]) * (16 + index))
    (assets / "qualification-summary.json").write_bytes(b'{"matched": 256}\n')
    write_sums(assets)
    contract = {
        "schema_version": 1,
        "model": {
            "release_tag": TAG,
            **{key: sha256(assets / name) for key, name in PINNED.items()},
        },
    }
    config = root / "shipping.json"
    config.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return assets, config


def run(assets: pathlib.Path, config: pathlib.Path, *extra: str) -> int:
    return subprocess.run(
        [sys.executable, str(TOOL), "--assets-dir", str(assets), "--config", str(config), *extra],
        check=False,
        capture_output=True,
        text=True,
    ).returncode


def corrupt(path: pathlib.Path) -> bytes:
    """Flip one bit in the last byte and return the original bytes."""
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 0x01]))
    return original


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        assets, config = build(root)

        # A self-consistent release verifies, with and without a tag assertion.
        assert run(assets, config) == 0, "a self-consistent release must verify"
        assert run(assets, config, "--release-tag", TAG) == 0

        # The release tag is part of the contract, not a cosmetic label.
        assert run(assets, config, "--release-tag", "model-000000000000") == 2

        # A single flipped byte in a pinned asset must fail.
        target = assets / PINNED["model_sha256"]
        original = corrupt(target)
        assert run(assets, config) == 2, "a corrupted pinned asset must fail"
        target.write_bytes(original)
        write_sums(assets)
        assert run(assets, config) == 0

        # Regenerating MODEL_SHA256SUMS so it matches the corrupted bytes must
        # not launder the asset: the contract pin is the authority, not the
        # manifest shipped next to the payload. This is the whole reason the
        # tool exists rather than telling users to run `sha256sum -c`.
        corrupt(target)
        write_sums(assets)
        assert run(assets, config) == 2, "a regenerated manifest must not launder a corrupted asset"
        target.write_bytes(original)
        write_sums(assets)
        assert run(assets, config) == 0

        # A manifest entry with no file must fail rather than be skipped.
        (assets / "qualification-summary.json").unlink()
        assert run(assets, config) == 2, "a manifest entry with no file must fail"
        (assets / "qualification-summary.json").write_bytes(b'{"matched": 256}\n')
        write_sums(assets)

        # A pinned asset that the manifest omits must fail even though its
        # digest matches the contract: a consumer walking the manifest would
        # never reach it.
        sums = assets / "MODEL_SHA256SUMS"
        kept = [
            line
            for line in sums.read_text(encoding="utf-8").splitlines()
            if "xiaowo-model.pt" not in line
        ]
        sums.write_text("\n".join(kept) + "\n", encoding="utf-8")
        assert run(assets, config) == 2, "a pinned asset absent from the manifest must fail"
        write_sums(assets)

        # An absent manifest is not an implicit pass.
        sums.unlink()
        assert run(assets, config) == 2, "an absent manifest must fail"
        write_sums(assets)

        # A contract that pins a digest the release does not carry must fail.
        contract = json.loads(config.read_text(encoding="utf-8"))
        contract["model"]["model_sha256"] = "0" * 64
        config.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        assert run(assets, config) == 2, "a contract pin the release cannot satisfy must fail"

    print("test_model_release_pin: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
