#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMMUTABLE_IMAGE_RE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_base_image(value: str) -> str:
    value = value.strip()
    if IMMUTABLE_IMAGE_RE.fullmatch(value) is None:
        raise ValueError("base image must use name@sha256:<64 lowercase hex>")
    return value


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    args = parser.parse_args()
    base = validate_base_image(args.base_image)
    dockerfile = ROOT / "training" / "Dockerfile"
    lock = ROOT / "training" / "requirements.lock"
    subprocess.check_call(
        [
            "docker",
            "build",
            "--pull=false",
            "--build-arg",
            f"KWS_TRAINING_BASE={base}",
            "-f",
            str(dockerfile),
            "-t",
            args.tag,
            str(ROOT),
        ]
    )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format={{.Id}}", args.tag], text=True
    ).strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ValueError("docker returned a non-content-addressed image id")
    receipt = {
        "schema_version": 1,
        "source_sha": git_sha(),
        "base_image": base,
        "dockerfile_sha256": sha256_file(dockerfile),
        "requirements_lock_sha256": sha256_file(lock),
        "image_id": image_id,
        "tag": args.tag,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote training image receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
