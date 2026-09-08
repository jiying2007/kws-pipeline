#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def inspect_adapter(path: pathlib.Path) -> tuple[dict, pathlib.Path, dict]:
    adapter = json.loads(path.read_text(encoding="utf-8"))
    if adapter.get("schema_version") != 1:
        raise ValueError("AFE adapter schema_version must be 1")
    executable = pathlib.Path(str(adapter.get("executable_path", ""))).resolve(strict=True)
    if not executable.is_file():
        raise ValueError("AFE executable_path must resolve to a file")
    template = adapter.get("command_argv")
    if (
        not isinstance(template, list)
        or not template
        or any(not isinstance(item, str) or not item for item in template)
    ):
        raise ValueError("AFE command_argv must be a non-empty string array")
    joined = "\n".join(template)
    required = ("{executable}", "{input}", "{output}", "{result}")
    missing = [token for token in required if token not in joined]
    if missing:
        raise ValueError(f"AFE command template is missing placeholders: {missing}")

    config_paths = [
        pathlib.Path(str(item)).resolve(strict=True)
        for item in adapter.get("config_files", [])
    ]
    if not config_paths:
        raise ValueError("AFE adapter config_files must not be empty")
    config_rows = []
    seen_names: set[str] = set()
    for config in config_paths:
        if not config.is_file():
            raise ValueError(f"AFE config is not a file: {config}")
        name = config.name
        if name in seen_names:
            raise ValueError(f"AFE config bundle has duplicate basename: {name}")
        seen_names.add(name)
        config_rows.append(
            {"name": name, "sha256": sha256_file(config), "bytes": config.stat().st_size}
        )
    config_rows.sort(key=lambda row: row["name"])
    config_bundle_sha = canonical_sha(config_rows)

    source_sha = adapter.get("pipeline_source_sha")
    if source_sha is not None:
        source_sha = str(source_sha).lower()
        if len(source_sha) != 40 or any(ch not in "0123456789abcdef" for ch in source_sha):
            raise ValueError("pipeline_source_sha must be 40 lowercase hex characters")

    for field in ("sku", "microphone_revision", "enclosure_revision", "audio_route", "toolchain"):
        value = adapter.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"AFE adapter {field} must be non-empty text")

    public = {
        "schema_version": 1,
        "backend": "command",
        "shipping_authority": True,
        "executable_sha256": sha256_file(executable),
        "config_bundle_sha256": config_bundle_sha,
        "config_files": config_rows,
        "command_argv": template,
        "pipeline_source_sha": source_sha,
        "sku": adapter["sku"],
        "microphone_revision": adapter["microphone_revision"],
        "enclosure_revision": adapter["enclosure_revision"],
        "audio_route": adapter["audio_route"],
        "toolchain": adapter["toolchain"],
    }
    public["identity_sha256"] = canonical_sha(public)
    return adapter, executable, public


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    _, _, identity = inspect_adapter(args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(identity, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
