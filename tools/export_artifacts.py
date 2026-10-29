"""Stdlib-only path and write boundary for model export artifacts."""
from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Mapping


def paths_alias(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Catch lexical/resolved aliases and existing hard links in a stable workspace."""
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except FileNotFoundError:
        return False


def validate_export_paths(*, checkpoint: pathlib.Path, tokens: pathlib.Path,
                          output: pathlib.Path, provenance: pathlib.Path) -> None:
    inputs = (("checkpoint", checkpoint), ("tokens", tokens))
    outputs = (("output", output), ("provenance", provenance))
    for output_name, output_path in outputs:
        for input_name, input_path in inputs:
            if paths_alias(output_path, input_path):
                raise ValueError(f"{output_name} aliases {input_name}; export would overwrite an input")
    if paths_alias(output, provenance):
        raise ValueError("output aliases provenance; model and sidecar require distinct files")


def write_export_artifacts(payloads: Mapping[pathlib.Path, bytes]) -> None:
    """Stage every payload before replacement; each replacement is atomic.

    This is not a transaction across multiple files: a later replace failure can
    leave an earlier artifact updated. The caller must treat any error as a
    failed export and verify the model/sidecar hash binding before consumption.
    Callers also own synchronization against concurrent path/file mutation.
    """
    destinations = list(payloads)
    for index, destination in enumerate(destinations):
        for previous in destinations[:index]:
            if paths_alias(destination, previous):
                raise ValueError("export artifact destinations alias")
    staged: list[tuple[pathlib.Path, pathlib.Path]] = []
    try:
        for destination, payload in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, prefix=f".{destination.name}.",
                suffix=".tmp", delete=False,
            ) as stream:
                temporary = pathlib.Path(stream.name)
                staged.append((temporary, destination))
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
