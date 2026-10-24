#!/usr/bin/env python3
"""Stage the unchanged historical near-pronunciation source closure for tests.

This never updates scientific pins or silently accepts current-source drift.
Only the five explicitly retained fixtures replace current files. No downloads,
model execution, training, or changes to the caller's source tree are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile

FROZEN_FILES = frozenset({
    'src/decoder.c', 'tests/test_decoder.c',
    'tools/kws_trace_io.c', 'tools/kws_trace_io.h',
    'tools/kws_decoder_replay.c',
})
OVERLAY = 'research/near_pronunciation_v1/decoder_overlay.json'
FIXTURES = 'tests/fixtures/near_pronunciation_v1'


def stage_source(source_root: Path, output_dir: Path) -> dict[str, str]:
    source_root = source_root.resolve(strict=True)
    output_dir = output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError('historical source output already exists')
    pins = json.loads((source_root / OVERLAY).read_bytes())['source_sha256']
    if not isinstance(pins, dict) or not FROZEN_FILES.issubset(pins):
        raise ValueError('historical source closure omits retained fixtures')
    originals = {}
    for name, expected in pins.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or str(rel) != name:
            raise ValueError('noncanonical historical source path')
        origin = source_root / FIXTURES if name in FROZEN_FILES else source_root
        path = (origin / name).resolve(strict=True)
        if (not path.is_relative_to(source_root) or
                not path.is_relative_to(origin.resolve(strict=True))):
            raise ValueError('historical source path escapes its source root')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'historical source hash mismatch: {name}')
        originals[name] = data
    # Validate every input before creating output. A failed staging attempt must
    # never leave a partially usable source tree for a later compiler invocation.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.historical-decoder-', dir=output_dir.parent))
    try:
        for name, data in originals.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError('historical source output appeared during staging')
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        pins = stage_source(args.source_root, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'historical decoder source refused: {exc}\n')
    print(json.dumps({'scope': 'historical-near-pronunciation-only',
                      'pinned_files': len(pins), 'current_decoder_qualification': False},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
