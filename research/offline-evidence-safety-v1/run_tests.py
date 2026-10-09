#!/usr/bin/env python3
"""Portable synthetic-only CI. Explicit public fetch, or verified offline helper."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from fetch_public_helper import cached_helper, fetch_helper, verify_helper

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--helper', type=Path, help='existing hash-pinned public admission.py (offline)')
    choice.add_argument('--fetch-helper', action='store_true',
                        help='download 59.4 MB pinned public archive; load only the one helper')
    parser.add_argument('--helper-cache-dir', type=Path,
                        help='reuse only size/hash-verified helper bytes with --fetch-helper')
    args = parser.parse_args()
    if args.helper_cache_dir is not None and not args.fetch_helper:
        parser.error('--helper-cache-dir requires --fetch-helper')
    if sys.flags.optimize != 0:
        raise SystemExit('Run harness without -O/-OO; tests exercise both internally.')
    lock = json.loads((ROOT / 'PUBLIC-DEPENDENCY.json').read_text())
    with tempfile.TemporaryDirectory(prefix='offline-evidence-tests-') as tmp:
        if args.helper:
            with args.helper.open('rb') as stream:
                raw = verify_helper(stream.read(lock['helper_bytes'] + 1), lock)
        elif args.helper_cache_dir is not None:
            raw = cached_helper(lock, args.helper_cache_dir, tmp)
        else:
            raw = fetch_helper(lock, tmp)
        helper = Path(tmp) / 'admission.py'
        helper.write_bytes(raw)
        env = dict(os.environ)
        env.update(PUBLIC_ADMISSION_PATH=str(helper), PYTHONPATH=str(ROOT / 'src'),
                   PYTHONDONTWRITEBYTECODE='1')
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                                 '-s', str(ROOT / 'tests'), '-v'], env=env, cwd=ROOT)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
