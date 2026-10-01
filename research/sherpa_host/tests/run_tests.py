#!/usr/bin/env python3
"""Offline checks of real CLI helpers with a fake API; never loads ORT/models."""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(argv, **kwargs):
    subprocess.run([str(a) for a in argv], check=True, timeout=180, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    run([sys.executable, ROOT / "tests/test_contracts.py"])
    run(["sh", "-n", ROOT / "cli/run-kws"])
    run([sys.executable, ROOT / "cli/tests/launcher_fake_tests.py", ROOT / "cli"])
    with tempfile.TemporaryDirectory(prefix="sherpa-host-helpers-") as tmp:
        temp = Path(tmp); fixture_root = temp / "cli"; (fixture_root / "tests/fixtures").mkdir(parents=True)
        shutil.copytree(ROOT / "cli/config", fixture_root / "config")
        pcm = struct.pack("<5h", -32768, -1, 0, 1, 32767)
        fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
        wave = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(pcm)) + pcm
        (fixture_root / "tests/fixtures/valid tiny.wav").write_bytes(b"RIFF" + struct.pack("<I", len(wave)) + wave)
        flags = ["-std=c++17", "-O1", "-Wall", "-Wextra", "-Werror"]
        if args.sanitize:
            flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
        run(["g++", *flags, "-I" + str(ROOT / "cli/src"),
             "-I" + str(ROOT.parent / "sherpa_pcm/vendor"),
             ROOT / "cli/tests/core_tests.cc", "-o", temp / "core-tests"])
        environment = dict(os.environ)
        environment["TMPDIR"] = str(temp)
        environment.setdefault("ASAN_OPTIONS", "detect_leaks=1:halt_on_error=1")
        environment.setdefault("UBSAN_OPTIONS", "halt_on_error=1")
        run([temp / "core-tests", fixture_root], env=environment)


if __name__ == "__main__":
    main()
