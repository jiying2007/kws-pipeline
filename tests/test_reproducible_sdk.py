from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        first = root / "a"
        second = root / "b"
        first.mkdir()
        second.mkdir()
        (first / "lib.a").write_bytes(b"same")
        (second / "lib.a").write_bytes(b"same")
        command = [
            sys.executable,
            str(ROOT / "tools" / "check_reproducible_sdk.py"),
            str(first),
            str(second),
        ]
        assert subprocess.run(command, check=False).returncode == 0
        (second / "lib.a").write_bytes(b"different")
        assert subprocess.run(command, check=False).returncode == 1
    print("test_reproducible_sdk: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
