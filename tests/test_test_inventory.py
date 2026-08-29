from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tests = root / "tests"
        tests.mkdir()
        (tests / "test_ok.py").write_text("print('ok')\n", encoding="utf-8")
        workflow = root / "ci.yml"
        workflow.write_text("run: python3 tests/test_ok.py\n", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "tools" / "test_inventory.py"),
            "--root",
            str(tests),
            "--workflow",
            str(workflow),
        ]
        assert subprocess.run(command, check=False).returncode == 0
        (tests / "test_missing.py").write_text("print('missing')\n", encoding="utf-8")
        assert subprocess.run(command, check=False).returncode == 1
    print("test_test_inventory: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
