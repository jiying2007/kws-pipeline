from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "test_ok.py").write_text("print('ok')\n", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "test_inventory.py"),
                "--root",
                str(root),
            ],
            check=False,
        )
        assert completed.returncode == 0
    print("test_test_inventory: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
