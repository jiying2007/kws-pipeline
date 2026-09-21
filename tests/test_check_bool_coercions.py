from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_bool_coercions.py"

# A scanner that silently finds nothing is worse than no scanner: it reports
# green while bool("false") keeps turning recorded failures into passes. So
# these cases pin what the signal is -- external value, kept result -- as well
# as what it deliberately is not.

# Flagged: row comes from a parameter, and the result is returned.
EXTERNAL = """\
def streak(records):
    total = 0
    for row in records:
        total += 1 if bool(row.get("calibration_gate")) else 0
    return total
"""

# Not flagged: the value is computed here, so the coercion is a no-op.
LOCAL = """\
def streak(records):
    qualified = base_gate(records) and domain_gate(records)
    return bool(qualified)
"""

# Not flagged: the result only picks a branch, it is never kept.
BRANCH_ONLY = """\
def streak(records):
    for row in records:
        if bool(row.get("calibration_gate")):
            print("ok")
"""


def run(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )


def write(root: pathlib.Path, body: str) -> None:
    directory = root / "tools"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sample.py").write_text(body, encoding="utf-8")


def baseline(root: pathlib.Path) -> pathlib.Path:
    return root / "configs" / "bool-coercion-baseline.json"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        write(root, EXTERNAL)

        # No baseline yet: refusing is better than silently waiving everything.
        done = run(root)
        assert done.returncode == 2, done.stderr
        assert "baseline is missing" in done.stderr, done.stderr

        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        assert "1 entry" in done.stdout, done.stdout
        assert len(json.loads(baseline(root).read_text(encoding="utf-8"))["waived"]) == 1

        done = run(root)
        assert done.returncode == 0, done.stderr
        assert "open=0" in done.stdout, done.stdout

        # A value built in the function is already the type it says it is.
        write(root, LOCAL)
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "no longer exist" in done.stderr, done.stderr

        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        assert "0 entry" in done.stdout, done.stdout

        # Choosing a branch is not producing evidence.
        write(root, BRANCH_ONLY)
        done = run(root)
        assert done.returncode == 0, done.stdout
        assert "open=0" in done.stdout, done.stdout

        # A new coercion on external evidence is caught, and named.
        write(root, EXTERNAL + "\n\ndef other(value):\n    kept = bool(value.get('gate'))\n    return kept\n")
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "new bool coercion" in done.stderr, done.stderr
        assert "kept = bool(value.get('gate'))" in done.stderr, done.stderr

    print("test_check_bool_coercions: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
