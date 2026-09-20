from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "check_twin_parity.py"

# A parity scanner that silently finds nothing is worse than no scanner: it
# reports green while the twins drift apart. So these cases use a synthetic
# pair where the divergence is known, and pin all four outcomes -- detected,
# waived, newly diverged, and stale.

LEFT = """\
def check(value):
    if isinstance(value, dict):
        return value
    return None
"""

RIGHT = """\
def check(value):
    return value
"""


def run(root: pathlib.Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), "--root", str(root), *args],
        capture_output=True,
        text=True,
    )


def write(root: pathlib.Path, name: str, body: str) -> pathlib.Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def pair(root: pathlib.Path, left: str = LEFT, right: str = RIGHT) -> None:
    write(root, "tools/thing_rnn_tool.py", left)
    write(root, "tools/thing_gru_tool.py", right)


def baseline(root: pathlib.Path) -> pathlib.Path:
    return root / "configs" / "twin-parity-baseline.json"


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        pair(root)

        # No baseline yet: refusing is better than silently waiving everything.
        done = run(root)
        assert done.returncode == 2, done.stderr
        assert "baseline is missing" in done.stderr, done.stderr

        # The divergence is real: one twin checks the type, the other does not.
        done = run(root, "--report")
        assert done.returncode == 2, done.stderr

        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        assert "1 waived" in done.stdout, done.stdout
        waived = json.loads(baseline(root).read_text(encoding="utf-8"))["waived"]
        assert len(waived) == 1, waived
        assert "isinstance:dict" in waived[0], waived

        # Waived divergence: green.
        done = run(root)
        assert done.returncode == 0, done.stderr
        assert "open=0" in done.stdout, done.stdout

        # A check appears in only one twin and is not waived: fail, and name it.
        write(root, "tools/thing_rnn_tool.py", LEFT + "\ndef other(value):\n    return value\n")
        write(root, "tools/thing_gru_tool.py", RIGHT + "\ndef other(value):\n    if value is not True:\n        raise ValueError('no')\n")
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "new twin divergence" in done.stderr, done.stderr
        assert "is-not:True" in done.stderr, done.stderr

        # A function present on only one side is a different finding, not drift:
        # one twin may reuse the other's copy. It is reported once as a missing
        # function, never as each of its guards -- that would read like N missing
        # checks and hide the fact that there is nothing to compare.
        write(root, "tools/thing_rnn_tool.py", LEFT)
        write(root, "tools/thing_gru_tool.py", RIGHT + "\ndef only_here(value):\n    if value is not True:\n        raise ValueError('no')\n")
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "missing-function" in done.stderr, done.stderr
        assert "only_here" in done.stderr, done.stderr
        assert "is-not:True" not in done.stderr, done.stderr

        # A waived divergence that no longer exists is a stale waiver: fail too,
        # otherwise the baseline rots into a list of things that were once true.
        write(root, "tools/thing_rnn_tool.py", "def check(value):\n    return value\n")
        write(root, "tools/thing_gru_tool.py", "def check(value):\n    return value\n")
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "no longer diverge" in done.stderr, done.stderr

        # Twin tokens are collapsed before comparing, so a function named for
        # its family still matches across the pair.
        write(root, "tools/thing_rnn_tool.py", "def check_rnn_value(value):\n    if isinstance(value, list):\n        return value\n    return None\n")
        write(root, "tools/thing_gru_tool.py", "def check_gru_value(value):\n    return value\n")
        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        waived = json.loads(baseline(root).read_text(encoding="utf-8"))["waived"]
        assert any("check_*_value" in item for item in waived), waived

    # A family binding is a bare `!=` on a field read, so none of the
    # isinstance / is / in guards see it -- and a twin that drops it can bind
    # the other lane's one-shot evidence. Two defects of exactly this shape
    # shipped before the scanner learned it.
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        pair(
            root,
            "def check(value):\n    if value.get('model_family') != 'rnn':\n        raise ValueError('no')\n",
            "def check(value):\n    return value\n",
        )
        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        waived = json.loads(baseline(root).read_text(encoding="utf-8"))["waived"]
        assert any("neq-get:model_family" in item for item in waived), waived

        # str() around the read is the same field read. Without peeling the
        # coercion, a twin that wraps and a twin that does not would look like
        # a divergence that is not there.
        pair(
            root,
            "def check(value):\n    if str(value.get('status')) == 'opened':\n        raise ValueError('no')\n",
            "def check(value):\n    if value.get('status') == 'opened':\n        raise ValueError('no')\n",
        )
        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        waived = json.loads(baseline(root).read_text(encoding="utf-8"))["waived"]
        assert waived == [], waived

        # A module constant counts the same as a literal: `!= POLICY` is how
        # most of this repo writes these checks.
        pair(
            root,
            "def check(value):\n    if value.get('policy') != POLICY:\n        raise ValueError('no')\n",
            "def check(value):\n    return value\n",
        )
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "neq-get:policy" in done.stderr, done.stderr

    print("test_check_twin_parity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
