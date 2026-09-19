from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_deferred_verdicts.py"
WORKFLOWS = ROOT / ".github" / "workflows"

# A deferred verdict that IS redeemed: the loop captures the code, and a later
# step's `if:` references it. This is the shape most of the repository uses.
REDEEMED = """\
name: redeemed
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - id: loop
        name: Run loop
        run: |
          set +e
          ./thing
          code=$?
          echo "exit_code=$code" >> "$GITHUB_OUTPUT"
          exit 0
      - name: Collect evidence
        if: always()
        run: echo collecting
      - name: Gate
        if: always()
        run: |
          test "${{ steps.loop.outputs.exit_code }}" = '0'
"""

# The hole this checker exists for: captured, never asserted.
ORPHAN = """\
name: orphan
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - id: develop
        name: Execute development loop
        run: |
          set +e
          ./thing
          code=$?
          echo "exit_code=$code" >> "$GITHUB_OUTPUT"
          exit 0
      - name: Enforce completion
        if: always()
        run: test -f out.json
"""

# Two jobs: the capture is in one, the assertion in the other. Cross-job
# redemption is legal, so this must pass.
CROSS_JOB = """\
name: cross
on: {workflow_dispatch: null}
jobs:
  produce:
    runs-on: ubuntu-24.04
    steps:
      - id: fresh
        run: |
          set +e
          ./x
          code=$?
          echo "exit_code=$code" >> "$GITHUB_OUTPUT"
          exit 0
  gate:
    needs: produce
    runs-on: ubuntu-24.04
    steps:
      - run: test "${{ needs.produce.outputs.x }}" = '0'
      - run: echo "${{ steps.fresh.outputs.exit_code }}"
"""

# A step that mentions exit_code but does not capture it must be ignored, or the
# checker would report every workflow that merely reads a captured code.
NO_CAPTURE = """\
name: nocapture
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: echo "exit_code is not used here"
"""


def run(root: pathlib.Path, *paths: pathlib.Path) -> int:
    return subprocess.run(
        [sys.executable, str(CHECKER), *[str(p) for p in paths]],
        check=False, capture_output=True, text=True,
    ).returncode


def write(root: pathlib.Path, name: str, text: str) -> pathlib.Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        redeemed = write(root, "redeemed.yml", REDEEMED)
        orphan = write(root, "orphan.yml", ORPHAN)
        cross = write(root, "cross.yml", CROSS_JOB)
        nocap = write(root, "nocapture.yml", NO_CAPTURE)

        assert run(root, redeemed) == 0, "a redeemed verdict must pass"
        assert run(root, cross) == 0, "cross-job redemption must pass"

        # The whole point: an unredeemed capture must fail and name the step.
        assert run(root, orphan) == 1, "an unredeemed capture must fail"
        done = subprocess.run(
            [sys.executable, str(CHECKER), str(orphan)],
            check=False, capture_output=True, text=True,
        )
        assert "id=develop" in done.stderr, done.stderr

        # A workflow with no captures at all is the wrong input, not a pass:
        # it is how a broken checker would silently report everything clean.
        assert run(root, nocap) == 2, "no captures must be an error, not a pass"

        # The same, when the argument is a directory whose files have none.
        empty = root / "empty"
        empty.mkdir()
        assert run(root, empty) == 2

        # A directory is scanned as a whole, and one bad file is enough.
        assert run(root, root) == 1, "a directory scan must catch the orphan"

        # Malformed YAML is an error, never a silent pass.
        bad = write(root, "bad.yml", "jobs: [unclosed\n")
        assert run(root, bad) == 2, "invalid YAML must be an error"

        # A missing path is an error, never a skip.
        assert run(root, root / "absent.yml") == 2

    # The repository's own workflows must satisfy the gate that now guards them.
    # This is the assertion that keeps the fix from regressing.
    assert run(ROOT, WORKFLOWS) == 0, "the repository workflows must have no unredeemed verdicts"

    print("test_deferred_verdicts: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
