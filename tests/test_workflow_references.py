from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = ROOT / "tools" / "check_workflow_references.py"
WORKFLOWS = ROOT / ".github" / "workflows"

EXISTING = """\
name: existing
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: python3 tools/exists.py
"""

MISSING = """\
name: missing
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: python3 tools/gone.py
"""

# governance/verification.json is not in the worktree and never should be: the
# step above copies it into a staging directory that is outside the repository,
# and the step below hashes it there. Reading the file text has to see the copy.
PRODUCED = """\
name: produced
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: cp build/control-plane/verification.json build/approval/governance/
      - run: sha256sum governance/verification.json
"""

# Paths a job writes at run time are not repository dependencies, and a path
# built from an expression cannot be resolved by reading. Both have to be
# invisible to the reference count instead of being reported as dangling, so the
# count stays at the one real reference.
RUNTIME = """\
name: runtime
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: python3 tools/exists.py
      - run: test -f "$RUNNER_TEMP/threshold-operating-curve.json"
      - run: test -x build/kws_wav_gru
      - run: python3 tools/"${{ inputs.driver }}"
"""

NO_REFERENCE = """\
name: bare
on: {workflow_dispatch: null}
jobs:
  build:
    runs-on: ubuntu-24.04
    steps:
      - run: echo hello
"""


def run(root: pathlib.Path, *paths: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root), *[str(p) for p in paths]],
        check=False, capture_output=True, text=True,
    )


def write(directory: pathlib.Path, name: str, text: str) -> pathlib.Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        (root / "tools").mkdir()
        (root / "tools" / "exists.py").write_text("# present\n", encoding="utf-8")
        (root / "wf").mkdir()
        existing = write(root / "wf", "existing.yml", EXISTING)
        missing = write(root / "wf", "missing.yml", MISSING)
        produced = write(root / "wf", "produced.yml", PRODUCED)
        runtime = write(root / "wf", "runtime.yml", RUNTIME)
        bare = write(root / "wf", "bare.yml", NO_REFERENCE)

        assert run(root, existing).returncode == 0, "a path that exists must pass"
        assert run(root, produced).returncode == 0, "a path the job writes must pass"

        gone = run(root, missing)
        assert gone.returncode == 1, "a deleted dependency must fail"
        assert "tools/gone.py" in gone.stderr, gone.stderr

        # Runtime paths and expression-built paths are not repository
        # dependencies: only the one real reference may be counted.
        quiet = run(root, runtime)
        assert quiet.returncode == 0, quiet.stderr
        assert "references=1" in quiet.stdout, quiet.stdout

        # A workflow with nothing to resolve is the wrong input, not a pass:
        # it is how a broken scanner would report every repository clean.
        assert run(root, bare).returncode == 2, "no references must error, not pass"

    # The repository's own wiring is the regression this guards. It has to be
    # clean and it has to actually resolve something, or the check above is
    # measuring nothing.
    real = run(ROOT, WORKFLOWS)
    assert real.returncode == 0, real.stderr
    assert "resolved=0" not in real.stdout, real.stdout
    print("test_workflow_references: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
