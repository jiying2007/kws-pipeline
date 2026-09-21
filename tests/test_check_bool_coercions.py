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
# these cases pin what the signal is -- an external *field* read whose result
# decides something -- as well as what it deliberately is not.

# Flagged: row comes from a parameter, and the result is kept.
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

# Flagged. This used to be deliberately ignored on the grounds that choosing a
# branch is not producing evidence, and that was the mistake: this line decides
# whether a gate raises. A gate that decides something IS the evidence.
CONDITION = """\
def require(work):
    manifest = json.loads(work.read_text(encoding="utf-8"))
    if bool(manifest.get("development_qualified")):
        raise ValueError("not qualified")
"""

# Flagged, and reported against the `if`, not against whatever the branch body
# happens to assign on the next line. Matching by line number reported this as
# `manifest = bool(...)`, a key that moved whenever anything above it changed.
NEGATED_CONDITION = """\
def load_state(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not bool(value.get("complete")):
        manifest = load_object(path)
        return manifest
    return None
"""

# Flagged: the filter of a comprehension is where the gate lives.
FILTER = """\
def eligible(manifest):
    records = manifest.get("records")
    return [
        row
        for row in records
        if isinstance(row, dict)
        and bool(row.get("calibration_gate"))
    ]
"""

# Not flagged: bool(items) asks whether a container is empty. That is a truth
# test, not a recorded verdict being flipped.
TRUTHINESS = """\
def safe_negative(sequence):
    return bool(sequence)
"""

# Flagged: the records come out of JSON, and the result is returned.
RETURNED = """\
def gate(work):
    manifest = json.loads(work.read_text(encoding="utf-8"))
    return 0 if bool(manifest.get("development_qualified")) else 1
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


def entries_for(body: str) -> list[str]:
    """Write one sample and return what the scanner records for it.

    The recorded key is `<path>|<function>|<target> = <expr>`; the path is the
    same throwaway file for every case here, so it is dropped.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        write(root, body)
        done = run(root, "--update")
        assert done.returncode == 0, done.stderr
        waived = json.loads(baseline(root).read_text(encoding="utf-8"))["waived"]
        return [item.split("|", 1)[1] for item in waived]


def main() -> int:
    # What is caught, and how it is named.
    assert entries_for(EXTERNAL) == ["streak|total = bool(row.get('calibration_gate'))"], entries_for(EXTERNAL)
    assert entries_for(RETURNED) == [
        "gate|<return> = bool(manifest.get('development_qualified'))"
    ], entries_for(RETURNED)
    assert entries_for(CONDITION) == [
        "require|<if> = bool(manifest.get('development_qualified'))"
    ], entries_for(CONDITION)
    assert entries_for(FILTER) == [
        "eligible|<filter> = bool(row.get('calibration_gate'))"
    ], entries_for(FILTER)

    # The `if` owns the coercion even when the body assigns on the next line.
    negated = entries_for(NEGATED_CONDITION)
    assert negated == ["load_state|<if> = bool(value.get('complete'))"], negated

    # What is deliberately not caught.
    assert entries_for(LOCAL) == [], entries_for(LOCAL)
    assert entries_for(TRUTHINESS) == [], entries_for(TRUTHINESS)

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

        done = run(root)
        assert done.returncode == 0, done.stderr
        assert "open=0" in done.stdout, done.stdout

        # The key carries no line number: an edit above a coercion would
        # otherwise move it and report one stale plus one new entry for a
        # change that never touched the coercion.
        write(root, "# a comment shifts every line below it\n" + EXTERNAL)
        done = run(root)
        assert done.returncode == 0, done.stderr
        assert "open=0" in done.stdout and "stale=0" in done.stdout, done.stdout
        write(root, EXTERNAL)

        # A new coercion on external evidence is caught, and named.
        write(
            root,
            EXTERNAL + "\n\ndef other(value):\n    kept = bool(value.get('gate'))\n    return kept\n",
        )
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "new bool coercion" in done.stderr, done.stderr
        assert "kept = bool(value.get('gate'))" in done.stderr, done.stderr

        # A coercion that disappears leaves a stale entry, which is how the
        # baseline is forced to shrink instead of only growing.
        write(root, LOCAL)
        done = run(root)
        assert done.returncode == 1, done.stdout
        assert "no longer exist" in done.stderr, done.stderr

    print("test_check_bool_coercions: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
