from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_rnn_evaluation_independence as independence  # noqa: E402

TOOL = ROOT / "tools" / "verify_rnn_evaluation_independence.py"
FRESH_SPLITS = ("calibration", "test", "qualification")
SHADOW_EVIDENCE = "development-only-shadow-qualification"

# What this verifier exists to prove: the fresh splits, and the shadow
# seeds, share no WAV. If they do, every number downstream is inflated, and
# nothing in the receipt reveals it, because the receipt only carries
# counts. So the tests pin the two ways that check goes quietly wrong: the
# index reader skipping rows it should have counted, and the disjointness
# check comparing sets that were never both populated. A split that reads as
# empty is the dangerous case, because an empty set has a trivially empty
# intersection.


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def row(split: str, label: str) -> dict:
    return {"split": split, "wav_sha256": digest(label)}


def write_index(path: pathlib.Path, rows: list) -> pathlib.Path:
    # Padded on purpose: blank lines are legal in a JSONL index and must be
    # skipped rather than counted as a row.
    body = "".join(json.dumps(item) + "\n" for item in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n" + body + "\n", encoding="utf-8")
    return path


def build_fresh_index(base: pathlib.Path, counts, overlap=None) -> pathlib.Path:
    rows: list = []
    for split, count in zip(FRESH_SPLITS, counts):
        for index in range(count):
            rows.append(row(split, f"{split}-{index}"))
    if overlap is not None:
        rows.append(row(overlap[1], overlap[0]))
    return write_index(base / "dataset" / "domain-index.jsonl", rows)


def build_shadow_tree(base: pathlib.Path, seeds, per_seed: int = 2) -> None:
    for seed in seeds:
        rows = [row("qualification", f"seed{seed}-{index}") for index in range(per_seed)]
        write_index(base / f"seed-{seed}" / "dataset" / "domain-index.jsonl", rows)


def write_summary(base: pathlib.Path, seeds, evidence=SHADOW_EVIDENCE, qualified=True) -> pathlib.Path:
    path = base / "summary.json"
    payload = {"evidence_class": evidence, "qualified": qualified, "seeds": seeds}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def expect_raises(needle: str, call) -> str:
    try:
        call()
    except (TypeError, ValueError) as exc:
        assert needle in str(exc), f"expected {needle!r}, got {exc!r}"
        return str(exc)
    raise AssertionError(f"expected a failure containing {needle!r}")


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def expect(needle: str, done: subprocess.CompletedProcess) -> None:
    assert done.returncode != 0, done.stdout + done.stderr
    # This verifier prints its errors on stdout, unlike most of tools/ which
    # uses stderr. Accept either: the assertion is about the failure, not
    # about which stream it arrived on.
    assert needle in done.stdout + done.stderr, f"expected {needle!r}, got: {done.stdout + done.stderr}"


def check_split_hashes(base: pathlib.Path) -> None:
    expect_raises("domain index missing", lambda: independence.split_hashes(base / "absent.jsonl", "test"))

    path = write_index(
        base / "dataset" / "domain-index.jsonl",
        [row("test", "a"), row("train", "b"), row("test", "c")],
    )
    assert independence.split_hashes(path, "test") == {digest("a"), digest("c")}

    # A JSON line that is not an object is an error, not a skipped row.
    broken = base / "broken.jsonl"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("[1, 2]\n", encoding="utf-8")
    expect_raises("expected object", lambda: independence.split_hashes(broken, "test"))

    for label, value in (("short", digest("a")[:63]), ("nonhex", "z" * 64)):
        path = write_index(base / f"{label}.jsonl", [{"split": "test", "wav_sha256": value}])
        expect_raises("invalid test WAV SHA", lambda p=path: independence.split_hashes(p, "test"))

    # A row missing the field entirely is rejected the same way.
    path = write_index(base / "fieldless.jsonl", [{"split": "test"}])
    expect_raises("invalid test WAV SHA", lambda: independence.split_hashes(path, "test"))

    path = write_index(base / "dup.jsonl", [row("test", "a"), row("test", "a")])
    expect_raises("duplicate test WAV SHA", lambda: independence.split_hashes(path, "test"))

    # The failure mode that would make the whole verifier decorative.
    path = write_index(base / "empty.jsonl", [row("train", "a")])
    expect_raises("no WAV identity evidence", lambda: independence.split_hashes(path, "test"))


def check_fresh(base: pathlib.Path) -> None:
    result = independence.verify_fresh(build_fresh_index(base, (2, 3, 4)))
    assert result["schema_version"] == 1
    assert result["policy"] == "frozen-rnn-fresh-internal-sha-independence-v1"
    assert result["model_family"] == "rnn"
    assert result["qualified"] is True
    assert result["all_fresh_splits_sha_disjoint"] is True
    assert result["splits"] == {"calibration": 2, "test": 3, "qualification": 4}
    assert result["wav_sha256_count"] == 9

    # One WAV reused across two splits. Every count still looks plausible,
    # which is exactly why the check has to exist.
    path = build_fresh_index(base / "a", (2, 3, 4), overlap=("calibration-0", "test"))
    message = expect_raises("overlaps", lambda: independence.verify_fresh(path))
    assert "test" in message, message

    path = build_fresh_index(base / "b", (2, 3, 4), overlap=("test-0", "qualification"))
    message = expect_raises("overlaps", lambda: independence.verify_fresh(path))
    assert "qualification" in message, message

    # Two of the three splits present: the third must not be treated as an
    # empty, therefore disjoint, set.
    path = write_index(
        base / "c" / "dataset" / "domain-index.jsonl",
        [row("calibration", "a"), row("test", "b")],
    )
    message = expect_raises("no WAV identity evidence", lambda: independence.verify_fresh(path))
    assert "qualification" in message, message


def check_shadow(base: pathlib.Path) -> None:
    seeds = list(range(8))
    build_shadow_tree(base / "ok", seeds)
    result = independence.verify_shadow(base / "ok", write_summary(base / "ok", seeds))
    assert result["schema_version"] == 1
    assert result["policy"] == "frozen-rnn-shadow-seed-sha-independence-v1"
    assert result["model_family"] == "rnn"
    assert result["qualified"] is True
    assert result["all_shadow_seeds_sha_disjoint"] is True
    assert result["seeds"] == seeds
    assert result["per_seed_wav_sha256_count"] == {str(seed): 2 for seed in seeds}
    assert result["wav_sha256_count"] == 16
    assert result["formal_qualification_used"] is False

    # The 8..16 seed window is inclusive on both ends.
    for count in (8, 16):
        home = base / f"n{count}"
        chosen = list(range(count))
        build_shadow_tree(home, chosen)
        independence.verify_shadow(home, write_summary(home, chosen))
    for count in (7, 17):
        home = base / f"n{count}"
        chosen = list(range(count))
        build_shadow_tree(home, chosen)
        expect_raises("shadow seeds invalid", lambda: independence.verify_shadow(home, write_summary(home, chosen)))

    home = base / "dupes"
    chosen = [1, 2, 3, 4, 5, 6, 7, 7]
    build_shadow_tree(home, chosen)
    expect_raises("shadow seeds invalid", lambda: independence.verify_shadow(home, write_summary(home, chosen)))

    home = base / "nonint"
    build_shadow_tree(home, [2, 3, 4, 5, 6, 7, 8])
    expect_raises(
        "invalid literal for int",
        lambda: independence.verify_shadow(home, write_summary(home, ["x", 2, 3, 4, 5, 6, 7, 8])),
    )

    # A string is iterable, so without a type check "12345678" is read as
    # the eight seeds 1..8 and the verifier certifies directories that the
    # summary never declared. The producer happens to guarantee a list, but
    # this is the defence-in-depth layer and must not inherit that. The GRU
    # twin of this verifier already rejects non-lists.
    home = base / "stringseeds"
    build_shadow_tree(home, list(range(1, 9)))
    expect_raises("must be a list", lambda: independence.verify_shadow(home, write_summary(home, "12345678")))

    # qualified must be the boolean True: a truthy 1 is not evidence.
    home = base / "evidence"
    build_shadow_tree(home, seeds)
    for evidence, qualified in (
        ("something-else", True),
        (SHADOW_EVIDENCE, False),
        (SHADOW_EVIDENCE, 1),
    ):
        expect_raises(
            "evidence mismatch",
            lambda e=evidence, q=qualified: independence.verify_shadow(home, write_summary(home, seeds, e, q)),
        )

    home = base / "overlap"
    build_shadow_tree(home, seeds)
    shared = home / "seed-7" / "dataset" / "domain-index.jsonl"
    shared.write_text(
        shared.read_text(encoding="utf-8") + json.dumps(row("qualification", "seed0-0")) + "\n",
        encoding="utf-8",
    )
    message = expect_raises("overlaps", lambda: independence.verify_shadow(home, write_summary(home, seeds)))
    assert "shadow seed" in message, message

    home = base / "missing"
    build_shadow_tree(home, seeds[:-1])
    expect_raises("domain index missing", lambda: independence.verify_shadow(home, write_summary(home, seeds)))


def check_cli(base: pathlib.Path) -> None:
    index = build_fresh_index(base / "fresh", (2, 3, 4))
    receipt = base / "fresh.json"
    done = run("--mode", "fresh", "--fresh-index", str(index), "--output", str(receipt))
    assert done.returncode == 0, done.stdout + done.stderr
    text = receipt.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert payload["qualified"] is True
    assert list(payload) == sorted(payload)
    assert json.loads(done.stdout)["qualified"] is True

    home = base / "shadow"
    seeds = list(range(8))
    build_shadow_tree(home, seeds)
    receipt = home / "shadow.json"
    done = run(
        "--mode", "shadow",
        "--shadow-root", str(home),
        "--shadow-summary", str(write_summary(home, seeds)),
        "--output", str(receipt),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(receipt.read_text(encoding="utf-8"))["wav_sha256_count"] == 16

    # Mode-specific inputs are required, not silently defaulted.
    expect("requires --fresh-index", run("--mode", "fresh", "--output", str(base / "a.json")))
    expect("requires root and summary", run("--mode", "shadow", "--shadow-root", str(home), "--output", str(base / "b.json")))
    expect("invalid choice", run("--mode", "other", "--output", str(base / "c.json")))
    expect("required", run("--mode", "fresh", "--fresh-index", str(index)))

    # An index with nothing in it must fail rather than certify zero rows.
    empty = write_index(base / "empty" / "dataset" / "domain-index.jsonl", [])
    expect(
        "no WAV identity evidence",
        run("--mode", "fresh", "--fresh-index", str(empty), "--output", str(base / "d.json")),
    )

    # A rejected run must leave no receipt behind: downstream steps read
    # that JSON, and a partial or leftover one looks like a certification.
    leaked = build_fresh_index(base / "leak", (2, 3, 4), overlap=("calibration-0", "test"))
    receipt = base / "leak.json"
    expect("overlaps", run("--mode", "fresh", "--fresh-index", str(leaked), "--output", str(receipt)))
    assert not receipt.exists()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        check_split_hashes(base / "reader")
        check_fresh(base / "fresh")
        check_shadow(base / "shadow")
        check_cli(base / "cli")
    print("test_verify_rnn_evaluation_independence: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
