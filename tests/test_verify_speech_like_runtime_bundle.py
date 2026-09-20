from __future__ import annotations

import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_speech_like_runtime_bundle as bundle  # noqa: E402

TOOL = ROOT / "tools" / "verify_speech_like_runtime_bundle.py"
ARCHIVE_ROOT = "speech-like-runtime"
REQUIRED = {"model": "model.kwm", "tokens": "tokens.txt", "lexicon": "lexicon.txt"}

# Three workflows call this verifier and nothing asserts its failure modes.
# It unpacks a downloaded runtime bundle, so the two things worth pinning are
# the archive contract (name, size, digest, required roles) and the member path
# check that keeps an archive from writing outside its root.


def default_members() -> dict:
    return {
        f"{ARCHIVE_ROOT}/model.kwm": b"model-bytes",
        f"{ARCHIVE_ROOT}/tokens.txt": b"tokens-bytes",
        f"{ARCHIVE_ROOT}/lexicon.txt": b"lexicon-bytes",
    }


def build_archive(path: pathlib.Path, members: dict) -> tuple[int, str]:
    with tarfile.open(path, "w:bz2") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    payload = path.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def reference(size: int, sha: str, candidate: str = "cand", asset: str = "bundle.tar.bz2") -> dict:
    return {
        "schema_version": 1,
        "evidence_class": bundle.REFERENCE_CLASS,
        "reference_candidates": [
            {
                "name": candidate,
                "runtime_asset_bundle": {
                    "asset": asset,
                    "expected_size_bytes": size,
                    "expected_sha256": sha,
                    "url": "https://example.invalid/bundle.tar.bz2",
                    "archive_root": ARCHIVE_ROOT,
                    "required_files": dict(REQUIRED),
                },
            }
        ],
    }


def fixture(base: pathlib.Path, name: str, members=None, asset: str = "bundle.tar.bz2"):
    home = base / name
    (home / "out").mkdir(parents=True, exist_ok=True)
    archive = home / asset
    size, sha = build_archive(archive, default_members() if members is None else members)
    home.mkdir(parents=True, exist_ok=True)
    return home, size, sha, archive


def write_reference(home: pathlib.Path, value: dict) -> pathlib.Path:
    path = home / "reference.json"
    path.write_bytes(json.dumps(value).encode())
    return path


def run(home: pathlib.Path, ref: pathlib.Path, archive: pathlib.Path, candidate: str = "cand"):
    return subprocess.run(
        [
            sys.executable, str(TOOL),
            "--reference", str(ref),
            "--candidate", candidate,
            "--archive", str(archive),
            "--output-dir", str(home / "out"),
            "--receipt", str(home / "receipt.json"),
        ],
        check=False, capture_output=True, text=True,
    )


def expect(needle: str, done) -> None:
    assert done.returncode != 0, done.stdout
    # Errors belong on stderr, like every other verifier in tools/ and like
    # the backend twin. Pinned per-stream, not accepting either, so that a
    # later move back to stdout fails here.
    assert "error:" in done.stderr, done.stderr
    assert needle in done.stderr, done.stderr


def check_member_paths() -> None:
    assert str(bundle.safe_member_path(f"{ARCHIVE_ROOT}/model.kwm", ARCHIVE_ROOT)) == (
        f"{ARCHIVE_ROOT}/model.kwm"
    )
    # A single dot is normalised away by PurePosixPath, so root/./x is root/x
    # and is harmless -- only the forms that escape or absolutise are unsafe.
    for unsafe in ("/etc/passwd", "..", "root/../evil", ""):
        try:
            bundle.safe_member_path(unsafe, ARCHIVE_ROOT)
        except ValueError as exc:
            assert "unsafe" in str(exc), f"{unsafe!r}: {exc}"
        else:
            raise AssertionError(f"{unsafe!r} must be rejected as unsafe")
    try:
        bundle.safe_member_path("other/model.kwm", ARCHIVE_ROOT)
    except ValueError as exc:
        assert "outside expected root" in str(exc), exc
    else:
        raise AssertionError("a member outside the archive root must be rejected")


def check_archive_contract(base: pathlib.Path) -> None:
    home, size, sha, archive = fixture(base, "good")
    ref = write_reference(home, reference(size, sha))
    done = run(home, ref, archive)
    assert done.returncode == 0, done.stderr
    receipt = home / "receipt.json"
    assert receipt.is_file(), "a verified bundle must write a receipt"
    value = json.loads(receipt.read_bytes().decode())
    assert value.get("evidence_class") == bundle.RECEIPT_CLASS, value
    assert (home / "out" / ARCHIVE_ROOT / "model.kwm").is_file(), sorted(
        (home / "out").rglob("*")
    )

    def fail(name: str, needle: str, mutate=None, **kwargs) -> None:
        home, size, sha, archive = fixture(base, name, **kwargs)
        value = reference(size, sha)
        if mutate is not None:
            mutate(value)
        expect(needle, run(home, write_reference(home, value), archive))

    def top(**changes):
        def apply(value: dict) -> None:
            value.update(changes)
        return apply

    def candidate_row(**changes):
        def apply(value: dict) -> None:
            value["reference_candidates"][0].update(changes)
        return apply

    def spec(**changes):
        def apply(value: dict) -> None:
            value["reference_candidates"][0]["runtime_asset_bundle"].update(changes)
        return apply

    fail("schema", "identity mismatch", top(schema_version=2))
    fail("class", "identity mismatch", top(evidence_class="other"))
    fail("candidates", "must be a list", top(reference_candidates="nope"))
    fail("bundle", "is missing runtime_asset_bundle", candidate_row(runtime_asset_bundle=None))
    fail("no-role", "must be a non-empty object", spec(required_files={}))
    fail("roles", "must contain model/tokens/lexicon", spec(required_files={"model": "m"}))
    fail("size", "size mismatch", spec(expected_size_bytes=1))
    fail("sha", "sha256 mismatch", spec(expected_sha256="0" * 64))
    fail("name", "filename mismatch", spec(asset="other.tar.bz2"))

    # A candidate that matches twice is as unusable as one that never matches.
    home, size, sha, archive = fixture(base, "duplicate")
    value = reference(size, sha)
    value["reference_candidates"].append(value["reference_candidates"][0])
    expect(
        "must match exactly once",
        run(home, write_reference(home, value), archive),
    )
    home, size, sha, archive = fixture(base, "absent")
    expect("must match exactly once", run(home, write_reference(home, reference(size, sha)), archive, candidate="nope"))

    # Archive shape: missing, empty, and short a required role.
    home, size, sha, archive = fixture(base, "missing-archive")
    archive.unlink()
    expect("runtime archive is missing", run(home, write_reference(home, reference(size, sha)), archive))

    home, size, sha, archive = fixture(base, "empty-archive", members={})
    expect("runtime archive is empty", run(home, write_reference(home, reference(size, sha)), archive))

    home, size, sha, archive = fixture(
        base,
        "short-archive",
        members={f"{ARCHIVE_ROOT}/model.kwm": b"m", f"{ARCHIVE_ROOT}/tokens.txt": b"t"},
    )
    short = run(home, write_reference(home, reference(size, sha)), archive)
    assert short.returncode != 0, short.stdout

    # The reason this file exists: an archive must not write outside its root.
    for label, member in (
        ("parent", "../evil.kwm"),
        ("absolute", "/etc/passwd"),
        ("nested", f"{ARCHIVE_ROOT}/../../evil.kwm"),
    ):
        home, size, sha, archive = fixture(
            base,
            f"traversal-{label}",
            members={**default_members(), member: b"evil"},
        )
        done = run(home, write_reference(home, reference(size, sha)), archive)
        assert done.returncode != 0, f"{label}: {done.stdout}"
        output = done.stdout + done.stderr
        assert "unsafe" in output or "outside expected root" in output, output

    # Extracting into a dirty directory would mix two bundles silently.
    home, size, sha, archive = fixture(base, "dirty-out")
    (home / "out" / "stale.txt").write_text("stale\n", encoding="utf-8")
    expect("output-dir must be empty", run(home, write_reference(home, reference(size, sha)), archive))


def main() -> int:
    check_member_paths()
    with tempfile.TemporaryDirectory() as td:
        check_archive_contract(pathlib.Path(td))
    print("test_verify_speech_like_runtime_bundle: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
