from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import verify_speech_like_backend_bundle as backend  # noqa: E402

TOOL = ROOT / "tools" / "verify_speech_like_backend_bundle.py"
REFERENCE_CLASS = "speech-like-provider-reference-v1"
RECEIPT_CLASS = "speech-like-sherpa-backend-bundle-v1"
ARCHIVE_ROOT = "sherpa"
PLATFORM = "linux-x64"
EXECUTABLE = f"{ARCHIVE_ROOT}/bin/sherpa-onnx-offline-tts"
LIBRARY = f"{ARCHIVE_ROOT}/lib/libonnxruntime.so"
SYMLINK = f"{ARCHIVE_ROOT}/lib/libonnxruntime.so.1"

# This verifier is the one thing standing between a downloaded tarball and
# an extracted tree: it decides both what may be unpacked and where it may
# land. Two failure modes matter more than the rest: a member path or a
# symlink target that escapes the archive root, and a receipt that asserts
# safe_archive_verified without the archive having been re-checked. Every
# fixture is a real tar.bz2, because the checks under test are about tar
# member metadata that only a real archive carries.


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def tar_info(name: str, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.mode = mode
    return info


def file_member(name: str, payload: bytes, mode: int = 0o644):
    info = tar_info(name, mode)
    info.size = len(payload)
    return info, io.BytesIO(payload)


def dir_member(name: str, mode: int = 0o755):
    info = tar_info(name, mode)
    info.type = tarfile.DIRTYPE
    info.size = 0
    return info, None


def sym_member(name: str, target: str):
    info = tar_info(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    info.size = 0
    return info, None


def fifo_member(name: str):
    info = tar_info(name)
    info.type = tarfile.FIFOTYPE
    info.size = 0
    return info, None


def good_members():
    return [
        dir_member(ARCHIVE_ROOT),
        dir_member(f"{ARCHIVE_ROOT}/bin"),
        file_member(EXECUTABLE, b"#!/bin/sh\nexit 0\n", 0o755),
        dir_member(f"{ARCHIVE_ROOT}/lib"),
        file_member(LIBRARY, b"ELF-lib-a"),
        sym_member(SYMLINK, "libonnxruntime.so"),
    ]


def write_archive(path: pathlib.Path, members) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:bz2") as archive:
        for info, payload in members:
            archive.addfile(info, payload)
    return path


def good_archive(base: pathlib.Path, name: str = "sherpa.tar.bz2") -> pathlib.Path:
    return write_archive(base / name, good_members())


def reference_payload(asset: str, size: int, sha: str) -> dict:
    return {
        "schema_version": 1,
        "evidence_class": REFERENCE_CLASS,
        "backend_bootstrap": {
            PLATFORM: {
                "asset": asset,
                "expected_size_bytes": size,
                "expected_sha256": sha,
                "url": "https://example.invalid/sherpa.tar.bz2",
                "archive_root": ARCHIVE_ROOT,
                "backend_executable": "bin/sherpa-onnx-offline-tts",
                "lib_dir": "lib",
            }
        },
    }


def write_reference(base: pathlib.Path, payload: dict) -> pathlib.Path:
    # Deliberately not "reference.json": the pristine reference lives there,
    # and callers re-read it to build each mutation. Writing mutations over
    # it would poison every subsequent case in the same directory.
    base.mkdir(parents=True, exist_ok=True)
    path = base / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def matching_reference(base: pathlib.Path, archive: pathlib.Path) -> pathlib.Path:
    base.mkdir(parents=True, exist_ok=True)
    payload = reference_payload(
        archive.name, archive.stat().st_size, digest(archive.read_bytes())
    )
    path = base / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def set_contract(payload: dict, key: str, value) -> dict:
    payload["backend_bootstrap"][PLATFORM][key] = value
    return payload


def drop_contract(payload: dict, key: str) -> dict:
    payload["backend_bootstrap"][PLATFORM].pop(key, None)
    return payload


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
    # Unlike its runtime twin, this verifier reports errors on stderr. Pin
    # the stream: a later switch to stdout would be invisible otherwise,
    # and the two twins would then disagree about where failures go.
    assert "error:" in done.stderr, f"expected the error on stderr, got: {done.stderr!r}"
    assert needle in done.stderr, f"expected {needle!r}, got: {done.stderr}"


def check_contract(base: pathlib.Path) -> None:
    archive = good_archive(base)
    reference = matching_reference(base, archive)
    contract = backend.backend_contract(reference, PLATFORM)
    assert contract["asset"] == archive.name

    # This tool reads the version with int(), so a string "1" is accepted.
    # Not pinned as a failure: both twins do it, and a producer emitting a
    # quoted version is not the threat this verifier guards against.
    payload = json.loads(reference.read_text(encoding="utf-8"))
    for version in (0, 2, "2"):
        payload["schema_version"] = version
        expect_raises("provider reference identity mismatch", lambda: backend.backend_contract(write_reference(base, payload), PLATFORM))
    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["evidence_class"] = "something-else"
    expect_raises("provider reference identity mismatch", lambda: backend.backend_contract(write_reference(base, payload), PLATFORM))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    del payload["backend_bootstrap"]
    expect_raises(f"backend bootstrap is not declared for {PLATFORM}", lambda: backend.backend_contract(write_reference(base, payload), PLATFORM))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["backend_bootstrap"] = []
    expect_raises(f"backend bootstrap is not declared for {PLATFORM}", lambda: backend.backend_contract(write_reference(base, payload), PLATFORM))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    expect_raises(f"backend bootstrap is not declared for other", lambda: backend.backend_contract(write_reference(base, payload), "other"))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    payload["backend_bootstrap"][PLATFORM] = "nope"
    expect_raises("backend bootstrap contract must be an object", lambda: backend.backend_contract(write_reference(base, payload), PLATFORM))


def check_member_paths() -> None:
    assert backend.safe_member_path(f"{ARCHIVE_ROOT}/bin/tool", ARCHIVE_ROOT).as_posix() == f"{ARCHIVE_ROOT}/bin/tool"

    # No "sherpa/./bin/tool" case: PurePosixPath normalises a single dot
    # away, so it really is the same path as "sherpa/bin/tool". Listing it
    # as unsafe would be asserting something false.
    for name in ("/etc/passwd", "../evil", f"{ARCHIVE_ROOT}/../../evil"):
        expect_raises("member path is unsafe", lambda n=name: backend.safe_member_path(n, ARCHIVE_ROOT))
    expect_raises("outside expected root", lambda: backend.safe_member_path("other/bin/tool", ARCHIVE_ROOT))


def check_symlinks() -> None:
    member = pathlib.PurePosixPath(f"{ARCHIVE_ROOT}/lib/libonnxruntime.so.1")
    assert backend.validate_symlink(member, "libonnxruntime.so", ARCHIVE_ROOT) == "libonnxruntime.so"
    # Still inside the archive root, so still fine.
    assert backend.validate_symlink(member, "../bin/tool", ARCHIVE_ROOT) == "../bin/tool"

    expect_raises("symlink target is unsafe", lambda: backend.validate_symlink(member, "/etc/passwd", ARCHIVE_ROOT))
    expect_raises("symlink escaped root", lambda: backend.validate_symlink(member, "../../../etc/passwd", ARCHIVE_ROOT))
    expect_raises("symlink escaped root", lambda: backend.validate_symlink(pathlib.PurePosixPath(f"{ARCHIVE_ROOT}/x"), "../evil", ARCHIVE_ROOT))


def check_inspect(base: pathlib.Path) -> None:
    archive = good_archive(base)
    reference = matching_reference(base, archive)
    receipt = backend.inspect_verified_archive(
        reference_path=reference, platform_key=PLATFORM, archive_path=archive
    )
    assert receipt["schema_version"] == 1
    assert receipt["evidence_class"] == RECEIPT_CLASS
    assert receipt["platform"] == PLATFORM
    assert receipt["reference_sha256"] == digest(reference.read_bytes())
    assert receipt["archive"]["name"] == archive.name
    assert receipt["archive"]["size_bytes"] == archive.stat().st_size
    assert receipt["archive"]["sha256"] == digest(archive.read_bytes())
    assert receipt["archive"]["source_url"].startswith("https://")
    assert receipt["archive_root"] == ARCHIVE_ROOT
    assert receipt["backend_executable"]["path"] == EXECUTABLE
    assert receipt["backend_executable"]["size_bytes"] == len(b"#!/bin/sh\nexit 0\n")
    assert receipt["backend_executable"]["sha256"] == digest(b"#!/bin/sh\nexit 0\n")
    assert receipt["backend_executable"]["mode"] == 0o755
    assert receipt["lib_dir"] == f"{ARCHIVE_ROOT}/lib"
    assert [item["path"] for item in receipt["libraries"]] == [LIBRARY]
    assert receipt["libraries"][0]["sha256"] == digest(b"ELF-lib-a")
    assert receipt["library_symlinks"] == [{"path": SYMLINK, "target": "libonnxruntime.so"}]
    assert receipt["safe_archive_verified"] is True

    # Contract fields: present, and non-empty text.
    payload = json.loads(reference.read_text(encoding="utf-8"))
    for key in ("asset", "url", "archive_root", "backend_executable", "lib_dir", "expected_sha256"):
        expect_raises("must be non-empty text", lambda k=key: backend.inspect_verified_archive(
            reference_path=write_reference(base, drop_contract(json.loads(reference.read_text(encoding="utf-8")), k)),
            platform_key=PLATFORM,
            archive_path=archive,
        ))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    set_contract(payload, "expected_size_bytes", 0)
    expect_raises("size/sha contract is invalid", lambda: backend.inspect_verified_archive(
        reference_path=write_reference(base, payload), platform_key=PLATFORM, archive_path=archive
    ))
    payload = json.loads(reference.read_text(encoding="utf-8"))
    set_contract(payload, "expected_sha256", "0" * 63)
    expect_raises("size/sha contract is invalid", lambda: backend.inspect_verified_archive(
        reference_path=write_reference(base, payload), platform_key=PLATFORM, archive_path=archive
    ))

    # Archive identity: name, presence, size, digest.
    payload = json.loads(reference.read_text(encoding="utf-8"))
    set_contract(payload, "asset", "other.tar.bz2")
    expect_raises("archive filename mismatch", lambda: backend.inspect_verified_archive(
        reference_path=write_reference(base, payload), platform_key=PLATFORM, archive_path=archive
    ))

    # Same filename, different directory: the name check runs first, so an
    # absent archive only reaches "is missing" if its name matches.
    expect_raises("backend archive is missing", lambda: backend.inspect_verified_archive(
        reference_path=reference, platform_key=PLATFORM, archive_path=base / "gone" / archive.name
    ))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    set_contract(payload, "expected_size_bytes", archive.stat().st_size + 1)
    expect_raises("archive size mismatch", lambda: backend.inspect_verified_archive(
        reference_path=write_reference(base, payload), platform_key=PLATFORM, archive_path=archive
    ))

    payload = json.loads(reference.read_text(encoding="utf-8"))
    set_contract(payload, "expected_sha256", "0" * 64)
    expect_raises("archive sha256 mismatch", lambda: backend.inspect_verified_archive(
        reference_path=write_reference(base, payload), platform_key=PLATFORM, archive_path=archive
    ))

    # Archive contents.
    def inspect_with(members, needle: str) -> None:
        home = base / f"case-{abs(hash(str(members))) % 10**8}"
        home.mkdir(parents=True, exist_ok=True)
        packed = write_archive(home / "sherpa.tar.bz2", members)
        expect_raises(needle, lambda: backend.inspect_verified_archive(
            reference_path=matching_reference(home, packed), platform_key=PLATFORM, archive_path=packed
        ))

    inspect_with([], "backend archive is empty")
    inspect_with([dir_member(ARCHIVE_ROOT)], "backend archive is missing executable")
    inspect_with(
        [dir_member(ARCHIVE_ROOT), dir_member(f"{ARCHIVE_ROOT}/bin"), file_member(EXECUTABLE, b"x")],
        "no regular files under lib_dir",
    )
    inspect_with(
        [dir_member(ARCHIVE_ROOT), dir_member(f"{ARCHIVE_ROOT}/bin"), file_member(EXECUTABLE, b"x"), fifo_member(f"{ARCHIVE_ROOT}/lib/pipe")],
        "unsupported member type",
    )
    inspect_with([dir_member(ARCHIVE_ROOT), file_member(EXECUTABLE, b"x"), file_member(EXECUTABLE, b"y")], "duplicate backend archive member")
    inspect_with(
        [dir_member(ARCHIVE_ROOT), file_member(EXECUTABLE, b"x"), file_member("../evil", b"x")],
        "member path is unsafe",
    )
    inspect_with(
        [
            dir_member(ARCHIVE_ROOT),
            dir_member(f"{ARCHIVE_ROOT}/bin"),
            file_member(EXECUTABLE, b"x"),
            file_member(LIBRARY, b"y"),
            sym_member(f"{ARCHIVE_ROOT}/lib/escape", "../../../etc/passwd"),
        ],
        "symlink escaped root",
    )
    inspect_with(
        [
            dir_member(ARCHIVE_ROOT),
            dir_member(f"{ARCHIVE_ROOT}/bin"),
            file_member(EXECUTABLE, b"x"),
            file_member(LIBRARY, b"y"),
            sym_member(f"{ARCHIVE_ROOT}/lib/absolute", "/etc/passwd"),
        ],
        "symlink target is unsafe",
    )


def check_extract_and_validate(base: pathlib.Path) -> None:
    archive = good_archive(base)
    reference = matching_reference(base, archive)
    out = base / "out"
    receipt = backend.extract_verified_bundle(
        reference_path=reference, platform_key=PLATFORM, archive_path=archive, output_dir=out
    )
    assert receipt["safe_archive_verified"] is True
    executable = out / EXECUTABLE
    assert executable.is_file()
    assert oct(executable.stat().st_mode & 0o777) == oct(0o755)
    assert (out / LIBRARY).read_bytes() == b"ELF-lib-a"
    assert os.readlink(out / SYMLINK) == "libonnxruntime.so"

    # A second run into a populated directory must refuse rather than merge.
    expect_raises("output-dir must be empty", lambda: backend.extract_verified_bundle(
        reference_path=reference, platform_key=PLATFORM, archive_path=archive, output_dir=out
    ))

    # Re-validating the extracted tree catches drift after the fact.
    backend.validate_extracted_bundle(receipt=receipt, output_dir=out)

    (out / LIBRARY).unlink()
    expect_raises("library file set mismatch", lambda: backend.validate_extracted_bundle(receipt=receipt, output_dir=out))

    # Same length on purpose: a different length trips the size check first
    # and the sha256 branch would never run.
    (out / LIBRARY).write_bytes(b"ELF-lib-b")
    expect_raises("library sha256 mismatch", lambda: backend.validate_extracted_bundle(receipt=receipt, output_dir=out))

    (out / LIBRARY).unlink()
    (out / LIBRARY).symlink_to("libonnxruntime.so")
    expect_raises("library file set mismatch", lambda: backend.validate_extracted_bundle(receipt=receipt, output_dir=out))
    (out / LIBRARY).unlink()
    (out / LIBRARY).write_bytes(b"ELF-lib-a")

    (out / SYMLINK).unlink()
    (out / SYMLINK).symlink_to("something-else")
    expect_raises("symlink set/targets mismatch", lambda: backend.validate_extracted_bundle(receipt=receipt, output_dir=out))
    (out / SYMLINK).unlink()
    (out / SYMLINK).symlink_to("libonnxruntime.so")

    (out / EXECUTABLE).unlink()
    expect_raises("extracted backend executable is missing", lambda: backend.validate_extracted_bundle(receipt=receipt, output_dir=out))

    # A traversal member is refused before anything is written.
    home = base / "traversal"
    home.mkdir(parents=True, exist_ok=True)
    packed = write_archive(
        home / "sherpa.tar.bz2",
        [dir_member(ARCHIVE_ROOT), file_member(EXECUTABLE, b"x"), file_member(LIBRARY, b"y"), file_member("../escaped", b"pwned")],
    )
    expect_raises("member path is unsafe", lambda: backend.extract_verified_bundle(
        reference_path=matching_reference(home, packed), platform_key=PLATFORM, archive_path=packed, output_dir=home / "out"
    ))
    assert not (home / "escaped").exists()


def check_cli(base: pathlib.Path) -> None:
    archive = good_archive(base)
    reference = matching_reference(base, archive)
    receipt_path = base / "receipt.json"
    out = base / "out"

    done = run(
        "--reference", str(reference),
        "--platform", PLATFORM,
        "--archive", str(archive),
        "--output-dir", str(out),
        "--receipt", str(receipt_path),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    text = receipt_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    payload = json.loads(text)
    assert list(payload) == sorted(payload)
    assert payload["evidence_class"] == RECEIPT_CLASS
    assert f"platform={PLATFORM}" in done.stdout
    assert "verify_only=False" in done.stdout

    # verify-only re-checks the archive and the extracted tree against the
    # receipt. Both modes are used by bootstrap_speech_like_stage_a.py.
    done = run(
        "--reference", str(reference),
        "--platform", PLATFORM,
        "--archive", str(archive),
        "--output-dir", str(out),
        "--receipt", str(receipt_path),
        "--verify-only",
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "verify_only=True" in done.stdout

    tampered = json.loads(text)
    tampered["libraries"][0]["sha256"] = "0" * 64
    (base / "tampered.json").write_text(json.dumps(tampered), encoding="utf-8")
    expect("receipt does not match verified archive", run(
        "--reference", str(reference),
        "--platform", PLATFORM,
        "--archive", str(archive),
        "--output-dir", str(out),
        "--receipt", str(base / "tampered.json"),
        "--verify-only",
    ))

    wrong_class = json.loads(text)
    wrong_class["evidence_class"] = "something-else"
    (base / "wrong-class.json").write_text(json.dumps(wrong_class), encoding="utf-8")
    expect("backend receipt identity mismatch", run(
        "--reference", str(reference),
        "--platform", PLATFORM,
        "--archive", str(archive),
        "--output-dir", str(out),
        "--receipt", str(base / "wrong-class.json"),
        "--verify-only",
    ))

    expect("backend receipt is missing", run(
        "--reference", str(reference),
        "--platform", PLATFORM,
        "--archive", str(archive),
        "--output-dir", str(out),
        "--receipt", str(base / "absent.json"),
        "--verify-only",
    ))

    expect("required", run(
        "--reference", str(reference),
        "--platform", PLATFORM,
        "--archive", str(archive),
        "--output-dir", str(out),
    ))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        check_contract(base / "contract")
        check_member_paths()
        check_symlinks()
        check_inspect(base / "inspect")
        check_extract_and_validate(base / "extract")
        check_cli(base / "cli")
    print("test_verify_speech_like_backend_bundle: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
