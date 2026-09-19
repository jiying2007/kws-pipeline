from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import corpus_identity  # noqa: E402

TOOL = ROOT / "tools" / "verify_corpus_identity.py"
SAMPLE_RATE = 16000

# Retained corpus identity: no workflow called this verifier and no test
# exercised it. It reads real WAV files, so the fixtures here are real files.


def write_wav(path: pathlib.Path, samples: list) -> bytes:
    payload = b"".join(struct.pack("<h", sample) for sample in samples)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(payload)
    return payload


def measure(path: pathlib.Path, payload: bytes) -> dict:
    """Independent measurement: the verifier's own numbers must match these."""
    data = path.read_bytes()
    return {
        "file_sha256": hashlib.sha256(data).hexdigest(),
        "pcm_sha256": hashlib.sha256(payload).hexdigest(),
        "frames": len(payload) // 2,
    }


def row_for(path: pathlib.Path, payload: bytes, recording: str) -> dict:
    measured = measure(path, payload)
    return {
        "recording": recording,
        "path": str(path),
        "file_sha256": measured["file_sha256"],
        "pcm_sha256": measured["pcm_sha256"],
        "frames": measured["frames"],
    }


def manifest(rows: list, digest: str | None = None) -> dict:
    return {
        "schema_version": 1,
        "recordings": rows,
        "corpus_sha256": digest if digest is not None else corpus_identity.corpus_digest(rows),
    }


def run(root: pathlib.Path, value: dict) -> subprocess.CompletedProcess:
    path = root / "corpus-identity.json"
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_bytes(json.dumps(value).encode())
    return subprocess.run(
        [sys.executable, str(TOOL), "--manifest", str(path)],
        check=False, capture_output=True, text=True,
    )


def expect(needle: str, done: subprocess.CompletedProcess) -> None:
    assert done.returncode != 0, done.stdout
    assert needle in done.stderr, f"expected {needle!r}, got: {done.stderr}"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)

        first = base / "a.wav"
        second = base / "b.wav"
        payload_a = write_wav(first, [0, 100, -100, 200])
        payload_b = write_wav(second, [1, 2, 3, 4, 5])
        rows = [row_for(first, payload_a, "a"), row_for(second, payload_b, "b")]

        # The measured values are computed here, not copied from the verifier:
        # a verifier that measured nothing would not match them.
        assert rows[0]["frames"] == 4, rows[0]
        assert rows[1]["frames"] == 5, rows[1]

        done = run(base, manifest(rows))
        assert done.returncode == 0, done.stderr
        assert "verified corpus identity" in done.stdout, done.stdout

        expect("schema_version must be 1", run(base, {**manifest(rows), "schema_version": 2}))
        expect("must be non-empty", run(base, {**manifest(rows), "recordings": []}))
        expect("must be non-empty", run(base, {**manifest(rows), "recordings": "nope"}))
        expect(
            "recordings[0] must be an object",
            run(base, {**manifest(rows), "recordings": ["nope", rows[1]]}),
        )
        expect(
            "corpus_sha256 does not match",
            run(base, manifest(rows, digest="0" * 64)),
        )

        # Each measured field is checked on its own, not just in aggregate.
        for field in ("file_sha256", "pcm_sha256", "frames"):
            tampered = json.loads(json.dumps(rows))
            tampered[0][field] = "0" * 64 if field.endswith("sha256") else 99
            expect(
                f"recordings[0].{field} does not match",
                run(base, manifest(tampered)),
            )

        # The drift this exists for: the bytes change, the manifest does not.
        write_wav(first, [0, 100, -100, 201])
        expect("recordings[0].file_sha256 does not match", run(base, manifest(rows)))

        # A recording that no longer exists must fail, not be skipped.
        first.unlink()
        gone = run(base, manifest(rows))
        assert gone.returncode == 2, gone.stderr
        assert "error:" in gone.stderr, gone.stderr

        # Malformed JSON is an error, not a silent pass.
        broken = run(base, "{not json")
        assert broken.returncode == 2, broken.stderr

    print("test_verify_corpus_identity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
