from __future__ import annotations

import pathlib
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import corpus_digest, inspect_pcm16_wav  # noqa: E402


def write_wav(path: pathlib.Path, samples: list[int]) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        payload = bytearray()
        for sample in samples:
            payload += int(sample).to_bytes(2, "little", signed=True)
        writer.writeframes(bytes(payload))


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        first = root / "a.wav"
        second = root / "b.wav"
        write_wav(first, [0, 100, -100, 7])
        write_wav(second, [0, 100, -100, 7])
        a = inspect_pcm16_wav(first)
        b = inspect_pcm16_wav(second)
        assert a["pcm_sha256"] == b["pcm_sha256"]
        rows = [
            {
                "recording": "a",
                "path": str(first),
                **a,
                "speaker_id": "s1",
            }
        ]
        digest = corpus_digest(rows)
        assert len(digest) == 64
        write_wav(second, [0, 100, -101, 7])
        c = inspect_pcm16_wav(second)
        assert c["pcm_sha256"] != a["pcm_sha256"]
    print("test_corpus_identity: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
