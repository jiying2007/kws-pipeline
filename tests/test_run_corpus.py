#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: pathlib.Path, seconds: int = 2) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 16000 * seconds)


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        runner = root / "runner.py"
        model = root / "model.kwm"
        keywords = root / "keywords.kwk"
        references = root / "references.jsonl"
        audio = root / "audio.wav"
        detections = root / "detections.jsonl"
        provenance = root / "provenance.json"
        corpus = root / "corpus.json"

        runner.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "print(json.dumps({'recording': sys.argv[4], 'keyword_id': 1, "
            "'time_s': 1.0, 'confidence': 0.8}))\n",
            encoding="utf-8",
        )
        runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
        model.write_bytes(b"model-fixture")
        keywords.write_bytes(b"pack-fixture")
        write_wav(audio)
        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": "audio.wav",
                    "duration_s": 2.0,
                    "speaker_id": "spk-1",
                    "session_id": "session-1",
                    "source_id": "source-1",
                    "expected": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--provenance", str(provenance),
                "--corpus-identity", str(corpus),
            ]
        )

        detection_rows = [json.loads(line) for line in detections.read_text(encoding="utf-8").splitlines()]
        assert detection_rows == [
            {"recording": "room-1", "keyword_id": 1, "time_s": 1.0, "confidence": 0.8}
        ]
        result = json.loads(provenance.read_text(encoding="utf-8"))
        identity = json.loads(corpus.read_text(encoding="utf-8"))
        assert result["schema_version"] == 2
        assert result["runner_sha256"] == sha256_file(runner)
        assert result["model_sha256"] == sha256_file(model)
        assert result["keyword_pack_sha256"] == sha256_file(keywords)
        assert result["references_sha256"] == sha256_file(references)
        assert result["detections_sha256"] == sha256_file(detections)
        assert result["audio_corpus_sha256"] == identity["corpus_sha256"]
        assert result["audio_files"] == identity["recordings"]
        assert result["recordings"] == 1
        assert result["detections"] == 1

        original = audio.read_bytes()
        audio.write_bytes(original[:-2] + b"\x01\x00")
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--provenance", str(provenance),
            ]
        )
        changed = json.loads(provenance.read_text(encoding="utf-8"))
        assert changed["audio_corpus_sha256"] != result["audio_corpus_sha256"]

    print("test_run_corpus: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
