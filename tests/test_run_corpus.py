#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import stat
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        audio.write_bytes(b"wav-fixture")
        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": "audio.wav",
                    "duration_s": 2.0,
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
                "--runner",
                str(runner),
                "--model",
                str(model),
                "--keywords",
                str(keywords),
                "--references",
                str(references),
                "--audio-root",
                str(root),
                "--detections",
                str(detections),
                "--provenance",
                str(provenance),
            ]
        )

        detection_rows = [
            json.loads(line)
            for line in detections.read_text(encoding="utf-8").splitlines()
        ]
        assert detection_rows == [
            {
                "recording": "room-1",
                "keyword_id": 1,
                "time_s": 1.0,
                "confidence": 0.8,
            }
        ]
        result = json.loads(provenance.read_text(encoding="utf-8"))
        assert result["schema_version"] == 1
        assert result["runner_sha256"] == sha256_file(runner)
        assert result["model_sha256"] == sha256_file(model)
        assert result["keyword_pack_sha256"] == sha256_file(keywords)
        assert result["references_sha256"] == sha256_file(references)
        assert result["detections_sha256"] == sha256_file(detections)
        assert result["recordings"] == 1
        assert result["detections"] == 1

    print("test_run_corpus: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
