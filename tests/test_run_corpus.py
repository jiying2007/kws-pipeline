#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
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

        posterior_dump = root / "posterior_dump.py"
        decoder_replay = root / "decoder_replay.py"
        posterior_cache = root / "posterior-cache"
        dump_count = root / "dump-count.txt"

        posterior_dump.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, json, os, pathlib, sys\n"
            "model=pathlib.Path(sys.argv[1]); audio=pathlib.Path(sys.argv[2]); trace=pathlib.Path(sys.argv[3])\n"
            "model_sha=hashlib.sha256(model.read_bytes()).hexdigest()\n"
            "audio_sha=hashlib.sha256(audio.read_bytes()).hexdigest()\n"
            "trace.write_bytes(('trace-v1:'+model_sha+':'+audio_sha).encode())\n"
            "trace_sha=hashlib.sha256(trace.read_bytes()).hexdigest()\n"
            "count=pathlib.Path(os.environ['DUMP_COUNT_FILE'])\n"
            "count.write_text(count.read_text()+'1\\n' if count.exists() else '1\\n')\n"
            "print(json.dumps({'schema_version':1,'evidence_class':'kws-posterior-trace-v1',"
            "'model_sha256':model_sha,'trace_sha256':trace_sha,'frames':3,'vocab_size':4}))\n",
            encoding="utf-8",
        )
        decoder_replay.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "pack=pathlib.Path(sys.argv[2]).read_bytes()\n"
            "keyword=2 if pack.endswith(b'v2') else 1\n"
            "print(json.dumps({'recording':sys.argv[4],'keyword_id':keyword,"
            "'time_s':1.0,'confidence':0.8}))\n",
            encoding="utf-8",
        )
        posterior_dump.chmod(posterior_dump.stat().st_mode | stat.S_IXUSR)
        decoder_replay.chmod(decoder_replay.stat().st_mode | stat.S_IXUSR)
        env = dict(os.environ)
        env["DUMP_COUNT_FILE"] = str(dump_count)

        cached_provenance = root / "cached-provenance.json"
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
                "--provenance", str(cached_provenance),
                "--posterior-dump", str(posterior_dump),
                "--decoder-replay", str(decoder_replay),
                "--posterior-cache", str(posterior_cache),
            ],
            env=env,
        )
        cached_first = json.loads(cached_provenance.read_text(encoding="utf-8"))
        cached_rows = [
            json.loads(line)
            for line in detections.read_text(encoding="utf-8").splitlines()
        ]
        assert cached_rows[0]["keyword_id"] == 1
        assert cached_first["evaluation_mode"] == "posterior-replay-cache-v1"
        assert cached_first["posterior_cache_hits"] == 0
        assert cached_first["posterior_cache_misses"] == 1
        assert len(cached_first["posterior_traces"]) == 1
        first_trace_sha = cached_first["posterior_traces"][0]["trace_sha256"]
        assert dump_count.read_text(encoding="utf-8").splitlines() == ["1"]

        keywords.write_bytes(b"pack-fixture-v2")
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
                "--provenance", str(cached_provenance),
                "--posterior-dump", str(posterior_dump),
                "--decoder-replay", str(decoder_replay),
                "--posterior-cache", str(posterior_cache),
                "--decoder-state-retention", "0.91",
                "--decoder-refractory-ms", "250",
            ],
            env=env,
        )
        cached_second = json.loads(cached_provenance.read_text(encoding="utf-8"))
        cached_rows = [
            json.loads(line)
            for line in detections.read_text(encoding="utf-8").splitlines()
        ]
        assert cached_rows[0]["keyword_id"] == 2
        assert cached_second["posterior_cache_hits"] == 1
        assert cached_second["posterior_cache_misses"] == 0
        assert cached_second["decoder_replay_overrides"] == {
            "state_retention": 0.91,
            "refractory_ms": 250,
        }
        assert cached_second["posterior_traces"][0]["trace_sha256"] == first_trace_sha
        assert dump_count.read_text(encoding="utf-8").splitlines() == ["1"]

        override_without_replay = subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval" / "run_corpus.py"),
                "--runner", str(runner),
                "--model", str(model),
                "--keywords", str(keywords),
                "--references", str(references),
                "--audio-root", str(root),
                "--detections", str(detections),
                "--decoder-state-retention", "0.91",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert override_without_replay.returncode != 0
        assert "decoder replay overrides require posterior replay cache mode" in (
            override_without_replay.stderr + override_without_replay.stdout
        )

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
