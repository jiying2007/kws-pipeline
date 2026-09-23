#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write_wav(path: pathlib.Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 32000)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="decoder-retention-curve-") as td:
        root = pathlib.Path(td)
        runner = root / "runner.py"
        dump = root / "posterior_dump.py"
        replay = root / "decoder_replay.py"
        model = root / "model.kwm"
        keywords = root / "keywords.kwk"
        config = root / "config.json"
        audio = root / "audio.wav"
        refs = root / "refs.jsonl"
        cache = root / "cache"
        output = root / "curve.json"
        work = root / "work"
        dump_count = root / "dump-count.txt"

        runner.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        runner.chmod(runner.stat().st_mode | stat.S_IXUSR)
        model.write_bytes(b"model-fixture")
        keywords.write_bytes(b"keyword-pack-fixture")
        write_wav(audio)
        config.write_text(
            json.dumps(
                {
                    "domain_gates": {
                        "max_frr": 0.0,
                        "max_far_per_hour": 0.0,
                        "max_p95_latency_ms": 800.0,
                        "max_far_frr": 0.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        refs.write_text(
            json.dumps(
                {
                    "recording": "negative",
                    "path": str(audio),
                    "duration_s": 2.0,
                    "speaker_id": "spk",
                    "session_id": "session",
                    "source_id": "source",
                    "expected": [],
                    "domain": {
                        "distance_band": "1m",
                        "distance_m": 1.0,
                        "azimuth_deg": 0.0,
                        "rt60_s": 0.2,
                        "snr_db": 20.0,
                        "noise_profile": "fixture",
                        "playback_sir_db": None,
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        dump.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib,json,os,pathlib,sys\n"
            "model=pathlib.Path(sys.argv[1]); audio=pathlib.Path(sys.argv[2]); trace=pathlib.Path(sys.argv[3])\n"
            "model_sha=hashlib.sha256(model.read_bytes()).hexdigest()\n"
            "audio_sha=hashlib.sha256(audio.read_bytes()).hexdigest()\n"
            "trace.write_bytes(('trace:'+model_sha+':'+audio_sha).encode())\n"
            "trace_sha=hashlib.sha256(trace.read_bytes()).hexdigest()\n"
            "count=pathlib.Path(os.environ['DUMP_COUNT_FILE'])\n"
            "count.write_text(count.read_text()+'1\\n' if count.exists() else '1\\n')\n"
            "print(json.dumps({'schema_version':1,'evidence_class':'kws-posterior-trace-v1',"
            "'model_sha256':model_sha,'trace_sha256':trace_sha,'frames':3,'vocab_size':4}))\n",
            encoding="utf-8",
        )
        dump.chmod(dump.stat().st_mode | stat.S_IXUSR)

        replay.write_text(
            "#!/usr/bin/env python3\n"
            "import json,sys\n"
            "idx=sys.argv.index('--state-retention')\n"
            "ret=float(sys.argv[idx+1])\n"
            "if ret < 0.95:\n"
            " print(json.dumps({'recording':sys.argv[4],'keyword_id':1,'time_s':1.0,'confidence':0.8}))\n",
            encoding="utf-8",
        )
        replay.chmod(replay.stat().st_mode | stat.S_IXUSR)

        env = dict(os.environ)
        env["DUMP_COUNT_FILE"] = str(dump_count)
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "diagnose_decoder_retention_curve.py"),
                "--runner",
                str(runner),
                "--model",
                str(model),
                "--keywords",
                str(keywords),
                "--config",
                str(config),
                "--calibration-references",
                str(refs),
                "--test-references",
                str(refs),
                "--posterior-dump",
                str(dump),
                "--decoder-replay",
                str(replay),
                "--posterior-cache",
                str(cache),
                "--retentions",
                "0.90",
                "0.97",
                "--work-dir",
                str(work),
                "--output",
                str(output),
            ],
            env=env,
        )

        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["development_only"] is True
        assert result["selection_feedback_allowed"] is False
        assert result["retentions"] == [0.9, 0.97]
        assert result["posterior_cache_hits"] == 3
        assert result["posterior_cache_misses"] == 1
        assert dump_count.read_text(encoding="utf-8").splitlines() == ["1"]
        low, high = result["operating_curve"]
        assert low["state_retention"] == 0.9
        assert low["calibration"]["far_per_hour"] > 0.0
        assert low["test"]["far_per_hour"] > 0.0
        assert low["strict"] is False
        assert high["state_retention"] == 0.97
        assert high["calibration"]["far_per_hour"] == 0.0
        assert high["test"]["far_per_hour"] == 0.0
        assert high["strict"] is True

    print("decoder retention curve: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
