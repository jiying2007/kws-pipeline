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
sys.path.insert(0, str(ROOT / "eval"))
from run_corpus import cached_trace_valid, ensure_cached_trace, trace_cache_paths  # noqa: E402


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
            "if pathlib.Path(sys.argv[3]).read_bytes().startswith(b'trace-v2:'): keyword=3\n"
            "print(json.dumps({'recording':sys.argv[4],'keyword_id':keyword,"
            "'time_s':1.0,'confidence':0.8}))\n",
            encoding="utf-8",
        )
        posterior_dump.chmod(posterior_dump.stat().st_mode | stat.S_IXUSR)
        decoder_replay.chmod(decoder_replay.stat().st_mode | stat.S_IXUSR)
        env = dict(os.environ)
        env["DUMP_COUNT_FILE"] = str(dump_count)

        cached_provenance = root / "cached-provenance.json"
        cached_command = [
            sys.executable, str(ROOT / "eval" / "run_corpus.py"),
            "--runner", str(runner), "--model", str(model),
            "--keywords", str(keywords), "--references", str(references),
            "--audio-root", str(root), "--detections", str(detections),
            "--provenance", str(cached_provenance),
            "--posterior-dump", str(posterior_dump),
            "--decoder-replay", str(decoder_replay),
            "--posterior-cache", str(posterior_cache),
        ]
        first_dump_sha = sha256_file(posterior_dump)
        identity_args = {
            "model_sha256": sha256_file(model),
            "audio_sha256": sha256_file(audio),
            "posterior_dump_sha256": first_dump_sha,
        }
        # A valid pre-producer-identity cache must not be reused or relabelled.
        legacy_trace = (posterior_cache / identity_args["model_sha256"]
                        / identity_args["audio_sha256"][:2]
                        / (identity_args["audio_sha256"] + ".kwtr"))
        legacy_trace.parent.mkdir(parents=True)
        legacy_trace.write_bytes(b"trace-v2:unidentified-producer")
        legacy_summary = {
            "schema_version": 1,
            "evidence_class": "kws-posterior-trace-cache-v1",
            "model_sha256": identity_args["model_sha256"],
            "audio_sha256": identity_args["audio_sha256"],
            "trace_sha256": sha256_file(legacy_trace), "frames": 3, "vocab_size": 4,
        }
        legacy_sidecar = legacy_trace.with_suffix(".json")
        legacy_sidecar.write_text(json.dumps(legacy_summary), encoding="utf-8")
        assert cached_trace_valid(legacy_trace, legacy_sidecar, **identity_args) is None
        subprocess.check_call(cached_command, env=env)
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
        assert cached_first["posterior_dump_sha256"] == first_dump_sha
        assert cached_first["posterior_traces"][0]["posterior_dump_sha256"] == first_dump_sha
        first_trace, first_sidecar, _ = trace_cache_paths(posterior_cache, **identity_args)
        assert first_trace.parent.parent.name == first_dump_sha
        assert json.loads(first_sidecar.read_text())["posterior_dump_sha256"] == first_dump_sha
        assert legacy_trace.read_bytes() == b"trace-v2:unidentified-producer"
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
                "--decoder-blank-retention", "0.85",
                "--decoder-fuzzy-child-cost-log", "-4.0",
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
            "blank_retention": 0.85,
            "fuzzy_child_cost_log": -4.0,
        }
        assert cached_second["posterior_traces"][0]["trace_sha256"] == first_trace_sha
        assert dump_count.read_text(encoding="utf-8").splitlines() == ["1"]

        def check_cached_run(*, hits: int, misses: int, keyword_id: int) -> dict:
            subprocess.check_call(cached_command, env=env)
            result = json.loads(cached_provenance.read_text(encoding="utf-8"))
            producer_sha = sha256_file(posterior_dump)
            assert result["posterior_cache_hits"] == hits
            assert result["posterior_cache_misses"] == misses
            assert result["posterior_dump_sha256"] == producer_sha
            assert result["posterior_traces"][0]["posterior_dump_sha256"] == producer_sha
            assert json.loads(detections.read_text())["keyword_id"] == keyword_id
            return result

        # Decoder-only changes still reuse acoustic traces.
        decoder_replay.write_text(decoder_replay.read_text() + "# decoder-only change\n")
        decoder_changed = check_cached_run(hits=1, misses=0, keyword_id=2)
        assert decoder_changed["decoder_replay_sha256"] != cached_second["decoder_replay_sha256"]
        assert dump_count.read_text().splitlines() == ["1"]

        # Same executable path, different bytes: regenerate and identify the new producer.
        posterior_dump.write_text(posterior_dump.read_text().replace("trace-v1:", "trace-v2:"))
        second_dump_sha = sha256_file(posterior_dump)
        producer_changed = check_cached_run(hits=0, misses=1, keyword_id=3)
        assert second_dump_sha != first_dump_sha
        assert producer_changed["posterior_traces"][0]["trace_sha256"] != first_trace_sha
        assert first_trace.is_file() and first_sidecar.is_file()
        check_cached_run(hits=1, misses=0, keyword_id=3)
        assert dump_count.read_text().splitlines() == ["1", "1"]

        identity_args["posterior_dump_sha256"] = second_dump_sha
        _, sidecar, _ = trace_cache_paths(posterior_cache, **identity_args)
        for bad_producer in (None, first_dump_sha):
            summary = json.loads(sidecar.read_text())
            if bad_producer is None:
                summary.pop("posterior_dump_sha256")
            else:
                summary["posterior_dump_sha256"] = bad_producer
            sidecar.write_text(json.dumps(summary), encoding="utf-8")
            check_cached_run(hits=0, misses=1, keyword_id=3)
        # Even a legacy entry copied into the new namespace cannot be admitted.
        summary = json.loads(sidecar.read_text())
        summary.update(schema_version=1, evidence_class="kws-posterior-trace-cache-v1")
        sidecar.write_text(json.dumps(summary), encoding="utf-8")
        check_cached_run(hits=0, misses=1, keyword_id=3)
        assert len(dump_count.read_text().splitlines()) == 5

        # A stale caller snapshot must fail before a hit or generator invocation.
        try:
            ensure_cached_trace(
                posterior_dump=posterior_dump, cache_root=posterior_cache,
                model=model, audio=audio,
                **{**identity_args, "posterior_dump_sha256": first_dump_sha},
            )
        except ValueError as exc:
            assert "executable changed during this run" in str(exc)
        else:
            raise AssertionError("stale producer identity accepted")
        assert len(dump_count.read_text().splitlines()) == 5

        # Mutation while dumping must not publish a cache entry under the old identity.
        stable_dump_source = posterior_dump.read_text()
        posterior_dump.write_text(
            stable_dump_source.replace("os.environ['DUMP_COUNT_FILE']", repr(str(dump_count)))
            + "self=pathlib.Path(__file__); self.write_text(self.read_text()+'# changed\\n')\n"
        )
        drift_args = {**identity_args, "posterior_dump_sha256": sha256_file(posterior_dump)}
        try:
            ensure_cached_trace(
                posterior_dump=posterior_dump, cache_root=posterior_cache,
                model=model, audio=audio, **drift_args,
            )
        except ValueError as exc:
            assert "executable changed during trace generation" in str(exc)
        else:
            raise AssertionError("mid-generation producer mutation accepted")
        drift_trace, drift_sidecar, _ = trace_cache_paths(posterior_cache, **drift_args)
        assert not drift_trace.exists() and not drift_sidecar.exists()
        assert not list(drift_trace.parent.glob("*.tmp"))

        # A later executable replacement must not relabel a trace already consumed.
        posterior_dump.write_text(stable_dump_source)
        replay_source = decoder_replay.read_text()
        decoder_replay.write_text(
            replay_source + f"producer=pathlib.Path({str(posterior_dump)!r}); "
            "producer.write_text(producer.read_text()+'# replaced after replay\\n')\n"
        )
        subprocess.check_call(cached_command, env=env)
        consumed = json.loads(cached_provenance.read_text())
        assert consumed["posterior_cache_hits"] == 1
        assert sha256_file(posterior_dump) != second_dump_sha
        assert consumed["posterior_dump_sha256"] == second_dump_sha
        assert consumed["posterior_traces"][0]["posterior_dump_sha256"] == second_dump_sha
        posterior_dump.write_text(stable_dump_source)
        decoder_replay.write_text(replay_source)

        # Model and WAV identity remain separate invalidation dimensions.
        model.write_bytes(model.read_bytes() + b"-changed")
        check_cached_run(hits=0, misses=1, keyword_id=3)
        audio_bytes = audio.read_bytes()
        audio.write_bytes(audio_bytes[:-2] + b"\x02\x00")
        check_cached_run(hits=0, misses=1, keyword_id=3)
        check_cached_run(hits=1, misses=0, keyword_id=3)
        audio.write_bytes(audio_bytes)

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

        references.write_text(
            json.dumps({"recording": "room-1", "path": "audio.wav",
                        "duration_s": 3.0, "expected": []}) + "\n",
            encoding="utf-8",
        )
        invalid_duration = subprocess.run(
            [sys.executable, str(ROOT / "eval" / "run_corpus.py"),
             "--runner", str(runner), "--model", str(model),
             "--keywords", str(keywords), "--references", str(references),
             "--audio-root", str(root), "--detections", str(detections)],
            check=False, text=True, capture_output=True,
        )
        assert invalid_duration.returncode == 2
        assert "reference duration_s does not match WAV duration" in invalid_duration.stderr

        # Identity admission validates numeric type, missing values and one-sample tolerance.
        sys.path.insert(0, str(ROOT / "eval"))
        from run_corpus import audio_identity
        row = {"recording": "room-1", "_execution_path": "audio.wav"}
        for duration in (2.0, 2.0 + 1.0 / 32000.0):
            assert audio_identity({**row, "duration_s": duration}, audio)["duration_s"] == 2.0
        for duration in (None, True, "2", float("nan"), float("inf"), -2,
                         2.0 + 2.0 / 16000.0):
            try:
                audio_identity({**row, "duration_s": duration}, audio)
            except ValueError as exc:
                assert "reference duration_s" in str(exc)
            else:
                raise AssertionError(duration)

    print("test_run_corpus: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
