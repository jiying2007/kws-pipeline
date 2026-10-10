from __future__ import annotations

import argparse
import copy
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

from qualification_fixture import write_model, write_pack, write_tokens

ROOT = pathlib.Path(__file__).resolve().parents[1]


def replay_cases(runner: pathlib.Path) -> None:
    """Only zero-weight unit model, deterministic biases and digital-zero PCM."""
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        model, pack, wav = (root / name for name in ("unit.kwm", "unit.kwk", "zeros.wav"))
        _, fingerprint = write_tokens(root / "tokens.txt")
        write_model(model, fingerprint)
        blob = bytearray(model.read_bytes())
        bias = struct.unpack_from("<I", blob, 64)[0]
        struct.pack_into("<5f", blob, bias, -8, 8, -8, -8, -8)
        model.write_bytes(blob)
        write_pack(pack, fingerprint)
        blob = bytearray(pack.read_bytes())
        struct.pack_into("<fH", blob, 28, 0.5, 1)
        struct.pack_into("<16H", blob, 40, 1, *([0] * 15))
        pack.write_bytes(blob)
        with wave.open(str(wav), "wb") as writer:
            writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            writer.writeframes(b"\0\0" * 1040)
        timeline = root / "timeline.tsv"
        stats = root / "stats.json"
        config = "5a" * 32
        spans = [[0, 320, 0, 0, 16, 0, 1, 320, config],
                 [320, 720, 1, 320 * 62500, 16, 0, 0, 320, config]]
        def run(rows=spans, *, chunk=160, valid=True, metadata=True):
            timeline.write_text("kws-afe-timeline-v1\n" + "".join(
                "\t".join(map(str, row)) + "\n" for row in rows), encoding="ascii")
            command = [str(runner), str(model), str(pack), str(wav), "unit",
                       "--stats-json", str(stats), "--block-samples", str(chunk)]
            if metadata:
                command += ["--metadata-tsv", str(timeline)]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            assert (result.returncode == 0) == valid, result.stderr
            if not valid:
                assert result.stdout == "", "bad sidecars must fail before emitting detections"
                return None
            return [json.loads(line) for line in result.stdout.splitlines()], json.loads(stats.read_text())
        outputs = [run(chunk=size) for size in (160, 320, 73)]
        assert outputs[0] == outputs[1] == outputs[2]
        events, counters = outputs[0]
        assert counters["speech_frames"] == 1 and counters["external_vad_frames"] == 3
        assert len(events) == 1
        assert events[0]["output_end_sample"] == 400
        assert events[0]["decision_capture_ns"] == 400 * 62500
        assert events[0]["raw_acoustic_end_ns"] == 80 * 62500
        # No sidecar still uses energy gating, unchanged default.
        assert run(metadata=False)[0] == []
        for column, value in ((0, 321), (1, 719), (2, 3), (3, 321 * 62500),
                              (4, 32), (5, 1), (6, "nan"), (6, "1.00000001"), (7, 160), (8, "ab" * 32)):
            bad = copy.deepcopy(spans)
            bad[1][column] = value
            run(bad, valid=False)
        tiny = copy.deepcopy(spans)
        tiny[1][6] = 1e-40
        assert run(tiny)[0] == outputs[0][0]
        gap = [[0, 200, 0, 0, 16, 0, 1, 320, config],
               [200, 840, 7, 280 * 62500, 18, 80, 1, 320, config]]
        events, counters = run(gap)
        assert counters["lost_samples"] == 80 and counters["discontinuities"] == 1
        assert events[0]["output_end_sample"] == 600
        assert events[0]["raw_acoustic_end_ns"] == 360 * 62500
        changed = copy.deepcopy(spans)
        changed[1][4], changed[1][6], changed[1][7], changed[1][8] = 20, 1, 160, "ab" * 32
        events, counters = run(changed)
        assert counters["discontinuities"] == 1 and events[0]["output_end_sample"] == 720
        reset = [[0, 400, 0, 0, 16, 0, 1, 320, config],
                 [400, 640, 0, 0, 24, 0, 1, 320, config]]
        events, counters = run(reset)
        assert [e["capture_epoch"] for e in events] == [0, 1]
        assert [e["decision_capture_ns"] for e in events] == [400 * 62500] * 2
        # Constant-delay final-sidecar writer fails closed on discontinuity.
        sys.path.insert(0, str(ROOT / "tools"))
        from run_final_afe_corpus import write_aligned_timeline
        sidecar = {"timing_contract": "afe-output-sample-v1", "latency_samples": 320,
                   "vad_segments": [{"start_sample": 0, "sample_count": 320, "probability": 1},
                                    {"start_sample": 320, "sample_count": 720, "probability": 0}]}
        write_aligned_timeline(timeline, sidecar, 1040, config)
        expected = "kws-afe-timeline-v1\n" + "".join("\t".join(map(str, r)) + "\n" for r in spans)
        assert timeline.read_text() == expected
        for change in ({"lost_samples": 1}, {"timing_contract": "unknown"},
                       {"vad_segments": []}, {"vad_segments": [{"start_sample": 1, "sample_count": 1040}]}):
            try:
                write_aligned_timeline(timeline, {**sidecar, **change}, 1040, config)
            except ValueError:
                pass
            else:
                raise AssertionError(change)
        # Synthetic command-adapter integration: preserve raw references while
        # creating a 320-sample delayed output and hash-bound VAD replay input.
        from final_afe_identity import inspect_adapter, sha256_file
        executable = root / "synthetic_afe.py"
        executable.write_text("#!/usr/bin/env python3\nimport json,sys,wave\n"
            "with wave.open(sys.argv[1], 'rb') as r: data=r.readframes(r.getnframes())\n"
            "with wave.open(sys.argv[2], 'wb') as w:\n"
            " w.setparams((1,2,16000,0,'NONE','not compressed')); w.writeframes(b'\\0\\0'*320+data)\n"
            "json.dump({'timing_contract':'afe-output-sample-v1','latency_samples':320,"
            "'vad_segments':[{'start_sample':0,'sample_count':1360,'probability':1}]},"
            "open(sys.argv[3],'w'))\n", encoding="utf-8")
        executable.chmod(0o700)
        config_file = root / "afe-config.json"
        config_file.write_text("{}")
        adapter = root / "adapter.json"
        adapter.write_text(json.dumps({"schema_version": 1, "executable_path": str(executable),
            "config_files": [str(config_file)],
            "command_argv": ["{executable}", "{input}", "{output}", "{result}"],
            **{field: "synthetic-fixture" for field in
               ("sku", "microphone_revision", "enclosure_revision", "audio_route", "toolchain")}}))
        _, _, identity = inspect_adapter(adapter)
        identity_file = root / "identity.json"
        identity_file.write_text(json.dumps(identity))
        manifest = root / "manifest.json"
        row = {"recording": "unit", "input_path": wav.name, "input_sha256": sha256_file(wav),
               "distance_m": 1, "azimuth_deg": 0, "tags": ["synthetic"],
               "expected": [{"keyword_id": 1, "start_s": 0.005, "end_s": 0.01}],
               **{field: "synthetic" for field in
                  ("speaker_id", "session_id", "source_id", "room_id", "device_id")}}
        manifest.write_text(json.dumps({"schema_version": 1, "qualification_id": "unit-only",
            "deployment_tag": "synthetic", "recordings": [row]}))
        result_dir = root / "afe-output"
        result = subprocess.run([sys.executable, str(ROOT / "tools/run_final_afe_corpus.py"),
            "--manifest", str(manifest), "--audio-root", str(root), "--adapter", str(adapter),
            "--expected-identity", str(identity_file), "--output-dir", str(result_dir)],
            check=False, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        reference = json.loads((result_dir / "references.post-afe.jsonl").read_text())
        assert reference["expected"] == [{"keyword_id": 1, "start_s": 0.025, "end_s": 0.03,
                                          "raw_start_s": 0.005, "raw_end_s": 0.01}]
        assert reference["metadata_sha256"] == sha256_file(result_dir / "post-afe" / reference["metadata_path"])
        assert reference["recalibration_required"] is True
        assert reference["raw_duration_s"] == 1040 / 16000
        detections = root / "detections.jsonl"
        result = subprocess.run([sys.executable, str(ROOT / "eval/run_corpus.py"),
            "--runner", str(runner), "--model", str(model), "--keywords", str(pack),
            "--references", str(result_dir / "references.post-afe.jsonl"),
            "--audio-root", str(result_dir / "post-afe"), "--detections", str(detections)],
            check=False, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        sys.path.insert(0, str(ROOT / "eval"))
        from score_events import validate_recordings, validate_detections, score
        from run_corpus import validate_constant_afe_timeline
        recordings = validate_recordings([reference])
        rows = [json.loads(line) for line in detections.read_text().splitlines()]
        report, _, _ = score(recordings, validate_detections(rows, recordings), 0.15, 0.5)
        assert abs(report["p50_post_afe_signed_end_offset_ms"] + 5) < 1e-9
        assert abs(report["p50_raw_capture_signed_end_offset_ms"] - 15) < 1e-9
        for field, value in (("capture_epoch", 1), ("decision_capture_ns", 30000000),
                             ("raw_acoustic_end_ns", 0), ("output_end_sample", 401)):
            bad = [{**rows[0], field: value}]
            try:
                validate_detections(bad, recordings)
            except ValueError:
                pass
            else:
                raise AssertionError(field)
        aligned_path = result_dir / "post-afe" / reference["metadata_path"]
        original_timeline = aligned_path.read_text()
        for column, value in ((3, "1"), (4, "18"), (5, "80"), (7, "160")):
            fields = original_timeline.splitlines()[1].split()
            fields[column] = value
            aligned_path.write_text("kws-afe-timeline-v1\n" + "\t".join(fields) + "\n")
            try:
                validate_constant_afe_timeline(aligned_path, reference, 1360)
            except ValueError:
                pass
            else:
                raise AssertionError(column)
        aligned_path.write_text(original_timeline)
    print("AFE timeline: chunk equivalence, delay mapping, gap/config/clock and malformed cases ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=pathlib.Path, default=ROOT / "build/kws_wav")
    args = parser.parse_args()
    replay_cases(args.runner.resolve(strict=True))
    header = (ROOT / "include" / "kws_pipeline" / "kws.h").read_text(encoding="utf-8")
    integration = (ROOT / "docs" / "AUDIO_DISCONTINUITY.md").read_text(encoding="utf-8")
    # This test turns green only when the public API is wired into the runtime.
    assert "kws_engine_notify_discontinuity" in header
    assert "discontinu" in integration.lower()
    print("test_audio_discontinuity_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
