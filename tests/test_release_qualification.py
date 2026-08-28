#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

from qualification_fixture import (
    sha256_file,
    write_json,
    write_model,
    write_pack,
    write_tokens,
    write_wav,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tokens = root / "tokens.txt"
        model = root / "base.kwm"
        pack = root / "xiaowo.kwk"
        config = root / "runtime.json"
        references = root / "references.jsonl"
        detections = root / "detections.jsonl"
        eval_summary = root / "eval-summary.json"
        eval_provenance = root / "eval-provenance.json"
        board_runner = root / "kws_board_bench"
        board_audio = root / "board-audio.wav"
        board_summary = root / "board-summary.json"
        evidence = root / "evidence.json"
        manifest = root / "qualification-manifest.json"
        policy = root / "policy.json"
        gate = root / "gate.json"

        _, fingerprint = write_tokens(tokens)
        model_bytes = write_model(model, fingerprint)
        pack_bytes = write_pack(pack, fingerprint)
        config.write_text('{"min_speech_dbfs":-55.0}\n', encoding="utf-8")
        references.write_text("reference fixture\n", encoding="utf-8")
        detections.write_text("detection fixture\n", encoding="utf-8")
        board_runner.write_bytes(b"board-runner-fixture")
        write_wav(board_audio, seconds=1)

        refs_hash = sha256_file(references)
        detections_hash = sha256_file(detections)
        model_hash = sha256_file(model)
        pack_hash = sha256_file(pack)
        board_runner_hash = sha256_file(board_runner)
        board_audio_hash = sha256_file(board_audio)
        write_json(
            eval_provenance,
            {
                "schema_version": 1,
                "runner_sha256": "1" * 64,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
                "references_sha256": refs_hash,
                "detections_sha256": detections_hash,
                "recordings": 24,
                "detections": 971,
            },
        )
        write_json(
            eval_summary,
            {
                "recordings": 24,
                "audio_hours": 24.0,
                "expected": 1000,
                "matched": 970,
                "false_rejects": 30,
                "false_accepts": 1,
                "frr": 0.03,
                "far_per_hour": 1.0 / 24.0,
                "p50_post_end_latency_ms": 180.0,
                "p95_post_end_latency_ms": 320.0,
                "per_keyword": {},
                "references_sha256": refs_hash,
                "detections_sha256": detections_hash,
            },
        )
        write_json(
            board_summary,
            {
                "schema_version": 1,
                "runner_sha256": board_runner_hash,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
                "audio_sha256": board_audio_hash,
                "block_samples": 320,
                "block_deadline_us": 20000.0,
                "audio_seconds": 60.0,
                "repeats": 10,
                "blocks": 3000,
                "model_bytes": model_bytes,
                "keyword_pack_bytes": pack_bytes,
                "arena_bytes": 16000,
                "total_process_us": 3000000.0,
                "mean_process_us": 1000.0,
                "p50_process_us": 900.0,
                "p95_process_us": 1800.0,
                "p99_process_us": 3000.0,
                "max_process_us": 4200.0,
                "rtf": 0.005,
                "p99_headroom": 20000.0 / 3000.0,
            },
        )
        write_json(
            evidence,
            {
                "target": "fixture-board",
                "board_revision": "A",
                "soc": "cortex-a32-fixture",
                "toolchain": "fixture-gcc",
                "compiler_flags": "-O3 -mcpu=cortex-a32",
                "governor": "performance",
                "audio_frontend": "audio-pipeline-fixture",
                "soak_hours": 8.0,
                "cpu_percent": 5.0,
                "rss_kib": 512.0,
                "stack_high_water_bytes": 32768.0,
                "max_temp_c": 55.0,
                "average_power_mw": 120.0,
            },
        )
        write_json(
            policy,
            {
                "schema_version": 1,
                "name": "fixture-policy",
                "min_audio_hours": 24.0,
                "min_expected_wakes": 1000,
                "max_frr": 0.05,
                "max_far_per_hour": 0.1,
                "max_p95_latency_ms": 500.0,
                "max_p99_process_us": 5000.0,
                "max_rtf": 0.25,
                "min_p99_headroom": 4.0,
                "min_soak_hours": 8.0,
            },
        )

        command = [
            sys.executable,
            str(ROOT / "tools" / "release_manifest.py"),
            "--model",
            str(model),
            "--keywords",
            str(pack),
            "--tokens",
            str(tokens),
            "--config",
            str(config),
            "--eval-summary",
            str(eval_summary),
            "--eval-provenance",
            str(eval_provenance),
            "--board-summary",
            str(board_summary),
            "--board-runner",
            str(board_runner),
            "--board-audio",
            str(board_audio),
            "--evidence",
            str(evidence),
            "--source-sha",
            "a" * 40,
            "--corpus-id",
            "fixture-corpus-v1",
            "--output",
            str(manifest),
        ]
        subprocess.check_call(command)
        result = json.loads(manifest.read_text(encoding="utf-8"))
        assert result["schema_version"] == 1
        assert result["source_sha"] == "a" * 40
        assert result["vocabulary"]["fingerprint"] == f"0x{fingerprint:016x}"
        assert result["artifacts"]["model"]["sha256"] == model_hash
        assert result["artifacts"]["board_runner"]["sha256"] == board_runner_hash
        assert result["artifacts"]["board_audio"]["sha256"] == board_audio_hash
        assert result["evaluation"]["references_sha256"] == refs_hash
        assert result["board"]["model_sha256"] == model_hash
        assert result["board"]["audio_sha256"] == board_audio_hash
        assert result["board"]["p99_process_us"] == 3000.0
        assert result["evidence"]["soak_hours"] == 8.0

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "qualification_gate.py"),
                "--manifest",
                str(manifest),
                "--policy",
                str(policy),
                "--output",
                str(gate),
            ]
        )
        gate_result = json.loads(gate.read_text(encoding="utf-8"))
        assert gate_result["qualified"] is True
        assert gate_result["violations"] == []

        failing_policy = json.loads(policy.read_text(encoding="utf-8"))
        failing_policy["max_frr"] = 0.01
        write_json(policy, failing_policy)
        failed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "qualification_gate.py"),
                "--manifest",
                str(manifest),
                "--policy",
                str(policy),
            ],
            check=False,
        )
        assert failed.returncode == 1

        provenance = json.loads(eval_provenance.read_text(encoding="utf-8"))
        provenance["model_sha256"] = "0" * 64
        write_json(eval_provenance, provenance)
        tampered_eval = subprocess.run(command, check=False)
        assert tampered_eval.returncode == 2
        provenance["model_sha256"] = model_hash
        write_json(eval_provenance, provenance)

        board = json.loads(board_summary.read_text(encoding="utf-8"))
        board["audio_sha256"] = "0" * 64
        write_json(board_summary, board)
        tampered_board = subprocess.run(command, check=False)
        assert tampered_board.returncode == 2

    print("test_release_qualification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
