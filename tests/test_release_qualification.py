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
    write_model_provenance,
    write_pack,
    write_tokens,
    write_wav,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from corpus_identity import evaluation_corpus_identity  # noqa: E402

CPU_PERCENT_SEMANTICS = "process_cpu_time / elapsed / online_cpu_capacity * 100"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tokens = root / "tokens.txt"
        training_tokens = root / "training-tokens.txt"
        training_manifest = root / "train.jsonl"
        training_audio = root / "train-a.wav"
        checkpoint = root / "base.pt"
        model = root / "base.kwm"
        model_provenance = root / "base.kwm.provenance.json"
        pack = root / "xiaowo.kwk"
        config = root / "runtime.json"
        eval_runner = root / "kws_wav"
        eval_audio = root / "room-1.wav"
        references = root / "references.jsonl"
        detections = root / "detections.jsonl"
        eval_summary = root / "eval-summary.json"
        eval_provenance = root / "eval-provenance.json"
        board_runner = root / "kws_board_bench"
        board_audio = root / "board-audio.wav"
        board_summary = root / "board-summary.json"
        evidence = root / "evidence.json"
        runtime_soak = root / "runtime-soak.json"
        raw_evidence = root / "target-raw.txt"
        power_raw = root / "power.csv"
        dataset_audit = root / "dataset-audit.json"
        manifest = root / "qualification-manifest.json"
        policy = root / "policy.json"
        gate = root / "gate.json"
        collector = ROOT / "tools" / "collect_target_evidence.py"

        _, fingerprint = write_tokens(tokens)
        _, training_fingerprint = write_tokens(training_tokens)
        assert training_fingerprint == fingerprint
        write_wav(training_audio, seconds=1)
        training_manifest.write_text(
            json.dumps(
                {
                    "audio": training_audio.name,
                    "tokens": [1, 2],
                    "speaker_id": "train-spk",
                    "session_id": "train-session",
                    "source_id": "train-source",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint.write_bytes(b"checkpoint-fixture-v2")
        model_bytes = write_model(model, fingerprint)
        checkpoint_hash = write_model_provenance(
            model_provenance,
            model,
            tokens,
            training_tokens,
            checkpoint,
            [training_manifest],
            fingerprint,
        )
        pack_bytes = write_pack(pack, fingerprint)
        write_json(
            config,
            {
                "sample_rate_hz": 16000,
                "frame_length_samples": 400,
                "frame_hop_samples": 320,
                "feature_dim": 32,
                "hidden_dim": 4,
                "vocab_target": "fixture-pinyin",
                "runtime": {
                    "min_speech_dbfs": -55.0,
                    "token_boost": 1.5,
                    "state_retention": 0.94,
                    "refractory_ms": 1200,
                },
            },
        )
        eval_runner.write_bytes(b"eval-runner-fixture")
        board_runner.write_bytes(b"board-runner-fixture")
        collector.write_bytes(b"collector-fixture")
        evidence_raw.write_bytes(b'{"sample":"fixture"}\n')
        write_wav(board_audio, seconds=1)
        write_wav(eval_audio, seconds=10)

        events = [
            {"keyword_id": 1, "start_s": 0.5 + i * 0.8, "end_s": 0.8 + i * 0.8}
            for i in range(10)
        ]
        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": eval_audio.name,
                    "duration_s": 10.0,
                    "speaker_id": "qual-spk",
                    "session_id": "qual-session",
                    "source_id": "qual-source",
                    "expected": events,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        detection_rows = [
            {
                "recording": "room-1",
                "keyword_id": 1,
                "time_s": events[i]["end_s"] + 0.1,
                "confidence": 0.8,
            }
            for i in range(9)
        ]
        detection_rows.append(
            {"recording": "room-1", "keyword_id": 1, "time_s": 9.5, "confidence": 0.7}
        )
        detections.write_text(
            "\n".join(json.dumps(row) for row in detection_rows) + "\n",
            encoding="utf-8",
        )

        refs_hash = sha256_file(references)
        detections_hash = sha256_file(detections)
        model_hash = sha256_file(model)
        pack_hash = sha256_file(pack)
        eval_runner_hash = sha256_file(eval_runner)
        board_runner_hash = sha256_file(board_runner)
        board_audio_hash = sha256_file(board_audio)
        eval_corpus = evaluation_corpus_identity(references, root)
        write_json(
            attestation_verification,
            {
                "schema_version": 1,
                "verified": True,
                "subject_kind": "kws-target-evidence",
                "issuer": "fixture-trusted-attestor",
                "trust_policy": "fixture-product-policy",
                "verified_at_utc": "2026-08-30T00:00:00Z",
                "subject_sha256": sha256_file(evidence_raw),
                "collector_sha256": sha256_file(collector),
                "board_runner_sha256": board_runner_hash,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
            },
        )
        write_json(
            eval_provenance,
            {
                "schema_version": 2,
                "runner_sha256": eval_runner_hash,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
                "references_sha256": refs_hash,
                "detections_sha256": detections_hash,
                "audio_corpus_sha256": eval_corpus["corpus_sha256"],
                "audio_files": eval_corpus["recordings"],
                "recordings": 1,
                "detections": 10,
            },
        )
        audio_hours = 10.0 / 3600.0
        write_json(
            eval_summary,
            {
                "recordings": 1,
                "audio_hours": audio_hours,
                "expected": 10,
                "matched": 9,
                "false_rejects": 1,
                "false_accepts": 1,
                "frr": 0.1,
                "far_per_hour": 1.0 / audio_hours,
                "p50_post_end_latency_ms": 100.0,
                "p95_post_end_latency_ms": 100.0,
                "per_keyword": {},
                "references_sha256": refs_hash,
                "detections_sha256": detections_hash,
            },
        )
        write_json(
            board_summary,
            {
                "schema_version": 1,
                "runtime_version": "0.2.0",
                "runtime_source_revision": "a" * 40,
                "runtime_config_digest": "b" * 64,
                "runtime_target": "arm-linux-gnueabihf",
                "runner_sha256": board_runner_hash,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
                "audio_sha256": board_audio_hash,
                "block_samples": 320,
                "block_deadline_us": 20000.0,
                "audio_seconds": 1.0,
                "repeats": 10,
                "blocks": 500,
                "model_bytes": model_bytes,
                "keyword_pack_bytes": pack_bytes,
                "arena_bytes": 16000,
                "total_process_us": 500000.0,
                "mean_process_us": 1000.0,
                "p50_process_us": 900.0,
                "p95_process_us": 1800.0,
                "p99_process_us": 3000.0,
                "max_process_us": 4200.0,
                "rtf": 0.05,
                "p99_headroom": 20000.0 / 3000.0,
            },
        )

        write_json(
            runtime_soak,
            {
                "schema_version": 2,
                "command": ["fixture-kws"],
                "pid": 123,
                "cpu_capacity_count": 2,
                "cpu_percent_semantics": CPU_PERCENT_SEMANTICS,
                "requested_hours": 8.0,
                "elapsed_seconds": 28800.0,
                "elapsed_hours": 8.0,
                "completed_requested_duration": True,
                "termination_returncode": -15,
                "sample_seconds": 60.0,
                "initial_cpu_seconds": 10.0,
                "samples": [
                    {
                        "elapsed_s": 0.0,
                        "rss_kib": 500.0,
                        "cpu_seconds": 10.0,
                        "temp_c": 50.0,
                    },
                    {
                        "elapsed_s": 28800.0,
                        "rss_kib": 512.0,
                        "cpu_seconds": 2890.0,
                        "temp_c": 55.0,
                    },
                ],
                "max_rss_kib": 512.0,
                "average_cpu_percent": 5.0,
                "max_temp_c": 55.0,
            },
        )
        raw_evidence.write_text("fixture target measurements\n", encoding="utf-8")
        power_raw.write_text("t,power_mw\n0,120\n", encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                str(collector),
                "--output", str(evidence),
                "--target", "fixture-board",
                "--board-revision", "A",
                "--soc", "cortex-a32-fixture",
                "--toolchain", "fixture-gcc",
                "--compiler-flags=-O3 -mcpu=cortex-a32",
                "--audio-frontend", "audio-pipeline-fixture",
                "--runtime-soak", str(runtime_soak),
                "--stack-high-water-bytes", "32768",
                "--average-power-mw", "120",
                "--raw-evidence", str(raw_evidence),
                "--power-raw", str(power_raw),
                "--instrument-id", "fixture-meter",
                "--calibration-id", "fixture-cal",
            ]
        )

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "training" / "audit_dataset.py"),
                "--split", f"train={training_manifest}",
                "--split", f"qualification={references}",
                "--require-metadata", "speaker_id",
                "--require-metadata", "session_id",
                "--require-metadata", "source_id",
                "--fail-within-split",
                "--report", str(dataset_audit),
            ]
        )

        valid_policy = {
            "schema_version": 2,
            "policy_id": "fixture-policy-v1",
            "name": "fixture-policy",
            "sku": "fixture-sku",
            "shipping_approved": True,
            "confidence_level": 0.95,
            "min_audio_hours": audio_hours,
            "min_expected_wakes": 10,
            "max_frr": 0.15,
            "max_frr_upper_bound": 0.40,
            "max_far_per_hour": 400.0,
            "max_far_upper_bound_per_hour": 3000.0,
            "max_p95_latency_ms": 500.0,
            "max_p99_process_us": 5000.0,
            "max_rtf": 0.25,
            "min_p99_headroom": 4.0,
            "min_soak_hours": 8.0,
            "max_cpu_percent": 10.0,
            "max_rss_kib": 2048.0,
            "max_stack_high_water_bytes": 65536.0,
            "max_temp_c": 70.0,
            "max_average_power_mw": 250.0,
        }
        write_json(policy, valid_policy)

        command = [
            sys.executable,
            str(ROOT / "tools" / "qualification_manifest.py"),
            "--model", str(model),
            "--model-provenance", str(model_provenance),
            "--checkpoint", str(checkpoint),
            "--training-tokens", str(training_tokens),
            "--training-manifest", str(training_manifest),
            "--dataset-audit", str(dataset_audit),
            "--keywords", str(pack),
            "--tokens", str(tokens),
            "--config", str(config),
            "--eval-runner", str(eval_runner),
            "--references", str(references),
            "--eval-audio-root", str(root),
            "--detections", str(detections),
            "--eval-summary", str(eval_summary),
            "--eval-provenance", str(eval_provenance),
            "--board-summary", str(board_summary),
            "--board-runner", str(board_runner),
            "--board-audio", str(board_audio),
            "--evidence", str(evidence),
            "--evidence-collector", str(collector),
            "--raw-evidence", str(runtime_soak),
            "--raw-evidence", str(raw_evidence),
            "--raw-evidence", str(power_raw),
            "--source-sha", "a" * 40,
            "--corpus-id", "home-kws-heldout-fixture-v2",
            "--output", str(manifest),
        ]
        subprocess.check_call(command)
        result = json.loads(manifest.read_text(encoding="utf-8"))
        assert result["schema_version"] == 2
        assert result["artifacts"]["model_checkpoint"]["sha256"] == checkpoint_hash
        assert result["model_lineage"]["training_corpus_sha256"]
        assert result["evaluation"]["audio_corpus_sha256"] == eval_corpus["corpus_sha256"]
        assert result["dataset_audit"]["sha256"] == sha256_file(dataset_audit)
        assert result["evidence"]["cpu_percent"] == 5.0
        assert result["evidence"]["rss_kib"] == 512.0

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "qualification_gate.py"),
                "--manifest", str(manifest),
                "--policy", str(policy),
                "--output", str(gate),
            ]
        )
        gate_result = json.loads(gate.read_text(encoding="utf-8"))
        assert gate_result["schema_version"] == 3
        assert gate_result["qualified"] is True
        assert gate_result["training_corpus_sha256"] == result["model_lineage"]["training_corpus_sha256"]
        assert gate_result["evaluation_corpus_sha256"] == eval_corpus["corpus_sha256"]

        original_evidence = evidence.read_text(encoding="utf-8")
        tampered_evidence = json.loads(original_evidence)
        tampered_evidence["cpu_percent"] = 6.0
        write_json(evidence, tampered_evidence)
        assert subprocess.run(command, check=False).returncode == 2
        evidence.write_text(original_evidence, encoding="utf-8")

        original_eval = eval_audio.read_bytes()
        eval_audio.write_bytes(original_eval[:-2] + b"\x00\x00")
        assert subprocess.run(command, check=False).returncode == 2
        eval_audio.write_bytes(original_eval)

        original_training = training_audio.read_bytes()
        training_audio.write_bytes(original_training[:-2] + b"\x00\x00")
        assert subprocess.run(command, check=False).returncode == 2
        training_audio.write_bytes(original_training)

        original_refs = references.read_text(encoding="utf-8")
        wrong_duration = json.loads(original_refs)
        wrong_duration["duration_s"] = 1000.0
        references.write_text(json.dumps(wrong_duration) + "\n", encoding="utf-8")
        assert subprocess.run(command, check=False).returncode == 2
        references.write_text(original_refs, encoding="utf-8")

        failing_policy = dict(valid_policy)
        failing_policy["max_cpu_percent"] = 4.0
        write_json(policy, failing_policy)
        assert subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "qualification_gate.py"),
                "--manifest", str(manifest),
                "--policy", str(policy),
            ],
            check=False,
        ).returncode == 1

    print("test_release_qualification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
