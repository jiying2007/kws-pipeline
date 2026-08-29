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


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tokens = root / "tokens.txt"
        training_tokens = root / "training-tokens.txt"
        training_manifest = root / "train.tsv"
        checkpoint = root / "base.pt"
        model = root / "base.kwm"
        model_provenance = root / "base.kwm.provenance.json"
        pack = root / "xiaowo.kwk"
        config = root / "runtime.json"
        eval_runner = root / "kws_wav"
        references = root / "references.jsonl"
        detections = root / "detections.jsonl"
        eval_summary = root / "eval-summary.json"
        eval_provenance = root / "eval-provenance.json"
        board_runner = root / "kws_board_bench"
        board_audio = root / "board-audio.wav"
        board_summary = root / "board-summary.json"
        evidence = root / "evidence.json"
        evidence_raw = root / "evidence.raw.jsonl"
        collector = root / "kws-evidence-collector"
        attestation_verification = root / "attestation-verification.json"
        manifest = root / "qualification-manifest.json"
        policy = root / "policy.json"
        gate = root / "gate.json"

        _, fingerprint = write_tokens(tokens)
        _, training_fingerprint = write_tokens(training_tokens)
        assert training_fingerprint == fingerprint
        training_manifest.write_text("train-a.wav\t1 2\n", encoding="utf-8")
        checkpoint.write_bytes(b"checkpoint-fixture-v1")

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

        events = [
            {
                "keyword_id": 1,
                "start_s": float(10 + i * 20),
                "end_s": float(11 + i * 20),
            }
            for i in range(10)
        ]
        references.write_text(
            json.dumps(
                {
                    "recording": "room-1",
                    "path": "room-1.wav",
                    "duration_s": 86400.0,
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
            {
                "recording": "room-1",
                "keyword_id": 1,
                "time_s": 10000.0,
                "confidence": 0.7,
            }
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
                "schema_version": 1,
                "runner_sha256": eval_runner_hash,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
                "references_sha256": refs_hash,
                "detections_sha256": detections_hash,
                "recordings": 1,
                "detections": 10,
            },
        )
        write_json(
            eval_summary,
            {
                "recordings": 1,
                "audio_hours": 24.0,
                "expected": 10,
                "matched": 9,
                "false_rejects": 1,
                "false_accepts": 1,
                "frr": 0.1,
                "far_per_hour": 1.0 / 24.0,
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
            evidence,
            {
                "target": "fixture-board",
                "schema_version": 2,
                "evidence_class": "product-board",
                "sku": "fixture-sku",
                "source_sha": "a" * 40,
                "collected_at_utc": "2026-08-30T00:00:00Z",
                "builder_id": "fixture-builder",
                "dut_id": "fixture-dut",
                "collector_id": "fixture-collector-v1",
                "collector_sha256": sha256_file(collector),
                "raw_evidence_sha256": sha256_file(evidence_raw),
                "attestation_verification_sha256": sha256_file(attestation_verification),
                "board_runner_sha256": board_runner_hash,
                "model_sha256": model_hash,
                "keyword_pack_sha256": pack_hash,
                "board_audio_sha256": board_audio_hash,
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
        valid_policy = {
            "schema_version": 2,
            "policy_id": "fixture-policy-v1",
            "name": "fixture-policy",
            "sku": "fixture-sku",
            "shipping_approved": True,
            "confidence_level": 0.95,
            "min_audio_hours": 24.0,
            "min_expected_wakes": 10,
            "max_frr": 0.15,
            "max_frr_upper_bound": 0.40,
            "max_far_per_hour": 0.1,
            "max_far_upper_bound_per_hour": 0.21,
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
            "--model",
            str(model),
            "--model-provenance",
            str(model_provenance),
            "--checkpoint",
            str(checkpoint),
            "--training-tokens",
            str(training_tokens),
            "--training-manifest",
            str(training_manifest),
            "--keywords",
            str(pack),
            "--tokens",
            str(tokens),
            "--config",
            str(config),
            "--eval-runner",
            str(eval_runner),
            "--references",
            str(references),
            "--detections",
            str(detections),
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
            "--evidence-raw",
            str(evidence_raw),
            "--collector",
            str(collector),
            "--attestation-verification",
            str(attestation_verification),
            "--sku",
            "fixture-sku",
            "--source-sha",
            "a" * 40,
            "--corpus-id",
            "fixture-corpus-v1",
            "--output",
            str(manifest),
        ]
        subprocess.check_call(command)
        result = json.loads(manifest.read_text(encoding="utf-8"))
        assert result["artifacts"]["model_provenance"]["sha256"] == sha256_file(
            model_provenance
        )
        assert result["artifacts"]["model_checkpoint"]["sha256"] == checkpoint_hash
        assert result["artifacts"]["training_tokens"]["sha256"] == sha256_file(
            training_tokens
        )
        assert result["artifacts"]["training_manifests"][0]["sha256"] == sha256_file(
            training_manifest
        )
        assert result["model_lineage"]["checkpoint_sha256"] == checkpoint_hash
        assert result["model_lineage"]["model_sha256"] == model_hash
        assert result["runtime"]["frontend_kind"] == result["model_lineage"]["frontend_kind"]
        assert result["runtime"]["frontend_name"] == result["model_lineage"]["frontend_name"]
        assert result["artifacts"]["eval_runner"]["sha256"] == eval_runner_hash
        assert result["artifacts"]["references"]["sha256"] == refs_hash
        assert result["artifacts"]["detections"]["sha256"] == detections_hash
        assert result["board"]["audio_sha256"] == board_audio_hash

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
        assert gate_result["schema_version"] == 2
        assert gate_result["qualified"] is True
        assert gate_result["qualification_level"] == "product-certified"
        assert gate_result["sku"] == "fixture-sku"
        assert gate_result["manifest_sha256"] == sha256_file(manifest)
        assert gate_result["policy_sha256"] == sha256_file(policy)
        assert gate_result["model_checkpoint_sha256"] == checkpoint_hash
        assert gate_result["statistics"]["frr_upper_bound"] > 0.1
        assert gate_result["statistics"]["far_upper_bound_per_hour"] > 1.0 / 24.0

        non_shipping = dict(valid_policy)
        non_shipping["shipping_approved"] = False
        write_json(policy, non_shipping)
        rejected = subprocess.run(
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
        assert rejected.returncode == 2
        write_json(policy, valid_policy)

        statistical_failure = dict(valid_policy)
        statistical_failure["max_frr_upper_bound"] = 0.20
        write_json(policy, statistical_failure)
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
        write_json(policy, valid_policy)

        attestation_tampered = json.loads(manifest.read_text(encoding="utf-8"))
        attestation_tampered["evidence"]["attestation"]["verified"] = False
        write_json(manifest, attestation_tampered)
        assert (
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "qualification_gate.py"),
                    "--manifest",
                    str(manifest),
                    "--policy",
                    str(policy),
                ],
                check=False,
            ).returncode
            == 2
        )
        subprocess.check_call(command)

        frontend_tampered = json.loads(manifest.read_text(encoding="utf-8"))
        tampered_kind = 1 if int(frontend_tampered["runtime"]["frontend_kind"]) == 0 else 0
        frontend_tampered["runtime"]["frontend_kind"] = tampered_kind
        frontend_tampered["runtime"]["frontend_name"] = {0: "logmel", 1: "pcen-lite"}[tampered_kind]
        write_json(manifest, frontend_tampered)
        assert (
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "qualification_gate.py"),
                    "--manifest",
                    str(manifest),
                    "--policy",
                    str(policy),
                ],
                check=False,
            ).returncode
            == 2
        )
        subprocess.check_call(command)

        failing_policy = dict(valid_policy)
        failing_policy["max_cpu_percent"] = 4.0
        write_json(policy, failing_policy)
        assert (
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "qualification_gate.py"),
                    "--manifest",
                    str(manifest),
                    "--policy",
                    str(policy),
                ],
                check=False,
            ).returncode
            == 1
        )
        write_json(policy, valid_policy)

        tampered = json.loads(manifest.read_text(encoding="utf-8"))
        tampered["model_lineage"]["model_sha256"] = "0" * 64
        write_json(manifest, tampered)
        assert (
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "qualification_gate.py"),
                    "--manifest",
                    str(manifest),
                    "--policy",
                    str(policy),
                ],
                check=False,
            ).returncode
            == 2
        )
        subprocess.check_call(command)

        provenance = json.loads(model_provenance.read_text(encoding="utf-8"))
        provenance["model"]["sha256"] = "0" * 64
        write_json(model_provenance, provenance)
        assert subprocess.run(command, check=False).returncode == 2
        write_model_provenance(
            model_provenance,
            model,
            tokens,
            training_tokens,
            checkpoint,
            [training_manifest],
            fingerprint,
        )

        checkpoint.write_bytes(b"tampered-checkpoint")
        assert subprocess.run(command, check=False).returncode == 2
        checkpoint.write_bytes(b"checkpoint-fixture-v1")

        original_training_manifest = training_manifest.read_text(encoding="utf-8")
        training_manifest.write_text(
            original_training_manifest + "extra.wav\t1\n", encoding="utf-8"
        )
        assert subprocess.run(command, check=False).returncode == 2
        training_manifest.write_text(original_training_manifest, encoding="utf-8")

        original_refs = references.read_text(encoding="utf-8")
        references.write_text(original_refs + "# tampered\n", encoding="utf-8")
        assert subprocess.run(command, check=False).returncode == 2
        references.write_text(original_refs, encoding="utf-8")

        board = json.loads(board_summary.read_text(encoding="utf-8"))
        board["audio_sha256"] = "0" * 64
        write_json(board_summary, board)
        assert subprocess.run(command, check=False).returncode == 2

    print("test_release_qualification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
