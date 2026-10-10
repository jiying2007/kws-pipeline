#!/usr/bin/env python3
from __future__ import annotations

import json
import copy
import argparse
import pathlib
import subprocess
import sys
import tempfile

from qualification_fixture import (
    runtime_soak_fixture,
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

from runtime_soak_contract import CPU_MEASUREMENT_CONTRACT_ID
from qualification_metrics import (validate_event_metrics, score, validate_recordings,
                                   validate_detections, EVENT_SCORING_CONTRACT)
from test_board_bench import board_summary_fixture



def verify_event_metadata() -> None:
    import hashlib
    def prepare(refs, detections):
        raw = {"references": "".join(json.dumps(row) + "\n" for row in refs),
               "detections": "".join(json.dumps(row) + "\n" for row in detections)}
        hashes = {f"{key}_sha256": hashlib.sha256(value.encode()).hexdigest()
                  for key, value in raw.items()}
        recordings = validate_recordings(refs)
        summary, _, _ = score(recordings, validate_detections(detections, recordings), 0.15, 0.5)
        return raw, hashes, summary
    def rejects(summary, raw, hashes):
        try:
            validate_event_metrics(summary, raw, hashes)
        except ValueError:
            return
        raise AssertionError("forged event metrics admitted")
    refs = [{"recording": "r", "duration_s": 10.0,
             "expected": [{"keyword_id": 1, "start_s": 1.0, "end_s": 2.0}]}]
    # All variants are pure annotations/detection metadata; no runner is invoked.
    for keyword, time, matched in ((1, 2.1, 1), (2, 9.0, 0), (1, 9.0, 0),
                                   (1, 0.85, 1), (1, 2.5, 1), (1, 2.5001, 0)):
        raw, hashes, canonical = prepare(refs, [{"recording": "r", "keyword_id": keyword,
                                                "time_s": time, "confidence": 0.9}])
        result = validate_event_metrics(canonical, raw, hashes)
        if result["matched"] != matched or result["scoring_contract"] != EVENT_SCORING_CONTRACT:
            raise AssertionError("canonical event policy changed")
        forged = copy.deepcopy(canonical)
        forged.update(matched=1 - matched, false_rejects=matched,
                      false_accepts=matched, frr=float(matched),
                      far_per_hour=matched / canonical["audio_hours"])
        rejects(forged, raw, hashes)
        for field, value in (("p50_post_end_latency_ms", -1.0),
                             ("p50_post_end_latency_ms", -1e-12),
                             ("p95_post_end_latency_ms", float("nan")),
                             ("p95_post_end_latency_ms", 777.0),
                             ("event_match_pre_tolerance_ms", 9999.0),
                             ("event_match_post_tolerance_ms", 9999.0),
                             ("per_keyword", {})):
            rejects(dict(canonical, **{field: value}), raw, hashes)
        rejects(canonical, dict(raw, detections=raw["detections"] + "\n"), hashes)
    raw, hashes, canonical = prepare(refs, [])
    validate_event_metrics(canonical, raw, hashes)
    if canonical["p95_post_end_latency_ms"] != 0.0 or canonical["false_rejects"] != 1:
        raise AssertionError("empty detection semantics changed")
    # Duplicate detections are one match and one false accept, not two matches.
    det = {"recording": "r", "keyword_id": 1, "time_s": 2.1, "confidence": 0.9}
    raw, hashes, canonical = prepare(refs, [det, det])
    validate_event_metrics(canonical, raw, hashes)
    if canonical["matched"] != 1 or canonical["false_accepts"] != 1:
        raise AssertionError("duplicate detection consumed twice")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-metadata-only", action="store_true")
    args = parser.parse_args()
    verify_event_metadata()
    if args.event_metadata_only:
        print("test_release_qualification: canonical event metadata contracts ok (no model execution)")
        return 0
    synthetic_holdout = (
        ROOT / "training" / "render_qualification_holdout.py"
    ).read_text(encoding="utf-8")
    assert '"qualification_evidence_scope": "synthetic-seed-rotated-holdout-v1"' in synthetic_holdout
    assert '"wav_identity_independent": True' in synthetic_holdout
    assert '"generator_family_independent": False' in synthetic_holdout
    assert '"release_authority": False' in synthetic_holdout
    assert (
        '"release_authority_policy": "real-human-and-target-dut-required-v1"'
        in synthetic_holdout
    )

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
        evidence_raw = root / "evidence-raw.jsonl"
        attestation_verification = root / "attestation-verification.json"
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
        score_recordings = validate_recordings([json.loads(references.read_text(encoding="utf-8"))])
        canonical, _, _ = score(score_recordings,
                                validate_detections(detection_rows, score_recordings), 0.15, 0.5)
        write_json(eval_summary, {**canonical, "references_sha256": refs_hash,
                                  "detections_sha256": detections_hash})

        benchmark = board_summary_fixture(board_audio, repeats=10)
        benchmark.update(runtime_target="arm-linux-gnueabihf", runner_sha256=board_runner_hash,
                         model_sha256=model_hash, keyword_pack_sha256=pack_hash,
                         model_bytes=model_bytes, keyword_pack_bytes=pack_bytes,
                         p50_process_us=900.0, p95_process_us=1800.0,
                         p99_process_us=3000.0, max_process_us=4200.0,
                         p99_headroom=20000.0 / 3000.0)
        write_json(board_summary, benchmark)


        write_json(
            runtime_soak,
            runtime_soak_fixture(8.0),
        )
        raw_evidence.write_text("fixture target measurements\n", encoding="utf-8")
        power_raw.write_text("t,power_mw\n0,120\n", encoding="utf-8")
        raw_paths = [runtime_soak, raw_evidence, power_raw, board_summary]
        evidence_raw.write_text(
            "".join(
                json.dumps(
                    {
                        "name": path.name,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    },
                    sort_keys=True,
                )
                + "\n"
                for path in raw_paths
            ),
            encoding="utf-8",
        )
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
                "--raw-evidence", str(board_summary),
                "--power-raw", str(power_raw),
                "--evidence-raw", str(evidence_raw),
                "--attestation-verification", str(attestation_verification),
                "--board-runner", str(board_runner),
                "--model", str(model),
                "--keyword-pack", str(pack),
                "--board-audio", str(board_audio),
                "--sku", "fixture-sku",
                "--source-sha", "a" * 40,
                "--builder-id", "fixture-builder",
                "--dut-id", "fixture-dut",
                "--collector-id", "fixture-collector",
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
            "schema_version": 3,
            "measurement_contract_id": CPU_MEASUREMENT_CONTRACT_ID,
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
            "--evidence-raw", str(evidence_raw),
            "--attestation-verification", str(attestation_verification),
            "--raw-evidence", str(runtime_soak),
            "--raw-evidence", str(raw_evidence),
            "--raw-evidence", str(power_raw),
            "--raw-evidence", str(board_summary),
            "--source-sha", "a" * 40,
            "--sku", "fixture-sku",
            "--corpus-id", "home-kws-heldout-fixture-v2",
            "--output", str(manifest),
        ]
        subprocess.check_call(command)
        result = json.loads(manifest.read_text(encoding="utf-8"))
        assert result["schema_version"] == 3
        assert result["sku"] == "fixture-sku"
        assert result["artifacts"]["model_checkpoint"]["sha256"] == checkpoint_hash
        assert result["model_lineage"]["training_corpus_sha256"]
        assert result["evaluation"]["audio_corpus_sha256"] == eval_corpus["corpus_sha256"]
        assert result["dataset_audit"]["sha256"] == sha256_file(dataset_audit)
        assert result["evidence"]["cpu_percent"] == 5.0
        assert result["evidence"]["rss_kib"] == 512.0
        assert result["evidence"]["raw_evidence_sha256"] == sha256_file(evidence_raw)
        assert result["evidence"]["attestation"]["verified"] is True

        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "qualification_gate.py"),
                "--manifest", str(manifest),
                "--references", str(references), "--detections", str(detections),
                "--board-audio", str(board_audio),
                "--policy", str(policy),
                "--output", str(gate),
            ]
        )
        gate_result = json.loads(gate.read_text(encoding="utf-8"))
        assert gate_result["schema_version"] == 4
        assert gate_result["qualified"] is True
        assert gate_result["training_corpus_sha256"] == result["model_lineage"]["training_corpus_sha256"]
        assert gate_result["evaluation_corpus_sha256"] == eval_corpus["corpus_sha256"]
        gate_command = [sys.executable, str(ROOT / "tools" / "qualification_gate.py"),
                        "--manifest", str(manifest),
                "--references", str(references), "--detections", str(detections),
                "--board-audio", str(board_audio), "--policy", str(policy)]
        def gate_exit(expected: int) -> None:
            checked = subprocess.run(gate_command, capture_output=True, text=True, check=False)
            assert checked.returncode == expected, checked.stdout + checked.stderr

        original_manifest = manifest.read_text(encoding="utf-8")
        for updates in ({"matched": 10, "false_rejects": 0, "false_accepts": 0,
                         "frr": 0.0, "far_per_hour": 0.0},
                        {"p95_post_end_latency_ms": 0.0},
                        {"scoring_contract": {}}):
            forged = json.loads(original_manifest)
            forged["evaluation"].update(updates)
            write_json(manifest, forged)
            gate_exit(2)
        manifest.write_text(original_manifest, encoding="utf-8")
        for section, key, replacement in (
            (None, "schema_version", 2),
            ("evidence", "schema_version", 2),
            ("evidence", "measurement_contract_id", "legacy-online-capacity"),
            ("evidence", "cpu_percent_semantics", "process_cpu_time / elapsed / online_cpu_capacity * 100"),
            ("evidence", "max_thread_count", 0),
        ):
            old = json.loads(original_manifest)
            (old if section is None else old[section])[key] = replacement
            write_json(manifest, old)
            gate_exit(2)
        manifest.write_text(original_manifest, encoding="utf-8")
        for key, replacement in (("schema_version", 2),
                                 ("measurement_contract_id", "legacy-online-capacity")):
            old_policy = dict(valid_policy)
            old_policy[key] = replacement
            write_json(policy, old_policy)
            gate_exit(2)
        write_json(policy, valid_policy)

        # Gate-only synthetic summaries test the one-core boundaries, including
        # legal >100% observations and a corresponding >100% explicit budget.
        for percent, threads, expected in ((10.0, 1, 0), (20.0, 1, 1), (150.0, 2, 1)):
            changed = json.loads(original_manifest)
            changed["evidence"].update(cpu_percent=percent, max_thread_count=threads,
                process_cpu_seconds=changed["evidence"]["wall_seconds"] * percent / 100.0)
            write_json(manifest, changed)
            gate_exit(expected)
        generous = dict(valid_policy, max_cpu_percent=150.0)
        write_json(policy, generous)
        gate_exit(0)
        write_json(policy, valid_policy)
        manifest.write_text(original_manifest, encoding="utf-8")


        original_evidence = evidence.read_text(encoding="utf-8")
        old_evidence = json.loads(original_evidence)
        old_evidence["schema_version"] = 2
        write_json(evidence, old_evidence)
        rejected = subprocess.run(command, capture_output=True, text=True, check=False)
        assert rejected.returncode == 2
        assert "target evidence schema_version must be 3" in rejected.stderr
        evidence.write_text(original_evidence, encoding="utf-8")

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
                "--references", str(references), "--detections", str(detections),
                "--board-audio", str(board_audio),
                "--policy", str(policy),
            ],
            check=False,
        ).returncode == 1

    print("test_release_qualification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
