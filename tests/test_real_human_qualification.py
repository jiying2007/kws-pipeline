#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: pathlib.Path, marker: int, seconds: float = 1.0) -> None:
    frames = int(16000 * seconds)
    samples = [((index + marker * 17) % 257) - 128 for index in range(frames)]
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(
            b"".join(struct.pack("<h", value * 32) for value in samples)
        )


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expect:
        raise AssertionError(
            f"expected exit {expect}, got {completed.returncode}:\n{completed.stdout}"
        )
    return completed


def main() -> int:
    production_policy = json.loads(
        (ROOT / "commercial/real-human-qualification.policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert production_policy["shipping_approved"] is False
    assert production_policy["deployment_tag"] == "deployment-c20f3eb88e43"
    assert production_policy["minimums"]["speakers"] == 40
    assert production_policy["minimums"]["sessions_per_speaker"] == 2
    assert production_policy["minimums"]["expected_wakes_per_keyword"] == 600
    assert production_policy["minimums"]["negative_audio_hours"] == 24.0
    assert production_policy["gates"]["max_frr"] == 0.05
    assert production_policy["gates"]["max_frr_per_keyword"] == 0.05
    assert production_policy["gates"]["max_frr_upper_bound_per_keyword"] == 0.08
    assert production_policy["gates"]["max_far_per_hour"] == 0.10
    assert {
        row["name"] for row in production_policy["critical_positive_slices"]
    } >= {"distance_3_5m", "rear", "playback", "double_talk", "robot_motion"}
    assert production_policy["privacy"]["raw_audio_public_upload_allowed"] is False
    assert production_policy["privacy"]["post_afe_audio_public_upload_allowed"] is False

    summary_schema = json.loads(
        (ROOT / "commercial/real-human-qualification-summary.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary_schema["properties"]["shipping_approved"]["const"] is False
    assert "critical_negative_slices" in summary_schema["required"]

    workflow = (ROOT / ".github/workflows/real-human-qualification.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow.split("permissions:", 1)[0]
    assert "runs-on: [self-hosted, kws-real-audio]" in workflow
    assert "corpus_manifest_path:" not in workflow
    assert "audio_root:" not in workflow
    assert "afe_adapter_path:" not in workflow
    assert "qualification_id:" in workflow and "afe_profile:" in workflow
    assert "KWS_REAL_HUMAN_ROOT" in workflow and "KWS_FINAL_AFE_ROOT" in workflow
    consume = "Consume fresh corpus before final AFE or KWS evaluation"
    afe_run = "Execute frozen final AFE over consumed held-out corpus"
    kws_run = "Run frozen KWS over post-AFE human audio"
    assert consume in workflow and afe_run in workflow and kws_run in workflow
    assert workflow.index(consume) < workflow.index(afe_run) < workflow.index(kws_run)
    assert "human-exposure-${corpus_sha:0:16}" in workflow
    assert "final_afe_identity_sha256" in workflow
    assert "--expected-identity build/real-human/final-afe-identity.json" in workflow
    assert "raw audio remains restricted" in workflow.lower()
    upload_section = workflow.split(
        "Retain qualification evidence without raw or post-AFE audio", 1
    )[1].split("Publish successful Phase-A receipt", 1)[0]
    assert "build/real-human/afe/post-afe" not in upload_section
    assert "*.wav" not in upload_section
    assert "shipping_approved:false" in workflow

    with tempfile.TemporaryDirectory(prefix="kws-real-human-contract-") as tmp_raw:
        tmp = pathlib.Path(tmp_raw)
        audio_root = tmp / "audio"
        audio_root.mkdir()
        for index, name in enumerate(("k1.wav", "k2.wav", "neg.wav"), 1):
            write_wav(audio_root / name, index)

        draft = {
            "schema_version": 1,
            "qualification_id": "fixture-q-0001",
            "deployment_tag": "deployment-fixture0000",
            "corpus_role": "fresh-held-out-qualification",
            "recordings": [],
        }
        rows = [
            (
                "k1",
                "k1.wav",
                "spk-a",
                "s1",
                [{"keyword_id": 1, "start_s": 0.10, "end_s": 0.30}],
                ["household"],
            ),
            (
                "k2",
                "k2.wav",
                "spk-a",
                "s1",
                [{"keyword_id": 2, "start_s": 0.10, "end_s": 0.30}],
                ["playback"],
            ),
            (
                "neg",
                "neg.wav",
                "ambient-a",
                "n1",
                [],
                ["household", "speech_confusion", "playback", "robot_motion"],
            ),
        ]
        for recording, filename, speaker, session, expected, tags in rows:
            draft["recordings"].append(
                {
                    "recording": recording,
                    "input_path": filename,
                    "speaker_id": speaker,
                    "session_id": session,
                    "source_id": f"src-{recording}",
                    "room_id": "room-a",
                    "device_id": "device-a",
                    "distance_m": 0.5,
                    "azimuth_deg": 0.0,
                    "snr_db": 20.0,
                    "tags": tags,
                    "expected": expected,
                    "consent_scope": "product-kws-qualification",
                    "retention_class": "restricted-raw-audio",
                }
            )
        draft_path = tmp / "draft.json"
        draft_path.write_text(json.dumps(draft, indent=2) + "\n", encoding="utf-8")
        manifest_path = tmp / "manifest.json"
        run(
            "python3",
            "tools/seal_real_human_corpus.py",
            "--draft",
            str(draft_path),
            "--audio-root",
            str(audio_root),
            "--output",
            str(manifest_path),
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["recordings"]) == 3
        for row in manifest["recordings"]:
            path = audio_root / row["input_path"]
            assert row["input_sha256"] == sha(path)
            assert row["input_bytes"] == path.stat().st_size
            assert row["capture"] == {
                "sample_rate_hz": 16000,
                "channels": 1,
                "sample_format": "pcm_s16le",
            }
            assert abs(row["duration_s"] - 1.0) < 1e-12

        policy = {
            "schema_version": 1,
            "deployment_tag": "deployment-fixture0000",
            "confidence_level": 0.95,
            "minimums": {
                "speakers": 1,
                "sessions_per_speaker": 1,
                "expected_wakes_total": 2,
                "expected_wakes_per_keyword": 1,
                "negative_audio_hours": 1.0 / 3600.0,
            },
            "gates": {
                "max_frr": 0.01,
                "max_frr_upper_bound": 0.90,
                "max_frr_per_keyword": 0.01,
                "max_frr_upper_bound_per_keyword": 0.90,
                "max_far_per_hour": 0.01,
                "max_far_upper_bound_per_hour": 20000.0,
                "max_p95_post_end_latency_ms": 500.0,
            },
            "critical_positive_slices": [],
            "critical_negative_exposure_hours": {
                "household": 1.0 / 3600.0
            },
            "identity_policy": {"forbidden_fields": ["name", "email", "phone"]},
        }
        policy_path = tmp / "policy.json"
        policy_path.write_text(
            json.dumps(policy, indent=2) + "\n", encoding="utf-8"
        )
        intake = tmp / "intake.json"
        run(
            "python3",
            "tools/validate_real_human_corpus.py",
            "--manifest",
            str(manifest_path),
            "--audio-root",
            str(audio_root),
            "--policy",
            str(policy_path),
            "--public-summary",
            str(intake),
        )
        intake_json = json.loads(intake.read_text(encoding="utf-8"))
        assert intake_json["raw_audio_hashes_verified"] is True
        assert intake_json["expected_by_keyword"] == {"1": 1, "2": 1}

        bad_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad_manifest["recordings"][0]["email"] = "forbidden@example.invalid"
        bad_path = tmp / "bad-manifest.json"
        bad_path.write_text(json.dumps(bad_manifest) + "\n", encoding="utf-8")
        run(
            "python3",
            "tools/validate_real_human_corpus.py",
            "--manifest",
            str(bad_path),
            "--audio-root",
            str(audio_root),
            "--policy",
            str(policy_path),
            "--public-summary",
            str(tmp / "bad.json"),
            expect=2,
        )

        fake_afe = tmp / "fake_afe.py"
        fake_afe.write_text(
            "#!/usr/bin/env python3\n"
            "import json, shutil, sys\n"
            "_, src, dst, result = sys.argv\n"
            "shutil.copyfile(src, dst)\n"
            "open(result, 'w', encoding='utf-8').write(json.dumps({'latency_samples': 0}) + '\\n')\n",
            encoding="utf-8",
        )
        fake_afe.chmod(0o755)
        config = tmp / "afe.cfg"
        config.write_text("fixture=true\n", encoding="utf-8")
        adapter = {
            "schema_version": 1,
            "executable_path": str(fake_afe),
            "command_argv": [
                "{executable}",
                "{input}",
                "{output}",
                "{result}",
            ],
            "config_files": [str(config)],
            "pipeline_source_sha": "1" * 40,
            "sku": "fixture",
            "microphone_revision": "mic-a",
            "enclosure_revision": "enc-a",
            "audio_route": "fixture-route",
            "toolchain": "fixture-python",
        }
        adapter_path = tmp / "adapter.json"
        adapter_path.write_text(
            json.dumps(adapter, indent=2) + "\n", encoding="utf-8"
        )
        identity_path = tmp / "afe-identity.json"
        run(
            "python3",
            "tools/final_afe_identity.py",
            "--adapter",
            str(adapter_path),
            "--output",
            str(identity_path),
        )
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        assert identity["shipping_authority"] is True
        assert "command_argv" not in identity
        assert len(identity["command_argv_sha256"]) == 64
        assert len(identity["identity_sha256"]) == 64

        afe_root = tmp / "afe-out"
        run(
            "python3",
            "tools/run_final_afe_corpus.py",
            "--manifest",
            str(manifest_path),
            "--audio-root",
            str(audio_root),
            "--adapter",
            str(adapter_path),
            "--expected-identity",
            str(identity_path),
            "--output-dir",
            str(afe_root),
        )
        afe_summary = json.loads(
            (afe_root / "afe-corpus-summary.json").read_text(encoding="utf-8")
        )
        assert afe_summary["afe"]["shipping_authority"] is True
        assert afe_summary["afe"]["identity_sha256"] == identity["identity_sha256"]
        assert afe_summary["latency_samples"] == {"min": 0, "max": 0}

        detections = tmp / "detections.jsonl"
        detections.write_text(
            json.dumps(
                {
                    "recording": "k1",
                    "keyword_id": 1,
                    "time_s": 0.30,
                    "confidence": 0.9,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "recording": "k2",
                    "keyword_id": 2,
                    "time_s": 0.30,
                    "confidence": 0.9,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        score_summary = tmp / "score.json"
        false_accepts = tmp / "fa.jsonl"
        false_rejects = tmp / "fr.jsonl"
        references = afe_root / "references.post-afe.jsonl"
        run(
            "python3",
            "eval/score_events.py",
            "--references",
            str(references),
            "--detections",
            str(detections),
            "--summary",
            str(score_summary),
            "--false-positives",
            str(false_accepts),
            "--false-rejects",
            str(false_rejects),
        )
        final = tmp / "qualification.json"
        run(
            "python3",
            "tools/score_real_human_qualification.py",
            "--manifest",
            str(manifest_path),
            "--policy",
            str(policy_path),
            "--intake-summary",
            str(intake),
            "--afe-summary",
            str(afe_root / "afe-corpus-summary.json"),
            "--references",
            str(references),
            "--score-summary",
            str(score_summary),
            "--false-accepts",
            str(false_accepts),
            "--false-rejects",
            str(false_rejects),
            "--detections",
            str(detections),
            "--output",
            str(final),
        )
        result = json.loads(final.read_text(encoding="utf-8"))
        assert result["qualified"] is True
        assert result["shipping_approved"] is False
        assert result["false_rejects"] == 0
        assert result["false_accepts_negative"] == 0
        assert abs(result["negative_audio_hours"] - 1.0 / 3600.0) < 1e-12
        assert result["per_keyword"]["1"]["frr"] == 0.0
        assert result["per_keyword"]["2"]["frr"] == 0.0
        assert abs(
            result["critical_negative_slices"]["household"]["audio_hours"]
            - 1.0 / 3600.0
        ) < 1e-12
        assert result["next_gate"] == "physical-target-board-performance-and-soak"

        # Once the identity has been frozen, changing any AFE config must fail
        # before the consumed holdout is evaluated with a different tuple.
        config.write_text("fixture=false\n", encoding="utf-8")
        run(
            "python3",
            "tools/run_final_afe_corpus.py",
            "--manifest",
            str(manifest_path),
            "--audio-root",
            str(audio_root),
            "--adapter",
            str(adapter_path),
            "--expected-identity",
            str(identity_path),
            "--output-dir",
            str(tmp / "drift-out"),
            expect=2,
        )

    print("test_real_human_qualification: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
