from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: pathlib.Path, value: int) -> None:
    samples = [value] * 16000
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


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
    schema = json.loads(
        (ROOT / "commercial" / "real-human-development-corpus.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["corpus_role"]["const"] == "development-feedback"
    assert schema["properties"]["recordings"]["items"]["properties"]["consent_scope"][
        "const"
    ] == "product-kws-development"

    workflow = (
        ROOT / ".github" / "workflows" / "restricted-development-dataset-iteration.yml"
    ).read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, kws-real-audio]" in workflow
    assert "KWS_REAL_HUMAN_DEVELOPMENT_ROOT" in workflow
    assert "KWS_FINAL_AFE_ROOT" in workflow
    assert "KWS_MODEL_CANDIDATE_ROOT" in workflow
    assert "candidate.json" in workflow
    assert "--variable model" in workflow
    assert "--protocol restricted-development-dataset-iteration-v1" in workflow
    assert "--pre-tolerance-ms 0" in workflow
    assert "--post-tolerance-ms 500" in workflow
    assert "feedback_allowed':True" in workflow
    assert "qualification_authority':False" in workflow
    assert "shipping_authority':False" in workflow
    assert "human-exposure-" not in workflow
    assert "Consume fresh corpus" not in workflow
    assert "Purge restricted audio before evidence retention" in workflow
    upload = workflow.split("Upload public-safe development evidence", 1)[1]
    assert "afe/post-afe/" not in upload
    assert "*.wav" not in upload

    with tempfile.TemporaryDirectory(prefix="kws-development-feedback-") as raw:
        root = pathlib.Path(raw)
        audio = root / "wav"
        audio.mkdir()
        for index, name in enumerate(("k1.wav", "k2.wav", "neg.wav"), 1):
            write_wav(audio / name, index * 50)

        rows = [
            ("k1", "k1.wav", "spk-a", [{"keyword_id": 1, "start_s": 0.1, "end_s": 0.3}]),
            ("k2", "k2.wav", "spk-b", [{"keyword_id": 2, "start_s": 0.1, "end_s": 0.3}]),
            ("neg", "neg.wav", "ambient-a", []),
        ]
        draft = {
            "schema_version": 1,
            "dataset_id": "development-fixture-0001",
            "corpus_role": "development-feedback",
            "recordings": [],
        }
        for recording, filename, speaker, expected in rows:
            draft["recordings"].append(
                {
                    "recording": recording,
                    "input_path": filename,
                    "speaker_id": speaker,
                    "session_id": "session-a",
                    "source_id": f"source-{recording}",
                    "room_id": "room-a",
                    "device_id": "device-a",
                    "distance_m": 1.0,
                    "azimuth_deg": 0.0,
                    "snr_db": 20.0,
                    "tags": ["household"],
                    "expected": expected,
                    "consent_scope": "product-kws-development",
                    "retention_class": "restricted-raw-audio",
                }
            )
        draft_path = root / "draft.json"
        sealed = root / "sealed.json"
        draft_path.write_text(json.dumps(draft), encoding="utf-8")
        run(
            "python3", "tools/seal_real_human_corpus.py",
            "--draft", str(draft_path),
            "--audio-root", str(audio),
            "--output", str(sealed),
        )
        manifest = json.loads(sealed.read_text(encoding="utf-8"))
        assert manifest["corpus_role"] == "development-feedback"
        assert manifest["dataset_id"] == "development-fixture-0001"
        assert all("input_sha256" in row for row in manifest["recordings"])

        summary = root / "development-summary.json"
        run(
            "python3", "tools/validate_real_human_development_corpus.py",
            "--manifest", str(sealed),
            "--audio-root", str(audio),
            "--public-summary", str(summary),
        )
        value = json.loads(summary.read_text(encoding="utf-8"))
        assert value["feedback_allowed"] is True
        assert value["repeatable"] is True
        assert value["qualification_authority"] is False
        assert value["shipping_authority"] is False
        assert value["expected_by_keyword"] == {"1": 1, "2": 1}
        assert value["negative_audio_hours"] > 0.0

        fake_afe = root / "fake_afe.py"
        fake_afe.write_text(
            "#!/usr/bin/env python3\n"
            "import json, shutil, sys\n"
            "_, src, dst, result = sys.argv\n"
            "shutil.copyfile(src, dst)\n"
            "open(result, 'w', encoding='utf-8').write(json.dumps({'latency_samples': 0}) + '\\n')\n",
            encoding="utf-8",
        )
        fake_afe.chmod(0o755)
        cfg = root / "afe.cfg"
        cfg.write_text("fixture=true\n", encoding="utf-8")
        adapter = root / "adapter.json"
        adapter.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "executable_path": str(fake_afe),
                    "command_argv": ["{executable}", "{input}", "{output}", "{result}"],
                    "config_files": [str(cfg)],
                    "pipeline_source_sha": "1" * 40,
                    "sku": "fixture",
                    "microphone_revision": "mic-a",
                    "enclosure_revision": "enc-a",
                    "audio_route": "fixture-route",
                    "toolchain": "fixture-python",
                }
            ),
            encoding="utf-8",
        )
        identity = root / "afe-identity.json"
        run(
            "python3", "tools/final_afe_identity.py",
            "--adapter", str(adapter),
            "--output", str(identity),
        )
        afe_out = root / "afe-out"
        run(
            "python3", "tools/run_final_afe_corpus.py",
            "--manifest", str(sealed),
            "--audio-root", str(audio),
            "--adapter", str(adapter),
            "--expected-identity", str(identity),
            "--output-dir", str(afe_out),
        )
        afe_summary = json.loads(
            (afe_out / "afe-corpus-summary.json").read_text(encoding="utf-8")
        )
        assert afe_summary["corpus_role"] == "development-feedback"
        assert afe_summary["corpus_id"] == "development-fixture-0001"
        assert afe_summary["dataset_id"] == "development-fixture-0001"
        assert "qualification_id" not in afe_summary
        assert "deployment_tag" not in afe_summary
        assert len(afe_summary["post_afe_corpus_sha256"]) == 64

        bad = json.loads(sealed.read_text(encoding="utf-8"))
        bad["recordings"][0]["email"] = "forbidden@example.invalid"
        bad_path = root / "bad.json"
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        run(
            "python3", "tools/validate_real_human_development_corpus.py",
            "--manifest", str(bad_path),
            "--audio-root", str(audio),
            "--public-summary", str(root / "bad-summary.json"),
            expect=2,
        )

    print("test_restricted_development_dataset: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
