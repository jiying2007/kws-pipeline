#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile

from qualification_fixture import write_wav

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "training" / "audit_dataset.py"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(AUDITOR), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def add_junk_chunk(source: pathlib.Path, destination: pathlib.Path) -> None:
    data = bytearray(source.read_bytes())
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    chunk = b"JUNK" + struct.pack("<I", 4) + b"meta"
    riff_size = struct.unpack_from("<I", data, 4)[0]
    struct.pack_into("<I", data, 4, riff_size + len(chunk))
    destination.write_bytes(data[:12] + chunk + data[12:])


def metadata_row(path: str, *, speaker: str, session: str, source: str) -> dict:
    return {
        "audio": path,
        "speaker_id": speaker,
        "session_id": session,
        "source_id": source,
        "room_id": "room-a",
        "device_id": "device-a",
        "target_ids": [1, 2],
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        train_dir = root / "train"
        eval_dir = root / "eval"
        train_dir.mkdir()
        eval_dir.mkdir()

        write_wav(train_dir / "a.wav", seconds=1)
        write_wav(train_dir / "b.wav", seconds=1)
        data = bytearray((train_dir / "b.wav").read_bytes())
        data[-2:] = (123).to_bytes(2, "little", signed=True)
        (train_dir / "b.wav").write_bytes(data)
        write_wav(eval_dir / "c.wav", seconds=1)
        data = bytearray((eval_dir / "c.wav").read_bytes())
        data[-2:] = (-321).to_bytes(2, "little", signed=True)
        (eval_dir / "c.wav").write_bytes(data)

        train_manifest = root / "train.tsv"
        train_manifest.write_text("train/a.wav\t1 2\ntrain/b.wav\t\n", encoding="utf-8")
        eval_manifest = root / "eval.jsonl"
        eval_manifest.write_text(
            json.dumps(
                {
                    "recording": "eval-c",
                    "path": "eval/c.wav",
                    "duration_s": 1.0,
                    "expected": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report = root / "audit.json"

        clean = run(
            "--split",
            f"train={train_manifest}",
            "--split",
            f"eval={eval_manifest}",
            "--report",
            str(report),
        )
        assert clean.returncode == 0, clean.stderr
        result = json.loads(report.read_text(encoding="utf-8"))
        assert result["schema_version"] == 3
        assert result["audio_identity"] == "decoded-mono-16khz-pcm16-sha256"
        assert result["clean"] is True
        assert result["cross_split_leaks"] == []
        assert result["identity_violations"] == []
        assert result["splits"]["train"]["examples"] == 2
        assert result["splits"]["eval"]["examples"] == 1

        # Real-human manifests can use the richer JSONL schema. PCM can differ
        # while speaker/session/source leakage still invalidates held-out evidence.
        train_jsonl = root / "train-real.jsonl"
        eval_jsonl = root / "eval-real.jsonl"
        train_jsonl.write_text(
            json.dumps(
                metadata_row(
                    "train/a.wav",
                    speaker="speaker-01",
                    session="session-01",
                    source="source-01",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        eval_jsonl.write_text(
            json.dumps(
                metadata_row(
                    "eval/c.wav",
                    speaker="speaker-01",
                    session="session-02",
                    source="source-02",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        identity_leak = run(
            "--split",
            f"train={train_jsonl}",
            "--split",
            f"qualification={eval_jsonl}",
            "--require-metadata",
            "speaker_id",
            "--require-metadata",
            "session_id",
            "--require-metadata",
            "source_id",
        )
        assert identity_leak.returncode == 1, identity_leak.stderr
        identity_result = json.loads(identity_leak.stdout)
        assert identity_result["cross_split_leaks"] == []
        assert any(
            leak["field"] == "speaker_id"
            for leak in identity_result["identity_violations"]
        )

        eval_jsonl.write_text(
            json.dumps(
                metadata_row(
                    "eval/c.wav",
                    speaker="speaker-02",
                    session="session-02",
                    source="source-02",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        room_allowed = run(
            "--split",
            f"train={train_jsonl}",
            "--split",
            f"qualification={eval_jsonl}",
            "--require-metadata",
            "speaker_id",
            "--require-metadata",
            "session_id",
            "--require-metadata",
            "source_id",
        )
        assert room_allowed.returncode == 0, room_allowed.stderr
        room_failed = run(
            "--split",
            f"train={train_jsonl}",
            "--split",
            f"qualification={eval_jsonl}",
            "--fail-room-overlap",
        )
        assert room_failed.returncode == 1
        assert any(
            leak["field"] == "room_id"
            for leak in json.loads(room_failed.stdout)["identity_violations"]
        )

        missing_required = run(
            "--split",
            f"eval={eval_manifest}",
            "--require-metadata",
            "speaker_id",
        )
        assert missing_required.returncode == 1
        assert json.loads(missing_required.stdout)["missing_metadata"]

        # Re-wrap identical PCM with an extra RIFF metadata chunk. Container
        # bytes differ, but decoded audio identity must still catch leakage.
        add_junk_chunk(train_dir / "a.wav", eval_dir / "rewrapped.wav")
        assert (train_dir / "a.wav").read_bytes() != (eval_dir / "rewrapped.wav").read_bytes()
        eval_manifest.write_text(
            json.dumps(
                {
                    "recording": "rewrapped-leak",
                    "path": "eval/rewrapped.wav",
                    "duration_s": 1.0,
                    "expected": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        leaked = run(
            "--split",
            f"train={train_manifest}",
            "--split",
            f"eval={eval_manifest}",
        )
        assert leaked.returncode == 1, leaked.stderr
        leaked_result = json.loads(leaked.stdout)
        assert leaked_result["clean"] is False
        assert len(leaked_result["cross_split_leaks"]) == 1
        leak = leaked_result["cross_split_leaks"][0]
        assert leak["splits"] == ["eval", "train"]
        assert len(leak["file_sha256"]) == 2
        assert isinstance(leak["pcm_sha256"], str) and len(leak["pcm_sha256"]) == 64

        duplicate_manifest = root / "dup.tsv"
        duplicate_manifest.write_text(
            "train/a.wav\t1 2\ntrain/a.wav\t1 2\n", encoding="utf-8"
        )
        duplicate = run(
            "--split",
            f"dup={duplicate_manifest}",
            "--fail-within-split",
        )
        assert duplicate.returncode == 1
        duplicate_result = json.loads(duplicate.stdout)
        assert len(duplicate_result["within_split_duplicates"]) == 1
        assert "pcm_sha256" in duplicate_result["within_split_duplicates"][0]

    print("test_dataset_audit: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
