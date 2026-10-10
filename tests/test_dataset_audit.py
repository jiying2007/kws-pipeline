#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

from qualification_fixture import write_wav

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "training" / "audit_dataset.py"
sys.path.insert(0, str(ROOT / "training"))
from audit_dataset import (  # noqa: E402
    audit_splits, inspect_wav, lineage_path, sha256_file,
    verify_manifest_lineage, write_lineage_sidecar,
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
         str(AUDITOR), *args],
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


def test_bound_lineage_sidecars() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        paths = {}
        for index, name in enumerate(("source-a", "source-b", "render-a", "render-b")):
            wav = root / f"{name}.wav"
            write_wav(wav, seconds=1)
            data = bytearray(wav.read_bytes())
            data[-2:] = (index + 100).to_bytes(2, "little", signed=True)
            wav.write_bytes(data)
            paths[name] = wav

        def record(name: str, source: str, family: str) -> dict:
            source_path = paths[source]
            return {
                "path": str(paths[name]), "target_ids": [1, 2],
                "wav_sha256": sha256_file(paths[name]),
                "source_path": str(source_path),
                "source_wav_sha256": sha256_file(source_path),
                "source_pcm_sha256": inspect_wav(source_path)[2],
                "source_family_id": family,
                "speech_like_provenance": {"provider_kind": "offline-tts",
                    "provider_name": "shared-engine", "source_id": "shared-engine-config"},
                "admission": {"mode": "synthetic-fixture-v1", "ctc_training_allowed": False},
            }

        a = record("render-a", "source-a", "family-a")
        b = record("render-b", "source-b", "family-b")
        train, test = root / "train.tsv", root / "test.tsv"
        train.write_text(a["path"] + "\t1 2\n", encoding="utf-8")
        test.write_text(b["path"] + "\t1 2\n", encoding="utf-8")
        write_lineage_sidecar(train, [a])
        write_lineage_sidecar(test, [b])
        specs = [("train", train), ("test", test)]
        report = audit_splits(specs, require_lineage=True)
        assert report["clean"]
        assert report["identity_disjointness"]["source_pcm_sha256"] == "verified"
        assert report["identity_disjointness"]["speaker_id"] == "unknown"
        assert report["identity_disjointness"]["source_family_id"] == "verified"
        assert report["clean_scope"] == "observed-identities-only"
        assert report["splits"]["test"]["source_lineage_verified_rows"] == 1
        assert len(report["splits"]["train"]["lineage_sidecar_sha256"]) == 64
        assert verify_manifest_lineage(train) == [a]
        jsonl = root / "train.jsonl"
        jsonl.write_text(json.dumps(a) + "\n", encoding="utf-8")
        assert verify_manifest_lineage(jsonl) == [a]

        # New augmentation/noise gives distinct final PCM, but original PCM is
        # still shared. Split-specific family names cannot conceal this leak.
        leaked = record("render-b", "source-a", "family-b")
        write_lineage_sidecar(test, [leaked])
        report = audit_splits(specs, require_lineage=True)
        assert report["cross_split_leaks"]
        assert not report["clean"]
        assert any(item["field"] == "source_pcm_sha256" for item in report["identity_violations"])
        write_lineage_sidecar(test, [b])

        for field, identity in (("source_family_id", "family-shared"),
                                ("family_id", "ancestor-shared"),
                                ("speaker_id", "person-a"), ("session_id", "session-a"),
                                ("reference_audio_sha256", "a" * 64),
                                ("derivation_family_id", "derivation-a")):
            write_lineage_sidecar(train, [{**a, field: identity}])
            write_lineage_sidecar(test, [{**b, field: identity}])
            report = audit_splits(specs, require_lineage=True)
            assert not report["clean"], field
            assert any(item["field"] == field for item in report["identity_violations"])
        write_lineage_sidecar(train, [a])
        write_lineage_sidecar(test, [b])

        def rejected(expected: str) -> None:
            try:
                verify_manifest_lineage(train)
            except ValueError as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError(f"invalid lineage accepted: {expected}")

        # The consumer sees exactly the target IDs covered by the sidecar.
        original_tsv = train.read_text()
        train.write_text(a["path"] + "\t2 1\n", encoding="utf-8")
        rejected("manifest binding mismatch")
        train.write_text(original_tsv, encoding="utf-8")
        write_lineage_sidecar(train, [{**a, "target_ids": [2, 1]}])
        rejected("row binding mismatch")
        for invalid_targets in ([True, 2], [1.0, 2]):
            write_lineage_sidecar(train, [{**a, "target_ids": invalid_targets}])
            rejected("target_ids must be an integer list")
        write_lineage_sidecar(train, [a, a])
        rejected("row count mismatch")
        write_lineage_sidecar(train, [a])

        source_bytes = paths["source-a"].read_bytes()
        paths["source-a"].write_bytes(source_bytes[:-2] + b"xx")
        rejected("source WAV/PCM identity mismatch")
        paths["source-a"].write_bytes(source_bytes)
        rendered_bytes = paths["render-a"].read_bytes()
        paths["render-a"].write_bytes(rendered_bytes[:-2] + b"xx")
        rejected("rendered WAV hash mismatch")
        paths["render-a"].write_bytes(rendered_bytes)

        invalid = copy.deepcopy(a)
        invalid["source_pcm_sha256"] = "b" * 64
        write_lineage_sidecar(train, [invalid])
        rejected("source WAV/PCM identity mismatch")
        lineage_path(train).unlink()
        rejected("source lineage is required")
        unknown = audit_splits(specs)
        assert unknown["identity_disjointness"]["source_pcm_sha256"] == "unknown"


class CrossFieldPCMTests(unittest.TestCase):
    """Synthetic WAVs only; assertions remain active under -O and -OO."""

    def test_all_final_source_original_pcm_pairs_share_one_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            paths = []
            for index in range(3):
                path = root / f"invented-{index}.wav"
                with wave.open(str(path), "wb") as output:
                    output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    output.writeframes(struct.pack("<4h", 1, 2, 3, index + 1))
                paths.append(path)
            common, first, second = paths
            common_pcm = inspect_wav(common)[2]

            def record(field, rendered, split):
                row = metadata_row(str(common if field == "pcm_sha256" else rendered),
                                   speaker=split, session=split, source=split)
                if field == "source_pcm_sha256":
                    row.update(source_path=str(common), source_wav_sha256=sha256_file(common),
                               source_pcm_sha256=common_pcm)
                elif field == "original_pcm_sha256":
                    row[field] = common_pcm
                return row

            fields = ("pcm_sha256", "source_pcm_sha256", "original_pcm_sha256")
            for left in fields:
                for right in fields:
                    with self.subTest(left=left, right=right):
                        specs = []
                        for split, field, path in (("train", left, first), ("heldout", right, second)):
                            manifest = root / (split + ".jsonl")
                            manifest.write_text(json.dumps(record(field, path, split)) + "\n")
                            specs.append((split, manifest))
                        result = audit_splits(specs, require_metadata=("speaker_id", "session_id", "source_id"))
                        self.assertFalse(result["clean"])
                        leak = next(item for item in result["cross_split_leaks"]
                                    if item["pcm_sha256"] == common_pcm)
                        self.assertEqual(leak["splits"], ["heldout", "train"])
                        observations = {item["split"]: item for item in leak["observations"]}
                        self.assertEqual(observations["train"]["field"], left)
                        self.assertEqual(observations["heldout"]["field"], right)
                        if "source_pcm_sha256" in (left, right):
                            self.assertEqual(result["identity_disjointness"]["source_pcm_sha256"], "violated")
                        for item in observations.values():
                            if item["field"] == "original_pcm_sha256":
                                self.assertEqual(item["evidence_basis"], "declared-original-pcm")
                                self.assertIsNone(item["observed_path"])
                            else:
                                self.assertEqual(item["observed_path"], str(common))

            # A bare TSV heldout row still contributes its decoded final PCM.
            train = root / "train.jsonl"
            train.write_text(json.dumps(record("source_pcm_sha256", first, "train")) + "\n")
            heldout = root / "heldout.tsv"
            heldout.write_text(str(common) + "\t1 2\n")
            result = audit_splits([("train", train), ("heldout", heldout)])
            self.assertFalse(result["clean"])
            self.assertEqual(len(result["cross_split_leaks"]), 1)
            cli = run("--split", f"train={train}", "--split", f"heldout={heldout}")
            self.assertEqual(cli.returncode, 1, cli.stderr)
            self.assertFalse(json.loads(cli.stdout)["clean"])

            # Hash-shaped family/reference metadata is a different namespace.
            unrelated = metadata_row(str(second), speaker="heldout", session="heldout", source="heldout")
            unrelated.update(source_family_id=common_pcm, reference_audio_sha256=common_pcm)
            other = root / "other.jsonl"
            other.write_text(json.dumps(unrelated) + "\n")
            self.assertTrue(audit_splits([("train", train), ("heldout", other)])["clean"])
            self.assertTrue(audit_splits([("train", train)])["clean"])


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CrossFieldPCMTests)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        return 1
    test_bound_lineage_sidecars()
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
