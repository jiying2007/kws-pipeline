#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "materialize_kws_v2_research_negative_sidecar.py"


def write_wav(path: pathlib.Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [value] * 1600
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def sha256_file(path: pathlib.Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResearchNegativeSidecarTest(unittest.TestCase):
    def test_materialized_sidecar_is_portable_and_split_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            generated = root / "generated"
            requests = []
            group_rows = {"train": [], "generalization-search": []}
            fixtures = [
                ("train", "train", "voice-train", 10),
                ("calibration", "generalization-search", "voice-cal", 20),
                ("test", "generalization-search", "voice-test", 30),
            ]
            for ordinal, (split, group, voice, value) in enumerate(fixtures):
                group_dir = generated / group
                wav = group_dir / f"recording-{ordinal:06d}.wav"
                write_wav(wav, value)
                source = f"kws-v2-research-negative:{split}:slot:u{ordinal}"
                requests.append(
                    {
                        "schema_version": 1,
                        "evidence_class": "kws-v2-research-negative-request-v1",
                        "evidence_scope": "research-only",
                        "split": split,
                        "category": "fixture",
                        "utterance_id": f"u{ordinal}",
                        "text": f"测试{ordinal}",
                        "voice_slot": f"{split}-slot",
                        "voice_id": voice,
                        "source_id": source,
                        "target_policy": "empty-target-nonwake",
                        "protected_evidence_used": False,
                    }
                )
                group_rows[group].append(
                    {
                        "schema_version": 1,
                        "evidence_class": "speech-like-synthetic-recording-v1",
                        "synthetic": True,
                        "provider_kind": "offline-tts",
                        "provider_name": "fixture",
                        "provider_version": "1",
                        "license_id": "Apache-2.0",
                        "voice_id": voice,
                        "source_id": source,
                        "locale": "zh-CN",
                        "text": f"测试{ordinal}",
                        "tokens": ["nonwake"],
                        "generation_config_sha256": "a" * 64,
                        "audio": wav.name,
                        "file_sha256": sha256_file(wav),
                        "pcm_sha256": "b" * 64,
                    }
                )

            request_index = root / "request-index.jsonl"
            request_index.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests),
                encoding="utf-8",
            )
            for group, rows in group_rows.items():
                path = generated / group / "manifest.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )

            output = root / "sidecar"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--generated-root",
                    str(generated),
                    "--request-index",
                    str(request_index),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            receipt = json.loads((output / "sidecar-receipt.json").read_text())
            self.assertEqual(receipt["total_recordings"], 3)
            self.assertFalse(receipt["qualification_split_consumed"])
            self.assertEqual(receipt["target_policy"], "empty-target-nonwake")

            voices = set()
            for split in ("train", "calibration", "test"):
                item = receipt["splits"][split]
                self.assertEqual(item["recordings"], 1)
                self.assertFalse(pathlib.Path(item["tsv"]).is_absolute())
                self.assertFalse(pathlib.Path(item["references"]).is_absolute())
                tsv = output / item["tsv"]
                audio, targets = tsv.read_text().rstrip("\n").split("\t", 1)
                self.assertEqual(targets, "")
                self.assertFalse(pathlib.Path(audio).is_absolute())
                self.assertTrue((tsv.parent / audio).is_file())
                reference = json.loads((output / item["references"]).read_text())
                self.assertFalse(pathlib.Path(reference["path"]).is_absolute())
                self.assertTrue(((output / item["references"]).parent / reference["path"]).is_file())
                split_voices = set(item["voice_ids"])
                self.assertFalse(voices & split_voices)
                voices |= split_voices


if __name__ == "__main__":
    unittest.main()
