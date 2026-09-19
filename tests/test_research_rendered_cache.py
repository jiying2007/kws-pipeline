#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "materialize_kws_v2_research_rendered_cache.py"


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_wav(path: pathlib.Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = [value] * 1600
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(struct.pack("<" + "h" * len(samples), *samples))


class ResearchRenderedCacheTest(unittest.TestCase):
    def build_source(self, root: pathlib.Path) -> pathlib.Path:
        source = root / "rendered"
        rows = []
        for ordinal, split in enumerate(("train", "calibration", "test")):
            wav = source / "clips" / split / f"{ordinal:03d}.wav"
            write_wav(wav, 100 + ordinal)
            targets = "1 2 3 4" if split == "train" else ""
            (source / f"{split}.tsv").write_text(
                f"{wav.resolve()}\t{targets}\n",
                encoding="utf-8",
            )
            if split != "train":
                (source / f"{split}.references.jsonl").write_text(
                    json.dumps(
                        {
                            "recording": f"{split}-0",
                            "audio_path": str(wav.resolve()),
                            "path": wav.name,
                            "duration_s": 0.1,
                            "expected": [],
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            rows.append(
                {
                    "split": split,
                    "path": str(wav.resolve()),
                    "wav_sha256": sha256_file(wav),
                }
            )
        index = source / "domain-index.jsonl"
        index.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "config_sha256": "a" * 64,
            "domain_index_sha256": sha256_file(index),
        }
        (source / "domain-summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        return source

    def test_cache_survives_relocation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            source = self.build_source(root)
            # Simulate a downloaded GitHub artifact whose manifests still
            # contain the producing runner's now-stale absolute prefix.
            train_wav = source / "clips" / "train" / "000.wav"
            (source / "train.tsv").write_text(
                f"/stale/runner/work/_temp/research-result/dataset/clips/train/000.wav\t1 2 3 4\n",
                encoding="utf-8",
            )
            self.assertTrue(train_wav.is_file())

            output = root / "portable"
            subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--source-dataset",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
            )
            receipt = json.loads((output / "cache-receipt.json").read_text())
            self.assertTrue(receipt["portable_paths_only"])
            self.assertFalse(receipt["qualification_split_consumed"])
            self.assertEqual(receipt["unique_wav_files"], 3)

            relocated = root / "relocated"
            shutil.move(str(output), relocated)
            for split in ("train", "calibration", "test"):
                line = (relocated / f"{split}.tsv").read_text().strip("\n")
                audio, _ = line.split("\t", 1)
                self.assertFalse(pathlib.Path(audio).is_absolute())
                self.assertTrue((relocated / audio).is_file())
                if split != "train":
                    ref = json.loads((relocated / f"{split}.references.jsonl").read_text())
                    self.assertFalse(pathlib.Path(ref["path"]).is_absolute())
                    self.assertTrue((relocated / ref["path"]).is_file())

    def test_audio_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            source = self.build_source(root)
            outside = root / "outside.wav"
            write_wav(outside, 999)
            (source / "train.tsv").write_text(
                f"{outside.resolve()}\t1 2 3 4\n",
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL),
                    "--source-dataset",
                    str(source),
                    "--output-dir",
                    str(root / "portable"),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("escapes rendered dataset root", completed.stderr)


if __name__ == "__main__":
    unittest.main()
