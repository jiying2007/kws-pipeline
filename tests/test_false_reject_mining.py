#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MINER = ROOT / "eval" / "mine_false_rejects.py"


def write_wav(path: pathlib.Path, seconds: int = 3) -> None:
    frames = bytearray()
    for index in range(16000 * seconds):
        sample = 4000 if ((index // 40) & 1) else -4000
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(frames)


def run_miner(
    false_rejects: pathlib.Path,
    keywords: pathlib.Path,
    tokens: pathlib.Path,
    root: pathlib.Path,
    output_dir: pathlib.Path,
    manifest: pathlib.Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-S", "-B",
            *(["-" + "O" * sys.flags.optimize] if sys.flags.optimize else []),
            str(MINER),
            "--false-rejects",
            str(false_rejects),
            "--keywords",
            str(keywords),
            "--tokens",
            str(tokens),
            "--audio-root",
            str(root),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(manifest),
            *extra,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )



class RequestedCropCompletenessTests(unittest.TestCase):
    """Tiny invented PCM containers only; no training, models or real audio."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="false-reject-crop-")
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.pcm = struct.pack("<16h", *range(1, 17))
        self.tokens = self.root / "tokens.txt"
        self.tokens.write_text("<blk> 0\nunit 1\ntail 2\n", encoding="utf-8")
        self.keywords = self.root / "keywords.tsv"
        self.keywords.write_text("1\tartificial\t0.55\tunit tail\n", encoding="utf-8")

    @staticmethod
    def container(pcm: bytes, *, before: bytes = b"", trailing: bytes = b"") -> bytes:
        fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
        body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt + before
        body += b"data" + struct.pack("<I", 32) + pcm
        return b"RIFF" + struct.pack("<I", len(body) + 32 - len(pcm)) + body + trailing

    def invoke(self, name: str, sources: list[tuple[bytes, int, int]], context: int = 0):
        refs = self.root / (name + ".jsonl")
        rows = []
        inputs = [self.tokens, self.keywords, refs]
        for index, (raw, start, end) in enumerate(sources):
            source = self.root / f"{name}-{index}.wav"
            source.write_bytes(raw)
            inputs.append(source)
            rows.append({"recording": f"r{index}", "path": str(source),
                         "keyword_id": 1, "start_s": start / 16000,
                         "end_s": end / 16000, "duration_s": 16 / 16000})
        refs.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
        output = self.root / (name + "-clips")
        manifest = self.root / (name + ".tsv")
        completed = run_miner(refs, self.keywords, self.tokens, self.root,
                              output, manifest, "--context-s", str(context / 16000))
        self.assertEqual(before, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in inputs})
        return completed, output, manifest

    def assert_clip(self, path: pathlib.Path, expected: bytes) -> None:
        with wave.open(str(path), "rb") as reader:
            self.assertEqual((reader.getnchannels(), reader.getsampwidth(), reader.getframerate()),
                             (1, 2, 16000))
            self.assertEqual(reader.getnframes(), len(expected) // 2)
            self.assertEqual(reader.readframes(reader.getnframes()), expected)

    def test_complete_requested_ranges_preserve_pcm_and_targets(self) -> None:
        junk = b"JUNK" + struct.pack("<I", 3) + b"abc\0"
        full = self.container(self.pcm)
        cases = (
            ("complete", full, 0, 16, 0, 0, 16),
            ("middle", full, 4, 8, 2, 2, 10),
            ("left-context", full, 0, 4, 2, 0, 6),
            ("right-context", full, 12, 16, 2, 10, 16),
            ("intact-prefix", self.container(self.pcm[:-4]), 0, 8, 0, 0, 8),
            ("last-complete-frame", self.container(self.pcm[:-4]), 13, 14, 0, 13, 14),
            ("padded-unknown", self.container(self.pcm, before=junk), 0, 16, 0, 0, 16),
            ("trailing-bytes", self.container(self.pcm, trailing=b"trailing"), 0, 16, 0, 0, 16),
        )
        for name, raw, start, end, context, crop_start, crop_end in cases:
            with self.subTest(case=name):
                completed, output, manifest = self.invoke(name, [(raw, start, end)], context)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("mined 1", completed.stdout)
                rows = manifest.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(rows), 1)
                path, target = rows[0].split("\t")
                self.assertEqual(target, "1 2")
                self.assertEqual(pathlib.Path(path), output / "missed_000000_kw1.wav")
                self.assert_clip(pathlib.Path(path), self.pcm[crop_start * 2:crop_end * 2])

    def test_short_requested_ranges_fail_before_new_outputs(self) -> None:
        cases = (
            ("even-short", self.pcm[:-4], 0, 16, 0),
            ("odd-short", self.pcm[:-1], 0, 16, 0),
            ("zero-short", b"", 0, 16, 0),
            ("offset-short", self.pcm[:-4], 12, 16, 0),
            ("context-short", self.pcm[:-4], 8, 12, 8),
        )
        for name, pcm, start, end, context in cases:
            with self.subTest(case=name):
                completed, output, manifest = self.invoke(
                    name, [(self.container(pcm), start, end)], context)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertIn("truncated false-reject replay PCM payload", completed.stderr)
                self.assertEqual(completed.stdout, "")
                self.assertFalse(output.exists())
                self.assertFalse(manifest.exists())

    def test_first_bad_row_preserves_existing_clip_and_manifest(self) -> None:
        output = self.root / "sentinel-clips"
        output.mkdir()
        clip = output / "missed_000000_kw1.wav"
        clip.write_bytes(b"existing clip sentinel")
        manifest = self.root / "sentinel.tsv"
        manifest.write_bytes(b"existing manifest sentinel\n")
        completed, _, _ = self.invoke("sentinel", [(self.container(self.pcm[:-4]), 0, 16)])
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("truncated false-reject replay PCM payload", completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assertEqual(clip.read_bytes(), b"existing clip sentinel")
        self.assertEqual(manifest.read_bytes(), b"existing manifest sentinel\n")

    def test_later_bad_row_keeps_prior_clip_but_does_not_publish_manifest(self) -> None:
        manifest = self.root / "later.tsv"
        manifest.write_bytes(b"prior manifest sentinel\n")
        completed, output, _ = self.invoke("later", [
            (self.container(self.pcm), 0, 16),
            (self.container(self.pcm[:-4]), 0, 16),
        ])
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertEqual(completed.stdout, "")
        self.assert_clip(output / "missed_000000_kw1.wav", self.pcm)
        self.assertFalse((output / "missed_000001_kw1.wav").exists())
        self.assertEqual(manifest.read_bytes(), b"prior manifest sentinel\n")

def main() -> int:
    strict = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RequestedCropCompletenessTests))
    if not strict.testsRun or strict.skipped or not strict.wasSuccessful():
        return 1
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        source = root / "continuous.wav"
        write_wav(source)
        false_rejects = root / "false-rejects.jsonl"
        false_rejects.write_text(
            json.dumps(
                {
                    "recording": "r1",
                    "path": source.name,
                    "keyword_id": 1,
                    "start_s": 1.0,
                    "end_s": 1.8,
                    "duration_s": 3.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        tokens = root / "tokens.txt"
        tokens.write_text("<blk> 0\nni3 1\nhao3 2\nxiao3 3\nwo1 4\n", encoding="utf-8")
        keywords = root / "keywords.tsv"
        keywords.write_text(
            "1\t你好小窝\t0.55\tni3 hao3 xiao3 wo1\n",
            encoding="utf-8",
        )
        output_dir = root / "replay"
        manifest = root / "replay.tsv"
        completed = run_miner(
            false_rejects,
            keywords,
            tokens,
            root,
            output_dir,
            manifest,
            "--context-s",
            "0.2",
        )
        assert completed.returncode == 0, completed.stderr
        rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        assert len(rows) == 1
        clip_path, targets = rows[0].split("\t")
        assert targets == "1 2 3 4"
        with wave.open(clip_path, "rb") as reader:
            assert reader.getnchannels() == 1
            assert reader.getframerate() == 16000
            assert reader.getsampwidth() == 2
            assert 19190 <= reader.getnframes() <= 19210

        invalid = root / "invalid.jsonl"
        invalid.write_text(
            json.dumps(
                {
                    "recording": "r1",
                    "path": source.name,
                    "keyword_id": 1,
                    "start_s": 2.8,
                    "end_s": 3.2,
                    "duration_s": 3.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        completed = run_miner(
            invalid,
            keywords,
            tokens,
            root,
            root / "invalid-replay",
            root / "invalid.tsv",
        )
        assert completed.returncode == 2
        assert "exceeds source duration" in completed.stderr

        empty = root / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        empty_manifest = root / "empty.tsv"
        completed = run_miner(
            empty,
            keywords,
            tokens,
            root,
            root / "empty-replay",
            empty_manifest,
        )
        assert completed.returncode == 0, completed.stderr
        assert empty_manifest.read_text(encoding="utf-8") == ""

    print("test_false_reject_mining: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
