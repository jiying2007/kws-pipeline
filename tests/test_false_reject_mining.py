#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import wave

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


def main() -> int:
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
