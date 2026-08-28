#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from kws_vocab import load_tokens, vocab_fingerprint  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        header = root / "keywords.h"
        manifest = root / "keywords.json"
        pack = root / "keywords.kwk"
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "tools" / "compile_keywords.py"),
                "--tokens",
                str(ROOT / "keywords" / "tokens.example.txt"),
                "--keywords",
                str(ROOT / "keywords" / "zh_cn_example.tsv"),
                "--out-header",
                str(header),
                "--out-json",
                str(manifest),
                "--out-pack",
                str(pack),
            ]
        )

        text = header.read_text(encoding="utf-8")
        assert "kws_generated_keyword_count" in text
        assert "{1u, 2u, 3u, 4u}" in text

        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        items = metadata["keywords"]
        expected_fingerprint = vocab_fingerprint(
            load_tokens(ROOT / "keywords" / "tokens.example.txt")
        )
        assert metadata["vocab_size"] == 5
        assert metadata["vocab_fingerprint"] == f"0x{expected_fingerprint:016x}"

        blob = pack.read_bytes()
        (
            magic,
            version,
            header_bytes,
            count,
            vocab_size,
            total_bytes,
            fingerprint,
        ) = struct.unpack("<4sHHHHIQ", blob[:24])
        assert magic == b"KWKP"
        assert version == 2
        assert header_bytes == 24
        assert count == len(items) == 2
        assert vocab_size == 5
        assert fingerprint == expected_fingerprint
        assert total_bytes == len(blob) == 24 + count * 44

        kid, threshold, num_tokens, reserved, *tokens = struct.unpack(
            "<IfHH16H", blob[24:68]
        )
        assert kid == items[0]["id"]
        assert abs(threshold - items[0]["threshold"]) < 1.0e-6
        assert num_tokens == 4
        assert reserved == 0
        assert tokens[:4] == [1, 2, 3, 4]
        assert tokens[4:] == [0] * 12

        reordered = root / "tokens-reordered.txt"
        reordered.write_text(
            "wo1 4\nxiao3 3\nhao3 2\nni3 1\n<blk> 0\n",
            encoding="utf-8",
        )
        assert vocab_fingerprint(load_tokens(reordered)) == expected_fingerprint

        changed = root / "tokens-changed.txt"
        changed.write_text(
            "<blk> 0\nni3 1\nxiao3 2\nhao3 3\nwo1 4\n",
            encoding="utf-8",
        )
        assert vocab_fingerprint(load_tokens(changed)) != expected_fingerprint

    print("test_keyword_compile: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
