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


def compile_one(tokens: pathlib.Path, keywords: pathlib.Path, root: pathlib.Path):
    header = root / "keywords.h"
    manifest = root / "keywords.json"
    pack = root / "keywords.kwk"
    subprocess.check_call(
        [
            sys.executable,
            str(ROOT / "tools" / "compile_keywords.py"),
            "--tokens",
            str(tokens),
            "--keywords",
            str(keywords),
            "--out-header",
            str(header),
            "--out-json",
            str(manifest),
            "--out-pack",
            str(pack),
        ]
    )
    return header, manifest, pack


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        token_file = ROOT / "keywords" / "tokens.example.txt"
        header, manifest, pack = compile_one(
            token_file, ROOT / "keywords" / "zh_cn_example.tsv", root
        )
        text = header.read_text(encoding="utf-8")
        assert "kws_generated_keyword_count" in text
        assert "kws_generated_vocab_fingerprint" in text
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        items = metadata["keywords"]
        expected_fingerprint = vocab_fingerprint(load_tokens(token_file))
        assert metadata["vocab_size"] == 5
        assert metadata["vocab_fingerprint"] == f"0x{expected_fingerprint:016x}"

        blob = pack.read_bytes()
        magic, version, header_bytes, count, vocab_size, total_bytes, fingerprint = struct.unpack(
            "<4sHHHHIQ", blob[:24]
        )
        assert magic == b"KWKP"
        assert version == 3
        assert header_bytes == 24
        assert count == len(items) == 2
        assert vocab_size == 5
        assert fingerprint == expected_fingerprint
        assert total_bytes == len(blob) == 24 + count * 48

        fields = struct.unpack("<IfHBBBBH16H", blob[24:72])
        kid, threshold, num_tokens, min_blanks, priority, policy, grace, reserved, *tokens = fields
        assert kid == items[0]["id"]
        assert abs(threshold - items[0]["threshold"]) < 1.0e-6
        assert num_tokens == 4
        assert (min_blanks, priority, policy, grace, reserved) == (0, 0, 0, 0, 0)
        assert tokens[:4] == [1, 2, 3, 4]
        assert tokens[4:] == [0] * 12

        # Explicit overlapping keyword policy must survive TSV -> JSON/header/pack.
        overlap = root / "overlap.tsv"
        overlap.write_text(
            "10\t小窝\t0.55\txiao3 wo1\t1\t2\tlongest\t0\n"
            "11\t小窝小窝\t0.60\txiao3 wo1 xiao3 wo1\t0\t3\tgrace\t4\n",
            encoding="utf-8",
        )
        policy_root = root / "policy"
        policy_root.mkdir()
        _, policy_json, policy_pack = compile_one(token_file, overlap, policy_root)
        policy_items = json.loads(policy_json.read_text(encoding="utf-8"))["keywords"]
        assert policy_items[0]["prefix_policy"] == "longest"
        assert policy_items[0]["min_trailing_blanks"] == 1
        assert policy_items[1]["prefix_policy"] == "grace"
        assert policy_items[1]["grace_frames"] == 4
        pblob = policy_pack.read_bytes()
        p0 = struct.unpack("<IfHBBBBH16H", pblob[24:72])
        p1 = struct.unpack("<IfHBBBBH16H", pblob[72:120])
        assert p0[3:8] == (1, 2, 1, 0, 0)
        assert p1[3:8] == (0, 3, 2, 4, 0)

        reordered = root / "tokens-reordered.txt"
        reordered.write_text("wo1 4\nxiao3 3\nhao3 2\nni3 1\n<blk> 0\n", encoding="utf-8")
        assert vocab_fingerprint(load_tokens(reordered)) == expected_fingerprint
        changed = root / "tokens-changed.txt"
        changed.write_text("<blk> 0\nni3 1\nxiao3 2\nhao3 3\nwo1 4\n", encoding="utf-8")
        assert vocab_fingerprint(load_tokens(changed)) != expected_fingerprint
        sparse = root / "tokens-sparse.txt"
        sparse.write_text("<blk> 0\nni3 2\n", encoding="utf-8")
        try:
            load_tokens(sparse)
        except ValueError as exc:
            assert "contiguous" in str(exc)
        else:
            raise AssertionError("sparse token IDs must be rejected")

    print("test_keyword_compile: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
