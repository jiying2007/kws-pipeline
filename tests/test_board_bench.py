#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import tempfile

from qualification_fixture import write_model, write_pack, write_tokens, write_wav


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", required=True, type=pathlib.Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        tokens = root / "tokens.txt"
        model = root / "model.kwm"
        pack = root / "keywords.kwk"
        wav = root / "audio.wav"
        _, fingerprint = write_tokens(tokens)
        model_bytes = write_model(model, fingerprint)
        pack_bytes = write_pack(pack, fingerprint)
        write_wav(wav, seconds=1)

        completed = subprocess.run(
            [str(args.runner), str(model), str(pack), str(wav), "2"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        result = json.loads(completed.stdout)
        assert result["schema_version"] == 1
        assert result["block_samples"] == 320
        assert result["block_deadline_us"] == 20000.0
        assert result["audio_seconds"] == 1.0
        assert result["repeats"] == 2
        assert result["blocks"] == 100
        assert result["model_bytes"] == model_bytes
        assert result["keyword_pack_bytes"] == pack_bytes
        assert result["arena_bytes"] > 0
        assert 0.0 <= result["p50_process_us"] <= result["p95_process_us"]
        assert result["p95_process_us"] <= result["p99_process_us"]
        assert result["p99_process_us"] <= result["max_process_us"]
        assert result["rtf"] >= 0.0
        assert result["p99_headroom"] >= 0.0

    print("test_board_bench: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
