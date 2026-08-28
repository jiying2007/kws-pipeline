#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib

from qualification_fixture import write_model, write_pack, write_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    model_dir = args.output / "model"
    pack_dir = args.output / "keyword_pack"
    model_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)

    tokens = args.output / "tokens.txt"
    _, fingerprint = write_tokens(tokens)
    write_model(model_dir / "valid.kwm", fingerprint)
    write_pack(pack_dir / "valid.kwk", fingerprint)
    print(f"wrote fuzz seeds under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
