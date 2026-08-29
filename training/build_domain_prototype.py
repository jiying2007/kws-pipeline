#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from fit_domain_prototype import fit_domain_prototype
from synthetic_audio import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--carriers", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--training-output", required=True, type=pathlib.Path)
    parser.add_argument("--frontend", choices=["logmel", "pcen-lite"], default="logmel")
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--variants-per-token", type=int, default=16)
    parser.add_argument("--input-scale", type=float, default=0.010)
    parser.add_argument("--output-scale", type=float, default=0.050)
    parser.add_argument("--blank-bias", type=float, default=1.8)
    parser.add_argument("--token-bias", type=float, default=-1.2)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    result = fit_domain_prototype(
        config=load_config(args.config.resolve()),
        tokens_path=args.tokens.resolve(),
        carriers_path=args.carriers.resolve(),
        output=args.output.resolve(),
        training_output=args.training_output.resolve(),
        feature_dim=args.feature_dim,
        variants_per_token=args.variants_per_token,
        projection_gain=args.input_scale * 127.0,
        output_scale=args.output_scale,
        blank_bias=args.blank_bias,
        token_bias=args.token_bias,
        seed=args.seed,
        frontend=args.frontend,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
