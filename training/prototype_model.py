from __future__ import annotations

import argparse
import json
import pathlib
import sys

from fit_prototype import fit_prototype


def build_prototype(
    *,
    tokens_path: pathlib.Path,
    carriers_path: pathlib.Path,
    output: pathlib.Path,
    feature_dim: int,
    input_scale: float,
    output_scale: float,
    blank_bias: float,
    token_bias: float,
) -> dict:
    """Fit a dependency-free synthetic prototype from training-only token samples.

    `input_scale` remains the public candidate knob used by existing configs.
    The fitted backend interprets it as an equivalent projection gain
    (`input_scale * 127`) because ABI-v2 stores one global int8 input scale.
    No calibration/test/qualification audio is read here.
    """
    config = {
        "generator": {
            "tts": {
                "token_ms": 180.0,
                "token_ms_jitter": 20.0,
                "gap_ms": 80.0,
                "lead_ms": 140.0,
                "tail_ms": 160.0,
                "amplitude": 12500.0,
                "pitch_jitter": 0.008,
            },
            "augment": {
                "gain_db_min": -3.0,
                "gain_db_max": 2.5,
                "snr_db_min": 24.0,
                "snr_db_max": 42.0,
                "echo_max_ms": 28.0,
                "echo_gain_max": 0.10,
                "noise_profiles": ["white", "fan", "motor", "media"],
            },
        }
    }
    return fit_prototype(
        config=config,
        tokens_path=tokens_path,
        carriers_path=carriers_path,
        output=output,
        training_output=output.parent / "prototype-fit",
        feature_dim=feature_dim,
        variants_per_token=8,
        projection_gain=input_scale * 127.0,
        output_scale=output_scale,
        blank_bias=blank_bias,
        token_bias=token_bias,
        seed=1337,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=pathlib.Path)
    parser.add_argument("--carriers", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--input-scale", type=float, default=0.010)
    parser.add_argument("--output-scale", type=float, default=0.050)
    parser.add_argument("--blank-bias", type=float, default=1.8)
    parser.add_argument("--token-bias", type=float, default=-1.2)
    args = parser.parse_args()
    result = build_prototype(
        tokens_path=args.tokens.resolve(),
        carriers_path=args.carriers.resolve(),
        output=args.output.resolve(),
        feature_dim=args.feature_dim,
        input_scale=args.input_scale,
        output_scale=args.output_scale,
        blank_bias=args.blank_bias,
        token_bias=args.token_bias,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
