#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "experiments" / "model_family" / "run_rnn_tuning.py"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-seed-offset", required=True, type=int)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args, remaining = parser.parse_known_args()
    if args.training_seed_offset <= 0:
        raise ValueError("training seed offset must be positive")

    spec = importlib.util.spec_from_file_location("rnn_tuning_delegate", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load RNN tuning delegate")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original_load_config = module.load_config
    original_seed_holder: dict[str, int] = {}

    def patched_load_config(path: pathlib.Path) -> dict:
        cfg = original_load_config(path)
        original_seed = int(cfg.get("seed", 1337))
        original_seed_holder["seed"] = original_seed
        # run_rnn_tuning.py historically adds 5_100_019 internally. Shift the
        # in-memory seed only for training so its effective optimizer seed is
        # exactly base_seed + requested offset. The on-disk config remains
        # untouched and shadow_qualification receives the canonical config.
        cfg["seed"] = original_seed + args.training_seed_offset - 5_100_019
        return cfg

    module.load_config = patched_load_config
    sys.argv = [str(TARGET), *remaining, "--output", str(args.output)]
    rc = int(module.main())

    summary_path = args.output / "tuning-summary.json"
    if not summary_path.is_file():
        raise ValueError("delegated RNN tuning did not emit summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    base_seed = int(original_seed_holder["seed"])
    summary.setdefault("parameters", {})["training_seed_offset"] = args.training_seed_offset
    summary["parameters"]["training_seed"] = base_seed + args.training_seed_offset
    summary["seed_probe_policy"] = "rnn-optimizer-seed-probe-v1"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
