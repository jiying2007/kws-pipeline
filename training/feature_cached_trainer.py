#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
CACHE_POLICY = "development-feature-cache-v1"
MAX_FEATURE_CACHE_ITEMS = 32768


def feature_cache_max_items(policy: dict) -> int:
    raw = policy.get("feature_cache_max_items", 0)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError("feature_cache_max_items must be an integer")
    if not 0 <= raw <= MAX_FEATURE_CACHE_ITEMS:
        raise ValueError(
            f"feature_cache_max_items must be 0..{MAX_FEATURE_CACHE_ITEMS}"
        )
    return raw


def rewrite_training_command(command: list[str], max_items: int) -> list[str]:
    if max_items <= 0 or len(command) < 2:
        return list(command)
    script = pathlib.Path(command[1]).resolve()
    if script == (TRAINING / "train_ctc.py").resolve():
        trainer = "rnn"
    elif script == (TRAINING / "train_gru_ctc.py").resolve():
        trainer = "gru"
    else:
        return list(command)
    return [
        command[0],
        str(pathlib.Path(__file__).resolve()),
        "--trainer",
        trainer,
        "--feature-cache-max-items",
        str(max_items),
        "--",
        *command[2:],
    ]


def install_development_feature_cache(loop_module: Any, policy: dict) -> dict:
    max_items = feature_cache_max_items(policy)
    original_run = loop_module.run

    def cached_run(command: list[str]) -> None:
        original_run(rewrite_training_command(command, max_items))

    loop_module.run = cached_run
    return {
        "policy": CACHE_POLICY,
        "max_items": max_items,
        "default_disabled": True,
        "training_math_changed": False,
    }


def cached_manifest_class(base_manifest: type, max_items: int) -> type:
    if not 0 <= max_items <= MAX_FEATURE_CACHE_ITEMS:
        raise ValueError("feature cache capacity is invalid")

    class FeatureCachedManifest(base_manifest):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._feature_cache: OrderedDict[int, Any] = OrderedDict()

        def __getitem__(self, idx: int):
            if max_items <= 0:
                return super().__getitem__(idx)
            if idx in self._feature_cache:
                value = self._feature_cache.pop(idx)
                self._feature_cache[idx] = value
                return value
            value = super().__getitem__(idx)
            self._feature_cache[idx] = value
            while len(self._feature_cache) > max_items:
                self._feature_cache.popitem(last=False)
            return value

    FeatureCachedManifest.__name__ = f"FeatureCached{base_manifest.__name__}"
    return FeatureCachedManifest


def _patch_training_environment(base_module: Any, max_items: int) -> None:
    original = base_module.training_environment
    wrapper = pathlib.Path(__file__).resolve()

    def environment() -> dict:
        value = original()
        code = value.get("training_code_sha256")
        if not isinstance(code, dict):
            raise ValueError("training environment code binding is missing")
        code[wrapper.relative_to(ROOT).as_posix()] = base_module.sha256_file(wrapper)
        value["training_code_sha256"] = dict(sorted(code.items()))
        value["feature_cache"] = {
            "policy": CACHE_POLICY,
            "max_items": int(max_items),
            "training_math_changed": False,
        }
        return value

    base_module.training_environment = environment


def _run_trainer(trainer: str, max_items: int, trainer_args: list[str]) -> int:
    sys.path.insert(0, str(TRAINING))
    import train_ctc as base  # noqa: E402

    base.Manifest = cached_manifest_class(base.Manifest, max_items)
    _patch_training_environment(base, max_items)
    if trainer == "rnn":
        sys.argv = [str(TRAINING / "train_ctc.py"), *trainer_args]
        base.main()
        return 0
    if trainer == "gru":
        import train_gru_ctc as gru  # noqa: E402

        gru.Manifest = base.Manifest
        gru.training_environment = base.training_environment
        sys.argv = [str(TRAINING / "train_gru_ctc.py"), *trainer_args]
        gru.main()
        return 0
    raise ValueError(f"unsupported cached trainer: {trainer}")


def self_test() -> None:
    class FakeManifest:
        def __init__(self):
            self.calls = 0

        def __getitem__(self, idx: int):
            self.calls += 1
            return (idx, self.calls)

    cached = cached_manifest_class(FakeManifest, 2)()
    first = cached[0]
    assert cached[1][0] == 1
    assert cached[0] == first
    assert cached.calls == 2
    assert cached[2][0] == 2
    assert cached[1][0] == 1
    assert cached.calls == 4

    uncached = cached_manifest_class(FakeManifest, 0)()
    uncached[0]
    uncached[0]
    assert uncached.calls == 2

    rnn = rewrite_training_command(
        [sys.executable, str(TRAINING / "train_ctc.py"), "--epochs", "8"], 8192
    )
    gru = rewrite_training_command(
        [sys.executable, str(TRAINING / "train_gru_ctc.py"), "--epochs", "8"], 8192
    )
    assert rnn[2:6] == ["--trainer", "rnn", "--feature-cache-max-items", "8192"]
    assert gru[2:6] == ["--trainer", "gru", "--feature-cache-max-items", "8192"]
    assert rnn[-3:] == ["--", "--epochs", "8"]
    assert feature_cache_max_items({"feature_cache_max_items": 8192}) == 8192
    try:
        feature_cache_max_items({"feature_cache_max_items": MAX_FEATURE_CACHE_ITEMS + 1})
    except ValueError:
        pass
    else:
        raise AssertionError("oversized feature cache was accepted")


def main() -> int:
    if "--self-test" in sys.argv[1:]:
        self_test()
        print("development feature cache self-test: PASS")
        return 0
    try:
        marker = sys.argv.index("--")
    except ValueError as exc:
        raise ValueError("cached trainer requires '--' before delegated trainer args") from exc
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trainer", choices=("rnn", "gru"), required=True)
    parser.add_argument("--feature-cache-max-items", required=True, type=int)
    args = parser.parse_args(sys.argv[1:marker])
    if not 0 <= args.feature_cache_max_items <= MAX_FEATURE_CACHE_ITEMS:
        raise ValueError(
            f"feature cache max items must be 0..{MAX_FEATURE_CACHE_ITEMS}"
        )
    return _run_trainer(args.trainer, args.feature_cache_max_items, sys.argv[marker + 1 :])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
