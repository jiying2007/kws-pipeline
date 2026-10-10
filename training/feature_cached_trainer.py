#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
CACHE_POLICY = "deterministic-feature-cache-v2"
MAX_FEATURE_CACHE_ITEMS = 32768
MAX_FEATURE_CACHE_BYTES = 1024 * 1024 * 1024
CACHE_SCOPE = "trainer-process"
BYTE_ACCOUNTING = "unique-tensor-storage"


def _capacity(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{label} must be 0..{maximum}")
    return value


def feature_cache_max_items(policy: dict) -> int:
    return _capacity(policy.get("feature_cache_max_items", 0),
                     "feature_cache_max_items", MAX_FEATURE_CACHE_ITEMS)


def feature_cache_max_bytes(policy: dict) -> int:
    # An enabled cache must declare its total retained tensor-storage budget.
    if feature_cache_max_items(policy) and "feature_cache_max_bytes" not in policy:
        raise ValueError("enabled feature cache requires feature_cache_max_bytes")
    return _capacity(policy.get("feature_cache_max_bytes", 0),
                     "feature_cache_max_bytes", MAX_FEATURE_CACHE_BYTES)


def cache_contract(max_items: int, max_bytes: int) -> dict:
    return {
        "policy": CACHE_POLICY,
        "max_items": _capacity(max_items, "max_items", MAX_FEATURE_CACHE_ITEMS),
        "max_bytes": _capacity(max_bytes, "max_bytes", MAX_FEATURE_CACHE_BYTES),
        "cache_scope": CACHE_SCOPE,
        "byte_accounting": BYTE_ACCOUNTING,
        "eviction_policy": "least-recently-used",
        "training_math_changed": False,
    }


def normalize_feature_cache(value: Any) -> dict:
    if not isinstance(value, dict):
        raise ValueError("training_environment.feature_cache must be an object")
    if value.get("training_math_changed") is not False:
        raise ValueError("training_environment.feature_cache must preserve training math")
    items = _capacity(value.get("max_items"), "max_items", MAX_FEATURE_CACHE_ITEMS)
    if value.get("policy") == "deterministic-feature-cache-v1":
        # Read frozen provenance as its historical contract. It never satisfies
        # the current promotion gate or gains a fabricated byte-budget claim.
        return {"policy": value["policy"], "max_items": items,
                "training_math_changed": False}
    if value.get("policy") != CACHE_POLICY:
        raise ValueError("training_environment.feature_cache policy mismatch")
    expected = cache_contract(items, value.get("max_bytes"))
    if any(value.get(key) != expected[key] for key in expected):
        raise ValueError("training_environment.feature_cache byte-budget contract mismatch")
    return expected


def rewrite_training_command(command: list[str], max_items: int, max_bytes: int) -> list[str]:
    cache_contract(max_items, max_bytes)
    if len(command) < 2:
        return list(command)
    script = pathlib.Path(command[1]).resolve()
    if script == (TRAINING / "train_gru_ctc.py").resolve():
        raise ValueError("train_gru_ctc.py is retired; only the RNN trainer is supported")
    if script != (TRAINING / "train_ctc.py").resolve() or not max_items or not max_bytes:
        return list(command)
    return [
        command[0], str(pathlib.Path(__file__).resolve()),
        "--trainer", "rnn", "--feature-cache-max-items", str(max_items),
        "--feature-cache-max-bytes", str(max_bytes), "--", *command[2:],
    ]


def install_development_feature_cache(loop_module: Any, policy: dict) -> dict:
    max_items = feature_cache_max_items(policy)
    max_bytes = feature_cache_max_bytes(policy)
    original_run = loop_module.run

    def cached_run(command: list[str]) -> None:
        original_run(rewrite_training_command(command, max_items, max_bytes))

    loop_module.run = cached_run
    return {**cache_contract(max_items, max_bytes), "default_disabled": True}


def _tensor_storages(value: Any) -> dict | None:
    """Measure retained storage, including views, without importing torch.

    Manifest returns an immutable tuple of CPU feature/target/(optional VAD)
    tensors. Non-tensor, accelerator or differentiable results bypass caching.
    Python object overhead and transient/model/optimizer memory are not included.
    """
    if not isinstance(value, tuple) or not value:
        return None
    storages = {}
    for tensor in value:
        if (getattr(getattr(tensor, "device", None), "type", None) != "cpu"
                or getattr(tensor, "requires_grad", True)):
            return None
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("feature cache tensor storage size is invalid")
        # CPU storage pointers stay live while their cached tensors are retained.
        key = storage.data_ptr()
        if key in storages and storages[key] != size:
            raise ValueError("feature cache tensor storage identity is inconsistent")
        storages[key] = size
    return storages


def cached_manifest_class(base_manifest: type, max_items: int, max_bytes: int) -> type:
    contract = cache_contract(max_items, max_bytes)
    # One LRU and one byte budget shared by every manifest in this trainer.
    entries: OrderedDict[tuple[int, int], tuple[Any, dict]] = OrderedDict()
    storage_refs: dict[int, list[int]] = {}
    stats = {"hits": 0, "misses": 0, "evictions": 0, "oversized": 0,
             "bypasses": 0, "resident_items": 0, "resident_bytes": 0,
             "peak_resident_bytes": 0}
    manifest_count = 0

    def evict() -> None:
        _, (_, storage_sizes) = entries.popitem(last=False)
        for key, size in storage_sizes.items():
            storage_refs[key][1] -= 1
            if not storage_refs[key][1]:
                del storage_refs[key]
                stats["resident_bytes"] -= size
        stats["evictions"] += 1
        stats["resident_items"] = len(entries)

    class FeatureCachedManifest(base_manifest):
        def __init__(self, *args, **kwargs):
            nonlocal manifest_count
            super().__init__(*args, **kwargs)
            self._feature_cache_identity = manifest_count
            manifest_count += 1

        @staticmethod
        def feature_cache_stats() -> dict:
            return {**contract, **stats}

        def __getitem__(self, idx: int):
            key = (self._feature_cache_identity, idx)
            if max_items and max_bytes and key in entries:
                value, _ = entries[key]
                entries.move_to_end(key)
                stats["hits"] += 1
                return value
            stats["misses"] += 1
            value = super().__getitem__(idx)
            if not max_items or not max_bytes:
                stats["bypasses"] += 1
                return value
            storage_sizes = _tensor_storages(value)
            if storage_sizes is None:
                stats["bypasses"] += 1
                return value
            if sum(storage_sizes.values()) > max_bytes:
                stats["oversized"] += 1
                return value

            def extra_bytes() -> int:
                return sum(size for storage, size in storage_sizes.items()
                           if storage not in storage_refs)

            while entries and (len(entries) >= max_items or
                               stats["resident_bytes"] + extra_bytes() > max_bytes):
                evict()
            for storage, size in storage_sizes.items():
                if storage not in storage_refs:
                    storage_refs[storage] = [size, 0]
                    stats["resident_bytes"] += size
                elif storage_refs[storage][0] != size:
                    raise ValueError("cached tensor storage changed after admission")
                storage_refs[storage][1] += 1
            entries[key] = (value, storage_sizes)
            stats["resident_items"] = len(entries)
            stats["peak_resident_bytes"] = max(stats["peak_resident_bytes"],
                                                stats["resident_bytes"])
            return value

    FeatureCachedManifest.__name__ = f"FeatureCached{base_manifest.__name__}"
    return FeatureCachedManifest


def _patch_training_environment(base_module: Any, max_items: int, max_bytes: int) -> None:
    original = base_module.training_environment
    wrapper = pathlib.Path(__file__).resolve()

    def environment() -> dict:
        value = original()
        code = value.get("training_code_sha256")
        if not isinstance(code, dict):
            raise ValueError("training environment code binding is missing")
        code[wrapper.relative_to(ROOT).as_posix()] = base_module.sha256_file(wrapper)
        value["training_code_sha256"] = dict(sorted(code.items()))
        value["feature_cache"] = cache_contract(max_items, max_bytes)
        return value

    base_module.training_environment = environment


def _run_trainer(trainer: str, max_items: int, max_bytes: int, trainer_args: list[str]) -> int:
    if trainer != "rnn":
        raise ValueError(f"unsupported cached trainer: {trainer}")
    cache_contract(max_items, max_bytes)
    sys.path.insert(0, str(TRAINING))
    import train_ctc as base  # noqa: E402

    manifest, environment, argv = base.Manifest, base.training_environment, sys.argv
    cached = cached_manifest_class(manifest, max_items, max_bytes)
    try:
        base.Manifest = cached
        _patch_training_environment(base, max_items, max_bytes)
        sys.argv = [str(TRAINING / "train_ctc.py"), *trainer_args]
        base.main()
        return 0
    finally:
        print("feature_cache_stats=" + json.dumps(cached.feature_cache_stats(), sort_keys=True))
        base.Manifest, base.training_environment, sys.argv = manifest, environment, argv


def self_test() -> None:
    # Full stdlib fake-tensor behavior tests are registered in the official CI.
    assert feature_cache_max_items({"feature_cache_max_items": 8192}) == 8192
    assert feature_cache_max_bytes({}) == 0
    assert normalize_feature_cache(cache_contract(2, 128)) == cache_contract(2, 128)
    rnn = rewrite_training_command(
        [sys.executable, str(TRAINING / "train_ctc.py"), "--epochs", "8"], 8192, 256
    )
    assert rnn[2:8] == ["--trainer", "rnn", "--feature-cache-max-items", "8192",
                       "--feature-cache-max-bytes", "256"]
    assert rnn[-3:] == ["--", "--epochs", "8"]


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
    parser.add_argument("--trainer", choices=("rnn",), required=True)
    parser.add_argument("--feature-cache-max-items", required=True, type=int)
    parser.add_argument("--feature-cache-max-bytes", required=True, type=int)
    args = parser.parse_args(sys.argv[1:marker])
    return _run_trainer(args.trainer, args.feature_cache_max_items,
                        args.feature_cache_max_bytes, sys.argv[marker + 1 :])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
