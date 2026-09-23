from __future__ import annotations

import copy
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from materialize_product_training_config import (  # noqa: E402
    canonical_sha256,
    validate_replay_semantic_identity,
)


def matching_summary() -> dict:
    return {
        "provider_profile": "sherpa-vits-resampled-to-16k-v1",
        "license_evidence_sha256": "98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74",
        "corpus_plan_sha256": "ea0d012e6f771e45c620d3f2df441c1d9d01dfed9be87ee397494a33ffbe8189",
        "command_policy_sha256": "816acec9d2ddfc6041f0a2345434d3caf8a74e59bc78f1b156335ed893e69efc",
        "provider_reference": {
            "reference_sha256": "a118c73053b538b513c630e3979e7cce773338b76a7678bcfe95612d8a925751",
            "candidate": "icefall-tts-aishell3-vits-low-2024-04-06",
        },
        "runtime_asset_binding": {
            "archive_sha256": "ab468db3a3308cdd861495e0db2f25d79418a0c00639f74944c7cdf5dd8c6ec1",
        },
        "backend_bundle_binding": {
            "archive_sha256": "c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4",
        },
        "audio_normalization": {
            "adapter_sha256": "b238e7726456ad8cb524efab774983a8a95b1960e9cd48c7a604e49fa41c1c99",
            "source_sample_rate_hz": 8000,
            "output_sample_rate_hz": 16000,
            "resampler_kind": "builtin-lanczos-2x-v1",
            # Execution-only values may legitimately differ across runners.
            "backend_executable_sha256": "a" * 64,
        },
        "provider_identity_sha256": "b" * 64,
    }


def main() -> int:
    contract_path = (
        ROOT / "configs/training/product-replay-provider-semantic-v1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_identity = contract["semantic_identity_sha256"]

    identity, payload = validate_replay_semantic_identity(
        matching_summary(),
        contract,
    )
    assert identity == expected_identity
    assert canonical_sha256(payload) == expected_identity

    # The stable semantic identity is deliberately independent from the
    # execution-only provider hash / hosted Python launcher identity.
    other_runner = matching_summary()
    other_runner["provider_identity_sha256"] = "c" * 64
    other_runner["audio_normalization"]["backend_executable_sha256"] = "d" * 64
    other_identity, other_payload = validate_replay_semantic_identity(
        other_runner,
        contract,
    )
    assert other_identity == identity
    assert other_payload == payload

    for field, value in (
        ("adapter_sha256", "e" * 64),
        ("runtime_asset_archive_sha256", "f" * 64),
        ("backend_bundle_archive_sha256", "1" * 64),
    ):
        drifted = matching_summary()
        if field == "adapter_sha256":
            drifted["audio_normalization"][field] = value
        elif field == "runtime_asset_archive_sha256":
            drifted["runtime_asset_binding"]["archive_sha256"] = value
        else:
            drifted["backend_bundle_binding"]["archive_sha256"] = value
        try:
            validate_replay_semantic_identity(drifted, contract)
        except ValueError as exc:
            assert "semantic binding drifted" in str(exc)
        else:
            raise AssertionError(f"semantic drift was accepted: {field}")

    corrupt_contract = copy.deepcopy(contract)
    corrupt_contract["replay_train_voice_slots"] = 7
    try:
        validate_replay_semantic_identity(
            matching_summary(),
            corrupt_contract,
        )
    except ValueError as exc:
        assert "contract hash mismatch" in str(exc)
    else:
        raise AssertionError("semantic contract mutation was not hash-bound")

    print("replay provider semantic identity: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
