#!/usr/bin/env python3
"""Validate and materialize a same-runner paired product-development experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile

from product_development_experiment import materialize, verify_spec

SCHEMA_VERSION = 1
SOURCE_POLICY = "exact-pr-head-paired-same-runner-v1"
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,111}")
FIELDS = {
    "schema_version",
    "experiment_id",
    "development_only",
    "source_policy",
    "protected_evidence_used",
    "reason",
    "common_overrides",
    "variable",
}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("paired experiment spec must be a JSON object")
    return value


def arm_spec(pair: dict, arm: str) -> dict:
    variable = pair["variable"]
    overrides = dict(pair["common_overrides"])
    overrides[variable["name"]] = variable[arm]
    return {
        "schema_version": 1,
        "experiment_id": f"{pair['experiment_id']}-{arm}",
        "development_only": True,
        "source_policy": "exact-pr-head",
        "protected_evidence_used": False,
        "reason": (
            f"{pair['reason']} Same-runner paired arm={arm}; "
            f"only {variable['name']} may differ between arms."
        ),
        "config_overrides": overrides,
    }


def verify_pair_spec(path: pathlib.Path) -> dict:
    value = load_object(path)
    if set(value) != FIELDS:
        raise ValueError(
            "paired experiment spec fields mismatch: "
            f"missing={sorted(FIELDS-set(value))} extra={sorted(set(value)-FIELDS)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("paired experiment schema_version must be 1")
    experiment_id = value["experiment_id"]
    if not isinstance(experiment_id, str) or ID_RE.fullmatch(experiment_id) is None:
        raise ValueError("paired experiment_id is invalid")
    if value["development_only"] is not True:
        raise ValueError("paired experiment must be development_only=true")
    if value["source_policy"] != SOURCE_POLICY:
        raise ValueError("paired experiment source_policy mismatch")
    if value["protected_evidence_used"] is not False:
        raise ValueError("paired experiment must not use protected evidence")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("paired experiment reason is required")
    common = value["common_overrides"]
    if not isinstance(common, dict) or not common:
        raise ValueError("paired common_overrides must be a non-empty object")
    variable = value["variable"]
    if not isinstance(variable, dict) or set(variable) != {"name", "control", "treatment"}:
        raise ValueError("paired variable must contain name/control/treatment")
    name = variable["name"]
    if not isinstance(name, str) or not name or name in common:
        raise ValueError("paired variable name is invalid or duplicated in common_overrides")
    if variable["control"] == variable["treatment"]:
        raise ValueError("paired control/treatment variable values must differ")

    # Reuse the ordinary experiment contract as the authority for every override.
    with tempfile.TemporaryDirectory(prefix="paired-spec-contract-") as tmp:
        root = pathlib.Path(tmp)
        normalized = {}
        for arm in ("control", "treatment"):
            spec_path = root / f"{arm}.json"
            spec_path.write_text(
                json.dumps(arm_spec(value, arm), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            normalized[arm] = verify_spec(spec_path)
    left = dict(normalized["control"]["config_overrides"])
    right = dict(normalized["treatment"]["config_overrides"])
    moved = sorted(
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    )
    if moved != [name]:
        raise ValueError(f"paired experiment must move exactly one variable: moved={moved}")
    result = dict(value)
    result["common_overrides"] = {
        key: normalized["control"]["config_overrides"][key] for key in common
    }
    result["variable"] = {
        "name": name,
        "control": normalized["control"]["config_overrides"][name],
        "treatment": normalized["treatment"]["config_overrides"][name],
    }
    return result


def materialize_pair(
    *,
    pair_spec_path: pathlib.Path,
    effective_config_path: pathlib.Path,
    output_root: pathlib.Path,
    base_sha: str,
    head_sha: str,
) -> dict:
    pair = verify_pair_spec(pair_spec_path)
    output_root.mkdir(parents=True, exist_ok=True)
    arm_receipts = {}
    for arm in ("control", "treatment"):
        root = output_root / arm
        spec_path = root / ".github/triggers/model-training-experiment.json"
        # Keep the executable config beside the governed effective config.
        # materialize_product_training_config.py intentionally writes external-base
        # paths relative to effective_config_path.parent (normally ROOT/.generated).
        # Nesting the executable config under the evidence root would silently
        # rebase those paths and make the same config invalid at runtime.
        config_path = (
            effective_config_path.parent
            / f"xiaowo.product-paired-{arm}.json"
        )
        evidence_config_path = root / ".generated/xiaowo.product-experiment.json"
        receipt_path = root / "build/product-development-experiment-receipt.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(
            json.dumps(arm_spec(pair, arm), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt = materialize(
            spec_path=spec_path,
            effective_config_path=effective_config_path,
            output_path=config_path,
            receipt_path=receipt_path,
            base_sha=base_sha,
            head_sha=head_sha,
        )
        evidence_config_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_config_path.write_bytes(config_path.read_bytes())
        if sha256_file(evidence_config_path) != receipt["experiment_config_sha256"]:
            raise ValueError("paired evidence config copy drifted from executable config")
        receipt["executable_config"] = str(config_path)
        receipt["evidence_config"] = str(evidence_config_path)
        arm_receipts[arm] = receipt

    if (
        arm_receipts["control"]["effective_config_sha256"]
        != arm_receipts["treatment"]["effective_config_sha256"]
    ):
        raise ValueError("paired arms materialized from different effective product bases")

    result = {
        "schema_version": 1,
        "evidence_class": "product-development-same-runner-pair-receipt-v1",
        "development_only": True,
        "release_authority": False,
        "source_policy": SOURCE_POLICY,
        "pair_spec_sha256": sha256_file(pair_spec_path),
        "effective_config_sha256": arm_receipts["control"]["effective_config_sha256"],
        "pr_base_sha": base_sha,
        "pr_head_sha": head_sha,
        "same_runner_required": True,
        "shared_command_tts_cache_required": True,
        "variable": pair["variable"],
        "common_overrides": pair["common_overrides"],
        "arms": {
            arm: {
                "experiment_id": arm_receipts[arm]["experiment_id"],
                "config_sha256": arm_receipts[arm]["experiment_config_sha256"],
                "spec_sha256": arm_receipts[arm]["spec_sha256"],
            }
            for arm in ("control", "treatment")
        },
    }
    receipt_path = output_root / "pair-receipt.json"
    receipt_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="paired-spec-") as tmp:
        root = pathlib.Path(tmp)
        path = root / "pair.json"
        value = {
            "schema_version": 1,
            "experiment_id": "runtime-replay-paired-v1",
            "development_only": True,
            "source_policy": SOURCE_POLICY,
            "protected_evidence_used": False,
            "reason": "exercise paired single-variable contract",
            "common_overrides": {
                "train.ctc_vad_align": True,
                "train.sequence_margin_negative_policy": "runtime-executable-v1",
            },
            "variable": {
                "name": "domain_iteration.base_failure_replay_enabled",
                "control": False,
                "treatment": True,
            },
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        verified = verify_pair_spec(path)
        assert verified["variable"]["control"] is False
        assert verified["variable"]["treatment"] is True
        bad = json.loads(json.dumps(value))
        bad["common_overrides"]["domain_iteration.base_failure_replay_enabled"] = False
        path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            verify_pair_spec(path)
        except ValueError as exc:
            assert "variable name" in str(exc)
        else:
            raise AssertionError("paired second-variable ambiguity was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")
    verify = sub.add_parser("verify")
    verify.add_argument("--pair-spec", required=True, type=pathlib.Path)
    build = sub.add_parser("materialize")
    build.add_argument("--pair-spec", required=True, type=pathlib.Path)
    build.add_argument("--effective-config", required=True, type=pathlib.Path)
    build.add_argument("--output-root", required=True, type=pathlib.Path)
    build.add_argument("--base-sha", required=True)
    build.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("product development paired spec self-test: PASS")
        return 0
    if args.command == "verify":
        value = verify_pair_spec(args.pair_spec.resolve())
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "materialize":
        value = materialize_pair(
            pair_spec_path=args.pair_spec.resolve(),
            effective_config_path=args.effective_config.resolve(),
            output_root=args.output_root.resolve(),
            base_sha=args.base_sha,
            head_sha=args.head_sha,
        )
        print(json.dumps(value, ensure_ascii=False, sort_keys=True))
        return 0
    parser.error("a command or --self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
