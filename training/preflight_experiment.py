#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re

TRIGGER_POLICY = "development-only-preflight"
SOURCE_POLICY = "exact-current-main"
DISABLED_PROFILE = "disabled"
PATH_PURITY_PROFILE = "path-purity-v1"
REQUEST_FIELDS = {
    "schema_version",
    "experiment_id",
    "trigger_policy",
    "source_policy",
    "profile",
    "parameters",
    "reason",
}
ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}")
SHA_RE = re.compile(r"[0-9a-f]{40}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def verify_experiment(path: pathlib.Path) -> dict:
    value = load_object(path, "preflight experiment")
    fields = set(value)
    if fields != REQUEST_FIELDS:
        raise ValueError(
            "preflight experiment fields mismatch: "
            f"missing={sorted(REQUEST_FIELDS-fields)} extra={sorted(fields-REQUEST_FIELDS)}"
        )
    if value["schema_version"] != 1:
        raise ValueError("preflight experiment schema_version must be 1")
    experiment_id = value["experiment_id"]
    if not isinstance(experiment_id, str) or ID_RE.fullmatch(experiment_id) is None:
        raise ValueError("preflight experiment_id is invalid")
    if value["trigger_policy"] != TRIGGER_POLICY:
        raise ValueError("preflight experiment trigger_policy mismatch")
    if value["source_policy"] != SOURCE_POLICY:
        raise ValueError("preflight experiment source_policy mismatch")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("preflight experiment reason is required")
    profile = value["profile"]
    parameters = value["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("preflight experiment parameters must be an object")
    if profile == DISABLED_PROFILE:
        if parameters:
            raise ValueError("disabled preflight experiment must have empty parameters")
    elif profile == PATH_PURITY_PROFILE:
        if set(parameters) != {"weight", "margin"}:
            raise ValueError("path-purity-v1 requires exactly weight and margin")
        weight = float(parameters["weight"])
        margin = float(parameters["margin"])
        if not math.isfinite(weight) or not 0.0 < weight <= 1.0:
            raise ValueError("path-purity-v1 weight must be finite and in (0,1]")
        if not math.isfinite(margin) or not 0.0 <= margin <= 0.5:
            raise ValueError("path-purity-v1 margin must be finite and in [0,0.5]")
    else:
        raise ValueError(f"unsupported preflight experiment profile: {profile}")
    return value


def apply_experiment(
    *,
    experiment_path: pathlib.Path,
    config_path: pathlib.Path,
    receipt_path: pathlib.Path,
    base_sha: str,
    head_sha: str,
) -> dict:
    if SHA_RE.fullmatch(base_sha) is None or SHA_RE.fullmatch(head_sha) is None:
        raise ValueError("preflight experiment requires canonical base/head SHA")
    experiment = verify_experiment(experiment_path)
    config = load_object(config_path, "preflight config")
    profile = str(experiment["profile"])
    parameters = dict(experiment["parameters"])
    enabled = profile != DISABLED_PROFILE
    if profile == PATH_PURITY_PROFILE:
        train = config.setdefault("train", {})
        if not isinstance(train, dict):
            raise ValueError("preflight config train must be an object")
        train["path_purity_loss_weight"] = float(parameters["weight"])
        train["path_purity_margin"] = float(parameters["margin"])
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )

    receipt = {
        "schema_version": 1,
        "evidence_class": "development-preflight-experiment-v1",
        "development_only": True,
        "formal_qualification_used": False,
        "enabled": enabled,
        "experiment_id": experiment["experiment_id"],
        "profile": profile,
        "parameters": parameters,
        "trigger_policy": experiment["trigger_policy"],
        "source_policy": experiment["source_policy"],
        "experiment_sha256": sha256_file(experiment_path),
        "preflight_config_sha256": sha256_file(config_path),
        "base_sha": base_sha,
        "head_sha": head_sha,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return receipt


def verify_receipt(
    *,
    experiment_path: pathlib.Path,
    config_path: pathlib.Path,
    receipt_path: pathlib.Path,
    base_sha: str,
    head_sha: str,
) -> dict:
    experiment = verify_experiment(experiment_path)
    receipt = load_object(receipt_path, "preflight experiment receipt")
    if receipt.get("schema_version") != 1 or receipt.get("evidence_class") != "development-preflight-experiment-v1":
        raise ValueError("unsupported preflight experiment receipt")
    expected = {
        "experiment_id": experiment["experiment_id"],
        "profile": experiment["profile"],
        "parameters": experiment["parameters"],
        "experiment_sha256": sha256_file(experiment_path),
        "preflight_config_sha256": sha256_file(config_path),
        "base_sha": base_sha,
        "head_sha": head_sha,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"preflight experiment receipt mismatch: {key}")
    if receipt.get("development_only") is not True or receipt.get("formal_qualification_used") is not False:
        raise ValueError("preflight experiment crossed development-only boundary")
    return receipt


def self_test() -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="preflight-experiment-test-") as tmp:
        root = pathlib.Path(tmp)
        experiment = root / "experiment.json"
        config = root / "config.json"
        receipt = root / "receipt.json"
        experiment.write_text(
            json.dumps({
                "schema_version": 1,
                "experiment_id": "path-purity-fixture",
                "trigger_policy": TRIGGER_POLICY,
                "source_policy": SOURCE_POLICY,
                "profile": PATH_PURITY_PROFILE,
                "parameters": {"weight": 0.1, "margin": 0.1},
                "reason": "fixture",
            }),
            encoding="utf-8",
        )
        config.write_text(json.dumps({"train": {"epochs": 12}}), encoding="utf-8")
        result = apply_experiment(
            experiment_path=experiment,
            config_path=config,
            receipt_path=receipt,
            base_sha="1"*40,
            head_sha="2"*40,
        )
        assert result["enabled"] is True
        applied = load_object(config, "config")
        assert applied["train"]["path_purity_loss_weight"] == 0.1
        assert verify_receipt(
            experiment_path=experiment,
            config_path=config,
            receipt_path=receipt,
            base_sha="1"*40,
            head_sha="2"*40,
        ) == result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")
    verify = sub.add_parser("verify")
    verify.add_argument("--experiment", required=True, type=pathlib.Path)
    apply = sub.add_parser("apply")
    apply.add_argument("--experiment", required=True, type=pathlib.Path)
    apply.add_argument("--config", required=True, type=pathlib.Path)
    apply.add_argument("--receipt", required=True, type=pathlib.Path)
    apply.add_argument("--base-sha", required=True)
    apply.add_argument("--head-sha", required=True)
    check = sub.add_parser("verify-receipt")
    check.add_argument("--experiment", required=True, type=pathlib.Path)
    check.add_argument("--config", required=True, type=pathlib.Path)
    check.add_argument("--receipt", required=True, type=pathlib.Path)
    check.add_argument("--base-sha", required=True)
    check.add_argument("--head-sha", required=True)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("preflight experiment self-test: PASS")
        return 0
    if args.command == "verify":
        print(json.dumps(verify_experiment(args.experiment), sort_keys=True))
        return 0
    kwargs = {
        "experiment_path": args.experiment.resolve(),
        "config_path": args.config.resolve(),
        "receipt_path": args.receipt.resolve(),
        "base_sha": args.base_sha,
        "head_sha": args.head_sha,
    }
    result = apply_experiment(**kwargs) if args.command == "apply" else verify_receipt(**kwargs)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
