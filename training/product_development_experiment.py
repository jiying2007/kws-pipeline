#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re

from objective_config import AUXILIARY_LOSS_WEIGHT_NAMES, auxiliary_loss_weights
from objective_contract import (
    ORDERED_TOKEN_SCOPES,
    SEQUENCE_MARGIN_NEGATIVE_POLICIES,
)

SCHEMA_VERSION = 1
EVIDENCE_CLASS = "product-development-pr-head-experiment-v1"
SOURCE_POLICY = "exact-pr-head"
EXPERIMENT_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
FIELDS = {
    "schema_version",
    "experiment_id",
    "development_only",
    "source_policy",
    "protected_evidence_used",
    "reason",
    "config_overrides",
}
ALLOWED_OVERRIDES = {
    **{f"train.{name}": ("float", 0.0, 1.0) for name in AUXILIARY_LOSS_WEIGHT_NAMES},
    "train.path_purity_loss_weight": ("float", 0.0, 1.0),
    "train.path_purity_margin": ("float", 0.0, 2.0),
    "train.ordered_token_scope": ("enum", tuple(sorted(ORDERED_TOKEN_SCOPES))),
    "train.sequence_margin_negative_policy": (
        "enum",
        tuple(sorted(SEQUENCE_MARGIN_NEGATIVE_POLICIES)),
    ),
    "train.epochs": ("int", 1, 72),
    "train.warm_start_epochs": ("int", 1, 72),
    "domain_iteration.max_rounds": ("int", 1, 4),
    "domain_iteration.min_rounds": ("int", 1, 4),
    "domain_iteration.adversarial_lexicon.refinement_epochs": ("int", 1, 24),
    "domain_iteration.base_failure_replay_enabled": ("bool",),
}


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


def verify_spec(path: pathlib.Path) -> dict:
    value = load_object(path, "experiment spec")
    fields = set(value)
    if fields != FIELDS:
        raise ValueError(
            "experiment spec fields mismatch: "
            f"missing={sorted(FIELDS - fields)} extra={sorted(fields - FIELDS)}"
        )
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("experiment schema_version must be 1")
    experiment_id = value["experiment_id"]
    if (
        not isinstance(experiment_id, str)
        or EXPERIMENT_ID_RE.fullmatch(experiment_id) is None
    ):
        raise ValueError("experiment_id is invalid")
    if value["development_only"] is not True:
        raise ValueError("experiment must be development_only=true")
    if value["source_policy"] != SOURCE_POLICY:
        raise ValueError("experiment source_policy must be exact-pr-head")
    if value["protected_evidence_used"] is not False:
        raise ValueError("experiment must not use protected evidence")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("experiment reason is required")
    overrides = value["config_overrides"]
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError("experiment config_overrides must be a non-empty object")
    unknown = sorted(set(overrides) - set(ALLOWED_OVERRIDES))
    if unknown:
        raise ValueError(f"experiment override is not allowed: {unknown}")
    normalized: dict[str, object] = {}
    for key, raw in overrides.items():
        contract = ALLOWED_OVERRIDES[key]
        kind = contract[0]
        if kind == "float":
            if key.removeprefix("train.") in AUXILIARY_LOSS_WEIGHT_NAMES:
                auxiliary_loss_weights({key.removeprefix("train."): raw})
            _, low, high = contract
            if isinstance(raw, bool):
                raise ValueError(f"unsupported experiment override type for {key}")
            number = float(raw)
            if not math.isfinite(number) or not low <= number <= high:
                raise ValueError(f"experiment override {key} must be in [{low},{high}]")
            normalized[key] = number
        elif kind == "int":
            _, low, high = contract
            if isinstance(raw, bool) or not isinstance(raw, int) or not low <= raw <= high:
                raise ValueError(
                    f"experiment override {key} must be an integer in [{low},{high}]"
                )
            normalized[key] = raw
        elif kind == "enum":
            _, choices = contract
            if not isinstance(raw, str) or raw not in choices:
                raise ValueError(
                    f"experiment override {key} must be one of {', '.join(choices)}"
                )
            normalized[key] = raw
        elif kind == "bool":
            if not isinstance(raw, bool):
                raise ValueError(f"experiment override {key} must be boolean")
            normalized[key] = raw
        else:
            raise ValueError(f"unsupported experiment override contract for {key}")
    result = dict(value)
    result["config_overrides"] = normalized
    return result


def set_nested(config: dict, dotted: str, value: object) -> None:
    parts = dotted.split(".")
    cursor = config
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"experiment override path is not an object: {dotted}")
        cursor = child
    cursor[parts[-1]] = value


def materialize(
    *,
    spec_path: pathlib.Path,
    effective_config_path: pathlib.Path,
    output_path: pathlib.Path,
    receipt_path: pathlib.Path,
    base_sha: str,
    head_sha: str,
) -> dict:
    if SHA_RE.fullmatch(base_sha) is None or SHA_RE.fullmatch(head_sha) is None:
        raise ValueError("experiment base/head SHA must be exact 40-hex")
    spec = verify_spec(spec_path)
    config = load_object(effective_config_path, "effective product config")
    product = config.get("product_candidate_data")
    if (
        not isinstance(product, dict)
        or product.get("policy") != "external-speech-like-product-base-v1"
        or product.get("tone_fallback_allowed") is not False
        or product.get("protected_evidence_used") is not False
    ):
        raise ValueError("experiment requires governed unprotected speech-like product base")

    train = config.setdefault("train", {})
    if not isinstance(train, dict):
        raise ValueError("effective train config must be an object")
    train["epochs"] = 12
    train["warm_start_epochs"] = 6

    iteration = config.setdefault("domain_iteration", {})
    if not isinstance(iteration, dict):
        raise ValueError("effective domain_iteration config must be an object")
    iteration["max_rounds"] = 2
    iteration["min_rounds"] = 2
    iteration["patience"] = 1
    iteration["stop_on_gate"] = True
    # A non-collapsed development candidate is not a qualified shipping model.
    iteration["nondegenerate_selection_enabled"] = True
    adversarial = iteration.get("adversarial_lexicon")
    if not isinstance(adversarial, dict) or adversarial.get("enabled") is not True:
        raise ValueError("experiment requires enabled adversarial refinement")
    adversarial["refinement_epochs"] = 6

    for key, value in spec["config_overrides"].items():
        set_nested(config, key, value)

    cold_epochs = int(train["epochs"])
    warm_epochs = int(train["warm_start_epochs"])
    if not 0 < warm_epochs <= cold_epochs:
        raise ValueError("experiment warm_start_epochs must be in [1, train.epochs]")
    min_rounds = int(iteration["min_rounds"])
    max_rounds = int(iteration["max_rounds"])
    if not 0 < min_rounds <= max_rounds <= 4:
        raise ValueError("experiment round bounds must satisfy 1 <= min <= max <= 4")
    refinement_epochs = int(adversarial["refinement_epochs"])
    if not 1 <= refinement_epochs <= 24:
        raise ValueError("experiment refinement_epochs must be in [1,24]")

    config["development_experiment"] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "experiment_id": spec["experiment_id"],
        "source_policy": SOURCE_POLICY,
        "development_only": True,
        "protected_evidence_used": False,
        "pr_base_sha": base_sha,
        "pr_head_sha": head_sha,
        "spec_sha256": sha256_file(spec_path),
        "config_overrides": dict(spec["config_overrides"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "experiment_id": spec["experiment_id"],
        "source_policy": SOURCE_POLICY,
        "development_only": True,
        "protected_evidence_used": False,
        "pr_base_sha": base_sha,
        "pr_head_sha": head_sha,
        "spec_sha256": sha256_file(spec_path),
        "effective_config_sha256": sha256_file(effective_config_path),
        "experiment_config_sha256": sha256_file(output_path),
        "config_overrides": dict(spec["config_overrides"]),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="product-development-experiment-") as tmp:
        root = pathlib.Path(tmp)
        spec = root / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_id": "path-purity-smoke-v1",
                    "development_only": True,
                    "source_policy": "exact-pr-head",
                    "protected_evidence_used": False,
                    "reason": "exercise experiment contract",
                    "config_overrides": {
                        "train.path_purity_loss_weight": 0.1,
                        "train.path_purity_margin": 0.1,
                        "train.epochs": 12,
                        "train.warm_start_epochs": 6,
                        "domain_iteration.max_rounds": 4,
                        "domain_iteration.min_rounds": 4,
                        "domain_iteration.adversarial_lexicon.refinement_epochs": 12,
                        "domain_iteration.base_failure_replay_enabled": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        verified = verify_spec(spec)
        assert verified["config_overrides"]["train.path_purity_loss_weight"] == 0.1
        assert verified["config_overrides"]["train.epochs"] == 12
        assert verified["config_overrides"]["train.warm_start_epochs"] == 6
        assert verified["config_overrides"]["domain_iteration.max_rounds"] == 4
        assert (
            verified["config_overrides"]["domain_iteration.base_failure_replay_enabled"]
            is True
        )

        effective = root / "effective.json"
        effective.write_text(
            json.dumps(
                {
                    "product_candidate_data": {
                        "policy": "external-speech-like-product-base-v1",
                        "tone_fallback_allowed": False,
                        "protected_evidence_used": False,
                    },
                    "train": {},
                    "domain_iteration": {
                        "adversarial_lexicon": {"enabled": True}
                    },
                }
            ),
            encoding="utf-8",
        )
        bad_schedule = dict(verified)
        bad_schedule["config_overrides"] = dict(verified["config_overrides"])
        bad_schedule["config_overrides"]["train.warm_start_epochs"] = 13
        spec.write_text(json.dumps(bad_schedule), encoding="utf-8")
        try:
            materialize(
                spec_path=spec,
                effective_config_path=effective,
                output_path=root / "bad-config.json",
                receipt_path=root / "bad-receipt.json",
                base_sha="1" * 40,
                head_sha="2" * 40,
            )
        except ValueError as exc:
            assert "warm_start_epochs" in str(exc)
        else:
            raise AssertionError("invalid warm-start schedule was accepted")

        bad = dict(verified)
        bad["protected_evidence_used"] = True
        spec.write_text(json.dumps(bad), encoding="utf-8")
        try:
            verify_spec(spec)
        except ValueError as exc:
            assert "protected evidence" in str(exc)
        else:
            raise AssertionError("protected experiment evidence was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")
    verify = sub.add_parser("verify")
    verify.add_argument("--spec", required=True, type=pathlib.Path)
    build = sub.add_parser("materialize")
    build.add_argument("--spec", required=True, type=pathlib.Path)
    build.add_argument("--effective-config", required=True, type=pathlib.Path)
    build.add_argument("--output", required=True, type=pathlib.Path)
    build.add_argument("--receipt", required=True, type=pathlib.Path)
    build.add_argument("--base-sha", required=True)
    build.add_argument("--head-sha", required=True)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("product development experiment self-test: PASS")
        return 0
    if args.command == "verify":
        value = verify_spec(args.spec.resolve())
        print(f"development experiment: {value['experiment_id']}")
        return 0
    if args.command == "materialize":
        receipt = materialize(
            spec_path=args.spec.resolve(),
            effective_config_path=args.effective_config.resolve(),
            output_path=args.output.resolve(),
            receipt_path=args.receipt.resolve(),
            base_sha=args.base_sha,
            head_sha=args.head_sha,
        )
        print(json.dumps(receipt, sort_keys=True))
        return 0
    parser.error("a command or --self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
