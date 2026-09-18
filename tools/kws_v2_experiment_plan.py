#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib

POLICY_ID = "kws-v2-staged-experiments-v1"
RESOURCE_POLICY = "model-family-resource-contract-v1"
GENERALIZATION_POLICY = "development-generalization-v1"


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def changed_paths(left: object, right: object, prefix: str = "") -> set[str]:
    if type(left) is not type(right):
        return {prefix or "<root>"}
    if isinstance(left, dict):
        keys = set(left) | set(right)
        result: set[str] = set()
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                result.add(path)
            else:
                result.update(changed_paths(left[key], right[key], path))
        return result
    if isinstance(left, list):
        return set() if left == right else {prefix}
    return set() if left == right else {prefix}


def normalize_policy(path: pathlib.Path) -> dict:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != POLICY_ID:
        raise ValueError("KWS v2 experiment policy identity mismatch")
    if value.get("evidence_scope") != "development-only":
        raise ValueError("experiment policy must be development-only")
    protected = value.get("protected_evidence")
    if not isinstance(protected, dict) or any(protected.get(key) is not False for key in ("fresh_used", "shadow_used", "formal_qualification_used")):
        raise ValueError("protected evidence flags must all be false")
    invariants = value.get("invariants")
    if not isinstance(invariants, dict) or invariants.get("single_primary_variable_per_stage") is not True:
        raise ValueError("single-primary-variable invariant is required")
    if invariants.get("protected_feedback_allowed") is not False:
        raise ValueError("protected feedback must remain forbidden")
    return value


def resolve_input(policy_path: pathlib.Path, raw: str) -> pathlib.Path:
    path = pathlib.Path(raw)
    if path.is_absolute():
        return path.resolve()
    # Policy lives in configs/training; repository root is two parents above it.
    repo_root = policy_path.resolve().parents[2]
    return (repo_root / path).resolve()


def validate_resource_contract(path: pathlib.Path, policy: dict) -> dict[tuple[str, int], str]:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != RESOURCE_POLICY:
        raise ValueError("resource contract identity mismatch")
    candidates: dict[tuple[str, int], str] = {}
    for row in value.get("candidates", []):
        if not isinstance(row, dict):
            raise ValueError("resource candidate must be an object")
        family = str(row.get("family", ""))
        hidden = int(row.get("hidden_dim", 0))
        name = str(row.get("name", ""))
        if family not in {"rnn", "gru"} or hidden <= 0 or not name:
            raise ValueError("resource candidate is invalid")
        candidates[(family, hidden)] = name

    stage_b = policy["stage_b_capacity"]["candidates"]
    expected = {
        (family, int(hidden))
        for family, values in stage_b.items()
        for hidden in values
    }
    missing = sorted(expected - set(candidates))
    if missing:
        raise ValueError(f"stage-B candidates are absent from resource contract: {missing}")
    if max(hidden for _, hidden in expected) > int(policy["stage_b_capacity"]["max_hidden_dim_current_runtime_contract"]):
        raise ValueError("stage-B hidden dimension exceeds current runtime contract")
    return candidates


def validate_generalization_policy(path: pathlib.Path) -> dict:
    value = load_object(path)
    if int(value.get("schema_version", 0)) != 1 or value.get("policy") != GENERALIZATION_POLICY:
        raise ValueError("development-generalization policy identity mismatch")
    search = value.get("tiers", {}).get("search")
    if not isinstance(search, dict):
        raise ValueError("generalization search tier is missing")
    if int(search.get("recommended_independent_seeds", 0)) < 8:
        raise ValueError("KWS v2 experiments require at least 8 recommended search seeds")
    return value


def validate_base_config(value: dict) -> None:
    model = value.get("model")
    if not isinstance(model, dict):
        raise ValueError("base config model is missing")
    if int(model.get("feature_dim", 0)) != 32:
        raise ValueError("v1 staged experiments require feature_dim=32")
    if int(model.get("hidden_dim", 0)) != 64:
        raise ValueError("stage A requires the canonical hidden_dim=64 baseline")
    frontends = model.get("frontends")
    if frontends != ["logmel"]:
        raise ValueError("stage A base config must start from the canonical logmel frontend")


def candidate_record(
    *,
    candidate_id: str,
    family: str,
    frontend: str,
    hidden_dim: int,
    primary_variable: str,
    config_path: pathlib.Path,
    base_sha: str,
    stage_base_sha: str,
    mutation_paths: set[str],
    resource_name: str,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "model_family": family,
        "frontend": frontend,
        "feature_dim": 32,
        "hidden_dim": hidden_dim,
        "primary_variable": primary_variable,
        "config": config_path.name,
        "config_path_contract": "matrix-relative-v1",
        "config_sha256": sha256_file(config_path),
        "canonical_base_config_sha256": base_sha,
        "stage_base_config_sha256": stage_base_sha,
        "mutation_paths_from_stage_base": sorted(mutation_paths),
        "resource_contract_candidate": resource_name,
        "generalization_tier": "search",
        "protected_evidence_used": False,
    }


def build_stage_a(
    *,
    policy: dict,
    base: dict,
    base_sha: str,
    out: pathlib.Path,
    resources: dict[tuple[str, int], str],
) -> list[dict]:
    stage = policy["stage_a_frontend"]
    allowed = set(stage["mutable_paths"])
    fixed_hidden = int(stage["fixed_hidden_dim"])
    if fixed_hidden != 64:
        raise ValueError("stage A fixed_hidden_dim must remain 64")
    records: list[dict] = []
    for family in stage["families"]:
        for frontend in stage["frontends"]:
            candidate = copy.deepcopy(base)
            candidate["model"]["frontends"] = [frontend]
            path = out / f"stage-a-{family}-h64-{frontend}.json"
            write_json(path, candidate)
            mutations = changed_paths(base, candidate)
            if not mutations.issubset(allowed):
                raise ValueError(f"stage A mutated forbidden paths: {sorted(mutations - allowed)}")
            records.append(
                candidate_record(
                    candidate_id=f"v2-a-{family}-h64-{frontend}",
                    family=family,
                    frontend=frontend,
                    hidden_dim=64,
                    primary_variable="frontend",
                    config_path=path,
                    base_sha=base_sha,
                    stage_base_sha=base_sha,
                    mutation_paths=mutations,
                    resource_name=resources[(family, 64)],
                )
            )
    return records


def build_stage_b(
    *,
    policy: dict,
    base: dict,
    base_sha: str,
    frontend: str,
    out: pathlib.Path,
    resources: dict[tuple[str, int], str],
) -> list[dict]:
    if frontend not in set(policy["stage_a_frontend"]["frontends"]):
        raise ValueError("stage B frontend must be a stage-A candidate")
    stage = policy["stage_b_capacity"]
    if int(stage["max_hidden_dim_current_runtime_contract"]) != 64:
        raise ValueError("current runtime hidden-dimension ceiling drifted")
    stage_base = copy.deepcopy(base)
    stage_base["model"]["frontends"] = [frontend]
    stage_base_path = out / f"stage-b-base-{frontend}.json"
    write_json(stage_base_path, stage_base)
    stage_base_sha = sha256_file(stage_base_path)
    allowed = set(stage["mutable_paths"])
    records: list[dict] = []
    for family, hidden_values in stage["candidates"].items():
        for hidden in hidden_values:
            hidden = int(hidden)
            candidate = copy.deepcopy(stage_base)
            candidate["model"]["hidden_dim"] = hidden
            path = out / f"stage-b-{family}-h{hidden}-{frontend}.json"
            write_json(path, candidate)
            mutations = changed_paths(stage_base, candidate)
            if not mutations.issubset(allowed):
                raise ValueError(f"stage B mutated forbidden paths: {sorted(mutations - allowed)}")
            records.append(
                candidate_record(
                    candidate_id=f"v2-b-{family}-h{hidden}-{frontend}",
                    family=family,
                    frontend=frontend,
                    hidden_dim=hidden,
                    primary_variable="hidden_dim",
                    config_path=path,
                    base_sha=base_sha,
                    stage_base_sha=stage_base_sha,
                    mutation_paths=mutations,
                    resource_name=resources[(family, hidden)],
                )
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize staged single-variable KWS v2 GRU/RNN experiments.")
    parser.add_argument("--policy", required=True, type=pathlib.Path)
    parser.add_argument("--stage", required=True, choices=("A", "B", "C"))
    parser.add_argument("--frontend", choices=("logmel", "pcen-lite"))
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    policy_path = args.policy.resolve()
    policy = normalize_policy(policy_path)
    inputs = policy["inputs"]
    base_path = resolve_input(policy_path, inputs["base_config"])
    resource_path = resolve_input(policy_path, inputs["resource_contract"])
    generalization_path = resolve_input(policy_path, inputs["generalization_policy"])
    base = load_object(base_path)
    validate_base_config(base)
    resources = validate_resource_contract(resource_path, policy)
    generalization = validate_generalization_policy(generalization_path)

    if args.stage == "C":
        if policy["stage_c_loss_ablation"].get("enabled") is not False:
            raise ValueError("stage C v1 must stay disabled until frontend/capacity baseline is frozen")
        raise ValueError("stage C is intentionally blocked until stages A/B are frozen")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_sha = sha256_file(base_path)
    if args.stage == "A":
        if args.frontend is not None:
            raise ValueError("--frontend is not accepted for stage A")
        candidates = build_stage_a(
            policy=policy,
            base=base,
            base_sha=base_sha,
            out=out,
            resources=resources,
        )
    else:
        if args.frontend is None:
            raise ValueError("stage B requires the frozen stage-A --frontend choice")
        candidates = build_stage_b(
            policy=policy,
            base=base,
            base_sha=base_sha,
            frontend=args.frontend,
            out=out,
            resources=resources,
        )

    matrix = {
        "schema_version": 1,
        "policy": POLICY_ID,
        "evidence_class": "kws-v2-staged-experiment-matrix-v1",
        "evidence_scope": "development-only",
        "stage": args.stage,
        "canonical_base_config": base_path.relative_to(policy_path.resolve().parents[2]).as_posix(),
        "canonical_base_config_path_contract": "repository-relative-v1",
        "canonical_base_config_sha256": base_sha,
        "resource_contract_sha256": sha256_file(resource_path),
        "generalization_policy_sha256": sha256_file(generalization_path),
        "recommended_generalization_search_seeds": int(
            generalization["tiers"]["search"]["recommended_independent_seeds"]
        ),
        "corpus_requirement": inputs["corpus_requirement"],
        "requires_predeclared_generalization_plan": inputs["require_predeclared_plan"] is True,
        "candidates": candidates,
        "single_primary_variable_per_stage": True,
        "fresh_used": False,
        "shadow_used": False,
        "formal_qualification_used": False,
    }
    write_json(out / "experiment-matrix.json", matrix)
    print(
        f"kws-v2-experiment-plan: stage={args.stage} candidates={len(candidates)} "
        f"search-seeds={matrix['recommended_generalization_search_seeds']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
