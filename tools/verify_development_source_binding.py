#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile

SHA40_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def require_sha40(value: str, label: str) -> str:
    if SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be exact lowercase 40-hex")
    return value


def repository_relative(root: pathlib.Path, value: str) -> pathlib.Path:
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"invalid repository-relative training code path: {value}")
    path = (root / pathlib.Path(*relative.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"training code path escapes product root: {value}") from exc
    return path


def verify(
    candidate: pathlib.Path,
    product_root: pathlib.Path,
    source_run_head: str,
    product_head: str,
    binding: pathlib.Path | None,
) -> dict:
    candidate = candidate.resolve()
    product_root = product_root.resolve()
    source_run_head = require_sha40(source_run_head, "source_run_head")
    product_head = require_sha40(product_head, "product_head")
    freeze = load_object(candidate / "freeze-manifest.json")

    binding_mode = "canonical-run-head"
    binding_class = None
    if binding is None:
        if source_run_head != product_head:
            raise ValueError(
                "development source binding is required when workflow head differs from product head"
            )
    else:
        binding = binding.resolve()
        value = load_object(binding)
        if int(value.get("schema_version", 0)) != 1:
            raise ValueError("development source binding schema_version must be 1")
        binding_class = value.get("evidence_class")
        if (
            not isinstance(binding_class, str)
            or not binding_class.endswith("development-once-source-binding")
        ):
            raise ValueError("development source binding evidence_class is invalid")
        if value.get("development_only") is not True:
            raise ValueError("development source binding must be development-only")
        for field in ("fresh_used", "shadow_used", "formal_qualification_used"):
            if value.get(field) is not False:
                raise ValueError(f"development source binding requires {field}=false")
        if str(value.get("workflow_head", "")) != source_run_head:
            raise ValueError("development source binding workflow_head mismatch")
        if str(value.get("product_head", "")) != product_head:
            raise ValueError("development source binding product_head mismatch")
        declared_policy = value.get("development_policy")
        if declared_policy is not None:
            if not isinstance(declared_policy, str) or not declared_policy:
                raise ValueError("development source binding development_policy is invalid")
            policy_path = repository_relative(product_root, declared_policy)
            if not policy_path.is_file():
                raise ValueError("declared development policy is missing from exact product head")
            expected_policy = str(freeze.get("development_policy_sha256") or "")
            if SHA256_RE.fullmatch(expected_policy) is None:
                raise ValueError("frozen candidate development policy digest is invalid")
            if sha256_file(policy_path) != expected_policy:
                raise ValueError("development policy SHA drifted from exact product head")
        binding_mode = "one-shot-product-head"

    code = freeze.get("training_code_sha256")
    if not isinstance(code, dict) or not code:
        raise ValueError("frozen candidate training_code_sha256 is missing")
    verified_paths: list[str] = []
    for name, expected in sorted(code.items(), key=lambda item: str(item[0])):
        if not isinstance(name, str) or not name:
            raise ValueError("frozen candidate training code path is invalid")
        if not isinstance(expected, str) or SHA256_RE.fullmatch(expected) is None:
            raise ValueError(f"frozen candidate training code digest is invalid: {name}")
        path = repository_relative(product_root, name)
        if not path.is_file():
            raise ValueError(f"training code missing from exact product head: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"training code SHA drifted from exact product head: {name}")
        verified_paths.append(name)

    if binding is not None:
        value = load_object(binding)
        declared_wrapper = value.get("training_wrapper")
        if declared_wrapper is not None:
            if not isinstance(declared_wrapper, str) or not declared_wrapper:
                raise ValueError("development source binding training_wrapper is invalid")
            if declared_wrapper not in code:
                raise ValueError("declared training wrapper is not frozen in training_code_sha256")

    provenance = load_object(candidate / "model-provenance.json")
    training = provenance.get("training")
    if not isinstance(training, dict):
        raise ValueError("model provenance training section is missing")
    environment = training.get("environment")
    if not isinstance(environment, dict) or environment.get("recorded") is not True:
        raise ValueError("model provenance training environment is not recorded")
    repository_sha = str(environment.get("repository_sha") or "")
    if repository_sha != product_head:
        raise ValueError(
            "model provenance repository_sha does not match exact development product head"
        )

    return {
        "schema_version": 1,
        "policy": "development-product-source-binding-v1",
        "binding_mode": binding_mode,
        "binding_evidence_class": binding_class,
        "source_run_head": source_run_head,
        "product_head": product_head,
        "model_provenance_repository_sha": repository_sha,
        "training_code_files_verified": len(verified_paths),
        "training_code_paths": verified_paths,
    }


def self_test() -> None:
    run_head = "1" * 40
    product_head = "2" * 40
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        product = root / "product"
        candidate = root / "candidate"
        product.mkdir()
        candidate.mkdir()
        source = product / "training" / "example.py"
        source.parent.mkdir(parents=True)
        source.write_text("print('bound')\n", encoding="utf-8")
        policy = product / "configs" / "policy.json"
        policy.parent.mkdir(parents=True)
        policy.write_text("{}\n", encoding="utf-8")
        (candidate / "freeze-manifest.json").write_text(
            json.dumps(
                {
                    "development_policy_sha256": sha256_file(policy),
                    "training_code_sha256": {
                        "training/example.py": sha256_file(source),
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (candidate / "model-provenance.json").write_text(
            json.dumps(
                {
                    "training": {
                        "environment": {
                            "recorded": True,
                            "repository_sha": product_head,
                        }
                    }
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        binding = root / "development-source-binding.json"
        binding.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "evidence_class": "gru-development-once-source-binding",
                    "workflow_head": run_head,
                    "product_head": product_head,
                    "development_only": True,
                    "fresh_used": False,
                    "shadow_used": False,
                    "formal_qualification_used": False,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result = verify(candidate, product, run_head, product_head, binding)
        assert result["binding_mode"] == "one-shot-product-head"
        assert result["training_code_files_verified"] == 1

        try:
            verify(candidate, product, run_head, product_head, None)
        except ValueError as exc:
            assert "required when workflow head differs" in str(exc)
        else:
            raise AssertionError("missing one-shot source binding was accepted")

        canonical = verify(candidate, product, product_head, product_head, None)
        assert canonical["binding_mode"] == "canonical-run-head"

        source.write_text("print('tampered')\n", encoding="utf-8")
        try:
            verify(candidate, product, run_head, product_head, binding)
        except ValueError as exc:
            assert "training code SHA drifted" in str(exc)
        else:
            raise AssertionError("tampered product source was accepted")

        source.write_text("print('bound')\n", encoding="utf-8")
        bound = load_object(binding)
        bound["development_policy"] = "configs/policy.json"
        bound["training_wrapper"] = "training/example.py"
        binding.write_text(json.dumps(bound, sort_keys=True) + "\n", encoding="utf-8")
        result = verify(candidate, product, run_head, product_head, binding)
        assert result["binding_evidence_class"] == "gru-development-once-source-binding"

        bound["evidence_class"] = "untrusted"
        binding.write_text(json.dumps(bound, sort_keys=True) + "\n", encoding="utf-8")
        try:
            verify(candidate, product, run_head, product_head, binding)
        except ValueError as exc:
            assert "evidence_class" in str(exc)
        else:
            raise AssertionError("invalid source binding evidence class was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--product-root", type=pathlib.Path)
    parser.add_argument("--source-run-head")
    parser.add_argument("--product-head")
    parser.add_argument("--binding", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("development product source binding self-test: PASS")
        return 0
    required = (args.candidate, args.product_root, args.source_run_head, args.product_head)
    if any(value is None for value in required):
        parser.error(
            "--candidate, --product-root, --source-run-head and --product-head are required"
        )
    result = verify(
        args.candidate,
        args.product_root,
        str(args.source_run_head),
        str(args.product_head),
        args.binding,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        raise SystemExit(2)
