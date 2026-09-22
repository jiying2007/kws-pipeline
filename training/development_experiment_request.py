#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile

EVIDENCE_CLASS = "product-development-pr-experiment-invocation-v1"
REQUEST_FIELDS = {
    "schema_version",
    "enabled",
    "request_id",
    "trigger_policy",
    "purpose",
    "source_policy",
    "reason",
}
REQUEST_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}")
SHA_RE = re.compile(r"[0-9a-f]{40}")
TRIGGER_POLICY = "pull-request-marker-change"
PURPOSE = "product-speech-like-development-ab"
SOURCE_POLICY = "exact-pr-head-development-only"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("development experiment request must be a JSON object")
    return value


def verify_request(path: pathlib.Path) -> dict:
    value = read_json(path)
    fields = set(value)
    if fields != REQUEST_FIELDS:
        missing = sorted(REQUEST_FIELDS - fields)
        extra = sorted(fields - REQUEST_FIELDS)
        raise ValueError(
            f"development experiment request fields mismatch: missing={missing} extra={extra}"
        )
    if value["schema_version"] != 1:
        raise ValueError("development experiment request schema_version must be 1")
    if not isinstance(value["enabled"], bool):
        raise ValueError("development experiment request enabled must be a real boolean")
    request_id = value["request_id"]
    if not isinstance(request_id, str) or REQUEST_ID_RE.fullmatch(request_id) is None:
        raise ValueError("development experiment request_id is invalid")
    expected = {
        "trigger_policy": TRIGGER_POLICY,
        "purpose": PURPOSE,
        "source_policy": SOURCE_POLICY,
    }
    for key, wanted in expected.items():
        if value[key] != wanted:
            raise ValueError(f"development experiment request {key} mismatch")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("development experiment request reason is required")
    return value


def require_sha(value: str, label: str) -> str:
    if SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be exact 40-hex")
    return value


def write_receipt(
    *,
    base_sha: str,
    head_sha: str,
    pr_number: int,
    request_path: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    base_sha = require_sha(base_sha, "development experiment base SHA")
    head_sha = require_sha(head_sha, "development experiment head SHA")
    if base_sha == head_sha:
        raise ValueError("development experiment base/head SHA must differ")
    if pr_number <= 0:
        raise ValueError("development experiment PR number must be positive")
    request = verify_request(request_path)
    if request["enabled"] is not True:
        raise ValueError("development experiment receipt requires enabled request")

    payload = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "event_name": "pull_request",
        "pr_number": pr_number,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "source_policy": SOURCE_POLICY,
        "request": request,
        "request_sha256": sha256_file(request_path),
        "promotion_authorized": False,
        "formal_qualification_authorized": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def self_test() -> None:
    disabled = {
        "schema_version": 1,
        "enabled": False,
        "request_id": "development-experiment-disabled",
        "trigger_policy": TRIGGER_POLICY,
        "purpose": PURPOSE,
        "source_policy": SOURCE_POLICY,
        "reason": "disabled by default",
    }
    enabled = {**disabled, "enabled": True, "request_id": "contract-test-development-experiment"}
    with tempfile.TemporaryDirectory(prefix="development-experiment-request-") as tmp:
        root = pathlib.Path(tmp)
        request = root / "request.json"
        request.write_text(json.dumps(disabled), encoding="utf-8")
        assert verify_request(request)["enabled"] is False
        request.write_text(json.dumps(enabled), encoding="utf-8")
        receipt = write_receipt(
            base_sha="1" * 40,
            head_sha="2" * 40,
            pr_number=123,
            request_path=request,
            output=root / "receipt.json",
        )
        assert receipt["source_policy"] == SOURCE_POLICY
        assert receipt["promotion_authorized"] is False
        assert receipt["formal_qualification_authorized"] is False

        invalid = dict(enabled)
        invalid["source_policy"] = "exact-current-main"
        request.write_text(json.dumps(invalid), encoding="utf-8")
        try:
            verify_request(request)
        except ValueError as exc:
            assert "source_policy mismatch" in str(exc)
        else:
            raise AssertionError("governed-main source policy was accepted in PR experiment lane")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    sub = parser.add_subparsers(dest="command")

    verify = sub.add_parser("verify")
    verify.add_argument("--request", required=True, type=pathlib.Path)

    receipt = sub.add_parser("write-receipt")
    receipt.add_argument("--base-sha", required=True)
    receipt.add_argument("--head-sha", required=True)
    receipt.add_argument("--pr-number", required=True, type=int)
    receipt.add_argument("--request", required=True, type=pathlib.Path)
    receipt.add_argument("--output", required=True, type=pathlib.Path)

    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("development experiment request self-test: PASS")
        return 0
    if args.command == "verify":
        value = verify_request(args.request.resolve())
        print(
            "development experiment request: "
            f"{value['request_id']} enabled={str(value['enabled']).lower()}"
        )
        return 0
    if args.command == "write-receipt":
        write_receipt(
            base_sha=args.base_sha,
            head_sha=args.head_sha,
            pr_number=args.pr_number,
            request_path=args.request.resolve(),
            output=args.output.resolve(),
        )
        print("development experiment receipt: PASS")
        return 0
    parser.error("a command or --self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
