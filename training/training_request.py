#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import tempfile

EVIDENCE_CLASS = "governed-model-training-invocation-v1"
REQUEST_FIELDS = {
    "schema_version",
    "request_id",
    "trigger_policy",
    "purpose",
    "source_policy",
    "reason",
}
REQUEST_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{7,127}")
HEAD_SHA_RE = re.compile(r"[0-9a-f]{40}")
TRIGGER_POLICY = "run-on-protected-main-change"
PURPOSE = "governed-product-candidate-training"
SOURCE_POLICY = "exact-current-main"
SUPPORTED_EVENTS = {"push", "workflow_dispatch"}


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: pathlib.Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("model training request must be a JSON object")
    return value


def verify_request(path: pathlib.Path) -> dict:
    value = read_json(path)
    fields = set(value)
    if fields != REQUEST_FIELDS:
        missing = sorted(REQUEST_FIELDS - fields)
        extra = sorted(fields - REQUEST_FIELDS)
        raise ValueError(
            f"model training request fields mismatch: missing={missing} extra={extra}"
        )
    if value["schema_version"] != 1:
        raise ValueError("model training request schema_version must be 1")
    request_id = value["request_id"]
    if not isinstance(request_id, str) or REQUEST_ID_RE.fullmatch(request_id) is None:
        raise ValueError("model training request_id is invalid")
    expected = {
        "trigger_policy": TRIGGER_POLICY,
        "purpose": PURPOSE,
        "source_policy": SOURCE_POLICY,
    }
    for key, wanted in expected.items():
        if value[key] != wanted:
            raise ValueError(f"model training request {key} mismatch")
    reason = value["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("model training request reason is required")
    return value


def write_receipt(
    *,
    event_name: str,
    head_sha: str,
    ref: str,
    request_path: pathlib.Path,
    output: pathlib.Path,
) -> dict:
    if event_name not in SUPPORTED_EVENTS:
        raise ValueError(f"unsupported governed training event: {event_name}")
    if HEAD_SHA_RE.fullmatch(head_sha) is None:
        raise ValueError("governed training head SHA must be exact 40-hex")
    if ref != "refs/heads/main":
        raise ValueError("governed training receipt requires refs/heads/main")

    request = None
    request_sha256 = None
    if event_name == "push":
        request = verify_request(request_path)
        request_sha256 = sha256_file(request_path)

    payload = {
        "schema_version": 1,
        "evidence_class": EVIDENCE_CLASS,
        "event_name": event_name,
        "head_sha": head_sha,
        "ref": ref,
        "source_policy": SOURCE_POLICY,
        "request": request,
        "request_sha256": request_sha256,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def self_test() -> None:
    request = {
        "schema_version": 1,
        "request_id": "contract-test-request",
        "trigger_policy": TRIGGER_POLICY,
        "purpose": PURPOSE,
        "source_policy": SOURCE_POLICY,
        "reason": "exercise versioned governed training request",
    }
    with tempfile.TemporaryDirectory(prefix="training-request-self-test-") as tmp:
        root = pathlib.Path(tmp)
        request_path = root / "request.json"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        assert verify_request(request_path) == request

        output = root / "push-receipt.json"
        push = write_receipt(
            event_name="push",
            head_sha="1" * 40,
            ref="refs/heads/main",
            request_path=request_path,
            output=output,
        )
        assert push["request"] == request
        assert push["request_sha256"] == sha256_file(request_path)
        assert json.loads(output.read_text(encoding="utf-8")) == push

        manual = write_receipt(
            event_name="workflow_dispatch",
            head_sha="2" * 40,
            ref="refs/heads/main",
            request_path=request_path,
            output=root / "manual-receipt.json",
        )
        assert manual["request"] is None
        assert manual["request_sha256"] is None

        bad = dict(request)
        bad["unexpected"] = True
        request_path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            verify_request(request_path)
        except ValueError as exc:
            assert "fields mismatch" in str(exc)
        else:
            raise AssertionError("unexpected request field was accepted")

        try:
            write_receipt(
                event_name="pull_request",
                head_sha="3" * 40,
                ref="refs/heads/main",
                request_path=request_path,
                output=root / "bad-event.json",
            )
        except ValueError as exc:
            assert "unsupported governed training event" in str(exc)
        else:
            raise AssertionError("unsupported event was accepted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--request", required=True, type=pathlib.Path)

    receipt_parser = subparsers.add_parser("write-receipt")
    receipt_parser.add_argument("--event-name", required=True)
    receipt_parser.add_argument("--head-sha", required=True)
    receipt_parser.add_argument("--ref", required=True)
    receipt_parser.add_argument("--request", required=True, type=pathlib.Path)
    receipt_parser.add_argument("--output", required=True, type=pathlib.Path)

    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("training request self-test: PASS")
        return 0
    if args.command == "verify":
        value = verify_request(args.request.resolve())
        print(f"versioned governed training request: {value['request_id']}")
        return 0
    if args.command == "write-receipt":
        write_receipt(
            event_name=args.event_name,
            head_sha=args.head_sha,
            ref=args.ref,
            request_path=args.request.resolve(),
            output=args.output.resolve(),
        )
        print(f"governed training invocation receipt: {args.output}")
        return 0
    parser.error("a command or --self-test is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
