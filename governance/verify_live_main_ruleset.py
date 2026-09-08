#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise SystemExit(f"live main ruleset verification failed: {message}")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"{path} is not a JSON object")
    return data


def rule_map(rules: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rules, list):
        fail(f"{label}.rules is not a list")
    result: dict[str, dict[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
            fail(f"{label}.rules contains an invalid rule")
        rule_type = rule["type"]
        if rule_type in result:
            fail(f"{label}.rules contains duplicate type {rule_type!r}")
        result[rule_type] = rule
    return result


def require_equal(label: str, live: Any, expected: Any) -> None:
    if live != expected:
        fail(f"{label}: expected {expected!r}, got {live!r}")


def require_declared_parameters(
    rule_type: str,
    live_rule: dict[str, Any],
    target_rule: dict[str, Any],
) -> None:
    target_parameters = target_rule.get("parameters")
    if target_parameters is None:
        return
    if not isinstance(target_parameters, dict):
        fail(f"target rule {rule_type!r} parameters are invalid")
    live_parameters = live_rule.get("parameters")
    if not isinstance(live_parameters, dict):
        fail(f"live rule {rule_type!r} parameters are missing")

    for key, expected in target_parameters.items():
        if rule_type == "required_status_checks" and key == "required_status_checks":
            continue
        if rule_type == "pull_request" and key == "allowed_merge_methods":
            live_methods = live_parameters.get(key)
            if not isinstance(live_methods, list):
                fail("live pull_request.allowed_merge_methods is not a list")
            require_equal(
                "pull_request.allowed_merge_methods",
                sorted(live_methods),
                sorted(expected),
            )
            continue
        require_equal(f"{rule_type}.{key}", live_parameters.get(key), expected)


def status_contexts(rule: dict[str, Any], label: str) -> list[str]:
    parameters = rule.get("parameters")
    if not isinstance(parameters, dict):
        fail(f"{label}.parameters is missing")
    rows = parameters.get("required_status_checks")
    if not isinstance(rows, list):
        fail(f"{label}.required_status_checks is not a list")

    contexts: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("context"), str):
            fail(f"{label}.required_status_checks contains an invalid row")
        integration_id = row.get("integration_id")
        if integration_id is not None:
            fail(
                f"{label} unexpectedly pins status {row['context']!r} "
                f"to integration_id={integration_id!r}"
            )
        contexts.append(row["context"])
    if len(contexts) != len(set(contexts)):
        fail(f"{label}.required_status_checks contains duplicate contexts")
    return sorted(contexts)


def verify(target: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    for key in ("name", "target", "enforcement"):
        require_equal(key, live.get(key), target.get(key))

    require_equal("bypass_actors", live.get("bypass_actors", []), target.get("bypass_actors", []))

    target_ref = target.get("conditions", {}).get("ref_name", {})
    live_ref = live.get("conditions", {}).get("ref_name", {})
    for key in ("include", "exclude"):
        expected = target_ref.get(key)
        actual = live_ref.get(key)
        if not isinstance(expected, list) or not isinstance(actual, list):
            fail(f"conditions.ref_name.{key} must be a list")
        require_equal(f"conditions.ref_name.{key}", sorted(actual), sorted(expected))

    target_rules = rule_map(target.get("rules"), "target")
    live_rules = rule_map(live.get("rules"), "live")
    require_equal("rule types", sorted(live_rules), sorted(target_rules))

    for rule_type, target_rule in target_rules.items():
        live_rule = live_rules[rule_type]
        require_declared_parameters(rule_type, live_rule, target_rule)

    expected_contexts = status_contexts(target_rules["required_status_checks"], "target")
    live_contexts = status_contexts(live_rules["required_status_checks"], "live")
    require_equal("required status contexts", live_contexts, expected_contexts)

    return {
        "qualified": True,
        "ruleset_name": live["name"],
        "enforcement": live["enforcement"],
        "required_status_contexts": live_contexts,
        "bypass_actor_count": len(live.get("bypass_actors", [])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that the live GitHub main ruleset matches the reviewed repository target."
    )
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--live", required=True, type=Path)
    args = parser.parse_args()

    summary = verify(load_json(args.target), load_json(args.live))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
