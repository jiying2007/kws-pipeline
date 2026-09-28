#!/usr/bin/env python3
"""Resolve the unique retained counterfactual pair for a completed product-development run."""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request

VARIABLE = "domain_iteration.base_failure_replay_enabled"
WORKFLOW = "product-development-experiment.yml"
SPEC_PATH = ".github/triggers/model-training-experiment.json"
FINAL_ARTIFACT_PREFIX = "product-development-experiment-"


def normalized_overrides(spec: dict) -> dict:
    overrides = spec.get("config_overrides")
    if not isinstance(overrides, dict):
        raise ValueError("experiment spec config_overrides must be an object")
    result = dict(overrides)
    result.pop(VARIABLE, None)
    return result


def pairable(spec: dict) -> bool:
    overrides = spec.get("config_overrides")
    return (
        isinstance(overrides, dict)
        and isinstance(overrides.get(VARIABLE), bool)
        and spec.get("development_only") is True
        and spec.get("source_policy") == "exact-pr-head"
        and spec.get("protected_evidence_used") is False
    )


def counterpart(left: dict, right: dict) -> bool:
    if not pairable(left) or not pairable(right):
        return False
    lo = left["config_overrides"]
    ro = right["config_overrides"]
    return (
        lo[VARIABLE] is not ro[VARIABLE]
        and normalized_overrides(left) == normalized_overrides(right)
    )


class Github:
    def __init__(self, repository: str, token: str) -> None:
        if "/" not in repository or not token:
            raise ValueError("GitHub repository/token is missing")
        self.repository = repository
        self.token = token
        self.api = "https://api.github.com"

    def get(self, path: str, *, missing_ok: bool = False):
        url = self.api + path
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "kws-pipeline-pair-resolver",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if missing_ok and exc.code == 404:
                return None
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API {exc.code}: {path}: {detail[:500]}") from exc

    def run(self, run_id: int) -> dict:
        value = self.get(f"/repos/{self.repository}/actions/runs/{run_id}")
        if not isinstance(value, dict):
            raise ValueError("workflow run response must be an object")
        return value

    def spec(self, head_sha: str) -> dict | None:
        encoded_path = "/".join(urllib.parse.quote(part, safe="") for part in SPEC_PATH.split("/"))
        value = self.get(
            f"/repos/{self.repository}/contents/{encoded_path}?ref={urllib.parse.quote(head_sha, safe='')}",
            missing_ok=True,
        )
        if value is None:
            return None
        if not isinstance(value, dict) or value.get("encoding") != "base64":
            raise ValueError("experiment spec content response is invalid")
        raw = base64.b64decode(str(value.get("content", "")), validate=False)
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("experiment spec must be an object")
        return parsed

    def pulls(self) -> list[dict]:
        value = self.get(
            f"/repos/{self.repository}/pulls?state=all&base=main&sort=updated&direction=desc&per_page=100"
        )
        if not isinstance(value, list):
            raise ValueError("pull list response must be an array")
        return [row for row in value if isinstance(row, dict)]

    def workflow_runs(self) -> list[dict]:
        value = self.get(
            f"/repos/{self.repository}/actions/workflows/{WORKFLOW}/runs?event=pull_request&per_page=100"
        )
        if not isinstance(value, dict) or not isinstance(value.get("workflow_runs"), list):
            raise ValueError("workflow run list response is invalid")
        return [row for row in value["workflow_runs"] if isinstance(row, dict)]

    def has_final_artifact(self, run_id: int, base_sha: str, head_sha: str) -> bool:
        value = self.get(f"/repos/{self.repository}/actions/runs/{run_id}/artifacts?per_page=100")
        if not isinstance(value, dict) or not isinstance(value.get("artifacts"), list):
            raise ValueError("workflow artifact list response is invalid")
        expected = f"{FINAL_ARTIFACT_PREFIX}{base_sha}-{head_sha}"
        return any(
            isinstance(row, dict)
            and row.get("name") == expected
            and row.get("expired") is not True
            for row in value["artifacts"]
        )


def run_for_head(runs: list[dict], head_sha: str, pr_number: int) -> dict | None:
    matches = []
    for row in runs:
        if row.get("head_sha") != head_sha or row.get("status") != "completed":
            continue
        pulls = row.get("pull_requests")
        if isinstance(pulls, list) and pulls:
            numbers = {
                int(item["number"])
                for item in pulls
                if isinstance(item, dict) and isinstance(item.get("number"), int)
            }
            if pr_number not in numbers:
                continue
        matches.append(row)
    if not matches:
        return None
    return max(matches, key=lambda row: (int(row.get("run_attempt", 0)), int(row.get("id", 0))))


def resolve(github: Github, current_run_id: int) -> dict:
    current = github.run(current_run_id)
    if current.get("name") != "product-development-experiment":
        return {"ready": False, "state": "ignored-workflow", "current_run_id": current_run_id}
    if current.get("status") != "completed":
        return {"ready": False, "state": "current-run-not-completed", "current_run_id": current_run_id}
    pulls = current.get("pull_requests")
    if not isinstance(pulls, list) or len(pulls) != 1 or not isinstance(pulls[0], dict):
        return {"ready": False, "state": "ignored-non-pr-run", "current_run_id": current_run_id}

    current_pr = pulls[0]
    pr_number = int(current_pr["number"])
    head = current_pr.get("head")
    base = current_pr.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        raise ValueError("current run PR refs are missing")
    head_sha = str(head.get("sha", ""))
    base_sha = str(base.get("sha", ""))
    if len(head_sha) != 40 or len(base_sha) != 40:
        raise ValueError("current run PR SHA evidence is invalid")

    current_spec = github.spec(head_sha)
    if current_spec is None or not pairable(current_spec):
        return {
            "ready": False,
            "state": "ignored-non-paired-experiment",
            "current_run_id": current_run_id,
            "current_pr": pr_number,
        }
    if not github.has_final_artifact(current_run_id, base_sha, head_sha):
        raise ValueError("completed paired experiment lacks retained final artifact")

    candidates = []
    for pr in github.pulls():
        if int(pr.get("number", -1)) == pr_number:
            continue
        pbase = pr.get("base")
        phead = pr.get("head")
        if not isinstance(pbase, dict) or not isinstance(phead, dict):
            continue
        if str(pbase.get("sha", "")) != base_sha:
            continue
        candidate_sha = str(phead.get("sha", ""))
        if len(candidate_sha) != 40:
            continue
        candidate_spec = github.spec(candidate_sha)
        if candidate_spec is None or not counterpart(current_spec, candidate_spec):
            continue
        candidates.append(
            {
                "pr_number": int(pr["number"]),
                "head_sha": candidate_sha,
                "spec": candidate_spec,
            }
        )

    if not candidates:
        return {
            "ready": False,
            "state": "waiting-counterpart-pr",
            "current_run_id": current_run_id,
            "current_pr": pr_number,
            "base_sha": base_sha,
        }

    runs = github.workflow_runs()
    ready = []
    waiting = []
    for candidate in candidates:
        row = run_for_head(runs, candidate["head_sha"], candidate["pr_number"])
        if row is None:
            waiting.append(candidate["pr_number"])
            continue
        run_id = int(row["id"])
        if not github.has_final_artifact(run_id, base_sha, candidate["head_sha"]):
            waiting.append(candidate["pr_number"])
            continue
        ready.append({**candidate, "run_id": run_id})

    if not ready:
        return {
            "ready": False,
            "state": "waiting-counterpart-run",
            "current_run_id": current_run_id,
            "current_pr": pr_number,
            "base_sha": base_sha,
            "waiting_prs": sorted(waiting),
        }
    if len(ready) != 1:
        raise ValueError(
            "paired experiment resolution is ambiguous: "
            + ",".join(str(row["pr_number"]) for row in ready)
        )

    other = ready[0]
    current_value = current_spec["config_overrides"][VARIABLE]
    if current_value is False:
        control_run_id, treatment_run_id = current_run_id, other["run_id"]
        control_pr, treatment_pr = pr_number, other["pr_number"]
    else:
        control_run_id, treatment_run_id = other["run_id"], current_run_id
        control_pr, treatment_pr = other["pr_number"], pr_number

    return {
        "ready": True,
        "state": "paired",
        "base_sha": base_sha,
        "control_run_id": control_run_id,
        "treatment_run_id": treatment_run_id,
        "control_pr": control_pr,
        "treatment_pr": treatment_pr,
        "trigger_run_id": current_run_id,
    }


def emit_outputs(result: dict, path: pathlib.Path | None) -> None:
    if path is None:
        return
    lines = [
        f"ready={'true' if result.get('ready') is True else 'false'}",
        f"state={result.get('state', 'unknown')}",
    ]
    for key in ("control_run_id", "treatment_run_id", "control_pr", "treatment_pr", "base_sha"):
        if key in result:
            lines.append(f"{key}={result[key]}")
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def self_test() -> None:
    base = {
        "schema_version": 1,
        "development_only": True,
        "source_policy": "exact-pr-head",
        "protected_evidence_used": False,
        "config_overrides": {
            "train.ctc_vad_align": True,
            "train.sequence_margin_negative_policy": "runtime-executable-v1",
            VARIABLE: False,
        },
    }
    treatment = json.loads(json.dumps(base))
    treatment["config_overrides"][VARIABLE] = True
    assert pairable(base)
    assert counterpart(base, treatment)
    same = json.loads(json.dumps(base))
    assert not counterpart(base, same)
    drift = json.loads(json.dumps(treatment))
    drift["config_overrides"]["train.ctc_vad_align"] = False
    assert not counterpart(base, drift)
    unrelated = json.loads(json.dumps(base))
    unrelated["config_overrides"].pop(VARIABLE)
    assert not pairable(unrelated)
    assert not counterpart(base, unrelated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--github-output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("product development auto-pair resolver self-test: PASS")
        return 0
    if args.run_id is None:
        parser.error("--run-id is required")
    if not args.repository or not args.token:
        parser.error("repository and token are required")

    try:
        result = resolve(Github(args.repository, args.token), args.run_id)
        code = 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        result = {
            "ready": False,
            "state": "resolver-error",
            "current_run_id": args.run_id,
            "error": f"{type(exc).__name__}: {exc}",
        }
        code = 2

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    emit_outputs(result, args.github_output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
