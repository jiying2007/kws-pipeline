from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET = ROOT / "governance" / "main-ruleset-target.json"
VERIFIER = ROOT / "governance" / "verify_live_main_ruleset.py"


def run_case(live: dict, *, expect_success: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="kws-ruleset-") as tmp:
        path = pathlib.Path(tmp) / "live.json"
        path.write_text(json.dumps(live, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [
                "python3",
                str(VERIFIER),
                "--target",
                str(TARGET),
                "--live",
                str(path),
            ],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if expect_success:
            assert proc.returncode == 0, proc.stdout
            summary = json.loads(proc.stdout)
            assert summary["qualified"] is True
            assert summary["ruleset_name"] == "kws-main-terminal"
            assert summary["bypass_actor_count"] == 0
        else:
            assert proc.returncode != 0, proc.stdout
            assert "live main ruleset verification failed:" in proc.stdout


def rule(live: dict, rule_type: str) -> dict:
    return next(row for row in live["rules"] if row["type"] == rule_type)


def main() -> int:
    target = json.loads(TARGET.read_text(encoding="utf-8"))

    # A live GitHub ruleset may carry server metadata and null integration IDs;
    # those do not change the reviewed policy semantics.
    live = copy.deepcopy(target)
    live.update(
        {
            "id": 123456,
            "source_type": "Repository",
            "source": "jiying2007/kws-pipeline",
            "node_id": "RRS_example",
        }
    )
    rule(live, "pull_request")["parameters"]["automatic_copilot_code_review_enabled"] = False
    for row in rule(live, "required_status_checks")["parameters"]["required_status_checks"]:
        row["integration_id"] = None
    run_case(live, expect_success=True)

    weakened = copy.deepcopy(live)
    weakened["enforcement"] = "evaluate"
    run_case(weakened, expect_success=False)

    weakened = copy.deepcopy(live)
    weakened["bypass_actors"] = [
        {"actor_id": 1, "actor_type": "RepositoryRole", "bypass_mode": "always"}
    ]
    run_case(weakened, expect_success=False)

    weakened = copy.deepcopy(live)
    weakened["rules"] = [row for row in weakened["rules"] if row["type"] != "non_fast_forward"]
    run_case(weakened, expect_success=False)

    weakened = copy.deepcopy(live)
    rule(weakened, "pull_request")["parameters"]["required_review_thread_resolution"] = False
    run_case(weakened, expect_success=False)

    weakened = copy.deepcopy(live)
    checks = rule(weakened, "required_status_checks")["parameters"]["required_status_checks"]
    rule(weakened, "required_status_checks")["parameters"]["required_status_checks"] = [
        row for row in checks if row["context"] != "coverage"
    ]
    run_case(weakened, expect_success=False)

    weakened = copy.deepcopy(live)
    checks = rule(weakened, "required_status_checks")["parameters"]["required_status_checks"]
    checks[0]["integration_id"] = 999
    run_case(weakened, expect_success=False)

    print("test_live_ruleset_contract: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
