#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
from qualification_fixture import runtime_soak_fixture
from runtime_soak_contract import CPU_MEASUREMENT_CONTRACT_ID
from score_target_dut_qualification import validate_resource_budget
DIAGNOSTIC_PATH = ROOT / ".target-shipping-contract-diagnostic.json"
DIAGNOSTIC_STDOUT_LIMIT = 4096


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_diagnostic(
    *,
    line: int,
    command,
    expected_rc,
    actual_rc,
    stdout: str,
) -> None:
    if DIAGNOSTIC_PATH.exists():
        return
    DIAGNOSTIC_PATH.write_text(
        json.dumps(
            {
                "line": line,
                "command": command,
                "expected_rc": expected_rc,
                "actual_rc": actual_rc,
                "stdout": stdout[-DIAGNOSTIC_STDOUT_LIMIT:],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run(*args: str, expect: int = 0) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != expect:
        write_diagnostic(
            line=sys._getframe(1).f_lineno,
            command=list(args),
            expected_rc=expect,
            actual_rc=completed.returncode,
            stdout=completed.stdout,
        )
        raise AssertionError(
            f"command: {' '.join(args)}\n"
            f"expected exit {expect}, got {completed.returncode}:\n{completed.stdout}"
        )
    return completed


def write_runtime_soak(path: pathlib.Path, hours: float) -> None:
    path.write_text(json.dumps(runtime_soak_fixture(hours), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_bundle(
    root: pathlib.Path,
    *,
    source_sha: str,
    phase_a_tag: str,
    deployment_tag: str,
    afe_sha: str,
    afe_identity: str,
    cpu_percent: float = 5.0,
    capacity: int = 2,
    threads: int = 1,
) -> pathlib.Path:
    bundle = root / "bundle"
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    model = root / "model.kwm"
    keywords = root / "keywords.kwk"
    model.write_bytes(b"fixture-model")
    keywords.write_bytes(b"fixture-keywords")
    runner = bundle / "board-runner"
    audio = bundle / "board-audio.wav"
    runner.write_bytes(b"fixture-board-runner")
    audio.write_bytes(b"non-human-board-audio")

    resource_budget = bundle / "resource-budget.json"
    resource_budget.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "approved",
                "budget_id": "fixture-budget-v1",
                "sku": "fixture-sku",
                "board_revision": "A",
                "measurement_contract_id": CPU_MEASUREMENT_CONTRACT_ID,
                "authority": "fixture-product-owner",
                "approved_at_utc": "2026-09-08T00:00:00Z",
                "limits": {
                    "max_cpu_percent": 10.0,
                    "max_rss_kib": 2048.0,
                    "max_stack_high_water_bytes": 65536.0,
                    "max_temp_c": 70.0,
                    "max_average_power_mw": 250.0,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    soak = raw / "runtime-soak.json"
    power = raw / "power.csv"
    continuity = raw / "audio-continuity.json"
    soak.write_text(json.dumps(runtime_soak_fixture(24.0, cpu_percent=cpu_percent, capacity=capacity, threads=threads)) + "\n", encoding="utf-8")
    power.write_text("t,power_mw\n0,123\n", encoding="utf-8")
    continuity.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "xrun_count": 0,
                "discontinuity_count": 0,
                "lost_samples": 0,
                "backpressure_count": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    evidence_raw = bundle / "evidence-raw.jsonl"
    raw_paths = [soak, power, continuity]
    evidence_raw.write_text(
        "".join(
            json.dumps(
                {"name": path.name, "sha256": sha(path), "bytes": path.stat().st_size},
                sort_keys=True,
            )
            + "\n"
            for path in raw_paths
        ),
        encoding="utf-8",
    )

    collector = ROOT / "tools/collect_target_evidence.py"
    attestation = bundle / "attestation-verification.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "verified": True,
                "subject_kind": "kws-target-evidence",
                "issuer": "fixture-trust-layer",
                "trust_policy": "fixture-physical-policy",
                "verified_at_utc": "2026-09-08T00:00:00Z",
                "subject_sha256": sha(evidence_raw),
                "collector_sha256": sha(collector),
                "board_runner_sha256": sha(runner),
                "model_sha256": sha(model),
                "keyword_pack_sha256": sha(keywords),
                "audio_frontend_identity_sha256": afe_identity,
                "resource_budget_sha256": sha(resource_budget),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    target_evidence = bundle / "target-evidence.json"
    run(
        sys.executable,
        "tools/collect_target_evidence.py",
        "--output", str(target_evidence),
        "--target", "fixture-target",
        "--board-revision", "A",
        "--soc", "fixture-soc",
        "--toolchain", "fixture-gcc",
        "--compiler-flags=-O3",
        "--audio-frontend", "final-audio-pipeline",
        "--audio-frontend-sha256", afe_sha,
        "--audio-frontend-identity-sha256", afe_identity,
        "--resource-budget", str(resource_budget),
        "--runtime-soak", str(soak),
        "--stack-high-water-bytes", "4096",
        "--average-power-mw", "123",
        "--raw-evidence", str(continuity),
        "--power-raw", str(power),
        "--evidence-raw", str(evidence_raw),
        "--attestation-verification", str(attestation),
        "--board-runner", str(runner),
        "--model", str(model),
        "--keyword-pack", str(keywords),
        "--board-audio", str(audio),
        "--sku", "fixture-sku",
        "--source-sha", source_sha,
        "--builder-id", "builder-a",
        "--dut-id", "dut-a",
        "--collector-id", "station-a",
        "--instrument-id", "meter-a",
        "--calibration-id", "cal-a",
    )

    blocks = 100
    total_us = 100000.0
    p99 = 1000.0
    board_summary = {
        "schema_version": 1,
        "runner_sha256": sha(runner),
        "model_sha256": sha(model),
        "keyword_pack_sha256": sha(keywords),
        "audio_sha256": sha(audio),
        "runtime_version": "fixture",
        "runtime_source_revision": source_sha,
        "runtime_config_digest": "a" * 64,
        "runtime_target": "fixture-target",
        "block_samples": 320,
        "block_deadline_us": 20000.0,
        "audio_seconds": 1.0,
        "repeats": 2,
        "blocks": blocks,
        "model_bytes": model.stat().st_size,
        "keyword_pack_bytes": keywords.stat().st_size,
        "arena_bytes": 8192,
        "total_process_us": total_us,
        "mean_process_us": total_us / blocks,
        "p50_process_us": 800.0,
        "p95_process_us": 900.0,
        "p99_process_us": p99,
        "max_process_us": 1100.0,
        "rtf": total_us / 2_000_000.0,
        "p99_headroom": 20000.0 / p99,
    }
    (bundle / "board-summary.json").write_text(
        json.dumps(board_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (bundle / "target-profile.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile_id": "fixture-target-a",
                "human_qualification_tag": phase_a_tag,
                "deployment_tag": deployment_tag,
                "resource_budget_id": "fixture-budget-v1",
                "board_audio_class": "non-human-public-safe",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> int:
    production = json.loads(
        (ROOT / "commercial/target-qualification.policy.json").read_text(encoding="utf-8")
    )
    assert production["schema_version"] == 3
    assert production["shipping_approved"] is False
    assert production["per_dut_hard_gates"]["min_soak_hours"] == 24.0
    assert production["resource_budget"]["required"] is True
    assert "max_average_power_mw" not in production["per_dut_hard_gates"]
    assert production["cohort"]["min_unique_duts"] == 3
    assert production["cohort"]["min_long_soak_hours"] == 72.0
    assert production["cohort"]["min_long_soak_duts"] == 1
    assert production["cohort"]["require_same_resource_budget"] is True
    assert production["board_audio_policy"]["human_derived_audio_forbidden"] is True

    human_workflow = (ROOT / ".github/workflows/real-human-qualification.yml").read_text(encoding="utf-8")
    dut_workflow = (ROOT / ".github/workflows/target-dut-qualification.yml").read_text(encoding="utf-8")
    cohort_workflow = (ROOT / ".github/workflows/target-cohort-promotion.yml").read_text(encoding="utf-8")
    approval_workflow = (ROOT / ".github/workflows/shipping-approval.yml").read_text(encoding="utf-8")
    run("bash", "-n", "governance/require_current_main.sh")
    for text in (human_workflow, dut_workflow, cohort_workflow, approval_workflow):
        header = text.split("permissions:", 1)[0]
        assert "workflow_dispatch:" in header
        assert "pull_request:" not in header
        assert "push:" not in header
        assert "train_ctc.py" not in text
        assert "iterate_domain.py" not in text
        assert "governance/require_current_main.sh" in text
    assert "runs-on: [self-hosted, kws-target-board]" in dut_workflow
    assert "board_audio_class" in dut_workflow
    assert "resource_budget" in dut_workflow
    assert "target-cohort-qualified-" in cohort_workflow
    assert "shipping-approved-" in approval_workflow
    assert "shipping_approved': True" in approval_workflow
    assert "verify_live_main_ruleset.py" in approval_workflow or "require_current_main.sh" in approval_workflow
    assert "public-phase-a-receipt.json" in approval_workflow
    assert "public-target-cohort-receipt.json" in approval_workflow
    # Keep every public-receipt path on the current unit/version before it can
    # reach publishing. The terminal inline Python check is also executed with
    # synthetic objects, without any GitHub, approval or publication operation.
    for text in (dut_workflow, cohort_workflow, approval_workflow):
        assert '.schema_version == 3 and .measurement_contract_id == "process-cpu-one-core-v1"' in text
        assert '.schema_version == 2 and .measurement_contract_id == "process-cpu-one-core-v1"' in text
    assert '{schema_version:2,phase:"physical-target-dut-qualification"' in dut_workflow
    assert '{schema_version:2,phase:"physical-target-cohort-qualification"' in cohort_workflow
    assert '.resource_budget.measurement_contract_id == "process-cpu-one-core-v1" and .metrics.measurement_contract_id == "process-cpu-one-core-v1"' in cohort_workflow
    start = approval_workflow.index("          for label, value, version in (")
    end = approval_workflow.index("          if governance.get", start)
    guard = compile(textwrap.dedent(approval_workflow[start:end]), "shipping-approval-cpu-contract", "exec")
    good = {name: {"schema_version": version, "measurement_contract_id": CPU_MEASUREMENT_CONTRACT_ID}
            for name, version in (("policy", 3), ("budget", 2), ("cohort", 2), ("cohort_receipt", 2))}
    exec(guard, dict(good))
    for name in good:
        for field, bad_value in (("schema_version", 1), ("measurement_contract_id", None),
                                 ("measurement_contract_id", "legacy-online-capacity")):
            bad = json.loads(json.dumps(good))
            bad[name][field] = bad_value
            try:
                exec(guard, bad)
            except SystemExit as exc:
                assert "unsupported CPU measurement contract" in str(exc)
            else:
                raise AssertionError(f"shipping approval accepted old {name}/{field}")


    with tempfile.TemporaryDirectory(prefix="kws-target-shipping-") as td:
        root = pathlib.Path(td)
        source_sha = "b" * 40
        corpus_sha = "c" * 64
        afe_identity = "d" * 64
        afe_executable = "e" * 64
        deployment_tag = "deployment-fixture0000"
        human_tag = "human-qualified-0123456789abcdef"
        bundle = make_bundle(
            root,
            source_sha=source_sha,
            phase_a_tag=human_tag,
            deployment_tag=deployment_tag,
            afe_sha=afe_executable,
            afe_identity=afe_identity,
        )
        model = root / "model.kwm"
        keywords = root / "keywords.kwk"
        phase_a_summary = root / "phase-a-summary.json"
        phase_a_summary.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "phase": "real-human-final-afe-acoustic-qualification",
                    "qualified": True,
                    "shipping_approved": False,
                    "deployment_tag": deployment_tag,
                    "corpus_sha256": corpus_sha,
                    "afe": {
                        "identity_sha256": afe_identity,
                        "executable_sha256": afe_executable,
                        "sku": "fixture-sku",
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        phase_a_receipt = root / "phase-a-receipt.json"
        phase_a_receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "phase": "real-human-final-afe-acoustic-qualification",
                    "qualified": True,
                    "shipping_approved": False,
                    "deployment_tag": deployment_tag,
                    "deployment_target": source_sha,
                    "corpus_sha256": corpus_sha,
                    "final_afe_identity_sha256": afe_identity,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        deployment = root / "deployment.json"
        deployment.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "commercial-candidate",
                    "shipping_approved": False,
                    "deployment_tag": deployment_tag,
                    "source_sha": source_sha,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        policy = json.loads(json.dumps(production))
        policy["deployment_tag"] = deployment_tag
        policy_path = root / "policy.json"
        policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")
        dut_summary = root / "dut-summary.json"
        run(
            sys.executable,
            "tools/score_target_dut_qualification.py",
            "--bundle", str(bundle),
            "--policy", str(policy_path),
            "--phase-a-receipt", str(phase_a_receipt),
            "--phase-a-summary", str(phase_a_summary),
            "--deployment-manifest", str(deployment),
            "--model", str(model),
            "--keywords", str(keywords),
            "--output", str(dut_summary),
        )
        dut = json.loads(dut_summary.read_text(encoding="utf-8"))
        assert dut["qualified"] is True
        assert dut["shipping_approved"] is False
        assert dut["metrics"]["soak_hours"] == 24.0
        assert dut["continuity"]["xrun_count"] == 0
        assert dut["final_afe_identity_sha256"] == afe_identity
        assert dut["resource_budget"]["budget_id"] == "fixture-budget-v1"
        assert dut["resource_budget"]["sha256"] == sha(bundle / "resource-budget.json")
        assert dut["next_gate"] == "physical-target-cohort-qualification"
        assert dut["schema_version"] == 2
        assert dut["metrics"]["measurement_contract_id"] == CPU_MEASUREMENT_CONTRACT_ID
        assert dut["metrics"]["audio_seconds"] is None
        budget = json.loads((bundle / "resource-budget.json").read_text())
        required_limits = production["resource_budget"]["required_limit_fields"]
        schema = json.loads((ROOT / "commercial/target-resource-budget.schema.json").read_text())
        assert schema["properties"]["schema_version"]["const"] == 2
        assert schema["properties"]["measurement_contract_id"]["const"] == CPU_MEASUREMENT_CONTRACT_ID
        assert "maximum" not in schema["properties"]["limits"]["properties"]["max_cpu_percent"]
        assert validate_resource_budget(budget, required_limits)["measurement_contract_id"] == CPU_MEASUREMENT_CONTRACT_ID
        for section, key, replacement in ((None, "schema_version", 1),
                (None, "measurement_contract_id", "legacy-online-capacity"),
                ("limits", "max_cpu_percent", float("nan")),
                ("limits", "max_cpu_percent", 0)):
            old_budget = json.loads(json.dumps(budget))
            (old_budget if section is None else old_budget[section])[key] = replacement
            try:
                validate_resource_budget(old_budget, required_limits)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid CPU budget accepted")
        budget["limits"]["max_cpu_percent"] = 150.0
        assert validate_resource_budget(budget, required_limits)["limits"]["max_cpu_percent"] == 150.0


        # Exercise collector -> retained raw/attestation -> DUT scorer, rather
        # than merely comparing two hand-written CPU numbers.
        for name, percent, capacity, threads, exit_code in (
            ("one-core-boundary", 10.0, 1, 1, 0),
            ("two-core-boundary", 10.0, 2, 1, 0),
            ("two-core-overbudget", 20.0, 2, 1, 1),
            ("two-thread-overbudget", 150.0, 2, 2, 1),
        ):
            case_root = root / name
            case_root.mkdir()
            case_bundle = make_bundle(case_root, source_sha=source_sha,
                phase_a_tag=human_tag, deployment_tag=deployment_tag,
                afe_sha=afe_executable, afe_identity=afe_identity,
                cpu_percent=percent, capacity=capacity, threads=threads)
            case_output = case_root / "dut-summary.json"
            run(sys.executable, "tools/score_target_dut_qualification.py",
                "--bundle", str(case_bundle), "--policy", str(policy_path),
                "--phase-a-receipt", str(phase_a_receipt),
                "--phase-a-summary", str(phase_a_summary),
                "--deployment-manifest", str(deployment),
                "--model", str(case_root / "model.kwm"),
                "--keywords", str(case_root / "keywords.kwk"),
                "--output", str(case_output), expect=exit_code)
            measured = json.loads(case_output.read_text())
            assert measured["metrics"]["cpu_percent"] == percent
            assert measured["metrics"]["max_thread_count"] == threads
            assert measured["qualified"] is (exit_code == 0)
            if exit_code:
                assert "cpu-budget" in measured["failures"]


        drift = json.loads((bundle / "target-evidence.json").read_text(encoding="utf-8"))
        drift["audio_frontend_identity_sha256"] = "f" * 64
        drift_path = bundle / "target-evidence.json"
        drift_path.write_text(json.dumps(drift, indent=2) + "\n", encoding="utf-8")
        run(
            sys.executable,
            "tools/score_target_dut_qualification.py",
            "--bundle", str(bundle),
            "--policy", str(policy_path),
            "--phase-a-receipt", str(phase_a_receipt),
            "--phase-a-summary", str(phase_a_summary),
            "--deployment-manifest", str(deployment),
            "--model", str(model),
            "--keywords", str(keywords),
            "--output", str(root / "drift-summary.json"),
            expect=2,
        )
        drift["audio_frontend_identity_sha256"] = afe_identity
        drift_path.write_text(json.dumps(drift, indent=2) + "\n", encoding="utf-8")

        cohort_paths = []
        for index, hours in enumerate((24.0, 24.0, 72.0), 1):
            value = json.loads(json.dumps(dut))
            value["dut_id"] = f"dut-{index}"
            value["metrics"]["soak_hours"] = hours
            value["evidence_sha256"]["target_evidence"] = hashlib.sha256(
                f"evidence-{index}".encode()
            ).hexdigest()
            path = root / f"dut-{index}.json"
            path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            cohort_paths.append(path)
        cohort_out = root / "cohort.json"
        args = [sys.executable, "tools/score_target_cohort.py", "--policy", str(policy_path)]
        for path in cohort_paths:
            args.extend(["--summary", str(path)])
        args.extend(["--output", str(cohort_out)])
        run(*args)
        cohort = json.loads(cohort_out.read_text(encoding="utf-8"))
        assert cohort["qualified"] is True
        assert cohort["dut_count"] == 3
        assert cohort["long_soak_duts"] == 1
        assert cohort["resource_budget_sha256"] == dut["resource_budget"]["sha256"]
        assert cohort["next_gate"] == "shipping-approval-promotion"

        original = cohort_paths[0].read_text(encoding="utf-8")
        for section, key, replacement in (
            (None, "schema_version", 1),
            ("resource_budget", "measurement_contract_id", "legacy-online-capacity"),
            ("metrics", "cpu_percent_semantics", "process_cpu_time / elapsed / online_cpu_capacity * 100"),
            ("metrics", "cpu_percent", 2.5),
        ):
            old = json.loads(original)
            (old if section is None else old[section])[key] = replacement
            cohort_paths[0].write_text(json.dumps(old), encoding="utf-8")
            run(*args, expect=2)
        cohort_paths[0].write_text(original, encoding="utf-8")

        duplicate = json.loads(cohort_paths[2].read_text(encoding="utf-8"))
        duplicate["evidence_sha256"]["target_evidence"] = json.loads(
            cohort_paths[0].read_text(encoding="utf-8")
        )["evidence_sha256"]["target_evidence"]
        cohort_paths[2].write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
        run(*args, expect=2)

    print("test_target_shipping_contract: ok")
    return 0


if __name__ == "__main__":
    DIAGNOSTIC_PATH.unlink(missing_ok=True)
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        tb = exc.__traceback__
        while tb is not None and tb.tb_next is not None:
            tb = tb.tb_next
        line = tb.tb_lineno if tb is not None else 1
        if not DIAGNOSTIC_PATH.exists():
            write_diagnostic(
                line=line,
                command=None,
                expected_rc=None,
                actual_rc=None,
                stdout="",
            )
        message = str(exc) or "assertion failed"
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(
            f"::error file=tests/test_target_shipping_contract.py,line={line},title=target shipping contract::{message}"
        )
        raise