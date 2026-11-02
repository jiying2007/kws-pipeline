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
from test_board_bench import board_summary_fixture, write_pcm_fixture
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
    if args and args[0] in ("python3", sys.executable):
        args = (sys.executable, "-S", "-B", *args[1:])
    completed = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
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
    write_pcm_fixture(audio, 16000)

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

    board_summary = board_summary_fixture(audio)
    board_summary.update(runner_sha256=sha(runner), model_sha256=sha(model),
                         keyword_pack_sha256=sha(keywords), runtime_source_revision=source_sha,
                         model_bytes=model.stat().st_size, keyword_pack_bytes=keywords.stat().st_size)
    summary_path = bundle / "board-summary.json"
    summary_path.write_text(json.dumps(board_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raw_summary_path = raw / "board-summary.json"
    raw_summary_path.write_bytes(summary_path.read_bytes())

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
    raw_paths = [soak, power, continuity, raw_summary_path]
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
        "--raw-evidence", str(raw_summary_path),
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



def verify_end_to_end_contract(approval_workflow: str) -> None:
    import ast
    import math
    from qualification_latency_contract import (
        normalize_end_to_end, raw_sample_latency_diagnostic,
        require_measured_end_to_end, require_receipt_projection,
        unmeasured_end_to_end,
    )

    def require(value, why):
        if not value:
            raise AssertionError(why)

    not_run = unmeasured_end_to_end()
    for bad in (None, {}, True, [], {"qualified": True},
                dict(not_run, qualified=True), dict(not_run, limit_ms=True),
                dict(not_run, status="PASS"), dict(not_run, contract="other"),
                dict(not_run, metrics={"max_ms": 1}),
                dict(not_run, clock_epoch="unbound")):
        try:
            normalize_end_to_end(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported callback declaration accepted")

    timing = {"total_recordings": 2, "afe_contract_recordings": 2,
              "raw_metrics_available": True, "matched_events": 2}
    score = {"timing_coordinates": timing,
             "p95_raw_capture_signed_end_offset_ms": 500.0,
             "p95_post_afe_signed_end_offset_ms": 200.0}
    for raw, wanted in ((500.0, "PASS"), (500.001, "FAIL"), (600.0, "FAIL")):
        result = raw_sample_latency_diagnostic(dict(score, p95_raw_capture_signed_end_offset_ms=raw))
        require(result["status"] == wanted and result["product_qualification"] is False,
                "sample-coordinate diagnostic boundary")
    for bad_timing in ({}, dict(timing, afe_contract_recordings=1),
                       dict(timing, matched_events=0), dict(timing, matched_events=True),
                       dict(timing, total_recordings=True)):
        result = raw_sample_latency_diagnostic(dict(score, timing_coordinates=bad_timing))
        require(result["status"] == "NOT_EVALUABLE" and
                result["p95_raw_signed_end_offset_ms"] is None, "missing timing fabricated a metric")
    for bad in (True, None, float("nan"), float("inf")):
        result = raw_sample_latency_diagnostic(dict(score, p95_raw_capture_signed_end_offset_ms=bad))
        require(result["status"] == "NOT_EVALUABLE", "invalid metric accepted")
    require(require_receipt_projection({}, {}, "f" * 64) == not_run,
            "legacy phase readability changed")
    summary = {"end_to_end_latency": not_run}
    receipt = {"end_to_end_latency": not_run, "qualification_summary_sha256": "f" * 64}
    require(require_receipt_projection(summary, receipt, "f" * 64) == not_run,
            "new unmeasured projection")
    for broken in ({}, dict(receipt, qualification_summary_sha256="0" * 64),
                   dict(receipt, end_to_end_latency=None),
                   dict(receipt, end_to_end_latency=dict(not_run, status="PASS"))):
        try:
            require_receipt_projection(summary, broken, "f" * 64)
        except ValueError:
            pass
        else:
            raise AssertionError("receipt identity mismatch accepted")

    # Parse the actual trusted workflow inline Python, not a duplicate guard.
    step = approval_workflow.split("- name: Assemble terminal shipping approval manifest", 1)[1]
    inline = textwrap.dedent(step.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0])
    tree = ast.parse(inline)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "require_measured_end_to_end"]
    require(len(calls) == 1, "shipping assembly must use exactly one reviewed final gate")
    call = calls[0]
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "manifest"
                           for target in node.targets)]
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr in ("write_text", "write_bytes")]
    require(len(assignments) == 1 and call.lineno < assignments[0].lineno and
            all(call.lineno < node.lineno for node in writes), "final gate is after output")
    guard_nodes = [node for node in tree.body
                   if (isinstance(node, ast.Assign) and
                       any(isinstance(target, ast.Name) and target.id in ("human_e2e", "cohort_e2e")
                           for target in node.targets))
                   or (isinstance(node, ast.Expr) and node.value is call)]
    require(len(guard_nodes) == 3, "unexpected final shipping guard shape")
    guard = compile(ast.Module(body=guard_nodes, type_ignores=[]),
                    "shipping-approval-actual-latency-guard", "exec")
    for payload in ({}, {"end_to_end_latency": not_run},
                    {"end_to_end_latency": dict(not_run, status="PASS", qualified=True)},
                    {"end_to_end_latency": dict(not_run, metrics={"callback_max_ms": 1,
                                                                  "clock_epoch": "unbound"})}):
        projected = dict(payload, qualification_summary_sha256="f" * 64)
        namespace = {"human": payload, "human_receipt": projected,
                     "cohort": payload, "cohort_receipt": projected,
                     "root": pathlib.Path("unused-synthetic-root"),
                     "sha": lambda unused: "f" * 64,
                     "require_receipt_projection": require_receipt_projection,
                     "require_measured_end_to_end": require_measured_end_to_end}
        try:
            exec(guard, namespace)
        except ValueError as exc:
            require("end-to-end" in str(exc), "wrong failure precondition")
        else:
            raise AssertionError("legacy/sample/forged evidence reached product shipping")

    publish_step = approval_workflow.split("- name: Publish immutable shipping approval", 1)[1]
    publish_inline = textwrap.dedent(publish_step.split("python3 - <<'PY'\n", 1)[1].split("\n          PY", 1)[0])
    publish_tree = ast.parse(publish_inline)
    publish_calls = [node for node in publish_tree.body
                     if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == "require_measured_end_to_end"]
    require(len(publish_calls) == 1 and
            publish_step.index("require_measured_end_to_end(*values)") < publish_step.index("gh release create"),
            "publication can bypass the final callback guard")
    imports = [node for node in publish_tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "qualification_latency_contract"]
    require(len(imports) == 1 and
            {"require_receipt_projection", "require_measured_end_to_end"} <= {item.name for item in imports[0].names},
            "publication does not use the same reviewed helper")
    with tempfile.TemporaryDirectory(prefix="kws-shipping-latency-") as td:
        temp = pathlib.Path(td)
        (temp / "tools").mkdir()
        (temp / "tools/qualification_latency_contract.py").write_bytes(
            (ROOT / "tools/qualification_latency_contract.py").read_bytes())
        approval = temp / "build/approval"
        approval.mkdir(parents=True)
        sentinel = approval / "shipping-approval-manifest.json"
        sentinel.write_bytes(b"retain-existing-evidence\n")
        for payload in ({}, {"end_to_end_latency": not_run},
                        {"end_to_end_latency": dict(not_run, status="PASS", qualified=True)}):
            for folder, receipt_name, summary_name in (
                ("phase-a", "public-phase-a-receipt.json", "real-human-qualification-summary.json"),
                ("phase-b", "public-target-cohort-receipt.json", "target-cohort-qualification-summary.json"),
            ):
                target = approval / folder
                target.mkdir(exist_ok=True)
                body = json.dumps(payload).encode("utf-8")
                (target / summary_name).write_bytes(body)
                receipt = dict(payload, qualification_summary_sha256=hashlib.sha256(body).hexdigest())
                (target / receipt_name).write_text(json.dumps(receipt), encoding="utf-8")
            rejected = subprocess.run([sys.executable, "-S", "-B", "-c", publish_inline],
                cwd=temp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False, timeout=30)
            require(rejected.returncode != 0 and "end-to-end" in rejected.stdout,
                    "actual publication Python accepted unmeasured/forged callback evidence")
            require(sentinel.read_bytes() == b"retain-existing-evidence\n",
                    "rejected publication changed existing evidence")

    print("end-to-end contract: actual shipping guard rejects unmeasured/forged evidence")


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
    verify_end_to_end_contract(approval_workflow)
    run("/bin/bash", "-n", "governance/require_current_main.sh")
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
        # A plausible, internally consistent faster summary is still invalid if
        # it is not the exact raw result bound by the external attestation.
        summary_path = bundle / "board-summary.json"
        original_board = summary_path.read_bytes()
        forged_board = json.loads(original_board)
        for field in ("total_process_us", "mean_process_us", "p50_process_us",
                      "p95_process_us", "p99_process_us", "max_process_us", "rtf"):
            forged_board[field] /= 2.0
        forged_board["p99_headroom"] *= 2.0
        summary_path.write_text(json.dumps(forged_board) + "\n", encoding="utf-8")
        rejected = run(sys.executable, "tools/score_target_dut_qualification.py",
            "--bundle", str(bundle), "--policy", str(policy_path),
            "--phase-a-receipt", str(phase_a_receipt),
            "--phase-a-summary", str(phase_a_summary),
            "--deployment-manifest", str(deployment),
            "--model", str(model), "--keywords", str(keywords),
            "--output", str(root / "forged-board.json"), expect=2)
        if "board summary must be retained" not in rejected.stdout:
            raise AssertionError("DUT scorer did not enforce board-summary attestation")
        summary_path.write_bytes(original_board)
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

        from qualification_latency_contract import unmeasured_end_to_end
        if dut.get("end_to_end_latency") != unmeasured_end_to_end():
            raise AssertionError("legacy Phase A was not propagated as NOT_RUN")
        phase_snap = (phase_a_summary.read_bytes(), phase_a_receipt.read_bytes())
        try:
            phase_obj = json.loads(phase_snap[0])
            phase_obj["end_to_end_latency"] = unmeasured_end_to_end()
            phase_a_summary.write_text(json.dumps(phase_obj), encoding="utf-8")
            receipt_obj = json.loads(phase_snap[1])
            receipt_obj["end_to_end_latency"] = unmeasured_end_to_end()
            receipt_obj["qualification_summary_sha256"] = sha(phase_a_summary)
            phase_a_receipt.write_text(json.dumps(receipt_obj), encoding="utf-8")
            latency_args = (sys.executable, "tools/score_target_dut_qualification.py",
                "--bundle", str(bundle), "--policy", str(policy_path),
                "--phase-a-receipt", str(phase_a_receipt), "--phase-a-summary", str(phase_a_summary),
                "--deployment-manifest", str(deployment), "--model", str(model),
                "--keywords", str(keywords), "--output", str(root / "latency-dut.json"))
            run(*latency_args)
            propagated = json.loads((root / "latency-dut.json").read_text())
            if propagated.get("qualified") is not True or propagated.get("end_to_end_latency") != unmeasured_end_to_end():
                raise AssertionError("explicit NOT_RUN roundtrip altered phase scope")
            for changed in (
                dict(receipt_obj, qualification_summary_sha256="0" * 64),
                dict(receipt_obj, end_to_end_latency=None),
                dict(receipt_obj, end_to_end_latency=dict(unmeasured_end_to_end(), status="PASS", qualified=True)),
                {key: value for key, value in receipt_obj.items() if key != "end_to_end_latency"},
            ):
                phase_a_receipt.write_text(json.dumps(changed), encoding="utf-8")
                rejected = run(*latency_args, expect=2)
                if "end-to-end" not in rejected.stdout:
                    raise AssertionError("latency receipt case failed for unrelated reason")
        finally:
            phase_a_summary.write_bytes(phase_snap[0])
            phase_a_receipt.write_bytes(phase_snap[1])

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


        if cohort.get("end_to_end_latency") != unmeasured_end_to_end():
            raise AssertionError("cohort did not retain unmeasured callback boundary")
        latency_original = cohort_paths[0].read_bytes()
        try:
            for replacement in (None, True, {}, dict(unmeasured_end_to_end(), status="PASS"),
                                dict(unmeasured_end_to_end(), contract="unbound-clock")):
                changed = json.loads(latency_original)
                changed["end_to_end_latency"] = replacement
                cohort_paths[0].write_text(json.dumps(changed), encoding="utf-8")
                rejected = run(*args, expect=2)
                if "end-to-end" not in rejected.stdout:
                    raise AssertionError("DUT latency tamper missed precise consumer gate")
        finally:
            cohort_paths[0].write_bytes(latency_original)

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
