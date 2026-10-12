"""Offline host/workflow integration: synthetic receipts, never setup/inference.

Release fixtures live only inside individual tests. The actual source release
may be disabled or activated. Real CLI probes use invalid execution context and
stop before any checkout, network, runtime creation or model operation.
Fixtures reuse the stdlib semantic suite's WAV/claim builder, not model output.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import contract
import tts_common as common
import tts_gate as gate
import tts_host as host
import tts_public_schema as schema
import tts_worker as worker
from test_tts_semantics import archive

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HELPERS = HERE.parent / "qwen6_tts"
IDENTITY = {"repository": gate.REPOSITORY, "executed_head_sha": "a" * 40,
            "run_id": 123, "run_attempt": 1, "job_key": "k1-tts"}
SECRET = "HOST-PRIVATE-SENTINEL-DO-NOT-DISCLOSE"
HOST_SECRETS = {key: SECRET for key in (
    "GITHUB_TOKEN", "GH_TOKEN", "KWS_DATA_TOKEN", "KWS_EVIDENCE_TOKEN",
    "ACTIONS_RUNTIME_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_TEST_TOKEN",
    "HF_TOKEN", "AWS_SECRET_ACCESS_KEY", "PRIVATE_PASSWORD", "SSH_AUTH_SOCK",
    "PYTHONPATH", "LD_PRELOAD")}


def private_setup(verified=False, error=None):
    """Fabricated metadata only; pins are read from retained local JSON locks."""
    packages, sources, assets, locks = schema.inventory()
    result = {"schema": "fixed30-setup-receipt-v1", "python": dict(schema.PYTHON),
        "locks": locks, "initial_free_bytes": 12 * schema.GIB,
        "network_configuration": "offline_configured_not_kernel_isolated",
        "commands": [], "wheels": {}, "disk_checkpoints": [],
        "python_executable": "/private/" + SECRET, "downloads": [{"private_url": SECRET}],
        "builds": [{"private_path": SECRET}], "started_unix_seconds": 1000}
    if error is not None:
        result["exception"] = {"type": error, "message": SECRET}
        result["phase"] = {"stage": "build_wheel", "package": "sox"}
        result["commands"] = [{"argv": [SECRET], "log": SECRET, "stage": "build_wheel",
            "package": "sox", "command_role": "build", "status": "failed", "returncode": 1}]
    if verified:
        for name, pin in packages.items():
            result["wheels"][name] = {"path": "/private/" + SECRET,
                "name": pin["name"], "version": pin["version"], "bytes": pin["size_bytes"],
                "sha256": pin["sha256"], "metadata_sha256": pin["metadata_sha256"] or "d" * 64,
                "metadata_bytes": pin.get("metadata_bytes") or 123, "requires_dist": [SECRET]}
        disk = {"installed_bytes": 0, "buildtmp_bytes": 0, "workspace_bytes": 0,
                "free_bytes": 12 * schema.GIB}
        result["verification"] = {"installed": {"private": SECRET},
            "source_members": [dict(row, bytes=1) for row in sources],
            "model_assets": {"qwen_tts": assets}, "disk": disk, "python": dict(schema.PYTHON)}
        result["bootstrap_pip"] = {"source": "exact_python_bundled_ensurepip_not_independent_lock",
            "filename": "pip-25.0.1-py3-none-any.whl", "version": "25.0.1", "bytes": 1,
            "sha256": "d" * 64, "metadata_sha256": "e" * 64}
        result["commands"] = [{"argv": [SECRET], "log": SECRET, "stage": "verify_setup",
            "package": None, "command_role": "verification", "status": "success", "returncode": 0}]
        result["import_preflight"] = {"schema": "native-import-preflight-v1", "profile": "tts",
            "stage": "complete", "status": "success", "weight_loads": 0,
            "model_constructions": 0, "forward_calls": 0, "raw_text_disclosed": False,
            "versions": {"numpy": packages["numpy"]["version"]}, "native_io": {
                "frames": 12, "float32_values_sha256": "a" * 64, "float_wav_sha256": "b" * 64,
                "pcm_wav_sha256": "c" * 64, "pcm16_values": [-32768, -32768, -16384, -2, -1, 0, 0, 1, 16384, 32767, 32767, 32767],
                "bundled_native_module": "_soundfile_data/libsndfile_x86_64.so",
                "bundled_native_sha256": "d" * 64, "libsndfile_version": "1.2.2",
                "empty_nonfinite_rejection_checked": True}}
    return result


def private_load(runtime):
    _, _, assets, _ = schema.inventory()
    return {"sidecars": [{"path": row["filename"], "sha256": row["sha256"]}
                         for row in assets if row["filename"].endswith(".json")],
            "loaders": [{"loader": "AutoModel", "path": str(runtime / "models/qwen_tts"),
                "safety_kwargs": {"local_files_only": "True", "trust_remote_code": "False",
                    "force_download": "False", "use_safetensors": "True", "weights_only": "True",
                    "attn_implementation": "eager", "dtype": "torch.float32", "device_map": "cpu"}}],
            "network_events": []}


def supervisor_receipt(options, returncode=0):
    return {"returncode": returncode, "process_returncode": returncode,
        "filesystem_sampling": {"schema": "qwen6-filesystem-sampling-v1",
            "semantics": "non_atomic_observed_bytes_not_peak_or_quota",
            "maximum_scans_per_check_scope": 2, "scopes": {}},
        "filesystem_error": None,
        "termination_reason": "normal_exit" if returncode == 0 else "process_failure_or_kernel_limit",
        "wall_seconds": 0.01, "reaped_children_cpu_seconds": 0.0, "sample_count": 0,
        "sample_interval_seconds": 0.25,
        "max_sampled": dict.fromkeys(("rss_bytes", "live_cpu_seconds", "workspace_bytes", "installed_bytes", "buildtmp_bytes")),
        "cpu_limit_seconds_per_process": options["cpu_seconds"],
        "wall_limit_seconds": options["wall_seconds"], "affinity_core_limit": options["cores"],
        "rss_mode": "sampled_process_group_threshold_not_continuous_hard_limit",
        "rss_threshold_bytes": options["rss_bytes"],
        "cpu_mode": "kernel_RLIMIT_CPU_per_process_not_aggregate_tree_limit", "setup": options["setup"]}


class HostTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="k1-host-offline-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "k1-punctuation6-tts-v1"
        self.output = self.runtime / "publication-members"
        self.event_path = self.root / "event.json"
        common.write_once(self.event_path, {"created": True, "deleted": False,
            "ref": "refs/heads/" + gate.SOURCE_BRANCH, "after": IDENTITY["executed_head_sha"],
            "repository": {"full_name": gate.REPOSITORY, "private": False}})
        self.env = {"GITHUB_REPOSITORY": gate.REPOSITORY, "GITHUB_EVENT_NAME": "push",
            "GITHUB_REF": "refs/heads/" + gate.SOURCE_BRANCH, "GITHUB_JOB": "k1-tts",
            "GITHUB_RUN_ATTEMPT": "1", "GITHUB_RUN_ID": "123", "GITHUB_SHA": "a" * 40,
            "RUNNER_ENVIRONMENT": "github-hosted", "RUNNER_OS": "Linux", "RUNNER_ARCH": "X64",
            "GITHUB_EVENT_PATH": str(self.event_path), "RUNNER_TEMP": str(self.root),
            "K1_JOB_STARTED_UNIX_SECONDS": "999", **HOST_SECRETS}
        plan = HERE / "plan.proposed.json"
        self.source_freeze = {"schema": "k1-punctuation6-tts-source-freeze-v1",
            "files": {"research/k1-punctuation6-v1/plan.proposed.json":
                      {"bytes": plan.stat().st_size, "sha256": common.sha(plan)}}}
        self.source = {"source_freeze_sha256": hashlib.sha256(contract.canonical(self.source_freeze)).hexdigest(),
                       "frozen_files": len(self.source_freeze["files"])}
        self.calls = []
        self.setup_data = private_setup(verified=True)
        self.load_data = private_load(self.runtime)
        self.failure = None
        self.raise_in = None
        self.generation_mode = "complete"
        self.corrupt_completed_audio = False
        self.emitted = {}

    def fake_supervise(self, command, **options):
        name = "setup" if options["setup"] else "generation"
        self.calls.append((name, list(command), deepcopy(options)))
        common.assert_no_host_secrets(options["env"])
        self.assertEqual(options["cwd"], ROOT)
        self.assertEqual(options["workspace"], self.runtime)
        self.assertEqual(options["installed"], self.runtime / "venv")
        self.assertEqual(options["buildtmp"], self.runtime / "buildtmp")
        self.assertTrue((self.output / "claim.json").is_file())
        self.assertTrue((self.output / "source.json").is_file())
        self.assertEqual(gate.read_json(self.output / "resources.json")["status"], "started")
        options["log_path"].write_text(SECRET, encoding="utf-8")
        options["stderr_path"].write_text("RuntimeError: " + SECRET + "\n", encoding="utf-8")
        if name == "setup":
            common.write_once(self.runtime / "setup-receipt.json", self.setup_data)
        else:
            common.write_once(self.runtime / "model-load.private.json", self.load_data)
            if self.generation_mode in ("complete", "failed"):
                members = archive() if self.generation_mode == "complete" else archive(completed=1, failed=True)
                for path, raw in members.items():
                    if path in ("claim.json", "source.json", "resources.json"):
                        continue
                    target = self.output / path
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
                    self.emitted[path] = raw
            elif self.generation_mode == "interrupted":
                cell = contract.cells()[0]
                (self.output / "claims").mkdir()
                common.write_once(self.output / "outcomes.json", worker.new_outcomes())
                claim_path = "claims/" + cell["cell_id"] + ".started.json"
                raw = archive()[claim_path]
                (self.output / claim_path).write_bytes(raw)
                self.emitted[claim_path] = raw
            if self.corrupt_completed_audio:
                (self.output / "raw/k1p6-001.wav").write_bytes(b"invalid synthetic complete WAV")
        if self.raise_in == name:
            raise type("PrivateFailure" + SECRET, (RuntimeError,), {})(SECRET)
        return supervisor_receipt(options, 1 if self.failure == name else 0)

    def harness(self, *, released=True):
        stack = ExitStack()
        self.addCleanup(stack.close)
        fake = types.ModuleType("supervision")
        fake.supervise = mock.Mock(side_effect=self.fake_supervise)
        self.supervise = fake.supervise
        stack.enter_context(mock.patch.dict(sys.modules, {"supervision": fake}))
        stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
        stack.enter_context(mock.patch.dict(os.environ, self.env, clear=True))
        # These guards fail closed if a future host bypasses the fake boundary.
        stack.enter_context(mock.patch("subprocess.Popen", side_effect=AssertionError("no real child processes")))
        stack.enter_context(mock.patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("no real HTTP")))
        for name, value in (("python_implementation", "CPython"), ("python_version", "3.12.14"),
                            ("system", "Linux"), ("machine", "x86_64")):
            stack.enter_context(mock.patch.object(host.platform, name, return_value=value))
        stack.enter_context(mock.patch.object(host.sys, "byteorder", "little"))
        self.wall = stack.enter_context(mock.patch.object(host.time, "time", return_value=1000.0))
        self.monotonic = stack.enter_context(mock.patch.object(host.time, "monotonic", return_value=100.0))
        if released:
            stack.enter_context(mock.patch.object(host, "verify_release", return_value=({}, self.source)))
            stack.enter_context(mock.patch.object(host, "read_json", side_effect=lambda path, **kwargs:
                deepcopy(self.source_freeze) if Path(path) == HERE / "tts-source-freeze.json"
                else gate.read_json(path, **kwargs)))
        else:
            disabled_root = self.root / "disabled-source"
            folder = disabled_root / "research/k1-punctuation6-v1"
            folder.mkdir(parents=True)
            release = gate.read_json(HERE / "tts-release.json")
            release["enabled"] = False
            common.write_once(folder / "tts-release.json", release)
            stack.enter_context(mock.patch.object(host, "verify_release",
                side_effect=lambda: gate.verify_release(disabled_root)))
        self.checkout = stack.enter_context(mock.patch.object(host, "verify_clean_checkout",
            return_value={"tree": "c" * 40, "commit": "a" * 40}))
        self.get = mock.Mock(side_effect=AssertionError("only mocked preflight permitted"))
        self.reader = stack.enter_context(mock.patch.object(host, "public_reader", return_value=(
            self.get, {"body_bytes": 321, "requests": 5, "retries": 0})))
        self.preflight = stack.enter_context(mock.patch.object(host, "read_only_preflight", return_value={
            "preflight_verified": True, "numeric_job_id": 124, "source_branch_workflow_runs": 1,
            "repository_public": True, "result_ref_absent": True, "read_only_requests": 5}))
        return stack

    def records(self):
        source = gate.read_json(self.output / "source.json")
        resources = gate.read_json(self.output / "resources.json")
        freeze = dict(self.source_freeze, source_freeze_sha256=self.source["source_freeze_sha256"])
        self.assertTrue(schema.validate_source(source, IDENTITY, freeze,
            gate.read_json(HERE / "model-provenance.json"), common.sha(HERE / "plan.proposed.json")))
        self.assertTrue(schema.validate_resources(resources, IDENTITY))
        public = b"".join(path.read_bytes() for path in self.output.rglob("*.json"))
        self.assertNotIn(SECRET.encode(), public)
        self.assertNotIn(str(self.runtime).encode(), public)
        return source, resources, gate.read_json(self.output / "outcomes.json")

    def test_real_disabled_gate_precedes_every_runtime_side_effect(self):
        with self.harness(released=False), mock.patch.object(Path, "mkdir") as mkdir:
            with self.assertRaisesRegex(ValueError, "new TTS execution is disabled"):
                host.execute({})
            mkdir.assert_not_called()
        self.checkout.assert_not_called()
        self.reader.assert_not_called()
        self.preflight.assert_not_called()
        self.supervise.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_invalid_context_rejected_before_checkout_network_or_runtime(self):
        with self.harness():
            self.env["GITHUB_RUN_ATTEMPT"] = "2"
            with self.assertRaisesRegex(ValueError, "fixed standard hosted context"):
                host.execute(self.env)
        self.checkout.assert_not_called()
        self.reader.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_actual_activated_release_rejects_nonexecution_before_all_side_effects(self):
        if gate.read_json(HERE / "tts-release.json")["enabled"] is not True:
            self.skipTest("this source snapshot is deliberately disabled")
        with self.harness(), mock.patch.object(host, "verify_release", side_effect=gate.verify_release):
            self.env["GITHUB_REF"] = "refs/heads/codex/nonexecution-review"
            with self.assertRaisesRegex(ValueError, "fixed standard hosted context"):
                host.execute(self.env)
        self.checkout.assert_not_called()
        self.reader.assert_not_called()
        self.preflight.assert_not_called()
        self.supervise.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_dirty_checkout_and_wrong_platform_stop_before_public_preflight(self):
        for checkpoint in ("checkout", "platform"):
            with self.subTest(checkpoint=checkpoint), self.harness():
                if checkpoint == "checkout":
                    self.checkout.side_effect = ValueError("dirty checkout")
                else:
                    host.platform.python_version.return_value = "3.12.15"
                with self.assertRaises(ValueError):
                    host.execute(self.env)
                self.reader.assert_not_called()
                self.supervise.assert_not_called()
                self.assertFalse(self.runtime.exists())

    def test_provisioning_limit_rejects_expired_future_and_nonfinite_start(self):
        for start in ("700", "699", "1001", "nan", "inf", "-inf"):
            with self.subTest(start=start), self.harness():
                self.env["K1_JOB_STARTED_UNIX_SECONDS"] = start
                with self.assertRaisesRegex(ValueError, "remaining provisioning budget"):
                    host.execute(self.env)
                self.reader.assert_not_called()
                self.supervise.assert_not_called()
                self.assertFalse(self.runtime.exists())

    def test_preflight_deadline_uses_smaller_remaining_provisioning_budget(self):
        for start, deadline in (("999", 220.0), ("800", 200.0), ("700.25", 100.25)):
            with self.subTest(start=start), self.harness():
                self.env["K1_JOB_STARTED_UNIX_SECONDS"] = start
                self.preflight.side_effect = RuntimeError("synthetic preflight stop")
                with self.assertRaisesRegex(RuntimeError, "synthetic preflight stop"):
                    host.execute(self.env)
                self.reader.assert_called_once_with(IDENTITY, deadline)
                self.preflight.assert_called_once_with(IDENTITY, self.get)
                self.supervise.assert_not_called()
                self.assertFalse(self.runtime.exists())

    def test_preflight_cannot_spend_past_provisioning_limit_then_create_runtime(self):
        with self.harness():
            self.env["K1_JOB_STARTED_UNIX_SECONDS"] = "800"
            self.wall.side_effect = [1000.0, 1000.0, 1100.01]
            with self.assertRaisesRegex(ValueError, "provisioning budget before asset acquisition"):
                host.execute(self.env)
        self.preflight.assert_called_once()
        self.supervise.assert_not_called()
        self.assertFalse(self.runtime.exists())

    def test_complete_host_uses_isolated_secret_free_two_stage_commands(self):
        with self.harness():
            self.assertEqual(host.execute(), self.runtime)
        self.assertEqual([name for name, _, _ in self.calls], ["setup", "generation"])
        expected_keys = {"PATH", "HOME", "LANG", "LC_ALL", "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE", "ORT_DISABLE_TELEMETRY", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "TOKENIZERS_PARALLELISM",
            "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_XET"}
        for index, (name, command, options) in enumerate(self.calls):
            interpreter = Path(sys.executable) if index == 0 else self.runtime / "venv/bin/python"
            script = HELPERS / "setup_locked.py" if index == 0 else HERE / "tts_worker.py"
            args = ["--root", self.runtime] if index == 0 else ["--runtime", self.runtime, "--output", self.output]
            self.assertEqual(command, common.isolated_command(interpreter, script, args))
            self.assertEqual(command[1:4], ["-I", "-B", "-c"])
            self.assertEqual(set(options["env"]), expected_keys)
            self.assertFalse(set(options["env"]) & set(HOST_SECRETS))
            self.assertNotIn(SECRET, json.dumps(options["env"]))
            self.assertEqual(options["env"]["HOME"], str(self.runtime / "home"))
            self.assertEqual(options["env"]["PATH"], str(interpreter.parent) + ":/usr/bin:/bin")
            self.assertEqual(options["wall_seconds"], 1200)
            self.assertEqual(options["job_deadline"], 2500)
            self.assertEqual(options["cpu_seconds"], 2400 if index == 0 else 3600)
            self.assertEqual(options["rss_bytes"], (4 if index == 0 else 7) * schema.GIB)
            self.assertEqual(options["cores"], 4)
        source, resources, outcomes = self.records()
        self.assertEqual(source["runtime_setup"]["status"], "verified")
        self.assertEqual(source["runtime_model_load"]["status"], "loaded_validated")
        self.assertEqual(resources["status"], "completed_six_once")
        self.assertEqual(outcomes["status"], "completed_six_once")
        self.assertEqual(outcomes["consumed_attempts"], 6)
        for path, raw in self.emitted.items():
            self.assertEqual((self.output / path).read_bytes(), raw)
        self.assertEqual(set(self.output.iterdir()), {self.output / name for name in
            ("claim.json", "source.json", "resources.json", "outcomes.json", "eos.json", "claims", "raw", "derived")})
        for receipt in resources["stages"].values():
            self.assertEqual(receipt["stderr_diagnostic"]["exception_types"], ["RuntimeError"])
            self.assertIs(receipt["stderr_diagnostic"]["raw_text_disclosed"], False)

    def test_shared_stage_deadline_reduces_generation_allowance(self):
        with self.harness():
            self.monotonic.side_effect = [100.0, 100.0, 100.0, 1500.0, 1501.0]
            host.execute(self.env)
        self.assertEqual([options["wall_seconds"] for _, _, options in self.calls], [1200, 1000])
        self.assertEqual(self.records()[1]["wall_seconds"], 1401)

    def test_setup_failure_projects_error_and_never_starts_generation(self):
        self.failure = "setup"
        self.setup_data = private_setup(error="RuntimeError")
        with self.harness(), self.assertRaisesRegex(ValueError, "supervised stage failed; no retry"):
            host.execute(self.env)
        self.assertEqual([row[0] for row in self.calls], ["setup"])
        source, resources, outcomes = self.records()
        self.assertEqual(source["runtime_setup"]["status"], "error")
        self.assertEqual(source["runtime_setup"]["exception_type"], "RuntimeError")
        self.assertEqual(source["runtime_model_load"], {"status": "not_available"})
        self.assertEqual(resources["status"], "stopped_no_retry")
        self.assertEqual(resources["error_class"], "ValueError")
        self.assertEqual(outcomes["consumed_attempts"], 0)
        self.assertEqual([row["status"] for row in outcomes["cells"]], ["not_run"] * 6)
        self.assertEqual(gate.read_json(self.output / "eos.json")["rows"], [])

    def test_generation_failure_retains_loaded_evidence_and_six_cell_denominator(self):
        self.failure, self.generation_mode = "generation", "failed"
        with self.harness(), self.assertRaisesRegex(ValueError, "supervised stage failed; no retry"):
            host.execute(self.env)
        source, resources, outcomes = self.records()
        self.assertEqual(source["runtime_setup"]["status"], "verified")
        self.assertEqual(source["runtime_model_load"]["status"], "loaded_validated")
        self.assertEqual(resources["status"], "stopped_no_retry")
        self.assertEqual(set(resources["stages"]), {"setup", "generation"})
        self.assertEqual(outcomes["consumed_attempts"], 2)
        self.assertEqual([row["status"] for row in outcomes["cells"]],
                         ["complete", "failed_no_retry"] + ["not_run"] * 4)
        for path, raw in self.emitted.items():
            self.assertEqual((self.output / path).read_bytes(), raw)

    def test_zero_stage_exit_cannot_replace_complete_six_outcome_evidence(self):
        self.generation_mode = "failed"
        with self.harness(), self.assertRaisesRegex(ValueError, "all six outcomes required"):
            host.execute(self.env)
        _, resources, outcomes = self.records()
        self.assertEqual(resources["status"], "stopped_no_retry")
        self.assertEqual([row["returncode"] for row in resources["stages"].values()], [0, 0])
        self.assertEqual(outcomes["consumed_attempts"], 2)
        self.assertEqual(outcomes["status"], "stopped_no_retry")
        self.assertEqual(len(self.calls), 2)

    def test_supervisor_exception_is_safe_and_finally_still_projects_receipt(self):
        self.raise_in = "setup"
        self.setup_data = private_setup(error="RuntimeError")
        with self.harness(), self.assertRaises(RuntimeError):
            host.execute(self.env)
        source, resources, outcomes = self.records()
        self.assertEqual(resources["error_class"], "OtherException")
        self.assertEqual(resources["stages"], {})
        self.assertEqual(source["runtime_setup"]["status"], "error")
        self.assertEqual(outcomes["consumed_attempts"], 0)

    def test_secret_bearing_projection_extras_fail_closed_without_raw_metadata(self):
        self.setup_data["unreviewed_private_history"] = SECRET
        self.load_data["loaders"][0]["safety_kwargs"]["private_token"] = SECRET
        with self.harness():
            host.execute(self.env)
        source, resources, _ = self.records()
        for key in ("runtime_setup", "runtime_model_load"):
            self.assertEqual(source[key], {"status": "projection_failed", "error_class": "ValueError"})
        self.assertEqual(resources["evidence_projection_error_class"], "ValueError")
        self.assertIs(resources["complete_provenance_available"], False)

    def test_failure_after_durable_claim_reconciles_consumption_without_retry(self):
        self.failure, self.generation_mode = "generation", "interrupted"
        with self.harness(), self.assertRaisesRegex(ValueError, "supervised stage failed; no retry"):
            host.execute(self.env)
        _, resources, outcomes = self.records()
        self.assertEqual(resources["status"], "stopped_no_retry")
        self.assertEqual(outcomes["consumed_attempts"], 1)
        self.assertEqual([row["status"] for row in outcomes["cells"]], ["failed_no_retry"] + ["not_run"] * 5)
        self.assertEqual(outcomes["finalization"]["reconciled_cells"],
                         [{"cell_id": "k1p6-001", "previous_status": "not_run"}])
        self.assertIs(outcomes["finalization"]["calls_resumed"], False)
        self.assertEqual(len(self.calls), 2)
        for path, raw in self.emitted.items():
            self.assertEqual((self.output / path).read_bytes(), raw)

    def test_existing_runtime_cannot_be_reused_or_start_any_stage(self):
        self.runtime.mkdir()
        marker = self.runtime / "existing-private-record"
        marker.write_bytes(b"keep original")
        with self.harness(), self.assertRaises(FileExistsError):
            host.execute(self.env)
        self.supervise.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"keep original")
        self.assertEqual(list(self.runtime.iterdir()), [marker])

    def test_finalizer_rejection_cannot_leave_successful_resource_status(self):
        self.corrupt_completed_audio = True
        with self.harness(), self.assertRaisesRegex(ValueError, "complete cell audio cannot be quarantined"):
            host.execute(self.env)
        _, resources, outcomes = self.records()
        self.assertEqual(resources["status"], "stopped_no_retry")
        self.assertEqual(resources["error_class"], "ValueError")
        self.assertEqual([row["returncode"] for row in resources["stages"].values()], [0, 0])
        self.assertEqual(outcomes["status"], "completed_six_once")
        self.assertEqual((self.output / "outcomes.json").read_bytes(), self.emitted["outcomes.json"])
        self.assertEqual(len(self.calls), 2)

    def test_finalizer_rejection_preserves_primary_failure_class(self):
        self.corrupt_completed_audio, self.raise_in = True, "generation"
        with self.harness(), self.assertRaisesRegex(ValueError, "complete cell audio cannot be quarantined"):
            host.execute(self.env)
        _, resources, _ = self.records()
        self.assertEqual(resources["status"], "stopped_no_retry")
        self.assertEqual(resources["error_class"], "OtherException")
        self.assertEqual(set(resources["stages"]), {"setup"})
        self.assertEqual((self.output / "outcomes.json").read_bytes(), self.emitted["outcomes.json"])
        self.assertEqual(len(self.calls), 2)


class WorkflowTests(unittest.TestCase):
    def test_exact_workflow_host_cli_rejects_nonexecution_context_in_isolated_python(self):
        workflow = (ROOT / gate.WORKFLOW).read_text(encoding="utf-8")
        commands = [line.strip()[5:] for line in workflow.splitlines()
                    if line.strip().startswith("run: ") and line.strip().endswith("/tts_host.py")]
        self.assertEqual(len(commands), 1)
        command = shlex.split(commands[0])
        self.assertEqual(command[:4], ["python", "-I", "-B", "-c"])
        self.assertEqual(command[-1], "research/k1-punctuation6-v1/tts_host.py")
        # The real source can be activated; invalid context must still stop it.
        if gate.read_json(HERE / "tts-release.json")["enabled"] is True:
            gate.verify_release()
        with tempfile.TemporaryDirectory(prefix="k1-workflow-offline-") as directory:
            event = Path(directory) / "event.json"
            event.write_text("{}", encoding="utf-8")
            env = {"PATH": "/usr/bin:/bin", "HOME": directory, "RUNNER_TEMP": directory,
                   "LANG": "C.UTF-8", "GITHUB_TOKEN": SECRET, "GH_TOKEN": SECRET,
                   "KWS_DATA_TOKEN": SECRET, "ACTIONS_RUNTIME_TOKEN": SECRET,
                   "GITHUB_EVENT_PATH": str(event)}
            result = subprocess.run([sys.executable, *command[1:]], cwd=ROOT, env=env,
                capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout, "")
            self.assertEqual(json.loads(result.stderr), {"status": "stopped_no_retry", "error_class": "ValueError"})
            self.assertNotIn("ImportError", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertNotIn(SECRET, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [event])
            self.assertEqual(event.read_text(encoding="utf-8"), "{}")

    def test_workflow_keeps_pinned_actions_separate_token_steps_and_fixed_limits(self):
        workflow = (ROOT / gate.WORKFLOW).read_text(encoding="utf-8")
        actions = re.findall(r"^\s+- uses: (\S+)$", workflow, re.MULTILINE)
        self.assertEqual(len(actions), 6)
        for action in actions:
            self.assertRegex(action, r"^actions/(checkout|setup-python)@[0-9a-f]{40}$")
        self.assertEqual(workflow.count("persist-credentials: false"), 3)
        self.assertIn("cancel-in-progress: false", workflow)
        self.assertNotIn("workflow_dispatch", workflow)
        self.assertNotIn("upload-artifact", workflow)
        job = workflow.split("  k1-tts:\n", 1)[1]
        self.assertIn("timeout-minutes: 50", job)
        self.assertIn("github.event.created == true", job)
        self.assertIn("github.run_attempt == 1", job)
        self.assertLess(job.index("Start provisioning allowance"), job.index("actions/checkout@"))
        self.assertIn("timeout-minutes: 2", job)
        self.assertIn("timeout-minutes: 3", job)
        host_step = job.split("      - name: Run exact six once if separately released\n", 1)[1].split("      - name:", 1)[0]
        self.assertNotIn("env:", host_step)
        self.assertNotIn("TOKEN", host_step)
        self.assertEqual(job.count("KWS_DATA_TOKEN: ${{ github.token }}"), 1)


if __name__ == "__main__":
    unittest.main()
