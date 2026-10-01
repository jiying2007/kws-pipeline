#!/usr/bin/env python3
"""Offline fixtures for the exact no-checkout inline hosted resource probe."""
from __future__ import annotations

import ast
import contextlib
import io
import json
import pathlib
import types
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/github-hosted-resource-probe.yml"
SOURCE = WORKFLOW.read_text(encoding="utf-8")
PREFIX = "          "
START = PREFIX + "python3 -I -S - <<'PY'\n"
assert SOURCE.count(START) == 1
INLINE = SOURCE.split(START, 1)[1]
assert INLINE.endswith(PREFIX + "PY\n")
INLINE = INLINE[:-len(PREFIX + "PY\n")]
assert all(not line or line.startswith(PREFIX) for line in INLINE.splitlines())
INLINE = "\n".join(line[len(PREFIX):] for line in INLINE.splitlines()) + "\n"
PROBE = types.ModuleType("hosted_resource_probe_fixture")
exec(compile(INLINE, "<hosted-resource-probe>", "exec"), PROBE.__dict__)

ENV_KEYS = {
    "KWS_PROBE_HEAD_SHA", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
    "GITHUB_EVENT_NAME", "ImageOS", "ImageVersion", "GITHUB_STEP_SUMMARY",
}
READ_PATHS = (
    "/proc/meminfo", "/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/cpu.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
)
FIELD_KEYS = {
    "schema", "evidence_class", "identity", "runner_image", "cpu", "memory",
    "cgroup", "root_filesystem_available_bytes", "runtime",
    "distribution_metadata_scope", "installed_distributions", "interpretation",
}


class GuardedEnvironment:
    def __init__(self, values):
        self.values = values
        self.reads = set()

    def get(self, key, default=None):
        assert key in ENV_KEYS, f"non-allowlisted environment key: {key}"
        self.reads.add(key)
        return self.values.get(key, default)

    def __iter__(self):
        raise AssertionError("environment enumeration forbidden")


class ProbeTests(unittest.TestCase):
    def fixture(self, env=None, files=None, affinity=None):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        environment = GuardedEnvironment(env or {})
        files = files or {}

        def read(path):
            self.assertIn(path, READ_PATHS)
            return files.get(path, "unknown")

        stack.enter_context(mock.patch.object(PROBE.os, "environ", environment))
        stack.enter_context(mock.patch.object(PROBE, "read_fixed", side_effect=read))
        stack.enter_context(mock.patch.object(PROBE.platform, "machine", return_value="x86_64"))
        stack.enter_context(mock.patch.object(PROBE.platform, "python_version", return_value="3.12.3"))
        stack.enter_context(mock.patch.object(PROBE.os, "confstr", return_value="glibc 2.39"))
        stack.enter_context(mock.patch.object(PROBE.os, "cpu_count", return_value=4))
        stack.enter_context(mock.patch.object(PROBE.os, "sched_getaffinity", return_value={0, 1} if affinity is None else affinity))
        stack.enter_context(mock.patch.object(PROBE.shutil, "disk_usage", return_value=types.SimpleNamespace(free=123456)))
        stack.enter_context(mock.patch.object(PROBE, "metadata_version", side_effect=PROBE.importlib.metadata.PackageNotFoundError))
        return environment

    def test_workflow_boundaries(self):
        header = SOURCE.split(START, 1)[0]
        self.assertEqual(header.count("runs-on:"), 1)
        self.assertIn("    runs-on: ubuntu-24.04\n", header)
        self.assertIn("    timeout-minutes: 3\n", header)
        self.assertIn("\npermissions: {}\n", header)
        self.assertIn("\nconcurrency:\n  group: github-hosted-resource-probe\n  cancel-in-progress: false\n", header)
        self.assertIn("  workflow_dispatch:\n  pull_request:\n", header)
        self.assertIn("    types: [opened, synchronize, reopened]\n", header)
        self.assertEqual(
            [line.strip() for line in header.splitlines() if line.strip().startswith("- '")],
            ["- '.github/workflows/github-hosted-resource-probe.yml'", "- 'tests/test_github_hosted_resource_probe.py'"],
        )
        for forbidden in ("push:", "schedule:", "pull_request_target:", "uses:", "secrets.", "inputs.", "self-hosted", "matrix:"):
            self.assertNotIn(forbidden, header)
        self.assertEqual(header.count("      - name:"), 1)
        self.assertEqual(header.count("${{"), 1)
        self.assertIn("KWS_PROBE_HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}", header)
        self.assertNotIn("${{", INLINE)
        ci = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        self.assertIn("run: python3 tests/test_github_hosted_resource_probe.py", ci)

    def test_stdlib_only_and_fixed_io_allowlist(self):
        tree = ast.parse(INLINE)
        imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {"importlib.metadata", "json", "os", "platform", "re", "shutil", "site", "sys"})
        self.assertFalse(any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"eval", "exec", "__import__", "compile"})
            if isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, {"system", "popen", "walk", "listdir", "scandir", "environb", "uname", "getlogin"})
        self.assertEqual(PROBE.READ_PATHS, READ_PATHS)
        self.assertEqual(PROBE.DISTRIBUTIONS, ("torch", "torchaudio", "onnxruntime", "transformers", "numpy", "safetensors"))
        env_reads = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "environ"]
        get_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                     and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "environ"]
        self.assertEqual(len(env_reads), len(get_calls))
        self.assertEqual({node.args[0].value for node in get_calls}, ENV_KEYS)
        for path in ("/etc/passwd", "/proc/self/environ", "/home/private/audio.wav"):
            with mock.patch("builtins.open", side_effect=AssertionError("unexpected file read")):
                self.assertEqual(PROBE.read_fixed(path), "unknown")

    def test_record_field_whitelist_and_identity(self):
        self.fixture(env={"KWS_PROBE_HEAD_SHA": "a" * 40, "GITHUB_SHA": "b" * 40,
                          "GITHUB_RUN_ID": "987654321", "GITHUB_RUN_ATTEMPT": "1",
                          "GITHUB_EVENT_NAME": "pull_request", "ImageOS": "ubuntu24",
                          "ImageVersion": "20261001.1.0"})
        record = PROBE.collect()
        self.assertEqual(set(record), FIELD_KEYS)
        self.assertEqual(record["schema"], "kws.github-hosted-resource-probe.v1")
        self.assertEqual(set(record["identity"]), {"run_id", "run_attempt", "event", "head_sha", "workflow_sha"})
        self.assertEqual(record["identity"]["head_sha"], "a" * 40)
        self.assertEqual(record["identity"]["workflow_sha"], "b" * 40)
        self.assertEqual(set(record["runner_image"]), {"ImageOS", "ImageVersion"})
        self.assertEqual(set(record["cpu"]), {"architecture", "logical_count", "affinity_count", "affinity_ids", "affinity_ids_truncated"})
        self.assertEqual(set(record["memory"]), {"total_bytes", "available_bytes"})
        self.assertEqual(set(record["cgroup"]), {"scope", "v2_memory_max_bytes", "v2_cpu_max", "v1_memory_limit_bytes", "v1_cpu_quota_us", "v1_cpu_period_us"})
        self.assertEqual(set(record["cgroup"]["v2_cpu_max"]), {"quota_us", "period_us"})
        self.assertEqual(set(record["runtime"]), {"python", "glibc"})
        self.assertEqual(set(record["installed_distributions"]), set(PROBE.DISTRIBUTIONS))
        for value in record["installed_distributions"].values():
            self.assertEqual(set(value), {"version", "status"})
        self.assertIn("does not prove imports", record["interpretation"])

    def test_unknown_and_sensitive_environment(self):
        environment = self.fixture(env={"SECRET_TOKEN": "do-not-disclose", "USER": "private-user",
                                        "HOME": "/home/private", "GITHUB_TOKEN": "do-not-disclose"})
        with mock.patch.object(PROBE.platform, "machine", side_effect=OSError("private filename")), \
             mock.patch.object(PROBE.os, "sched_getaffinity", side_effect=OSError("private process")), \
             mock.patch.object(PROBE.os, "cpu_count", return_value=None), \
             mock.patch.object(PROBE.shutil, "disk_usage", side_effect=OSError("private mount")):
            record = PROBE.collect()
        self.assertEqual(set(record["identity"].values()), {"unknown"})
        self.assertEqual(set(record["memory"].values()), {"unknown"})
        self.assertEqual(set(record["cpu"].values()), {"unknown"})
        self.assertEqual(record["root_filesystem_available_bytes"], "unknown")
        payload, _ = PROBE.render(record)
        self.assertNotIn("private", payload)
        self.assertNotIn("do-not-disclose", payload)
        self.assertTrue(environment.reads <= ENV_KEYS)

    def test_malformed_identity_never_becomes_script_or_output(self):
        self.fixture(env={key: "$(id)\n::error::private" for key in ENV_KEYS})
        record = PROBE.collect()
        self.assertEqual(set(record["identity"].values()), {"unknown"})
        self.assertEqual(set(record["runner_image"].values()), {"unknown"})
        self.assertNotIn("private", PROBE.render(record)[0])

    def test_text_and_affinity_bounds(self):
        self.fixture(env={"ImageVersion": "A" * 10000}, affinity=set(range(1000)))
        record = PROBE.collect()
        self.assertEqual(record["runner_image"]["ImageVersion"], "A" * 128)
        self.assertEqual(record["cpu"]["affinity_count"], 1000)
        self.assertEqual(record["cpu"]["affinity_ids"], list(range(128)))
        self.assertIs(record["cpu"]["affinity_ids_truncated"], True)
        self.assertEqual(PROBE.text_value("1!2.0+cpu"), "1!2.0+cpu")
        for value in (None, "", "\x00secret", "```", "/private", "non-ascii-值"):
            self.assertEqual(PROBE.text_value(value), "unknown")
        for value in (True, -1, 2**64, None):
            self.assertEqual(PROBE.integer(value), "unknown")

    def test_memory_units_and_unknown(self):
        self.assertEqual(PROBE.memory_bytes("MemTotal:       8000 kB\nMemAvailable:   4000 kB\nOther: secret\n"),
                         {"total_bytes": 8192000, "available_bytes": 4096000})
        self.assertEqual(PROBE.memory_bytes("MemTotal: 8 MB\nMemAvailable: -1 kB\n"),
                         {"total_bytes": "unknown", "available_bytes": "unknown"})

    def test_cgroup_max_v1_and_partial_visibility(self):
        self.fixture(files={READ_PATHS[1]: "max\n", READ_PATHS[2]: "max 100000\n",
                            READ_PATHS[3]: "9223372036854771712\n", READ_PATHS[4]: "-1\n"})
        result = PROBE.collect()["cgroup"]
        self.assertEqual(result["v2_memory_max_bytes"], "unlimited")
        self.assertEqual(result["v2_cpu_max"], {"quota_us": "unlimited", "period_us": 100000})
        self.assertEqual(result["v1_memory_limit_bytes"], 9223372036854771712)
        self.assertEqual(result["v1_cpu_quota_us"], "unlimited")
        self.assertEqual(result["v1_cpu_period_us"], "unknown")
        self.assertIn("not_effective_process_limit", result["scope"])
        self.assertEqual(PROBE.cpu_max("200000 100000"), {"quota_us": 200000, "period_us": 100000})
        self.assertEqual(PROBE.cpu_max("malformed"), {"quota_us": "unknown", "period_us": "unknown"})
        for value in ("", "unlimited", "-1", "9" * 100):
            self.assertEqual(PROBE.cgroup_limit(value), "unknown")

    def test_metadata_version_only_without_importing_packages(self):
        self.fixture()

        def version(name):
            if name == "torch":
                return "2.8.0+cpu"
            if name == "torchaudio":
                raise OSError("secret filepath")
            if name == "numpy":
                return "1" * 10000
            raise PROBE.importlib.metadata.PackageNotFoundError(name)

        with mock.patch.object(PROBE, "metadata_version", side_effect=version) as read:
            result = PROBE.distribution_versions()
        self.assertEqual([call.args[0] for call in read.call_args_list], list(PROBE.DISTRIBUTIONS))
        self.assertEqual(result["torch"], {"version": "2.8.0+cpu", "status": "metadata_only"})
        self.assertEqual(result["torchaudio"], {"version": "unknown", "status": "unreadable"})
        self.assertEqual(result["transformers"], {"version": "unknown", "status": "not_installed"})
        self.assertEqual(result["numpy"]["version"], "1" * 128)
        self.assertNotIn("secret", json.dumps(result))

    def test_real_distribution_metadata_without_module_or_pth_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            metadata = root / "torch-2.8.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.8.0+cpu\n", encoding="utf-8")
            (root / "torch.py").write_text("raise AssertionError('must never import model')\n", encoding="utf-8")
            (root / "unsafe.pth").write_text("import missing_must_never_execute_pth\n", encoding="utf-8")
            (root / "sitecustomize.py").write_text("raise AssertionError('must never execute sitecustomize')\n", encoding="utf-8")
            with mock.patch.object(PROBE.site, "getsitepackages", return_value=[directory]):
                self.assertEqual(PROBE.metadata_version("torch"), "2.8.0+cpu")
                with self.assertRaises(PROBE.importlib.metadata.PackageNotFoundError):
                    PROBE.metadata_version("transformers")
                self.assertNotIn("torch", PROBE.sys.modules)
        self.fixture()
        self.assertIn("without_pth_or_user_site", PROBE.collect()["distribution_metadata_scope"])

    def test_fixed_reads_bounded_and_errors_not_reported(self):
        with mock.patch("builtins.open", mock.mock_open(read_data="x" * 65537)) as source:
            self.assertEqual(PROBE.read_fixed(READ_PATHS[0]), "unknown")
            source().read.assert_called_once_with(65537)
        with mock.patch("builtins.open", side_effect=PermissionError("secret filepath")):
            self.assertEqual(PROBE.read_fixed(READ_PATHS[0]), "unknown")

    def test_json_and_summary_caps_fail_closed(self):
        self.fixture(affinity=set(range(1000)))
        record = PROBE.collect()
        payload, summary = PROBE.render(record)
        self.assertEqual(json.loads(payload), record)
        self.assertLessEqual(len(payload.encode()), 32768)
        self.assertLessEqual(len(summary.encode()), 32768)
        self.assertEqual(summary, "```json\n" + payload + "```\n")
        for record in ({"oversized": "x" * 32768}, {"oversized": "值" * 32768}):
            with self.assertRaisesRegex(ValueError, "exceeds byte cap"):
                PROBE.render(record)

    def test_main_emits_only_bounded_json_and_summary(self):
        self.fixture(env={"GITHUB_STEP_SUMMARY": "/private-output-path"})
        output = io.StringIO()
        with mock.patch("builtins.open", mock.mock_open()) as destination, contextlib.redirect_stdout(output):
            self.assertEqual(PROBE.main(), 0)
        payload = output.getvalue()
        self.assertEqual(set(json.loads(payload)), FIELD_KEYS)
        self.assertNotIn("private-output-path", payload)
        destination.assert_called_once_with("/private-output-path", "a", encoding="utf-8")
        summary = destination().write.call_args.args[0]
        self.assertLessEqual(len(summary.encode()), 32768)
        self.assertEqual(summary, "```json\n" + payload + "```\n")

    def test_summary_error_does_not_leak_path(self):
        self.fixture(env={"GITHUB_STEP_SUMMARY": "/private-output-path"})
        output = io.StringIO()
        with mock.patch("builtins.open", side_effect=OSError("private path")), contextlib.redirect_stdout(output):
            self.assertEqual(PROBE.main(), 1)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
