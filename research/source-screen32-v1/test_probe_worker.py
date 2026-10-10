"""Offline fictional kernel fixtures only; no real scope or container execution."""
import ast
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("screen32_probe_worker_test_target", HERE / "probe_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
scope = worker.runtime_scope


def kernel():
    return {"/proc/self/cgroup": "0::/\n",
            "/proc/self/mountinfo": "1 0 0:1 / / ro - overlay overlay ro\n"
                                    "2 1 0:2 / /sys/fs/cgroup ro - cgroup2 cgroup ro\n",
            "/proc/self/status": "NoNewPrivs:\t1\nSeccomp:\t2\nNSpid:\t1\n" +
                "".join(name + ":\t0000000000000000\n" for name in
                        ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")),
            **{"/sys/fs/cgroup/" + name: value for name, value in
               {"memory.max": str(scope.MEMORY), "memory.swap.max": "0",
                "cpu.max": "400000 100000", "pids.max": "256"}.items()}}


class ProbeWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output, self.runtime = self.root / "output", self.root / "runtime"
        self.output.mkdir()
        self.runtime.mkdir()

    @contextlib.contextmanager
    def observed(self, *, values=None, interfaces=("eth0", "lo"), uid=1001,
                 implementation="CPython", version="3.12.14"):
        values = kernel() if values is None else values
        network = types.SimpleNamespace(iterdir=lambda: (types.SimpleNamespace(name=name) for name in interfaces))
        with mock.patch.object(scope, "read", side_effect=lambda path: values[path]), \
                mock.patch.object(scope.os, "geteuid", return_value=uid), \
                mock.patch.object(scope, "Path", return_value=network), \
                mock.patch.object(worker.platform, "python_implementation", return_value=implementation), \
                mock.patch.object(worker.platform, "python_version", return_value=version):
            yield

    def run_worker(self):
        return worker.run(output=self.output, runtime=self.runtime)

    def assert_no_receipt(self):
        self.assertEqual(list(self.output.iterdir()), [])

    def test_complete_bounded_receipt_with_actual_verifier_on_fictional_kernel(self):
        actual = scope.verify_runtime_scope
        with self.observed(), mock.patch.object(scope, "verify_runtime_scope", wraps=actual) as verify, \
                mock.patch.object(worker.os, "fsync", wraps=worker.os.fsync) as fsync:
            value = self.run_worker()
        verify.assert_called_once_with(network_required=False)
        self.assertEqual(fsync.call_count, 2)
        raw = (self.output / "probe-kernel.json").read_bytes()
        self.assertLessEqual(len(raw), worker.MAX_RECEIPT_BYTES)
        self.assertEqual(json.loads(raw), value)
        self.assertEqual(set(value), {"schema", "status", "kernel", "python_version", "worker_sha256", "runtime_scope_sha256",
                                      "model_calls", "runtime_installs", "human_gold", "training_admitted"})
        self.assertEqual(value["schema"], "screen32-probe-kernel-v1")
        self.assertEqual(value["status"], "complete")
        self.assertEqual(value["python_version"], "3.12.14")
        self.assertEqual(value["kernel"]["schema"], "screen32-kernel-scope-v1")
        self.assertEqual(value["kernel"]["network_interfaces"], ["eth0", "lo"])
        self.assertFalse(value["kernel"]["network_disabled"])
        for key, name in (("worker_sha256", "probe_worker.py"), ("runtime_scope_sha256", "runtime_scope.py")):
            self.assertEqual(value[key], hashlib.sha256((HERE / name).read_bytes()).hexdigest())
        self.assertEqual((value["model_calls"], value["runtime_installs"]), (0, 0))
        self.assertIs(value["human_gold"], False)
        self.assertIs(value["training_admitted"], False)
        self.assertEqual(list(self.runtime.iterdir()), [])
        self.assertEqual([path.name for path in self.output.iterdir()], ["probe-kernel.json"])

    def test_kernel_failure_precedes_version_runtime_source_and_output_work(self):
        values = kernel()
        values["/sys/fs/cgroup/memory.max"] = "max"
        with self.observed(values=values), \
                mock.patch.object(worker.platform, "python_version") as version, \
                mock.patch.object(worker, "empty_directory") as directory, \
                mock.patch.object(worker, "source_sha256") as digest, \
                mock.patch.object(worker, "write_receipt") as publish:
            with self.assertRaisesRegex(RuntimeError, "kernel.*limits mismatch"):
                self.run_worker()
            for operation in (version, directory, digest, publish):
                operation.assert_not_called()
        self.assert_no_receipt()

    def test_each_unlimited_or_wrong_kernel_limit_rejected(self):
        for name, bad in (("memory.max", "max"), ("memory.max", str(scope.MEMORY + 1)),
                          ("memory.swap.max", "1"), ("cpu.max", "max 100000"),
                          ("pids.max", "512")):
            values = kernel()
            values["/sys/fs/cgroup/" + name] = bad
            with self.subTest(name=name, bad=bad), self.observed(values=values):
                with self.assertRaises(RuntimeError):
                    self.run_worker()
            self.assert_no_receipt()

    def test_nonroot_namespace_mount_and_privilege_guards_remain_live(self):
        for key, bad in (("/proc/self/cgroup", "0::/other\n"),
                         ("/proc/self/mountinfo", kernel()["/proc/self/mountinfo"].replace("/ / ro", "/ / rw")),
                         ("/proc/self/status", kernel()["/proc/self/status"].replace("Seccomp:\t2", "Seccomp:\t0"))):
            values = kernel()
            values[key] = bad
            with self.subTest(key=key), self.observed(values=values), self.assertRaises(RuntimeError):
                self.run_worker()
            self.assert_no_receipt()
        with self.observed(uid=0), self.assertRaises(RuntimeError):
            self.run_worker()
        self.assert_no_receipt()

    def test_missing_offline_duplicate_malformed_and_huge_network_rejected(self):
        inventories = ((), ("lo",), ("eth0",), ("lo", "lo"), ("eth0", "eth0", "lo"),
                       ("lo", "x" * 16), ("lo", "bad\nname"), ("lo", 1),
                       ("lo",) + tuple("eth" + str(index) for index in range(worker.MAX_INTERFACES)))
        for interfaces in inventories:
            with self.subTest(interfaces=interfaces), self.observed(interfaces=interfaces):
                with self.assertRaises((RuntimeError, TypeError)):
                    self.run_worker()
            self.assert_no_receipt()

    def test_wrong_interpreter_version_and_prerelease_rejected(self):
        for implementation, version in (("PyPy", "3.12.14"), ("CPython", "3.12.3"),
                                         ("CPython", "3.13.0"), ("CPython", "3.12.14rc1")):
            with self.subTest(implementation=implementation, version=version), \
                    self.observed(implementation=implementation, version=version), \
                    self.assertRaisesRegex(RuntimeError, "exact CPython"):
                self.run_worker()
            self.assert_no_receipt()

    def test_any_runtime_entry_including_hidden_install_or_model_rejected(self):
        for name in ("venv", "models", ".hidden"):
            path = self.runtime / name
            path.mkdir()
            with self.observed(), self.assertRaisesRegex(RuntimeError, "empty directory"):
                self.run_worker()
            path.rmdir()
            self.assert_no_receipt()

    def test_missing_or_symlink_runtime_and_output_rejected(self):
        for selected in (self.runtime, self.output):
            selected.rmdir()
            with self.observed(), self.assertRaises(OSError):
                self.run_worker()
            target = self.root / "target"
            target.mkdir()
            selected.symlink_to(target, target_is_directory=True)
            with self.observed(), self.assertRaises(OSError):
                self.run_worker()
            self.assertEqual(list(target.iterdir()), [])
            selected.unlink()
            target.rmdir()
            selected.mkdir()
            self.assert_no_receipt()

    def test_existing_receipt_or_other_output_is_never_overwritten(self):
        for name in ("probe-kernel.json", ".probe-kernel.json.tmp", "other"):
            path = self.output / name
            path.write_bytes(b"previous bytes")
            with self.observed(), self.assertRaisesRegex(RuntimeError, "empty directory"):
                self.run_worker()
            self.assertEqual(path.read_bytes(), b"previous bytes")
            path.unlink()

    def test_source_read_failure_leaves_no_receipt(self):
        with self.observed(), mock.patch.object(worker, "source_sha256", side_effect=OSError("fixture")), \
                self.assertRaises(OSError):
            self.run_worker()
        self.assert_no_receipt()

    def test_source_size_and_alias_guards(self):
        path = self.root / "source.py"
        path.write_bytes(b"x" * (worker.MAX_SOURCE_BYTES + 1))
        with self.assertRaisesRegex(RuntimeError, "bounded unaliased"):
            worker.source_sha256(path)
        path.unlink()
        path.symlink_to(HERE / "probe_worker.py")
        with self.assertRaisesRegex(RuntimeError, "bounded unaliased"):
            worker.source_sha256(path)

    def test_source_atime_change_is_allowed_but_identity_changes_are_rejected(self):
        path = self.root / "source.py"
        path.write_bytes(b"fixed source bytes")
        before = path.lstat()
        fields = {name: getattr(before, name) for name in
                  ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                   "st_mtime_ns", "st_ctime_ns", "st_atime_ns")}
        changed = types.SimpleNamespace(**{**fields, "st_atime_ns": fields["st_atime_ns"] + 1})
        with mock.patch.object(Path, "lstat", side_effect=[before, changed]):
            self.assertEqual(worker.source_sha256(path), hashlib.sha256(b"fixed source bytes").hexdigest())
        for name in ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"):
            changed = types.SimpleNamespace(**{**fields, name: fields[name] + 1})
            with self.subTest(field=name), mock.patch.object(Path, "lstat", side_effect=[before, changed]), \
                    self.assertRaisesRegex(RuntimeError, "source changed"):
                worker.source_sha256(path)
    def test_failed_write_file_fsync_replace_or_directory_fsync_leaves_no_success(self):
        cases = (("write", OSError("write")), ("fsync", OSError("file sync")),
                 ("replace", OSError("replace")), ("fsync", [None, OSError("directory sync")]))
        for operation, failure in cases:
            with self.subTest(operation=operation, failure=failure), self.observed(), \
                    mock.patch.object(worker.os, operation, side_effect=failure), self.assertRaises(OSError):
                self.run_worker()
            self.assert_no_receipt()

    def test_partial_write_leaves_no_success(self):
        with self.observed(), mock.patch.object(worker.os, "write", return_value=1), \
                self.assertRaisesRegex(RuntimeError, "incomplete probe"):
            self.run_worker()
        self.assert_no_receipt()

    def test_receipt_size_is_bounded_before_creating_output(self):
        with self.assertRaisesRegex(RuntimeError, "oversized probe"):
            worker.write_receipt(self.output, {"huge": "x" * worker.MAX_RECEIPT_BYTES})
        self.assert_no_receipt()

    def test_no_cli_arguments_and_no_injectable_kernel_success(self):
        with mock.patch.object(worker.sys, "argv", ["probe_worker.py", "--scope-ok"]), \
                mock.patch.object(worker, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "no arguments"):
                worker.main()
            run.assert_not_called()
        with mock.patch.object(worker.sys, "argv", ["probe_worker.py"]), \
                mock.patch.object(worker, "run") as run:
            worker.main()
            run.assert_called_once_with()
        for argument in ("kernel_scope", "scope", "verify", "network_required"):
            with self.subTest(argument=argument), self.assertRaises(TypeError):
                worker.run(**{argument: True})

    def test_imports_only_stdlib_and_fixed_scope_without_process_or_network_apis(self):
        tree = ast.parse((HERE / "probe_worker.py").read_text())
        imports = {alias.name.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertLessEqual(imports, sys.stdlib_module_names | {"runtime_scope"})
        self.assertFalse(imports & {"subprocess", "socket", "urllib", "http", "runpy", "importlib"})
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertFalse(calls & {"system", "popen", "fork", "execv", "spawnv", "connect", "urlopen"})


if __name__ == "__main__":
    unittest.main()
