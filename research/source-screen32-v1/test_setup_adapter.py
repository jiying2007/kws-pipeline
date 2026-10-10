"""Stdlib-only adapter binding tests. No installation, acquisition or native calls."""
import ast
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("screen32_setup_adapter_test_target", ROOT / "setup_adapter.py")
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class SetupBindings(unittest.TestCase):
    def setUp(self):
        self.scope = mock.patch.object(a, "_scope", return_value={"test_scope": True})
        self.scope_mock = self.scope.start()
        self.addCleanup(self.scope.stop)
        self.saved_modules = dict(sys.modules)
        self.addCleanup(self.restore_modules)

    def restore_modules(self):
        for name in ("runtime_gate", "generate_six", "import_preflight", "setup_diagnostics", "wheel_identity",
                     "source_screen_locked_tts", "source_screen_locked_asr"):
            if name in self.saved_modules:
                sys.modules[name] = self.saved_modules[name]
            else:
                sys.modules.pop(name, None)

    def bound(self, profile="tts", stage="setup"):
        return a.bind(profile, Path("/runtime"), stage)

    def test_import_is_stdlib_only(self):
        tree = ast.parse((ROOT / "setup_adapter.py").read_text())
        names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        names |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(names & {"torch", "numpy", "packaging", "qwen_tts", "qwen_asr", "soundfile"})

    def test_scope_is_first_and_fail_closed(self):
        self.scope_mock.side_effect = RuntimeError("not in admitted scope")
        with mock.patch.object(a, "_load") as load:
            with self.assertRaisesRegex(RuntimeError, "not in admitted scope"):
                self.bound()
            load.assert_not_called()

    def test_only_exact_runtime_and_profiles(self):
        for runtime in ("/code/runtime", "/tmp/runtime", "runtime"):
            with self.assertRaises(ValueError):
                a.bind("tts", runtime, "setup")
        with self.assertRaises(ValueError):
            a.bind("other", "/runtime", "setup")
        self.scope_mock.assert_not_called()

    def test_scope_network_exception_is_setup_only(self):
        self.scope.stop()
        observer = types.SimpleNamespace(verify_runtime_scope=mock.Mock(return_value={}))
        with mock.patch.object(a, "_load", return_value=observer):
            a._scope("setup")
            observer.verify_runtime_scope.assert_called_once_with(network_required=False)
            observer.verify_runtime_scope.reset_mock()
            a._scope("verify")
            observer.verify_runtime_scope.assert_called_once_with()
            with self.assertRaises(ValueError):
                a._scope("anything")
        self.scope.start()

    def test_exact_tts_conversion_and_total(self):
        helper = self.bound()
        runtime, models, identities = helper._read_locks()
        self.assertEqual(len(runtime["files"]), 92)
        model = models["qwen_tts"]
        self.assertEqual(model["model_id"], a.MODEL_ID)
        self.assertEqual(model["asset_lock"]["revision"], a.MODEL_REVISION)
        self.assertEqual(len(model["asset_lock"]["files"]), 13)
        self.assertEqual(helper.validate_locks(runtime, models), 4_956_729_799)
        self.assertEqual(identities["source_screen_model_lock"]["sha256"], a.MODEL_LOCK_SHA256)
        self.assertEqual(identities["runtime_lock"]["sha256"], a.RETAINED_PINS["tts"]["runtime-lock.json"])
        self.assertEqual(identities["bound_models_sha256"], helper.canonical_sha(models))
        self.assertEqual(identities["adapter"], a._identity(ROOT / "setup_adapter.py"))
        self.assertEqual(identities["runtime_scope"], a._identity(ROOT / "runtime_scope.py"))
        retained = json.loads((ROOT.parent / "qwen6_tts/model-locks.json").read_bytes())["qwen_tts"]
        self.assertEqual(model["source_lock"], retained["source_lock"])
        self.assertEqual(model["source_lock_sha256"], retained["source_lock_sha256"])
        self.assertEqual(len(model["source_lock"]), 17)

    def test_conversion_deterministic_and_does_not_mutate_source(self):
        helper = self.bound()
        source = json.loads((ROOT / "qwen-model-lock.json").read_bytes())
        retained = json.loads((ROOT.parent / "qwen6_tts/model-locks.json").read_bytes())
        before = copy.deepcopy((retained, source))
        value = a.converted_tts_models(retained, source, helper.canonical_sha)
        source["files"].reverse()
        self.assertEqual(value, a.converted_tts_models(retained, source, helper.canonical_sha))
        source["files"].reverse()
        self.assertEqual((retained, source), before)

    def test_reject_old_model_or_mixed_revision(self):
        helper = self.bound()
        source = json.loads((ROOT / "qwen-model-lock.json").read_bytes())
        retained = json.loads((ROOT.parent / "qwen6_tts/model-locks.json").read_bytes())
        for key, value in (("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"), ("revision", "0" * 40)):
            bad = copy.deepcopy(source)
            bad[key] = value
            with self.assertRaises(ValueError):
                a.converted_tts_models(retained, bad, helper.canonical_sha)
        source["files"][0]["observed_repo_revision"] = "0" * 40
        with self.assertRaises(ValueError):
            a.converted_tts_models(retained, source, helper.canonical_sha)

    def test_asr_exact_124_package_two_models(self):
        helper = self.bound("asr")
        runtime, models, identities = helper._read_locks()
        self.assertEqual(len(runtime["files"]), 124)
        self.assertEqual(set(models), {"qwen06", "sensevoice"})
        self.assertEqual(helper.validate_locks(runtime, models), 3_340_966_447)
        self.assertEqual(helper.DOWNLOAD_CAP, 4 * a.GIB)
        self.assertNotIn("source_screen_model_lock", identities)
        self.assertEqual(models, json.loads((ROOT.parent / "qwen6_asr/model-locks.json").read_bytes()))

    def test_exact_soundfile_bundle_repair_is_preserved(self):
        for profile, version, sha in (("tts", "0.13.1", "03267c4e493315294834a0870f31dbb3b28a95561b80b134f0bd3cf2d5f0e618"),
                                      ("asr", "0.14.0", "1e38bac1853412871318e82a1ba69a8be677619b56025bbfcccdb41b6cafe82d")):
            runtime, _, _ = self.bound(profile)._read_locks()
            row = next(row for row in runtime["files"] if row["name"] == "soundfile")
            self.assertEqual((row["version"], row["sha256"]), (version, sha))
            self.assertIn("manylinux_2_28_x86_64.whl", row["filename"])

    def test_tts_download_default_is_rebound_and_reserve_excluded(self):
        helper = self.bound()
        self.assertEqual(helper.DOWNLOAD_CAP, 6 * a.GIB - 128 * 1024 ** 2)
        self.assertEqual(helper.Downloader(opener=object()).cap, helper.DOWNLOAD_CAP)
        with self.assertRaises(ValueError):
            helper.Downloader(cap=6 * a.GIB, opener=object())
        self.assertEqual((helper.INSTALL_CAP, helper.BUILD_CAP, helper.WORKSPACE_CAP, helper.INITIAL_FREE),
                         (3 * a.GIB, 2 * a.GIB, 12 * a.GIB, 16 * a.GIB))

    def test_alternate_lock_paths_rejected(self):
        helper = self.bound()
        with self.assertRaises(ValueError):
            helper._read_locks(ROOT / "runtime-lock.json")

    def test_bad_retained_pin_fails_before_loading(self):
        with mock.patch.dict(a.RETAINED_PINS["tts"], {"setup_locked.py": "0" * 64}), mock.patch.object(a, "_load") as load:
            with self.assertRaisesRegex(ValueError, "pin changed"):
                self.bound()
            load.assert_not_called()

    def test_every_retained_helper_command_routes_back_to_adapter(self):
        helper = self.bound()
        for body, role in a.HELPER_BODIES.items():
            script = ROOT.parent / "qwen6_tts" / ("import_preflight.py" if role == "preflight" else "setup_locked.py")
            command = helper._helper_command("/runtime/venv/bin/python", script, body)
            self.assertEqual(command, a._child_command("/runtime/venv/bin/python", "tts", "/runtime", role))
            self.assertNotIn(str(script), command)
            self.assertEqual(command[1:4], ["-I", "-B", "-c"])
            self.assertNotIn(body, command)
        with self.assertRaises(ValueError):
            helper._helper_command("/runtime/venv/bin/python", script, "exec(input())")
        with self.assertRaises(ValueError):
            helper._helper_command("/runtime/venv/bin/python", ROOT / "old.py", next(iter(a.HELPER_BODIES)))

    def test_commands_wrap_only_known_modules(self):
        helper = self.bound()
        base = helper.Commands.__mro__[1]
        with mock.patch.object(base, "run", return_value="mocked") as run:
            runner = helper.Commands(Path("/runtime"), {}, 0)
            for module in ("ensurepip", "pip", "build"):
                result = runner.run(["/runtime/venv/bin/python", "-I", "-m", module, "fake-arg"], stage="bootstrap_pip")
                self.assertEqual(result, "mocked")
                self.assertEqual(run.call_args.args[0], a._child_command("/runtime/venv/bin/python", "tts", "/runtime", "module:" + module) + ["fake-arg"])
            for command in (["sh", "-c", "echo nope"], ["/runtime/venv/bin/python", "-I", "-m", "unreviewed"]):
                with self.assertRaises(ValueError):
                    runner.run(command, stage="bootstrap_pip")

    def test_commands_accept_exact_bound_child(self):
        helper = self.bound("asr")
        base = helper.Commands.__mro__[1]
        command = a._child_command("/runtime/venv/bin/python", "asr", "/runtime", "verify_after_models") + ["/runtime"]
        with mock.patch.object(base, "run", return_value="mocked") as run:
            helper.Commands(Path("/runtime"), {}, 0).run(command, stage="verify_setup")
            run.assert_called_once_with(command, stage="verify_setup")

    def test_no_setup_commands_in_verify(self):
        helper = self.bound(stage="verify")
        with self.assertRaises(ValueError):
            helper.Commands(Path("/runtime"), {}, 0).run(["/runtime/venv/bin/python", "-I", "-m", "pip"], stage="bootstrap_pip")
        with self.assertRaises(ValueError):
            helper._helper_command("/runtime/venv/bin/python", ROOT.parent / "qwen6_tts/setup_locked.py", next(iter(a.HELPER_BODIES)))

    def test_cache_paths_never_target_code(self):
        helper = self.bound()
        caches = sys.modules["runtime_gate"].cache_paths()
        self.assertEqual(set(caches), set(a.CACHE_KEYS))
        self.assertTrue(all(path.is_relative_to("/runtime/buildtmp") for path in caches.values()))
        environment = helper.offline_environment("/runtime")
        self.assertEqual(environment["ORT_DISABLE_TELEMETRY"], "1")
        self.assertEqual(environment["PIP_NO_INDEX"], "1")
        self.bound(stage="verify")
        self.assertTrue(all(path.is_relative_to("/scratch") for path in sys.modules["runtime_gate"].cache_paths().values()))

    def test_child_preflight_uses_bound_module_and_exact_arguments(self):
        fake = types.SimpleNamespace(execute=mock.Mock())
        with mock.patch.object(a, "bind", return_value=object()) as bind, mock.patch.dict(sys.modules, {"import_preflight": fake}):
            a._run_child("tts", "/runtime", "preflight", ["tts", "/runtime"])
            bind.assert_called_once_with("tts", Path("/runtime"), "setup")
            fake.execute.assert_called_once_with("tts", Path("/runtime"), Path("/runtime/import-preflight.json"))
            with self.assertRaises(ValueError):
                a._run_child("tts", "/runtime", "preflight", ["asr", "/runtime"])

    def test_child_verification_preserves_pre_model_mode(self):
        fake = types.SimpleNamespace(verify_setup=mock.Mock(return_value={}))
        with mock.patch.object(a, "bind", return_value=fake), contextlib.redirect_stdout(io.StringIO()):
            a._run_child("asr", "/runtime", "verify_before_models", ["/runtime"])
            fake.verify_setup.assert_called_once_with(Path("/runtime"), _during_setup=True, _before_models=True)
            fake.verify_setup.reset_mock()
            a._run_child("asr", "/runtime", "verify_after_models", ["/runtime"])
            fake.verify_setup.assert_called_once_with(Path("/runtime"), _during_setup=True, _before_models=False)

    def test_child_module_scope_precedes_third_party_entry(self):
        order = []
        with mock.patch.object(a, "bind", side_effect=lambda *args: order.append("scope")), mock.patch.object(a.runpy, "run_module", side_effect=lambda *args, **kwargs: order.append("module")), mock.patch.object(sys, "argv", []):
            a._run_child("tts", "/runtime", "module:build", ["--wheel"])
            self.assertEqual(order, ["scope", "module"])
            with self.assertRaises(ValueError):
                a._run_child("tts", "/runtime", "module:unreviewed", [])

    def test_main_verify_calls_exact_retained_verifier(self):
        fake = types.SimpleNamespace(verify_setup=mock.Mock(return_value={}))
        with mock.patch.object(a, "bind", return_value=fake) as bind, contextlib.redirect_stdout(io.StringIO()):
            a.main(["--profile", "tts", "--stage", "verify", "--runtime", "/runtime"])
            bind.assert_called_once_with("tts", Path("/runtime"), "verify")
            fake.verify_setup.assert_called_once_with(Path("/runtime"))

    def test_receipt_binds_source_scope_profile_and_download_budget(self):
        helper = self.bound()
        receipt = {"locks": helper._read_locks()[2]}
        with mock.patch.object(Path, "write_text") as write, mock.patch.object(Path, "replace"):
            helper._save_receipt(Path("/runtime/setup-receipt.json"), receipt)
            saved = json.loads(write.call_args.args[0])
        self.assertEqual(saved["source_screen_binding"], {
            "schema": "screen32-setup-binding-v1", "profile": "tts", "runtime": "/runtime",
            "scope": {"test_scope": True}, "model_id": a.MODEL_ID,
            "input_download_cap_bytes": a.TTS_INPUT_CAP})
        self.assertIn("source_screen_model_lock", saved["locks"])
        self.assertIn("runtime_lock", saved["locks"])
        with self.assertRaises(ValueError):
            helper._save_receipt(ROOT / "setup-receipt.json", receipt)

    def test_worker_verify_interface_is_read_only_and_offline(self):
        fake = types.SimpleNamespace(verify_setup=mock.Mock(return_value={"verified": True}))
        with mock.patch.object(a, "bind", return_value=fake) as bind:
            self.assertEqual(a.verify_runtime(Path("/runtime"), "tts"), {"verified": True})
            bind.assert_called_once_with("tts", Path("/runtime"), "verify")
            fake.verify_setup.assert_called_once_with(Path("/runtime"))

    def test_main_setup_checks_16gib_before_any_setup(self):
        fake = types.SimpleNamespace(INITIAL_FREE=16 * a.GIB, setup=mock.Mock(),
                                     shutil=types.SimpleNamespace(disk_usage=mock.Mock(return_value=types.SimpleNamespace(free=16*a.GIB-1))))
        with mock.patch.object(a, "bind", return_value=fake):
            with self.assertRaisesRegex(ValueError, "16 GiB"):
                a.main(["--profile", "tts", "--stage", "setup", "--runtime", "/runtime"])
            fake.setup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
