"""Hosted opt-in build entry: only synthetic mocked transport, no qualification."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from research_python_entry import clear_interpreter_library_path

EXECUTE = False


class ResearchFsmnStreamTest(unittest.TestCase):
    def test_bound_build_and_eight_mock_groups(self):
        if not EXECUTE:
            self.skipTest("explicit test entry --execute is required")
        root = Path(__file__).resolve().parents[1]
        driver = root / "research/fsmn2599-stream-c/build.py"
        spec = importlib.util.spec_from_file_location("fsmn_stream_build_guard", driver)
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)  # Definitions only; does not compile or run.
        with tempfile.TemporaryDirectory(prefix="fsmn-stream-entry-") as temporary:
            output, _ = guard.run_bounded(
                [sys.executable, "-B", str(driver), "--execute"], Path(temporary), 110, 100)
        report = json.loads(output)
        self.assertEqual(report["fixture_groups"], 8)
        self.assertEqual(report["fixture_runs"], 1)
        self.assertEqual(report["compiler_invocations"], 5)
        self.assertEqual(report["capture_fixture_groups"], 2)
        self.assertEqual(report["capture_fixture_runs"], 1)
        self.assertTrue(report["cpp_link_pass"])
        self.assertFalse(report["cpp_executed"])
        self.assertEqual(report["compile_budget_seconds"], 60)
        self.assertFalse(report["example_executed"])
        self.assertFalse(report["production_loaded"])
        self.assertFalse(report["qualification_allowed"])
        self.assertEqual(report["fixture_message"], guard.FIXTURE_MESSAGE)
        self.assertEqual(set(report["production_exports"]),
                         guard.ADAPTER | guard.DECODER_EXPORTS | guard.CORE_EXPORTS)

    def test_optin_profile_block2_public_matrix(self):
        if not EXECUTE:
            self.skipTest("explicit test entry --execute is required")
        clear_interpreter_library_path()
        driver = Path(__file__).resolve().parents[1] / "research/fsmn2599-stream-c/check_profile.py"
        spec = importlib.util.spec_from_file_location("optin_matrix", driver)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        report = module.execute()
        self.assertEqual(report["compiler_invocations"], 19)
        self.assertEqual(report["program_invocations"], 11)
        self.assertEqual([len(report["modes"][k]["exports"]) for k in ("default", "profile", "block2", "profile_block2")], [36,39,36,39])
        self.assertEqual(report["model_loads"], 0)
        self.assertEqual(report["model_forwards"], 0)
        self.assertFalse(report["qualification_allowed"])


class R1StreamSourceContracts(unittest.TestCase):
    """Source-only tests: no real weights, model forward, network or corpus."""
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "r1_port_contract", self.root / "research/r1_stream_c/port.py")
        self.port = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.port)
        self.temporary = tempfile.TemporaryDirectory(prefix="r1-source-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "output"

    def test_distinct_geometry_and_model_identity(self):
        self.port.prepare_sources(self.root, self.output)
        header = (self.output / "source/r1fsmn.h").read_text()
        self.assertIn("R1FSMN_OUTPUTS 2601u", header)
        self.assertIn("R1FSMN_FLOATS 757215u", header)
        self.assertIn(self.port.MODEL_SHA, (self.output / "source/model_identity.h").read_text())

    def test_no_synthetic_eof_tail(self):
        self.port.prepare_sources(self.root, self.output)
        source = (self.output / "source/r1_pcm.c").read_text()
        self.assertNotIn("for (unsigned i=0;i<4800;i++)", source)
        self.assertIn("if (s->have_low)", source)

    def test_source_preparation_never_synthesizes_weights(self):
        self.port.prepare_sources(self.root, self.output)
        self.assertEqual(list((self.output / "weights").iterdir()), [])
        self.assertTrue((self.output / "source/LICENSE.torchaudio").is_file())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(ValueError):
            self.port.prepare_sources(self.root, self.output)

    def test_missing_source_rejected_before_output(self):
        with self.assertRaises(ValueError):
            self.port.prepare_sources(Path(self.temporary.name), self.output)
        self.assertFalse(self.output.exists())

    def test_fake_model_card_rejected_before_import(self):
        package = Path(self.temporary.name) / "fake-model"
        package.mkdir()
        (package / "MODEL.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "untrusted model-card"):
            self.port.export_weights(package, self.output)
        self.assertNotIn("research_fsmn_weak", sys.modules)

    def test_full_c_compiles_without_loading_weights(self):
        import os
        import subprocess
        self.port.prepare_sources(self.root, self.output)
        environment = dict(os.environ)
        environment.pop("LD_LIBRARY_PATH", None)
        subprocess.run(["bash", str(self.output / "build.sh")], env=environment,
                       check=True, capture_output=True, timeout=30)
        self.assertTrue((self.output / "bin/r1_pcm").is_file())

    def test_optin_numeric_source_contracts(self):
        spec = importlib.util.spec_from_file_location(
            "r1_numeric_source_tests", self.root / "research/r1_stream_c/numeric_source_contracts.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.NumericSourceContracts)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.testsRun, 13)
        self.assertFalse(result.skipped)
        self.assertTrue(result.wasSuccessful(), repr(result.errors + result.failures))

    def test_cross_arch_source_contracts(self):
        spec = importlib.util.spec_from_file_location(
            "r1_cross_arch_source_tests",
            self.root / "research/r1_stream_c/cross_arch_source_contracts.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            module.CrossArchSourceContracts)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.testsRun, 28)
        self.assertFalse(result.skipped)
        self.assertTrue(result.wasSuccessful(), repr(result.errors + result.failures))


    def test_reference64_sdk_source_build(self):
        import os
        import subprocess
        driver = self.root / "research/r1_stream_c/deploy/materialize.py"
        spec = importlib.util.spec_from_file_location("r1_sdk_prepare", driver)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.prepare(self.root, self.output)
        self.assertEqual(report["model_forwards"], 0)
        self.assertFalse(report["weights_present"])
        self.assertEqual(report["legacy_fp32_gate"], "FAIL_UNCHANGED")
        build = Path(self.temporary.name) / "build-sdk"
        env = dict(os.environ)
        env.pop("LD_LIBRARY_PATH", None)
        for command in (["cmake", "-S", str(self.output), "-B", str(build)],
                        ["cmake", "--build", str(build), "--parallel", "2"],
                        ["ctest", "--test-dir", str(build), "--output-on-failure"]):
            subprocess.run(command, env=env, check=True, capture_output=True, timeout=30)
        with self.assertRaises(ValueError):
            module.prepare(self.root, self.output)


    def test_optin_qhead8_source_contracts(self):
        """Run archived-source INT8 head contracts; no models or PCM in CI."""
        spec = importlib.util.spec_from_file_location(
            "r1_qhead8_contracts",
            self.root / "research/r1_stream_c/quant_head8/test_contracts.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.Head8SourceContracts)
        result = unittest.TestResult()
        suite.run(result)
        self.assertEqual(result.testsRun, 9)
        self.assertFalse(result.skipped)
        self.assertTrue(result.wasSuccessful(), repr(result.errors + result.failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="build and run the eight mocked glue groups once")
    args = parser.parse_args()
    if args.execute:
        EXECUTE = True
        unittest.main(argv=[sys.argv[0]])
    else:
        parser.print_help()
