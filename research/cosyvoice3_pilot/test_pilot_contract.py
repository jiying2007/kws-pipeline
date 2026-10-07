"""Stdlib-only gates and failure-path tests; never install/import model runtime."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock
import wave

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot
import pilot_common as common
import pilot_worker as worker


class ContractTests(unittest.TestCase):
    def test_frozen_six_pairs(self):
        c = common.load_config()
        self.assertEqual([r["seed"] for r in c["design"]["phrase_rows"]], list(range(610201, 610207)))
        self.assertEqual(c["weights"]["total_bytes"], sum(r["size"] for r in c["weights"]["files"]))
        self.assertEqual(len(c["weights"]["files"]), 12)
        self.assertNotIn("local_path", c["reference"])

    def test_no_heavy_imports(self):
        self.assertFalse(any(x in sys.modules for x in ("torch", "numpy", "onnxruntime", "soundfile")))

    def test_all_paths_pinned(self):
        c = common.load_config()
        manifest = common.read_json(common.HERE / "source-allowlist.json")
        self.assertEqual(len({r["path"] for r in manifest["files"]}), len(manifest["files"]))
        for row in manifest["files"]:
            common.safe_relative(row["path"])
            self.assertIn(row["revision"], row["url"])
            self.assertIn(row["revision"], (c["source"]["commit"], c["source"]["matcha_submodule_commit"]))
            self.assertEqual(len(row["git_blob_sha1"]), 40)
            if row["path"].endswith(".py"):
                self.assertEqual(len(row["sha256"]), 64)

    def test_url_and_traversal_rejection(self):
        for name in ("../oops", "/oops", "a/../../b", "a\\b"):
            with self.assertRaises(common.GateError):
                common.safe_relative(name)
        for url in ("http://huggingface.co/a", "https://huggingface.co.evil.com/a", "https://user@huggingface.co/a", "https://evil.com/a"):
            with self.assertRaises(common.GateError):
                common.validate_url(url)
        common.validate_url("https://cas-bridge.xethub.hf.co/approved-signed-path")

    def test_runner_gate_no_local_execution(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(common.GateError):
            common.require_runner()

    def test_exact_file_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "sample"
            p.write_bytes(b"good")
            row = {"bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest(), "git_blob_sha1": hashlib.sha1(b"blob 4\0good").hexdigest()}
            common.verify_file(p, row)
            p.write_bytes(b"evil")
            with self.assertRaises(common.GateError):
                common.verify_file(p, row)

    def test_no_retry_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "claim.json"
            pilot.claim(p)
            with self.assertRaises(FileExistsError):
                pilot.claim(p)

    def test_model_config_remote_code_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "CosyVoice-BlankEN"
            root.mkdir()
            for name, data in (("config.json", {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}), ("generation_config.json", {}), ("tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})):
                common.write_json(root / name, data)
            common.validate_pretrained_configs(temp)
            for key in ("auto_map", "_attn_implementation_internal", "custom_pipelines", "custom_generate", "trust_remote_code"):
                common.write_json(root / "generation_config.json", {"nested": {key: "evil/repository"}})
                with self.assertRaises(common.GateError):
                    common.validate_pretrained_configs(temp)

    def test_job_budget_both_actual_and_effective(self):
        with tempfile.TemporaryDirectory() as temp:
            common.write_json(Path(temp) / "controller-state.json", {"baseline_free_bytes": 20_000_000_000})
            with mock.patch.object(common, "allocated_bytes", return_value=0), mock.patch.object(common, "disk_gate", return_value=12_000_000_000):
                with self.assertRaises(common.GateError):
                    common.job_gate(temp, common.MODEL_FREE_MIN)
            with mock.patch.object(common, "allocated_bytes", return_value=8_000_000_000), mock.patch.object(common, "disk_gate", return_value=20_000_000_000):
                with self.assertRaises(common.GateError):
                    common.job_gate(temp, common.MODEL_FREE_MIN)

    def test_dependency_failure_stops_before_model_download(self):
        paths = pilot.layout("/tmp/unused-pilot-test")
        with mock.patch.object(pilot, "qualified_source_gate", side_effect=common.GateError("dependency failed")), mock.patch.object(pilot, "download_one") as download:
            with self.assertRaises(common.GateError):
                pilot.acquire_model(paths, types.SimpleNamespace())
            download.assert_not_called()

    def test_no_nested_process_groups(self):
        source = (common.HERE / "pilot.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen":
                self.assertFalse({k.arg for k in node.keywords} & {"start_new_session", "process_group", "preexec_fn"})

    def test_inference_call_contract(self):
        tree = ast.parse((common.HERE / "pilot_worker.py").read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "inference_zero_shot"]
        self.assertEqual(len(calls), 1)
        kw = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords if k.arg in ("stream", "speed", "text_frontend")}
        self.assertEqual(kw, {"stream": False, "speed": 1.0, "text_frontend": False})
        source = (common.HERE / "pilot_worker.py").read_text()
        self.assertLess(source.index("bind_frontend(args.source)"), source.index("from cosyvoice.cli.cosyvoice import AutoModel"))
        self.assertLess(source.index('"token-contract.json"'), source.index("model.add_zero_shot_spk"))
        self.assertLess(source.index("random.seed(row["), source.index("model.inference_zero_shot"))
        self.assertNotIn("save_spkinfo(", source)
        self.assertNotIn("weights_only=False", source)

    def test_safety_environment(self):
        env = common.execution_env()
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["TORCH_FORCE_WEIGHTS_ONLY_LOAD"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_partial_failure_artifact_has_twelve_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = pilot.layout(Path(temp) / "job")
            paths["output"].mkdir(parents=True)
            paths["receipts"].mkdir(parents=True)
            common.write_json(paths["output"] / "clip-001.attempt.json", {"clip_id": "clip-001", "seed": 610201, "attempt": 1})
            (paths["output"] / "clip-001.native.npy").write_bytes(b"invalid partial")
            common.write_json(paths["receipts"] / "run-failure.json", {"status": "failed", "error": "timeout"})
            target = Path(temp) / "artifact"
            result = pilot.verify_output(paths, types.SimpleNamespace(artifact_dir=str(target)))
            self.assertEqual(result, "incomplete")
            summary = common.read_json(target / "pilot-outcomes.json")
            self.assertEqual(len(summary["outcomes"]), 12)
            self.assertEqual(summary["outcomes"][0]["status"], "attempted_failed")
            self.assertEqual(summary["outcomes"][1]["status"], "not_attempted")
            self.assertFalse((target / "clip-001.native.npy").exists())
            self.assertTrue((target / "run-failure.json").exists())

    def test_mismatched_id_cannot_publish_unvalidated_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = pilot.layout(Path(temp) / "job")
            paths["output"].mkdir(parents=True)
            paths["receipts"].mkdir(parents=True)
            common.write_json(paths["output"] / "clip-001.attempt.json", {"clip_id": "clip-001", "seed": 610201, "attempt": 1})
            common.write_json(paths["output"] / "clip-001.result.json", {"clip_id": "clip-002", "phrase_id": "p01", "condition": "control", "seed": 610201, "text": "你好小窝", "status": "generated"})
            (paths["output"] / "clip-001.wav").write_bytes(b"unvalidated bytes")
            target = Path(temp) / "artifact"
            with mock.patch.object(pilot, "validate_clip") as validate:
                pilot.verify_output(paths, types.SimpleNamespace(artifact_dir=str(target)))
                validate.assert_not_called()
            self.assertFalse((target / "clip-001.wav").exists())
            self.assertEqual(common.read_json(target / "pilot-outcomes.json")["outcomes"][0]["status"], "invalid_excluded")

    def test_private_reference_is_not_allowlisted(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = pilot.layout(Path(temp) / "job")
            paths["output"].mkdir(parents=True)
            (paths["output"] / "public-reference.wav").write_bytes(b"not permitted")
            with self.assertRaises(common.GateError):
                pilot.verify_output(paths, types.SimpleNamespace(artifact_dir=str(Path(temp) / "artifact")))

    def test_native_nonfinite_and_duration_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "clip.npy"
            for shape, sample in (((1,), float("nan")), ((480001,), 0.25)):
                header = repr({"descr": "<f4", "fortran_order": False, "shape": shape}).encode()
                path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<f", sample))
                with self.assertRaises(common.GateError):
                    pilot.read_native(path)


    def synthetic_supervisor(self, temp, body, limit=1024):
        root = Path(temp)
        (root / "pilot_worker.py").write_text(body)
        paths = pilot.layout(root / "job")
        with mock.patch.object(pilot, "HERE", root), mock.patch.object(pilot, "LOG_LIMIT", limit), mock.patch.object(pilot, "memory_gate"), mock.patch.object(pilot, "job_gate"), mock.patch.object(pilot, "process_tree_rss", return_value=0):
            return pilot.supervise(paths, "qualify")

    def test_supervisor_two_megabyte_burst_capped_with_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(common.GateError, "worker log exceeded"):
                self.synthetic_supervisor(temp, "import os; os.write(1, b'prefix:'+b'x'*(2*1024*1024))")
            paths = pilot.layout(Path(temp) / "job")
            self.assertEqual((paths["receipts"] / "qualify.log").read_bytes(), b"prefix:" + b"x" * (1024 - 7))
            receipt = common.read_json(paths["receipts"] / "qualify-supervisor.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertTrue(receipt["log_truncated"])
            self.assertEqual(receipt["log_bytes"], 1024)
            self.assertGreater(receipt["log_observed_bytes"], 1024)
            self.assertFalse(receipt["log_drain_timed_out"])
            self.assertTrue(receipt["no_retry"])

    def test_supervisor_normal_log_and_success_preserved(self):
        body = "import os, sys, json; from pathlib import Path; os.write(1, b'hello\\n'); os.write(2, b'error\\n'); Path(sys.argv[sys.argv.index('--result')+1]).write_text(json.dumps({'status':'qualified'}))"
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(self.synthetic_supervisor(temp, body), {"status": "qualified"})
            paths = pilot.layout(Path(temp) / "job")
            self.assertEqual((paths["receipts"] / "qualify.log").read_bytes(), b"hello\nerror\n")
            receipt = common.read_json(paths["receipts"] / "qualify-supervisor.json")
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["returncode"], 0)
            self.assertEqual(receipt["log_bytes"], 12)
            self.assertFalse(receipt["log_truncated"])
            self.assertTrue(receipt["log_eof"])

    def test_supervisor_normal_log_and_failure_exit_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(common.GateError, "worker exited 7"):
                self.synthetic_supervisor(temp, "import os; os.write(1, b'failure details\\n'); raise SystemExit(7)")
            paths = pilot.layout(Path(temp) / "job")
            self.assertEqual((paths["receipts"] / "qualify.log").read_bytes(), b"failure details\n")
            receipt = common.read_json(paths["receipts"] / "qualify-supervisor.json")
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["returncode"], 7)
            self.assertFalse(receipt["log_truncated"])
            self.assertTrue(receipt["log_eof"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
