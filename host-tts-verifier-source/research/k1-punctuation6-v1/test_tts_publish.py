"""Wrapper tests: synthetic fixed fixtures, current schemas and mocked HTTP only."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import tts_publish as w
import data_transport as t
import data_publisher as p
import tts_gate as gate
import tts_semantics as semantics
import test_data_publisher as protocol
import test_tts_semantics as synthetic


class NoEnvironment:
    def __getitem__(self, key): raise AssertionError("environment read before release")
    def __iter__(self): raise AssertionError("environment copied")


class CountingEnvironment(dict):
    def __init__(self, values): super().__init__(values); self.reads = []
    def __getitem__(self, key): self.reads.append(key); return super().__getitem__(key)
    def __iter__(self): raise AssertionError("must not iterate environment")


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("http.client.HTTPSConnection", side_effect=AssertionError("live network forbidden")))
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temp); self.here = self.root / "source"; self.here.mkdir()
        self.runner = self.root / "runner"; self.runner.mkdir()
        self.stack.enter_context(patch.object(w, "HERE", self.here))
        self.release = {"schema": "k1-punctuation6-publication-release-v1", "experiment": "k1-punctuation6-v1",
            "enabled": True, "purposes": {"fixture": True, "tts": True}, "independent_source_review": True,
            "root_publication_release": True, "prior_claim_history_sha256": "c" * 64, "commit_timestamp": "2026-10-12T00:00:00Z"}
        self.save_release()
        self.env, self.event, self.protocol_release, self.source_rows = protocol.setup()
        self.event_path = self.root / "event.json"
        self.env.update(GITHUB_EVENT_PATH=str(self.event_path), RUNNER_TEMP=str(self.runner), RUNNER_ENVIRONMENT="github-hosted",
            RUNNER_OS="Linux", RUNNER_ARCH="X64", KWS_DATA_TOKEN="offline-fixture-not-a-credential")
        self.event_path.write_bytes(w.canonical(self.event))
        self.api = None

    def save_release(self):
        (self.here / "publication-release.json").write_bytes(w.canonical(self.release))

    def ready(self, purpose="fixture", members=None):
        policy = t.policy(purpose)
        self.env.update(GITHUB_JOB=policy["job_key"], GITHUB_REF="refs/heads/" + policy["source_branch"])
        self.event["ref"] = self.env["GITHUB_REF"]
        self.protocol_release["purpose"] = purpose
        if purpose == "tts":
            self.env["GITHUB_RUN_ID"] = "701"
            runtime = self.runner / "k1-punctuation6-tts-v1"; runtime.mkdir()
            target = runtime / "publication-members"; target.mkdir()
            members = synthetic.archive(0) if members is None else copy.deepcopy(members)
            for name in ("claim.json", "source.json"):
                row = w.decode(members[name]); row["source_tree"] = self.protocol_release["source_tree"]
                members[name] = w.canonical(row)
            for name, raw in members.items():
                file = target / name; file.parent.mkdir(exist_ok=True, parents=True); file.write_bytes(raw)
            self.members = members
        self.event_path.write_bytes(w.canonical(self.event))
        self.stack.enter_context(patch.object(w, "components", return_value=(t, p, gate)))
        self.stack.enter_context(patch.object(gate, "verify_source", return_value={"source_freeze_sha256": "b" * 64, "frozen_files": 1}))
        self.stack.enter_context(patch.object(gate, "verify_release", return_value=({"enabled": True}, {"source_freeze_sha256": "b" * 64})))
        self.stack.enter_context(patch.object(w, "clean_checkout", return_value=self.protocol_release["source_tree"]))
        self.stack.enter_context(patch.object(semantics, "local_authorities", side_effect=synthetic.trusted_authorities))
        def api_factory(token, identity, release):
            self.assertEqual(token, "offline-fixture-not-a-credential")
            self.api = protocol.MockAPI(self.env, self.event, release, self.source_rows)
            self.api.run["id"] = identity["run_id"]
            self.api.job["run_id"] = identity["run_id"]
            self.api.job["id"] = 702 if purpose == "tts" else 456
            self.api.job["labels"] = ["ubuntu-24.04"]
            return self.api
        self.stack.enter_context(patch.object(p, "GitHubAPI", side_effect=api_factory))
        self.stack.enter_context(patch.object(p, "RawReader", side_effect=lambda commit, actual_purpose: self.api.raw_factory(commit, actual_purpose)))
        return CountingEnvironment(self.env)

    def test_disabled_gate_before_environment_components_or_source(self):
        self.release["enabled"] = False; self.save_release()
        with patch.object(w, "components", side_effect=AssertionError("components reached")):
            with self.assertRaises(w.Disabled): w.execute("fixture", NoEnvironment())
        self.assertEqual(list(self.runner.iterdir()), [])

    def test_bool_precise_purpose_gates_and_null_bindings(self):
        baseline = copy.deepcopy(self.release)
        for change in (lambda r: r.update(enabled=1), lambda r: r.update(enabled=None),
            lambda r: r["purposes"].update(fixture=False), lambda r: r["purposes"].update(fixture=1),
            lambda r: r.update(prior_claim_history_sha256=None), lambda r: r.update(commit_timestamp=None),
            lambda r: r.update(root_publication_release=1), lambda r: r.update(independent_source_review=False),
            lambda r: r["purposes"].update(extra=True), lambda r: r.update(extra="private")):
            self.release = copy.deepcopy(baseline); change(self.release); self.save_release()
            with self.assertRaises((ValueError, w.Disabled)): w.execute("fixture", NoEnvironment())
        self.assertEqual(list(self.runner.iterdir()), [])

    def test_source_failure_before_environment_token_or_runtime(self):
        with patch.object(w, "components", return_value=(t, p, gate)), patch.object(gate, "verify_source", side_effect=ValueError("private-path")):
            with self.assertRaises(ValueError): w.execute("fixture", NoEnvironment())
        self.assertEqual(list(self.runner.iterdir()), [])

    def test_fixture_full_mock_roundtrip_and_token_is_last_env_read(self):
        env = self.ready()
        result = w.execute("fixture", env)
        self.assertEqual(result["status"], "same_identity_readback_verified")
        self.assertLess(result["archive_bytes"], 3072)
        self.assertEqual(result["raw_gets"], 2)
        self.assertEqual(env.reads[-1], "KWS_DATA_TOKEN")
        self.assertEqual(env.reads.count("KWS_DATA_TOKEN"), 1)
        self.assertTrue((self.runner / "k1-punctuation6-writer-fixture-r1/publication.receipt.json").is_file())
        self.assertNotIn("tts_worker", w.execute.__code__.co_names)

    def test_fixture_never_imports_semantics_or_worker(self):
        env = self.ready()
        with patch.object(semantics, "validate_tts_members", side_effect=AssertionError("TTS validator touched")):
            result = w.execute("fixture", env)
        self.assertEqual(result["purpose"], "fixture")

    def test_tts_complete_partial_and_zero_call_mock_publication(self):
        # Separate runners avoid recreating any once-only identity locally.
        env = self.ready("tts", synthetic.archive(2, failed=True, partial_audio=True))
        result = w.execute("tts", env)
        self.assertEqual(result["status"], "same_identity_readback_verified")
        self.assertEqual(result["numeric_job_id"], 702)
        receipt = w.decode((self.runner / "k1-punctuation6-tts-v1/publication.receipt.json").read_bytes())
        self.assertFalse(receipt["retry_authorized"])
        self.assertEqual(env.reads[-1], "KWS_DATA_TOKEN")

    def test_tts_zero_call_receipt_retained(self):
        env = self.ready("tts", synthetic.archive(0))
        result = w.execute("tts", env)
        self.assertEqual(result["purpose"], "tts")
        self.assertFalse(any(name.startswith(("raw/", "derived/", "claims/")) for name in self.members))

    def test_tts_all_six_complete_receipt_retained(self):
        env = self.ready("tts", synthetic.archive(6))
        result = w.execute("tts", env)
        self.assertEqual(result["purpose"], "tts")
        self.assertEqual(len(self.members), 23)

    def test_tts_extra_file_rejected_before_token_or_api(self):
        env = self.ready("tts")
        root = self.runner / "k1-punctuation6-tts-v1/publication-members"
        (root / "private.log").write_text("secret")
        with self.assertRaises(ValueError): w.execute("tts", env)
        self.assertNotIn("KWS_DATA_TOKEN", env.reads)
        self.assertIsNone(self.api)

    def test_tts_symlink_hardlink_or_private_parent_blocked(self):
        env = self.ready("tts")
        root = self.runner / "k1-punctuation6-tts-v1/publication-members"
        original = (root / "eos.json").read_bytes(); (root / "eos.json").unlink()
        other = self.root / "private"; other.write_bytes(original)
        (root / "eos.json").symlink_to(other)
        with self.assertRaises((ValueError, OSError)): w.execute("tts", env)
        (root / "eos.json").unlink(); os.link(other, root / "eos.json")
        with self.assertRaises(ValueError): w.execute("tts", env)
        self.assertNotIn("KWS_DATA_TOKEN", env.reads)

    def test_tts_wrong_original_identity_or_source_blocks_before_token(self):
        env = self.ready("tts")
        root = self.runner / "k1-punctuation6-tts-v1/publication-members"
        row = w.decode((root / "claim.json").read_bytes()); row["source_tree"] = "f" * 40
        (root / "claim.json").write_bytes(w.canonical(row))
        with self.assertRaises(ValueError): w.execute("tts", env)
        self.assertNotIn("KWS_DATA_TOKEN", env.reads)

    def test_real_semantic_schema_rejects_untruthful_outcome_before_token(self):
        env = self.ready("tts")
        root = self.runner / "k1-punctuation6-tts-v1/publication-members"
        row = w.decode((root / "outcomes.json").read_bytes()); row["consumed_attempts"] = 6
        (root / "outcomes.json").write_bytes(w.canonical(row))
        with self.assertRaises(ValueError): w.execute("tts", env)
        self.assertNotIn("KWS_DATA_TOKEN", env.reads)

    def test_tts_execution_release_false_blocks_before_environment_or_members(self):
        self.stack.enter_context(patch.object(w, "components", return_value=(t, p, gate)))
        self.stack.enter_context(patch.object(gate, "verify_source", return_value={"source_freeze_sha256": "b" * 64}))
        self.stack.enter_context(patch.object(gate, "verify_release", side_effect=ValueError("execution disabled")))
        with self.assertRaises(ValueError): w.execute("tts", NoEnvironment())
        self.assertEqual(list(self.runner.iterdir()), [])

    def test_actual_runner_label_mismatch_stops_before_any_post(self):
        env = self.ready()
        original_factory = p.GitHubAPI.side_effect
        def wrong_label(*args):
            api = original_factory(*args); api.job["labels"] = ["self-hosted", "Linux"]; return api
        p.GitHubAPI.side_effect = wrong_label
        with self.assertRaises(ValueError): w.execute("fixture", env)
        self.assertFalse(any(call[0] == "POST" for call in self.api.calls))

    def test_original_numeric_job_binding_prevents_any_post(self):
        env = self.ready("tts")
        original_factory = p.GitHubAPI.side_effect
        def wrong_job(*args):
            api = original_factory(*args); api.job["id"] = 703; return api
        p.GitHubAPI.side_effect = wrong_job
        with self.assertRaises(ValueError) as caught: w.execute("tts", env)
        self.assertEqual(caught.exception.publication_receipt["status"], "publication_failed_or_unconfirmed_no_retry")
        self.assertFalse(any(call[0] == "POST" for call in self.api.calls))

    def test_uncertain_writer_failure_keeps_receipt_and_no_second_attempt(self):
        env = self.ready()
        original_factory = p.GitHubAPI.side_effect
        def failing(*args):
            api = original_factory(*args)
            api.mutations[("POST", p.BASE + "/git/refs", 201)] = p.PublicationError(503)
            return api
        p.GitHubAPI.side_effect = failing
        with self.assertRaises(p.PublicationError) as caught: w.execute("fixture", env)
        self.assertFalse(caught.exception.publication_receipt["new_generation_authorized"])
        self.assertTrue((self.runner / "k1-punctuation6-writer-fixture-r1/publication.receipt.json").exists())
        with self.assertRaises(FileExistsError): w.execute("fixture", env)

    def test_real_main_mock_failure_emits_preserved_safe_terminal(self):
        env = self.ready()
        original_factory = p.GitHubAPI.side_effect
        def fail(*args):
            api = original_factory(*args)
            api.mutations[("POST", p.BASE + "/git/refs", 201)] = p.PublicationError(403)
            return api
        p.GitHubAPI.side_effect = fail
        stream = io.StringIO()
        with patch.object(w.os, "environ", env), redirect_stdout(stream): code = w.main(["--purpose", "fixture"])
        result = json.loads(stream.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(result["publisher_stage"], "create_ref")
        self.assertEqual(result["http_status"], 403)
        self.assertEqual(result["identity"]["run_id"], 123)
        self.assertFalse(result["retry_authorized"])
        self.assertNotIn("offline-fixture-not-a-credential", stream.getvalue())
        self.assertNotIn("Traceback", stream.getvalue())

    def test_real_main_mock_success_emits_only_bounded_custody(self):
        env = self.ready("tts", synthetic.archive(2, failed=True, partial_audio=True))
        stream = io.StringIO()
        with patch.object(w.os, "environ", env), redirect_stdout(stream): code = w.main(["--purpose", "tts"])
        result = json.loads(stream.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "same_identity_readback_verified")
        self.assertEqual(result["numeric_job_id"], 702)
        self.assertTrue(result["semantic_inference_success_not_established_by_publisher"])
        self.assertLess(len(stream.getvalue().encode()), 16384)
        for forbidden in ("你好", "RIFF", "offline-fixture-not-a-credential", "Traceback"):
            self.assertNotIn(forbidden, stream.getvalue())

    def test_clean_git_subprocess_cannot_inherit_token(self):
        results = [type("Result", (), {"stdout": "a" * 40 + "\n"})(), type("Result", (), {"stdout": ""})(), type("Result", (), {"stdout": "b" * 40 + "\n"})()]
        with patch.dict(os.environ, {"KWS_DATA_TOKEN": "DO-NOT-PASS", "GITHUB_TOKEN": "DO-NOT-PASS"}), patch("subprocess.run", side_effect=results) as run:
            self.assertEqual(w.clean_checkout({"executed_head_sha": "a" * 40}), "b" * 40)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["env"], w.clean_git_env())
            self.assertNotIn("KWS_DATA_TOKEN", call.kwargs["env"])
            self.assertNotIn("GITHUB_TOKEN", call.kwargs["env"])
            self.assertEqual(call.kwargs["timeout"], 15)
            self.assertTrue(call.kwargs["capture_output"])

    def test_stdout_only_safe_terminal_no_traceback_private_text_or_token(self):
        error = p.PublicationError(503)
        error.publication_receipt = {"stage": "create_ref", "http_status": 503, "archive_sha256": "b" * 64,
            "error_type": "/private/home/API-KEY", "contents": "SECRET-WAV", "token": "FAKE-CREDENTIAL",
            "expected_result_commit": "d" * 40, "publication_confirmed": False, "readback_confirmed": False}
        stream = io.StringIO()
        with patch.object(w, "execute", side_effect=error), redirect_stdout(stream): code = w.main(["--purpose", "fixture"])
        raw = stream.getvalue(); result = json.loads(raw)
        self.assertEqual(code, 2); self.assertEqual(len(raw.splitlines()), 1)
        self.assertLess(len(raw.encode()), w.MAX_TERMINAL)
        self.assertEqual(result["publisher_stage"], "create_ref")
        self.assertEqual(result["http_status"], 503)
        for banned in ("private", "SECRET-WAV", "FAKE-CREDENTIAL", "Traceback", "contents", "token"):
            self.assertNotIn(banned, raw)

    def test_sanitizer_rejects_malformed_receipt_types(self):
        raw = w.safe_receipt({"stage": [], "identity": {}, "archive_bytes": True, "raw_gets": 100,
            "archive_sha256": "private-path", "http_status": "secret", "expected_result_tree": "z" * 40}, "fixture")
        self.assertEqual(raw, {})

    def test_disabled_real_cli_normal_optimized_double_optimized(self):
        # Exact wrapper bytes, deliberately without sibling modules: disabled gate
        # must execute under isolated Python, not fail at import resolution.
        script = self.here / "tts_publish.py"
        script.write_bytes(Path(w.__file__).read_bytes())
        self.release["enabled"] = False; self.save_release()
        for optimization in ([], ["-O"], ["-OO"]):
            result = subprocess.run([sys.executable, "-I", "-B", *optimization, str(script), "--purpose", "fixture"],
                env=w.clean_git_env(), capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stderr, "")
            terminal = json.loads(result.stdout)
            self.assertEqual(terminal["status"], "disabled")
            self.assertEqual(terminal["stage"], "release_gate")
            self.assertNotIn("ImportError", result.stdout)

    def test_invalid_cli_argument_is_not_echoed(self):
        stream = io.StringIO()
        with redirect_stdout(stream): code = w.main(["private-value"])
        self.assertEqual(code, 2); self.assertNotIn("private-value", stream.getvalue())
        self.assertIsNone(json.loads(stream.getvalue())["purpose"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
