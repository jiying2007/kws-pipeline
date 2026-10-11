"""Mock-HTTP-only trusted publisher tests. No live network or real credentials."""
import base64
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import branch_transport as b
import evidence_publisher as p
import log_transport as t

IDENTITY = {"repository": b.REPOSITORY, "run_id": 123, "run_attempt": 1,
    "job_key": "qwen12", "executed_head_sha": "a" * 40}


def fixtures(root, purpose="native"):
    policy = b.purpose_policy(purpose)
    env = {"GITHUB_REPOSITORY": b.REPOSITORY, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1", "GITHUB_SHA": "a" * 40, "GITHUB_REF": "refs/heads/" + policy["source_branch"],
        "GITHUB_JOB": policy["job_key"], "RUNNER_TEMP": str(root)}
    event = {"repository": {"full_name": b.REPOSITORY, "private": False}, "created": True, "deleted": False,
             "after": "a" * 40, "ref": env["GITHUB_REF"]}
    return env, event


class MockAPI:
    def __init__(self, env, event, purpose="native"):
        self.identity = p.context(env, event, purpose)
        self.policy = b.purpose_policy(purpose)
        self.paths = p.endpoints(self.identity, purpose)
        self.calls = []; self.blobs = {}; self.modify = {}
        self.repo = {"full_name": b.REPOSITORY, "private": False}
        self.run = {"id": 123, "run_attempt": 1, "head_sha": "a" * 40,
            "head_branch": self.policy["source_branch"], "event": "push", "path": p.WORKFLOW,
            "status": "in_progress", "repository": self.repo}
        self.job = {"id": 456, "run_id": 123, "head_sha": "a" * 40,
            "head_branch": self.policy["source_branch"], "name": self.policy["job_key"],
            "run_url": "https://api.github.com" + self.paths["run"], "status": "in_progress", "conclusion": None}
        self.jobs = {"total_count": 1, "jobs": [self.job]}
        self.source = {"sha": "a" * 40, "tree": {"sha": "b" * 40}}
        self.tree_rows = None

    def __call__(self, method, path, body=None, expected=200):
        self.calls.append((method, path, copy.deepcopy(body), expected))
        if path in self.modify:
            modifier = self.modify[path]
            if isinstance(modifier, BaseException): raise modifier
            return modifier(method, path, body, expected)
        if method == "GET":
            if path == self.paths["repo"]: return self.repo
            if path == self.paths["run"]: return self.run
            if path == self.paths["jobs"]: return self.jobs
            if path == self.paths["source"]: return self.source
            if path in (self.paths["ref"], self.paths["directory"]):
                if expected != 404: raise AssertionError("absence must require404")
                return None
        if method == "POST" and path == p.BASE + "/git/blobs":
            raw = base64.b64decode(body["content"], validate=True); digest = p.git_sha(raw)
            self.blobs[digest] = raw
            return {"sha": digest}
        if method == "POST" and path == p.BASE + "/git/trees":
            self.tree_rows = body["tree"]
            return {"sha": "c" * 40}
        if method == "POST" and path == p.BASE + "/git/commits":
            return {"sha": "d" * 40, "tree": {"sha": body["tree"]}, "parents": [{"sha": h} for h in body["parents"]]}
        if method == "POST" and path == p.BASE + "/git/refs":
            return {"ref": body["ref"], "object": {"type": "commit", "sha": body["sha"]}}
        raise AssertionError("unexpected mock route")


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / "terminal-outcomes.json").write_bytes(b'{"status":"partial_failure","completed":0}')
        self.payload, _, _ = t.build_package(self.root, ["terminal-outcomes.json"], IDENTITY)
        self.package = b.write_package(self.root, self.payload, IDENTITY, "native")
        self.env, self.event = fixtures(self.root)
        self.api = MockAPI(self.env, self.event)

    def tearDown(self): self.temp.cleanup()

    def call(self): return p.run_once(self.env, self.event, "native", self.api)

    def test_exact_atomic_create_after_all_validations_partial_native_preserved(self):
        result = self.call()
        self.assertEqual(result["status"], "evidence_branch_created")
        self.assertTrue(result["native_completion_not_established"])
        self.assertEqual(result["numeric_job_id"], 456)
        self.assertEqual(result["evidence_commit"], "d" * 40)
        self.assertEqual(self.api.calls[-1], ("POST", p.BASE + "/git/refs",
            {"ref": "refs/heads/evidence/qwen12-native-admission-recovery3", "sha": "d" * 40}, 201))
        self.assertEqual([method for method, _, _, _ in self.api.calls[:6]], ["GET"] * 6)
        tree_call = next(c for c in self.api.calls if c[1] == p.BASE + "/git/trees")
        self.assertEqual(tree_call[2]["base_tree"], "b" * 40)
        commit_call = next(c for c in self.api.calls if c[1] == p.BASE + "/git/commits")
        self.assertEqual(commit_call[2]["parents"], ["a" * 40])
        self.assertEqual({r["path"] for r in self.api.tree_rows}, {
            b.manifest_path("native"), b.chunk_path("native", 0), p.publication_path("native")})
        self.assertFalse(any(method in ("PATCH", "PUT", "DELETE") for method, _, _, _ in self.api.calls))
        publication = next(r for r in self.api.tree_rows if r["path"] == p.publication_path("native"))
        raw = self.api.blobs[publication["sha"]]
        outer = b.read_package(self.root, IDENTITY, "native")[0][b.manifest_path("native")]
        self.assertEqual(p.validate_publication(raw, IDENTITY, "native", 456, outer)["job_status_at_publication"], "in_progress")
        with self.assertRaises(ValueError): p.validate_publication(raw, IDENTITY, "native", 457, outer)
        with self.assertRaises(ValueError): p.validate_publication(raw, IDENTITY, "native", True, outer)
        with self.assertRaises(ValueError): p.validate_publication(raw, IDENTITY, "native", 456, outer + b" ")

    def test_never_retry_even_after_success(self):
        self.call(); before = len(self.api.calls)
        with self.assertRaises(FileExistsError): self.call()
        self.assertEqual(before, len(self.api.calls))

    def test_fresh_push_exact_identity_required_before_any_api(self):
        changes = [("env", "GITHUB_REPOSITORY", "someone/fork"), ("env", "GITHUB_EVENT_NAME", "pull_request"),
            ("env", "GITHUB_RUN_ID", "00123"), ("env", "GITHUB_RUN_ID", "True"),
            ("env", "GITHUB_RUN_ATTEMPT", "2"), ("env", "GITHUB_JOB", "other"),
            ("env", "GITHUB_REF", "refs/heads/main"), ("event", "created", 1),
            ("event", "created", False), ("event", "deleted", True), ("event", "after", "e" * 40),
            ("event", "ref", "refs/heads/other")]
        for target, key, value in changes:
            env, event = copy.deepcopy(self.env), copy.deepcopy(self.event)
            (env if target == "env" else event)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): p.publish(env, event, "native", self.api)
        self.assertFalse(self.api.calls)
        for value in [True, 0, None]:
            event = copy.deepcopy(self.event); event["repository"]["private"] = value
            with self.assertRaises(ValueError): p.publish(self.env, event, "native", self.api)

    def test_consumed_and_mixed_source_refs_rejected_before_any_api(self):
        for old in ("research/qwen12-native-admission-v1",
                    "research/qwen12-native-admission-recovery1",
                    "research/qwen12-native-admission-recovery2"):
            for part in ("env", "event", "both"):
                env,event=copy.deepcopy(self.env),copy.deepcopy(self.event)
                if part in ("env", "both"):env["GITHUB_REF"]="refs/heads/"+old
                if part in ("event", "both"):event["ref"]="refs/heads/"+old
                with self.subTest(branch=old,part=part),self.assertRaises(ValueError):
                    p.publish(env,event,"native",self.api)
        self.assertFalse(self.api.calls)

    def test_consumed_run_or_job_head_branch_rejected_before_writes(self):
        for old in ("research/qwen12-native-admission-v1",
                    "research/qwen12-native-admission-recovery1",
                    "research/qwen12-native-admission-recovery2"):
            for part in ("run", "job", "both"):
                api=MockAPI(self.env,self.event)
                if part in ("run", "both"):api.run["head_branch"]=old
                if part in ("job", "both"):api.job["head_branch"]=old
                with self.subTest(branch=old,part=part),self.assertRaises(ValueError):
                    p.publish(self.env,self.event,"native",api)
                self.assertFalse(any(call[0]=="POST" for call in api.calls))

    def test_consumed_evidence_endpoints_are_not_authorized(self):
        api=p.GitHubAPI("public-test-fixture-not-a-credential",IDENTITY,"native")
        paths=p.endpoints(IDENTITY,"native")
        self.assertEqual(paths["ref"],p.BASE+"/git/ref/heads/evidence/qwen12-native-admission-recovery3")
        self.assertEqual(paths["directory"],p.BASE+"/contents/research/qwen12-native-admission-v1/evidence/recovery3?ref="+"a"*40)
        for old,directory in (("v1","native-v1"),("recovery1","recovery1"),("recovery2","recovery2")):
            for path in (p.BASE+"/git/ref/heads/evidence/qwen12-native-admission-"+old,
                         p.BASE+"/contents/research/qwen12-native-admission-v1/evidence/"+directory+"?ref="+"a"*40):
                with self.subTest(path=path),patch.object(p.http.client,"HTTPSConnection") as connection:
                    with self.assertRaises(ValueError):api("GET",path,expected=404)
                    connection.assert_not_called()

    def test_api_repository_run_and_job_identity_negative_cases(self):
        cases = [("repo", "private", True), ("repo", "full_name", "other/repo"),
            ("run", "id", True), ("run", "id", 124), ("run", "run_attempt", True),
            ("run", "event", "pull_request"), ("run", "path", "another.yml"),
            ("run", "head_sha", "e" * 40), ("run", "head_branch", "main"),
            ("run", "status", "completed"), ("job", "run_id", 124), ("job", "run_attempt", 2),
            ("job", "run_attempt", True), ("job", "head_sha", "e" * 40), ("job", "id", True),
            ("job", "name", "qwen12 (matrix)"), ("job", "run_url", "https://evil.invalid/run"),
            ("job", "conclusion", "success"), ("job", "status", "completed")]
        for target, key, value in cases:
            api = MockAPI(self.env, self.event); getattr(api, target)[key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                p.publish(self.env, self.event, "native", api)
            self.assertFalse(any(c[0] == "POST" for c in api.calls))

    def test_ambiguous_missing_or_unbounded_jobs_rejected(self):
        for jobs in [{"total_count": 2, "jobs": [self.api.job, self.api.job]}, {"total_count": 0, "jobs": []},
            {"total_count": True, "jobs": [self.api.job]}, {"total_count": 101, "jobs": [self.api.job]},
            {"total_count": 2, "jobs": [self.api.job]}]:
            api = MockAPI(self.env, self.event); api.jobs = jobs
            with self.assertRaises(ValueError): p.publish(self.env, self.event, "native", api)
            self.assertFalse(any(c[0] == "POST" for c in api.calls))

    def test_existing_branch_or_directory_fail_without_writes(self):
        for key in ["ref", "directory"]:
            api = MockAPI(self.env, self.event); api.modify[api.paths[key]] = p.PublicationError(200)
            with self.assertRaises(p.PublicationError): p.publish(self.env, self.event, "native", api)
            self.assertFalse(any(c[0] == "POST" for c in api.calls))

    def test_conflict_uncertain_response_no_retry_preserves_small_receipt(self):
        for error in [p.PublicationError(422), p.PublicationError(), RuntimeError("secret header payload must never be saved")]:
            with tempfile.TemporaryDirectory() as root:
                b.write_package(root, self.payload, IDENTITY, "native")
                env, event = fixtures(root); api = MockAPI(env, event)
                api.modify[p.BASE + "/git/refs"] = error
                with self.assertRaises(type(error)): p.run_once(env, event, "native", api)
                self.assertEqual(sum(c[1] == p.BASE + "/git/refs" for c in api.calls), 1)
                before = len(api.calls)
                with self.assertRaises(FileExistsError): p.run_once(env, event, "native", api)
                self.assertEqual(len(api.calls), before)
                receipt = b.package_path(root, IDENTITY, "native").parent / "evidence-publication-receipt.json"
                raw = receipt.read_bytes(); self.assertNotIn(b"secret", raw)
                self.assertEqual(t.decode(raw)["status"], "publication_failed_or_unconfirmed_no_retry")
                self.assertEqual(b.read_package(root, IDENTITY, "native")[1], self.payload)

    def test_missing_invalid_package_never_posts(self):
        (self.package / "chunks/0000.bin").write_bytes(b"corrupt")
        with self.assertRaises(ValueError): self.call()
        self.assertFalse(self.api.calls)
        self.assertTrue((self.package.parent / "evidence-publication-receipt.json").is_file())

    def test_git_api_return_binding_before_ref(self):
        for route, value in [("blobs", {"sha": "f" * 40}), ("trees", {"sha": True}),
            ("commits", {"sha": "d" * 40, "tree": {"sha": "f" * 40}, "parents": [{"sha": "a" * 40}]}),
            ("commits", {"sha": "d" * 40, "tree": {"sha": "c" * 40}, "parents": [{"sha": "f" * 40}]})]:
            api = MockAPI(self.env, self.event); api.modify[p.BASE + "/git/" + route] = lambda *_, value=value: value
            with self.assertRaises(ValueError): p.publish(self.env, self.event, "native", api)
            self.assertFalse(any(c[1] == p.BASE + "/git/refs" for c in api.calls))

    def test_wrong_create_ref_response_never_reports_success(self):
        self.api.modify[p.BASE + "/git/refs"] = lambda *_: {"ref": "refs/heads/main", "object": {"type": "commit", "sha": "d" * 40}}
        with self.assertRaises(ValueError): self.call()
        receipt = t.decode((self.package.parent / "evidence-publication-receipt.json").read_bytes())
        self.assertEqual(receipt["evidence_commit"], "d" * 40)
        self.assertFalse(receipt["publication_confirmed"])
        self.assertEqual(receipt["stage"], "create_ref")
        self.assertEqual(receipt["status"], "publication_failed_or_unconfirmed_no_retry")

    def test_synthetic_rejects_relabeled_native_payload_before_api(self):
        env, event = fixtures(self.root, "synthetic")
        identity = p.context(env, event, "synthetic")
        payload, _, _ = t.build_package(self.root, ["terminal-outcomes.json"], identity)
        b.write_package(self.root, payload, identity, "synthetic")
        api = MockAPI(env, event, "synthetic")
        with self.assertRaises(ValueError): p.publish(env, event, "synthetic", api)
        self.assertFalse(api.calls)

    def test_synthetic_exact_near_cap_mock_publication(self):
        from emit_synthetic_branch_fixture import build_synthetic_package
        env, event = fixtures(self.root, "synthetic"); identity = p.context(env, event, "synthetic")
        payload, _ = build_synthetic_package(identity)
        b.write_package(self.root, payload, identity, "synthetic")
        api = MockAPI(env, event, "synthetic")
        result = p.run_once(env, event, "synthetic", api)
        self.assertEqual(result["file_count"], 34)
        self.assertEqual(result["evidence_branch"], "evidence/qwen12-native-transport-synthetic-v1")
        self.assertEqual(sum(c[1] == p.BASE + "/git/blobs" for c in api.calls), 34)

    def test_synthetic12m_full_48_chunk_mock_publication_and_readback(self):
        from emit_synthetic_branch_fixture import build_synthetic_package, verify_synthetic
        env, event = fixtures(self.root, "synthetic12m")
        identity = p.context(env, event, "synthetic12m")
        payload, _ = build_synthetic_package(identity, "synthetic12m")
        b.write_package(self.root, payload, identity, "synthetic12m")
        api = MockAPI(env, event, "synthetic12m")
        result = p.run_once(env, event, "synthetic12m", api)
        self.assertEqual(result["file_count"], 50)
        self.assertEqual(result["evidence_branch"], "evidence/qwen12-native-transport-synthetic-12m-v1")
        self.assertEqual(sum(c[1] == p.BASE + "/git/blobs" for c in api.calls), 50)
        files = {row["path"]: api.blobs[row["sha"]] for row in api.tree_rows}
        outer = files[b.manifest_path("synthetic12m")]
        manifest = b.parse_manifest(outer, identity, "synthetic12m")
        self.assertEqual(manifest["chunk_count"], 48)
        self.assertLessEqual(len(outer), b.MAX_MANIFEST)
        raw, _, _ = b.reassemble_file_set(outer,
            [(row["path"], files[row["path"]]) for row in manifest["chunks"]],
            identity, "synthetic12m", evidence_commit="d" * 40)
        self.assertEqual(raw, payload)
        verify_synthetic(raw, identity, "synthetic12m")
        p.validate_publication(files[p.publication_path("synthetic12m")],
            identity, "synthetic12m", 456, outer)

    def test_synthetic_scopes_and_payloads_cannot_be_relabelled(self):
        from emit_synthetic_branch_fixture import build_synthetic_package
        for original, other in [("synthetic", "synthetic12m"), ("synthetic12m", "synthetic")]:
            with self.subTest(original=original), tempfile.TemporaryDirectory() as directory:
                env, event = fixtures(directory, original)
                with self.assertRaises(ValueError): p.context(env, event, other)
                original_identity = p.context(env, event, original)
                payload, _ = build_synthetic_package(original_identity, original)
                # Both fixtures intentionally use the same job-key. The different
                # seed/member recipe still must fail before any remote API call.
                env, event = fixtures(directory, other)
                identity = p.context(env, event, other)
                b.write_package(directory, payload, identity, other)
                api = MockAPI(env, event, other)
                with self.assertRaises(ValueError): p.publish(env, event, other, api)
                self.assertFalse(api.calls)

    def test_freeze_whitelist_includes_actual_tools_and_workflow(self):
        import campaign_contract as c
        freeze = t.decode((p.HERE / "execution-freeze.json").read_bytes())
        for name in ["branch_transport.py", "evidence_publisher.py", "emit_synthetic_branch_fixture.py"]:
            freeze["host_code"]["research/qwen12-native-admission-v1/" + name] = c.identity(p.HERE / name)
        # Rebase changed files only inside this in-memory test fixture; production
        # integration regenerates and reviews the real committed source freeze.
        for name in freeze["host_code"]: freeze["host_code"][name] = c.identity(p.ROOT / name)
        release = {"source_freeze_sha256": c.identity(p.HERE / "execution-freeze.json")["sha256"],
                   "evidence_transport": b.TRANSPORT, "new_transcripts_publication_approved": True}
        def read(path): return freeze if path.name == "execution-freeze.json" else release
        with patch.object(c, "read_json", side_effect=read): p.frozen_sources()
        for val in [False, 1, "true"]:
            release["new_transcripts_publication_approved"] = val
            with patch.object(c, "read_json", side_effect=read), self.assertRaises(ValueError): p.frozen_sources()


class HTTPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.api = p.GitHubAPI("public-test-fixture-not-a-credential", IDENTITY, "native")

    def test_method_host_path_and_redirect_routes_cannot_be_selected(self):
        for method, path in [("PATCH", p.BASE + "/git/refs"), ("DELETE", p.BASE + "/git/refs"),
            ("GET", "https://evil.invalid"), ("POST", "/repos/other/repo/git/refs"),
            ("GET", p.BASE + "/git/ref/heads/main")]:
            with patch.object(p.http.client, "HTTPSConnection") as connection, self.assertRaises(ValueError):
                self.api(method, path)
            connection.assert_not_called()

    def invoke_response(self, status, raw):
        class Response:
            def __init__(self): self.status = status
            def read(self, cap):
                if cap != p.MAX_RESPONSE + 1: raise AssertionError("unbounded response read")
                return raw[:cap]
        class Connection:
            def request(self, method, path, **kwargs): pass
            def getresponse(self): return Response()
            def close(self): pass
        with patch.object(p.http.client, "HTTPSConnection", return_value=Connection()) as conn:
            result = self.api("GET", p.BASE)
            self.assertEqual(conn.call_args.args, ("api.github.com",))
            self.assertLessEqual(conn.call_args.kwargs["timeout"], 15)
            return result

    def test_http_fixed_host_valid_json(self): self.assertEqual(self.invoke_response(200, b'{}'), {})

    def test_redirect_error_and_response_oversize_not_followed_or_leaked(self):
        for status, raw in [(302, b'secret redirect'), (403, b'secret credential body'),
                            (200, b'x' * (p.MAX_RESPONSE + 1)), (200, b'invalid secret body')]:
            with self.subTest(status=status), self.assertRaises(p.PublicationError) as caught:
                self.invoke_response(status, raw)
            self.assertNotIn("secret", str(caught.exception))

    def test_timeout_and_transport_errors_sanitized_no_retry(self):
        with patch.object(p.http.client, "HTTPSConnection") as connection:
            connection.return_value.request.side_effect = RuntimeError("sensitive token request representation")
            with self.assertRaises(p.PublicationError) as caught: self.api("GET", p.BASE)
            self.assertNotIn("sensitive", str(caught.exception)); self.assertIsNone(caught.exception.status)
            self.assertEqual(connection.return_value.request.call_count, 1)

    def test_request_size_and_global_deadline(self):
        with patch.object(p.http.client, "HTTPSConnection") as connection:
            with self.assertRaises(ValueError): self.api("POST", p.BASE + "/git/blobs", {"content": "x" * p.MAX_REQUEST}, 201)
            self.api._deadline = 0
            with self.assertRaises(ValueError): self.api("GET", p.BASE)
            connection.assert_not_called()


if __name__ == "__main__": unittest.main(verbosity=2)
