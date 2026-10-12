"""Offline synthetic/mutation tests. Network is denied by default in every test."""
import base64
import copy
import io
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import data_transport as t
import data_publisher as p


def tree_response(rows):
    digest, directories = t.tree_identity(rows)
    return {"sha": digest, "truncated": False, "tree": [*copy.deepcopy(rows),
        *[{"path": name, "mode": "040000", "type": "tree", "sha": value} for name, value in sorted(directories.items())]]}


def setup(purpose="fixture"):
    source = [{"path": ".github/workflows/k1-punctuation6-tts-v1.yml", "type": "blob", "mode": "100644", "sha": t.git_sha(b"disabled")},
              {"path": "README.md", "type": "blob", "mode": "100644", "sha": t.git_sha(b"public fixture repository") }]
    release = {"schema": "k1-publisher-testable-release-v1", "purpose": purpose,
        "source_sha": "a" * 40, "source_tree": tree_response(source)["sha"],
        "source_closure_sha256": "b" * 64, "prior_claim_history_sha256": "c" * 64,
        "commit_timestamp": "2026-10-12T00:00:00Z"}
    policy = t.policy(purpose)
    env = {"GITHUB_REPOSITORY": t.REPOSITORY, "GITHUB_EVENT_NAME": "push", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "123", "GITHUB_SHA": release["source_sha"], "GITHUB_JOB": policy["job_key"],
        "GITHUB_REF": "refs/heads/" + policy["source_branch"]}
    event = {"repository": {"full_name": t.REPOSITORY, "private": False}, "ref": env["GITHUB_REF"],
             "created": True, "deleted": False, "after": release["source_sha"]}
    return env, event, release, source


class MockAPI:
    def __init__(self, env, event, release, source):
        self.identity = p.context(env, event, release); self.release = release
        self.paths = p.endpoints(self.identity, release); self.policy = t.policy(release["purpose"])
        self.calls = []; self.blobs = {}; self.source = source; self.mutations = {}; self.ref = None
        self.repo = {"full_name": t.REPOSITORY, "private": False}
        self.run = {"id": 123, "run_attempt": 1, "head_sha": release["source_sha"], "head_branch": self.policy["source_branch"],
            "event": "push", "path": self.policy["workflow"], "status": "in_progress", "repository": self.repo}
        self.history = {"total_count": 1, "workflow_runs": [self.run]}
        self.job = {"id": 456, "run_id": 123, "head_sha": release["source_sha"], "head_branch": self.policy["source_branch"],
            "name": self.policy["job_key"], "run_url": "https://api.github.com" + self.paths["run"], "status": "in_progress", "conclusion": None}
        self.jobs = {"total_count": 1, "jobs": [self.job]}

    def bind_result(self, commit, tree):
        self.expected_commit = commit; self.expected_tree = tree

    def __call__(self, method, route, body=None, expected=200):
        self.calls.append((method, route, copy.deepcopy(body), expected))
        mutation = self.mutations.get((method, route, expected))
        if isinstance(mutation, BaseException): raise mutation
        if mutation is not None: return mutation(method, route, body, expected)
        if method == "GET":
            if route == self.paths["repo"]: return self.repo
            if route == self.paths["run"]: return self.run
            if route == self.paths["history"]: return self.history
            if route == self.paths["jobs"]: return self.jobs
            if route == self.paths["source"]: return {"sha": self.release["source_sha"], "tree": {"sha": self.release["source_tree"]}}
            if route == self.paths["source_tree"]: return tree_response(self.source)
            if route == self.paths["ref"]:
                if expected == 404:
                    if self.ref: raise p.PublicationError(200)
                    return None
                return self.ref
            if route == p.BASE + "/git/commits/" + self.expected_commit: return self.commit
            if route == p.BASE + "/git/trees/" + self.expected_tree + "?recursive=1": return tree_response(self.result_rows)
        elif route == p.BASE + "/git/blobs":
            raw = base64.b64decode(body["content"], validate=True); digest = t.git_sha(raw)
            self.blobs[digest] = raw
            return {"sha": digest}
        elif route == p.BASE + "/git/trees":
            self.result_rows = self.source + body["tree"]
            return tree_response(self.result_rows)
        elif route == p.BASE + "/git/commits":
            expected_body, digest = p.commit_body(body["tree"], self.release)
            if body != expected_body: raise AssertionError("commit body differs")
            self.commit = {"sha": digest, "message": body["message"], "tree": {"sha": body["tree"]},
                "parents": [{"sha": v} for v in body["parents"]], "author": body["author"], "committer": body["committer"]}
            return self.commit
        elif route == p.BASE + "/git/refs":
            if self.ref: raise p.PublicationError(422)
            self.ref = {"ref": body["ref"], "object": {"type": "commit", "sha": body["sha"]}}
            return self.ref
        raise AssertionError("unexpected request")

    def raw_factory(self, commit, purpose):
        if commit != self.expected_commit: raise AssertionError("moving/mixed commit")
        self.raw_calls = []; prefix = t.policy(purpose)["directory"] + "/"
        files = {row["path"]: self.blobs[row["sha"]] for row in self.result_rows if row["path"].startswith(prefix)}
        def read(name):
            if name in self.raw_calls: raise AssertionError("duplicate read")
            self.raw_calls.append(name)
            return files[name]
        return read


class OfflineTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("http.client.HTTPSConnection", side_effect=AssertionError("live network prohibited")); self.guard.start()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.env, self.event, self.release, self.source = setup()
        self.identity = p.context(self.env, self.event, self.release)
        self.payload = t.fixture_archive(self.identity, self.release["source_closure_sha256"])
        self.api = MockAPI(self.env, self.event, self.release, self.source)

    def tearDown(self):
        self.temp.cleanup(); self.guard.stop()

    def publish(self, **kw):
        return p.publish(self.env, self.event, self.release, self.payload, self.api, self.api.raw_factory, **kw)

    def run_once(self):
        return p.run_once(self.env, self.event, self.release, self.payload, self.api, self.api.raw_factory, self.root)

    def test_normal_create_only_exact_readback(self):
        result = self.run_once()
        self.assertEqual(result["status"], "same_identity_readback_verified")
        self.assertEqual(result["raw_gets"], 2)
        self.assertEqual(result["archive_sha256"], t.sha(self.payload))
        self.assertTrue(result["semantic_inference_success_not_established_by_publisher"])
        self.assertFalse(result["retry_authorized"])
        self.assertEqual([c[0] for c in self.api.calls[:7]], ["GET"] * 7)
        self.assertFalse(any(c[0] in ("PATCH", "PUT", "DELETE") for c in self.api.calls))
        posts = [c for c in self.api.calls if c[0] == "POST"]
        self.assertEqual(posts[-1][1], p.BASE + "/git/refs")
        self.assertEqual(len(posts), 5)
        self.assertEqual(sorted(x["path"] for x in self.api.result_rows), sorted([x["path"] for x in self.source] + self.api.raw_calls))
        self.assertEqual(t.decode((self.root / "publication.receipt.json").read_bytes()), result)

    def test_single_final_lf_presentation_at_mock_post_and_get(self):
        for presentation_methods in ({"POST"}, {"GET"}, {"POST", "GET"}):
            class PresentedAPI(MockAPI):
                def __call__(self, method, route, body=None, expected=200):
                    value = super().__call__(method, route, body, expected)
                    if method in presentation_methods and (
                        route == p.BASE + "/git/commits" or route.startswith(p.BASE + "/git/commits/")
                        and route != self.paths["source"]
                    ):
                        value = copy.deepcopy(value)
                        if not value["message"].endswith("\n"):
                            raise AssertionError("mock must preserve the original commit message")
                        value["message"] = value["message"][:-1]
                    return value

            with self.subTest(presentation_methods=sorted(presentation_methods)):
                api = PresentedAPI(self.env, self.event, self.release, self.source)
                result = p.publish(self.env, self.event, self.release, self.payload, api, api.raw_factory)
                self.assertEqual(result["status"], "same_identity_readback_verified")
                self.assertEqual(result["raw_gets"], 2)
                self.assertEqual(result["result_commit"], api.expected_commit)
                self.assertEqual(api.ref["object"]["sha"], api.expected_commit)
                self.assertTrue(api.commit["message"].endswith("\n"))
                posts = [c for c in api.calls if c[0] == "POST"]
                self.assertEqual(len(posts), 5)
                self.assertEqual(posts[-1][1], p.BASE + "/git/refs")
                self.assertFalse(any(c[0] in ("PATCH", "PUT", "DELETE") for c in api.calls))

    def test_fixture_exact_small_and_fixed_json(self):
        self.assertLess(len(self.payload), 3072); self.assertGreater(len(self.payload), 1024)
        self.assertEqual(t.fixture_archive(self.identity, self.release["source_closure_sha256"]), self.payload)
        self.assertEqual(set(t.validate_archive(self.payload, self.identity, "fixture")), set(t.FIXTURE_MEMBERS))
        for payload in (self.payload + b"trailer", self.payload[:-1], self.payload.replace(b"Fixed", b"Other", 1)):
            with self.assertRaises(ValueError): t.verify_fixture(payload, self.identity, self.release["source_closure_sha256"])

    def test_fresh_push_mutations_fail_before_any_api(self):
        changes = [("env", "GITHUB_REPOSITORY", "jiying2007/kws-pipeline"), ("env", "GITHUB_EVENT_NAME", "create"),
            ("env", "GITHUB_RUN_ATTEMPT", "2"), ("env", "GITHUB_RUN_ID", "00123"), ("env", "GITHUB_RUN_ID", "True"),
            ("env", "GITHUB_REF", "refs/heads/main"), ("env", "GITHUB_JOB", "k1-tts"), ("env", "GITHUB_SHA", "d" * 40),
            ("event", "created", 1), ("event", "created", False), ("event", "deleted", True), ("event", "after", "d" * 40)]
        for target, key, value in changes:
            env, event = copy.deepcopy(self.env), copy.deepcopy(self.event)
            (env if target == "env" else event)[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.publish(env, event, self.release, self.payload, self.api, self.api.raw_factory)
        self.assertFalse(self.api.calls)

    def test_old_consumed_ref_and_other_purpose_rejected(self):
        for branch in ("research/qwen12-native-admission-remaining35-r1-v1", "research/qwen12-native-transport-preflight-12m-v1", "research/k1-punctuation6-tts-v1"):
            env, event = copy.deepcopy(self.env), copy.deepcopy(self.event)
            env["GITHUB_REF"] = event["ref"] = "refs/heads/" + branch
            with self.assertRaises(ValueError): p.context(env, event, self.release)
        for purpose in ("native", "synthetic", "synthetic12m", "retry", "anything"):
            with self.assertRaises(ValueError): t.policy(purpose)

    def test_consumed_fixture_v1_ref_rejected_by_current_fixture_context(self):
        consumed_ref = "refs/heads/research/k1-punctuation6-writer-fixture-v1"
        self.assertEqual(self.env["GITHUB_REF"], "refs/heads/" + t.policy("fixture")["source_branch"])
        self.assertNotEqual(self.env["GITHUB_REF"], consumed_ref)
        self.assertEqual(p.context(self.env, self.event, self.release), self.identity)
        env, event = copy.deepcopy(self.env), copy.deepcopy(self.event)
        env["GITHUB_REF"] = event["ref"] = consumed_ref
        with self.assertRaises(ValueError): p.context(env, event, self.release)
        self.assertFalse(self.api.calls)

    def test_run_job_repository_and_history_mutations_stop_before_posts(self):
        mutations = [("run", "event", "create"), ("run", "path", ".github/workflows/other.yml"),
            ("run", "head_sha", "f" * 40), ("run", "head_branch", "old"), ("run", "run_attempt", True),
            ("run", "status", "completed"), ("job", "id", True), ("job", "head_sha", "f" * 40),
            ("job", "name", "other"), ("job", "status", "completed"), ("job", "conclusion", "success"),
            ("repo", "private", True), ("repo", "full_name", "other/fork"), ("history", "total_count", 2)]
        for target, key, value in mutations:
            api = MockAPI(self.env, self.event, self.release, self.source)
            getattr(api, target)[key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                p.publish(self.env, self.event, self.release, self.payload, api, api.raw_factory)
            self.assertFalse(any(c[0] == "POST" for c in api.calls))

    def test_duplicate_or_paginated_jobs_rejected(self):
        for jobs in ({"total_count": 2, "jobs": [self.api.job, self.api.job]}, {"total_count": 101, "jobs": []}, {"total_count": 2, "jobs": [self.api.job]}):
            self.api.jobs = jobs
            with self.assertRaises(ValueError): self.publish()
        self.assertFalse(any(c[0] == "POST" for c in self.api.calls))

    def test_existing_result_ref_blocks_before_posts(self):
        self.api.ref = {"ref": "already exists"}
        with self.assertRaises(p.PublicationError): self.publish()
        self.assertFalse(any(c[0] == "POST" for c in self.api.calls))

    def test_destination_cannot_overwrite_source_or_parent_blob(self):
        for name in (t.policy("fixture")["directory"] + "/old.bin", "datasets"):
            self.api.source = self.source + [{"path": name, "mode": "100644", "type": "blob", "sha": "1" * 40}]
            self.release["source_tree"] = tree_response(self.api.source)["sha"]
            self.api.release = self.release; self.api.paths = p.endpoints(self.identity, self.release)
            with self.assertRaises(ValueError): self.publish()
            self.assertFalse(any(c[0] == "POST" for c in self.api.calls))

    def test_true_blob_hash_enforced(self):
        self.api.mutations[("POST", p.BASE + "/git/blobs", 201)] = lambda *_: {"sha": "e" * 40}
        with self.assertRaises(ValueError): self.publish()
        self.assertFalse(self.api.ref)

    def test_true_tree_hash_enforced(self):
        self.api.mutations[("POST", p.BASE + "/git/trees", 201)] = lambda *_: {"sha": "e" * 40}
        with self.assertRaises(ValueError): self.publish()
        self.assertFalse(self.api.ref)

    def test_true_commit_hash_enforced(self):
        self.api.mutations[("POST", p.BASE + "/git/commits", 201)] = lambda *_: {"sha": "e" * 40}
        with self.assertRaises(ValueError): self.publish()
        self.assertFalse(self.api.ref)

    def test_complete_tree_recomputation_rejects_truncation_and_false_subtree(self):
        for change in (lambda v: v.update(truncated=True), lambda v: v["tree"][-1].update(sha="e" * 40),
                       lambda v: v["tree"].append(copy.deepcopy(v["tree"][0]))):
            value = tree_response(self.source); change(value)
            with self.assertRaises(ValueError): p.verify_tree(value, self.release["source_tree"])

    def test_raw_corruption_after_created_ref_is_unconfirmed_not_regenerate(self):
        def corrupt(commit, purpose):
            read = self.api.raw_factory(commit, purpose)
            return lambda name: read(name) + b"corruption"
        with self.assertRaises(ValueError) as caught:
            p.run_once(self.env, self.event, self.release, self.payload, self.api, corrupt, self.root)
        receipt = caught.exception.publication_receipt
        self.assertTrue(receipt["publication_confirmed"])
        self.assertFalse(receipt["readback_confirmed"])
        self.assertFalse(receipt["new_generation_authorized"])
        before = len(self.api.calls)
        with self.assertRaises(FileExistsError): self.run_once()
        self.assertEqual(before, len(self.api.calls))

    def test_uncertain_ref_create_failure_preserves_receipt_no_retry(self):
        self.api.mutations[("POST", p.BASE + "/git/refs", 201)] = p.PublicationError(503)
        with self.assertRaises(p.PublicationError) as caught: self.run_once()
        receipt = caught.exception.publication_receipt
        self.assertEqual(receipt["stage"], "create_ref")
        self.assertEqual(receipt["http_status"], 503)
        self.assertFalse(receipt["retry_authorized"])
        self.assertTrue(receipt["expected_result_commit"])
        before = len(self.api.calls)
        with self.assertRaises(FileExistsError): self.run_once()
        self.assertEqual(before, len(self.api.calls))

    def test_success_is_once_only(self):
        self.run_once(); before = len(self.api.calls)
        with self.assertRaises(FileExistsError): self.run_once()
        self.assertEqual(before, len(self.api.calls))

    def test_fsync_started_and_receipt(self):
        with patch("os.fsync", wraps=p.os.fsync) as fsync:
            self.run_once()
        self.assertEqual(fsync.call_count, 4)

    def test_receipt_symlink_rejected(self):
        alias = self.root / "alias"; alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): p.write_once(alias, "claim", {})
        (self.root / "publication.started.json").symlink_to(self.root / "other")
        with self.assertRaises(FileExistsError): self.run_once()
        self.assertFalse(self.api.calls)

    def test_transport_at_exact_12m_and_48_chunks(self):
        # Incompressible bytes are transport-only data, never a released fixture.
        payload = random.Random(20261012).randbytes(t.MAX_ARCHIVE)
        files = t.build_file_set(payload, self.identity, "b" * 64, 456)
        manifest = files.pop(t.manifest_path())
        self.assertLessEqual(len(manifest), t.MAX_MANIFEST)
        self.assertEqual(len(files), 48)
        self.assertLess(len(manifest) + len(payload), t.MAX_BODY)
        self.assertEqual(t.reassemble(manifest, list(files.items()), self.identity, "b" * 64, 456), payload)
        with self.assertRaises(ValueError): t.build_file_set(payload + b"x", self.identity, "b" * 64, 456)
        with self.assertRaises(ValueError): t.verify_fixture(payload, self.identity, "b" * 64)

    def test_chunk_and_manifest_mutations(self):
        files = t.build_file_set(self.payload, self.identity, "b" * 64, 456)
        raw = files.pop(t.manifest_path()); rows = list(files.items())
        for changes in ({"numeric_job_id": True}, {"chunk_count": 49}, {"archive_bytes": t.MAX_ARCHIVE + 1},
                        {"fixture_only": False}, {"purpose": "tts"}, {"source_closure_sha256": "c" * 64}):
            value = t.decode(raw); value.update(changes)
            with self.assertRaises(ValueError): t.parse_manifest(t.canonical(value), self.identity, "b" * 64, 456)
        for chunks in ([], rows + rows, [(rows[0][0], rows[0][1] + b"x")], [("../other", rows[0][1])]):
            with self.assertRaises(ValueError): t.reassemble(raw, chunks, self.identity, "b" * 64, 456)
        with self.assertRaises(ValueError): t.parse_manifest(raw + b" ", self.identity, "b" * 64, 456)
        with self.assertRaises(ValueError): t.decode(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError): t.decode(b'{"a":NaN}')

    def test_real_tts_requires_semantic_validator_and_preserves_started_members(self):
        env, event, release, source = setup("tts"); identity = p.context(env, event, release)
        members = {name: t.canonical({"schema": "explicit-offline-test-stub", "identity": identity}) for name in t.TTS_JSON}
        members["claims/k1p6-001.started.json"] = t.canonical({"cell_id": "k1p6-001"})
        with self.assertRaises(ValueError): t.build_archive(members, identity, "tts")
        with self.assertRaises(ValueError): t.build_archive(members, identity, "tts", lambda *_: None)
        calls = []
        def validator(got, run):
            calls.append((got, run)); return True  # Only an offline test stub; no live release.
        payload = t.build_archive(members, identity, "tts", validator)
        self.assertEqual(t.validate_archive(payload, identity, "tts", validator), members)
        self.assertGreaterEqual(len(calls), 2)
        api = MockAPI(env, event, release, source)
        result = p.publish(env, event, release, payload, api, api.raw_factory, validator)
        self.assertEqual(result["purpose"], "tts")
        self.assertTrue(result["semantic_inference_success_not_established_by_publisher"])

    def test_chinese_utf8_started_claim_bytes_preserved_without_rewrite(self):
        env, event, release, _ = setup("tts"); identity = p.context(env, event, release)
        members = {name: b'{}' for name in t.TTS_JSON}
        raw = '{"cell_id":"k1p6-002","text":"你好，小窝"}'.encode("utf-8")
        self.assertEqual(t.canonical(t.decode(raw)), raw)
        members["claims/k1p6-002.started.json"] = raw
        payload = t.build_archive(members, identity, "tts", lambda *_: True)
        readback = t.validate_archive(payload, identity, "tts", lambda *_: True)
        self.assertEqual(readback["claims/k1p6-002.started.json"], raw)
        self.assertEqual(t.sha(readback["claims/k1p6-002.started.json"]), t.sha(raw))

    def test_tts_extra_members_private_absolute_paths_and_size_rejected(self):
        env, event, release, _ = setup("tts"); identity = p.context(env, event, release)
        members = {name: b'{}' for name in t.TTS_JSON}
        for name, raw in (("raw.log", b"secret"), ("weights.bin", b"x"), ("raw/k1p6-007.wav", b"x"),
                          ("raw/k1p6-001.wav", b"x" * (24000 * 12 * 4 + 257)),
                          ("source.json", b'{"path":"/home/user/private"}'),
                          ("claim.json", b'{"a":1,"a":2}')):
            bad = dict(members); bad[name] = raw
            with self.subTest(name=name), self.assertRaises(ValueError): t.build_archive(bad, identity, "tts", lambda *_: True)

    def test_zip_unexpected_or_duplicated_members_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z: z.writestr("../secret", b"x")
        with self.assertRaises(ValueError): t.validate_archive(buffer.getvalue(), self.identity, "fixture")
        with self.assertRaises(ValueError): t.validate_archive(self.payload + b"trailer", self.identity, "fixture")

    def test_empty_git_tree_known_identity(self):
        self.assertEqual(t.tree_identity([]), ("4b825dc642cb6eb9a060e54bf8d69288fbee4904", {}))
        self.assertEqual(t.git_sha(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")
        rows = [{"path": "a/b", "mode": "100644", "type": "blob", "sha": "1" * 40}]
        self.assertEqual(t.tree_identity(rows), t.tree_identity(list(reversed(rows))))
        for bad in (rows + rows, rows + [{"path": "a", "mode": "100644", "type": "blob", "sha": "2" * 40}]):
            with self.assertRaises(ValueError): t.tree_identity(bad)

    def test_postpublication_ref_mismatch_blocks_readback(self):
        self.api.mutations[("GET", self.api.paths["ref"], 200)] = lambda *_: {
            "ref": "refs/heads/" + t.policy("fixture")["result_branch"],
            "object": {"type": "commit", "sha": "e" * 40}}
        with self.assertRaises(ValueError) as caught: self.run_once()
        self.assertTrue(caught.exception.publication_receipt["publication_confirmed"])
        self.assertFalse(caught.exception.publication_receipt["readback_confirmed"])
        self.assertFalse(hasattr(self.api, "raw_calls"))

    def test_hard_host_alarm_installed_and_cleared(self):
        with patch("signal.setitimer", wraps=p.signal.setitimer) as timer:
            self.run_once()
        self.assertIn(((p.signal.ITIMER_REAL, p.TIMEOUT),), [(call.args,) for call in timer.call_args_list])
        self.assertEqual(timer.call_args_list[-1].args, (p.signal.ITIMER_REAL, 0))
        self.assertEqual(p.signal.getitimer(p.signal.ITIMER_REAL), (0.0, 0.0))

    def test_cli_unconditionally_disabled(self):
        with self.assertRaises(SystemExit): p.main()


class CommitPresentationTests(unittest.TestCase):
    # Minimal public fields from an observed GitHub GET /git/commits response.
    # This is immutable regression data, not a release or permission to retry.
    OBSERVED_GET = {
        "sha": "2d6f5bbbd744003a8ccf76eb571a20b460994dc0",
        "message": "Retain bounded K1 fixture data",
        "tree": {"sha": "be177d44f683ea53ce24280cf79719a503eb4103"},
        "parents": [{"sha": "72b41772eda8e56b99d526f03170d70ceda6d0d3"}],
        "author": {"name": "K1 bounded publisher", "email": "noreply@github.com", "date": "2026-10-12T01:59:07Z"},
        "committer": {"name": "K1 bounded publisher", "email": "noreply@github.com", "date": "2026-10-12T01:59:07Z"},
    }

    def setUp(self):
        guard = patch("http.client.HTTPSConnection", side_effect=AssertionError("live network prohibited"))
        guard.start(); self.addCleanup(guard.stop)
        release = {"purpose": "fixture", "source_sha": self.OBSERVED_GET["parents"][0]["sha"],
                   "commit_timestamp": self.OBSERVED_GET["author"]["date"]}
        self.body, self.expected_sha = p.commit_body(self.OBSERVED_GET["tree"]["sha"], release)

    def test_observed_get_omits_only_the_fixed_final_lf(self):
        self.assertEqual(self.expected_sha, self.OBSERVED_GET["sha"])
        self.assertEqual(self.body["message"], self.OBSERVED_GET["message"] + "\n")
        self.assertEqual(self.body["message"].count("\n"), 1)
        p.verify_commit(copy.deepcopy(self.OBSERVED_GET), self.expected_sha, self.body)

    def test_original_message_is_also_accepted_without_rewriting(self):
        value = copy.deepcopy(self.OBSERVED_GET); value["message"] = self.body["message"]
        before = copy.deepcopy(value)
        p.verify_commit(value, self.expected_sha, self.body)
        self.assertEqual(value, before)

    def test_message_changes_are_not_whitespace_normalized(self):
        exact = self.body["message"]; presented = exact[:-1]
        messages = {
            "different_words": exact.replace("Retain", "Preserve"),
            "double_internal_space": exact.replace("bounded K1", "bounded  K1"),
            "internal_tab": exact.replace("bounded K1", "bounded\tK1"),
            "internal_lf": exact.replace("bounded K1", "bounded\nK1"),
            "leading_space": " " + exact,
            "extra_final_lf": exact + "\n",
            "final_space": presented + " ",
            "space_before_lf": presented + " \n",
            "space_after_lf": exact + " ",
            "crlf": presented + "\r\n",
            "final_cr": presented + "\r",
            "internal_cr": exact.replace("bounded K1", "bounded\rK1"),
            "empty": "", "null": None, "bytes": exact.encode("utf-8"),
            "list": [presented],
        }
        for label, message in messages.items():
            value = copy.deepcopy(self.OBSERVED_GET); value["message"] = message
            with self.subTest(label=label), self.assertRaises(ValueError):
                p.verify_commit(value, self.expected_sha, self.body)

    def test_presented_message_still_requires_exact_git_identity(self):
        for message in (self.body["message"], self.body["message"][:-1]):
            value = copy.deepcopy(self.OBSERVED_GET); value["message"] = message
            mutations = {
                "sha": lambda v: v.update(sha="e" * 40),
                "tree": lambda v: v["tree"].update(sha="e" * 40),
                "parent": lambda v: v["parents"][0].update(sha="e" * 40),
                "no_parent": lambda v: v.update(parents=[]),
                "extra_parent": lambda v: v["parents"].append(copy.deepcopy(v["parents"][0])),
                "wrong_parent_type": lambda v: v.update(parents=tuple(v["parents"])),
            }
            for label, mutate in mutations.items():
                bad = copy.deepcopy(value); mutate(bad)
                with self.subTest(message=message, label=label), self.assertRaises(ValueError):
                    p.verify_commit(bad, self.expected_sha, self.body)

    def test_presented_message_still_requires_exact_author_and_committer(self):
        for message in (self.body["message"], self.body["message"][:-1]):
            for person in ("author", "committer"):
                for field, changed in (("name", "Other publisher"), ("email", "other@example.invalid"),
                                       ("date", "2026-10-12T01:59:08Z"),
                                       ("date", "2026-10-12T01:59:07+00:00")):
                    value = copy.deepcopy(self.OBSERVED_GET); value["message"] = message
                    value[person][field] = changed
                    with self.subTest(message=message, person=person, field=field, changed=changed), self.assertRaises(ValueError):
                        p.verify_commit(value, self.expected_sha, self.body)

    def test_git_sha_is_not_recomputed_from_the_presented_message(self):
        # The absent API LF changes Git bytes; only the original fixed bytes are authoritative.
        person = "K1 bounded publisher <noreply@github.com> 1791770347 +0000"
        raw = ("tree " + self.body["tree"] + "\nparent " + self.body["parents"][0]
               + "\nauthor " + person + "\ncommitter " + person + "\n\n" + self.body["message"]).encode("utf-8")
        self.assertEqual(t.git_sha(raw, "commit"), self.expected_sha)
        presented_sha = t.git_sha(raw[:-1], "commit")
        self.assertNotEqual(presented_sha, self.expected_sha)
        value = copy.deepcopy(self.OBSERVED_GET); value["sha"] = presented_sha
        with self.assertRaises(ValueError): p.verify_commit(value, self.expected_sha, self.body)


class Response:
    def __init__(self, raw, status=200): self.raw = raw; self.status = status; self.read_caps = []
    def read(self, cap): self.read_caps.append(cap); return self.raw[:cap]


class Connection:
    def __init__(self, response): self.response = response; self.requests = []; self.closed = False
    def request(self, *args, **kwargs): self.requests.append((args, kwargs))
    def getresponse(self): return self.response
    def close(self): self.closed = True


class HTTPBounds(OfflineTests):
    # Inherit default-deny setup, but suppress inherited test methods below.
    def client(self): return p.GitHubAPI("offline-fixture-not-a-credential", self.identity, self.release)

    def test_api_fixed_host_timeout_headers_and_read_cap(self):
        client = self.client(); response = Response(b'{}'); connection = Connection(response)
        with patch("http.client.HTTPSConnection", return_value=connection) as factory:
            self.assertEqual(client("GET", p.BASE), {})
        self.assertEqual(factory.call_args.args, ("api.github.com",))
        self.assertLessEqual(factory.call_args.kwargs["timeout"], 15)
        self.assertEqual(response.read_caps, [p.MAX_RESPONSE + 1])
        self.assertTrue(connection.closed)

    def test_api_no_redirect_retry_or_arbitrary_route(self):
        client = self.client()
        for method, route in (("PATCH", p.BASE + "/git/refs"), ("GET", "/repos/other/fork"), ("GET", "https://evil.example")):
            with self.assertRaises(ValueError): client(method, route)
        connection = Connection(Response(b'{}', 302))
        with patch("http.client.HTTPSConnection", return_value=connection) as factory:
            with self.assertRaises(p.PublicationError): client("GET", p.BASE)
            with self.assertRaises(ValueError): client("GET", p.BASE)
        self.assertEqual(factory.call_count, 1)

    def test_api_request_response_total_body_count_deadline_caps(self):
        client = self.client()
        with self.assertRaises(ValueError): client("POST", p.BASE + "/git/blobs", {"data": "x" * p.MAX_REQUEST}, 201)
        for field, value in (("count", p.MAX_API_REQUESTS), ("deadline", 0)):
            client = self.client(); setattr(client, field, value)
            with self.assertRaises(ValueError): client("GET", p.BASE)
        for body, prior in ((b"x" * (p.MAX_RESPONSE + 1), 0), (b"xxxx", p.MAX_API_BODY - 1)):
            client = self.client(); client.body_bytes = prior
            with patch("http.client.HTTPSConnection", return_value=Connection(Response(body))):
                with self.assertRaises(p.PublicationError): client("GET", p.BASE)

    def test_api_result_body_binding_and_total_upload_cap(self):
        client = self.client(); client.bind_result("d" * 40, "e" * 40)
        for body in ({"ref": "refs/heads/main", "sha": "d" * 40},
                     {"ref": "refs/heads/" + t.policy("fixture")["result_branch"], "sha": "f" * 40}):
            with self.assertRaises(ValueError): client("POST", p.BASE + "/git/refs", body, 201)
        client.request_bytes = p.MAX_API_UPLOAD
        with self.assertRaises(ValueError): client("POST", p.BASE + "/git/blobs", {"content": "YQ==", "encoding": "base64"}, 201)
        client = self.client(); client.bind_result("d" * 40, "e" * 40)
        body = {"ref": "refs/heads/" + t.policy("fixture")["result_branch"], "sha": "d" * 40}
        with patch("http.client.HTTPSConnection", return_value=Connection(Response(b'{}', 503))) as factory:
            with self.assertRaises(p.PublicationError): client("POST", p.BASE + "/git/refs", body, 201)
            with self.assertRaises(ValueError): client("POST", p.BASE + "/git/refs", body, 201)
        self.assertEqual(factory.call_count, 1)

    def test_raw_fixed_immutable_host_path_no_token_once(self):
        reader = p.RawReader("d" * 40, "fixture"); response = Response(b"abc"); connection = Connection(response)
        with patch("http.client.HTTPSConnection", return_value=connection) as factory:
            self.assertEqual(reader(t.manifest_path()), b"abc")
            with self.assertRaises(ValueError): reader(t.manifest_path())
        self.assertEqual(factory.call_args.args, ("raw.githubusercontent.com",))
        self.assertEqual(connection.requests[0][0], ("GET", "/jiying2007/kws-data/" + "d" * 40 + "/" + t.manifest_path()))
        self.assertNotIn("Authorization", connection.requests[0][1]["headers"])
        self.assertEqual(reader.body_bytes, 3)

    def test_raw_no_redirect_retry_paths_body_or_deadline_overrun(self):
        for status, body in ((302, b""), (503, b""), (200, b"x" * (t.MAX_MANIFEST + 1))):
            reader = p.RawReader("d" * 40, "fixture")
            with patch("http.client.HTTPSConnection", return_value=Connection(Response(body, status))) as factory:
                with self.assertRaises(p.PublicationError): reader(t.manifest_path())
                with self.assertRaises(ValueError): reader(t.manifest_path())
            self.assertEqual(factory.call_count, 1)
        reader = p.RawReader("d" * 40, "fixture")
        for name in ("https://evil.example", "../secret", t.chunk_path(0) + "?ref=main"):
            with self.assertRaises(ValueError): reader(name)
        reader.deadline = 0
        with self.assertRaises(ValueError): reader(t.manifest_path())
        reader = p.RawReader("d" * 40, "fixture"); reader.body_bytes = t.MAX_BODY
        with patch("http.client.HTTPSConnection", return_value=Connection(Response(b"x"))):
            with self.assertRaises(p.PublicationError): reader(t.manifest_path())


# Avoid accidentally running parent tests twice in the HTTP subclass.
for _name in list(OfflineTests.__dict__):
    if _name.startswith("test_") and _name not in HTTPBounds.__dict__:
        setattr(HTTPBounds, _name, None)

if __name__ == "__main__":
    unittest.main(verbosity=2)
