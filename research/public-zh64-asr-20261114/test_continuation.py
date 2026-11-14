"""No-network, no-Docker, no-model tests of the sixteen-cell continuation boundary."""
from pathlib import Path
import ast
import copy
import json
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continuation as c


def composition_fixture():
    value = {"providers": {family: [] for family in c.host.FAMILIES}, **c.UNADMITTED}
    for family, rows in value["providers"].items():
        for n in range(1, 65):
            pending = family == "whisper" and n >= 49
            rows.append(dict(audio_id=f"clip-{n:06d}", local_audio_id=f"clip-{(n - 1) % 16 + 1:06d}",
                source_id=f"source-{n}", request_id=f"request-{n}", wav_sha256=f"{n:064x}",
                pcm_sha256=f"{n + 64:064x}", frames=16000,
                status="not_run" if pending else "complete_native_observation",
                attempt_consumed=not pending, actual_text=None if pending else "saved-observation"))
    fresh = copy.deepcopy(value["providers"]["whisper"][48:])
    for row in fresh:
        row.update(status="complete_native_observation", attempt_consumed=True, actual_text="new-native-observation")
    return value, fresh


def parent_prefix_fixture(request):
    observations, _ = composition_fixture()
    counts = {family: {status: sum(row["status"] == status for row in rows)
        for status in ("complete_native_observation", "failed_no_retry", "not_run",
                       "rejected_native_evidence", "unknown_after_lifecycle_failure")}
        for family, rows in observations["providers"].items()}
    summary = dict(run_id=str(request["parent_run_id"]), executed_head_sha=request["parent_head_sha"],
        status="incomplete_no_retry", attempts_consumed=176, planned_asr_slots=192,
        claim_ref=request["parent_claim_ref"], retry=False, providers=counts,
        stages={"batch-3-whisper": dict(error_type="ValueError", status="failed_no_retry",
                                       message="current host memory reserve unavailable")}, **c.UNADMITTED)
    files = {"summary.json": c.canonical(summary), "observations.json": c.canonical(observations),
        "inference/batch-3/sensevoice/container-qwen06.json": c.canonical({"created_confirmed": True,
            "cleanup_failure": None, "operations": {"cleanup": {"status": "complete"}}})}
    files.update({"executed-control-source/" + name: c.read(HERE / name, c.MIB) for name in c.OLD_CONTROL})
    return files


class RequestAndExecutionTests(unittest.TestCase):
    def setUp(self):
        self.request, self.digest = c.frozen_request()

    def test_exact_scope_and_all_nearphone_roles(self):
        r = self.request
        self.assertEqual(r["max_asr_calls"], 16)
        self.assertEqual(r["already_complete_excluded"], {"qwen06": 64, "sensevoice": 64, "whisper": 48})
        self.assertEqual(r["minimum_host_available_bytes"], 14 * 1024 ** 3)
        self.assertEqual((r["stage_wall_seconds"], r["cell_wall_seconds"]), (1500, 300))
        self.assertEqual({row["global_audio_id"] for row in r["slots"]}, {f"clip-{i:06d}" for i in range(49, 65)})
        self.assertEqual([sum(row["research_split"] == role for row in r["slots"]) for role in ("train", "dev")], [8, 8])
        self.assertTrue(all(row["nearphone_quarantined"] and not row["optimizer_loss_authorized"] for row in r["slots"]))
        self.assertFalse(r["retry"])
        self.assertEqual((r["primary_model_loads"], r["new_audio_derivatives"], r["new_tts_calls"], r["new_training_runs"]), (0, 0, 0, 0))

    def test_request_cannot_expand_or_relabel_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            for key, value in (("max_asr_calls", 17), ("family", "sensevoice"), ("retry", True),
                               ("minimum_host_available_bytes", 12 * 1024 ** 3), ("training_admitted", True)):
                changed = dict(self.request, **{key: value})
                path.write_bytes(c.canonical(changed))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    c.frozen_request(path)
            changed = copy.deepcopy(self.request)
            changed["slots"][0]["research_split"] = "dev"
            path.write_bytes(c.canonical(changed))
            with self.assertRaises(ValueError):
                c.frozen_request(path)

    def test_formatting_cannot_create_another_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "request.json"
            path.write_bytes(c.canonical(self.request))
            _, compact = c.frozen_request(path)
            self.assertNotEqual(compact, self.digest)
            self.assertEqual(c.claim_ref(compact), c.claim_ref(self.digest))
            self.assertNotEqual(c.claim_ref(compact), c.host.claim_ref(compact))

    def test_original_control_and_claim_are_immutable(self):
        c.verify_control()
        valid = {"ref": self.request["parent_claim_ref"], "object": {"sha": self.request["parent_head_sha"], "type": "commit"}}
        c.parent_claim(valid, self.request)
        for changed in (dict(valid, ref=c.claim_ref(self.digest)),
                        dict(valid, object={"sha": "a" * 40, "type": "commit"})):
            with self.assertRaises(ValueError):
                c.parent_claim(changed, self.request)

    def test_main_attempt_one_actual_head_and_no_token(self):
        head = "a" * 40
        env = dict(GITHUB_REPOSITORY="jiying2007/kws-pipeline", GITHUB_EVENT_NAME="push",
            GITHUB_REF="refs/heads/main", GITHUB_RUN_ATTEMPT="1", GITHUB_ACTIONS="true",
            RUNNER_ENVIRONMENT="github-hosted", RUNNER_OS="Linux", RUNNER_ARCH="X64", GITHUB_SHA=head,
            GITHUB_RUN_ID="123", GITHUB_WORKFLOW_SHA=head,
            GITHUB_WORKFLOW_REF="jiying2007/kws-pipeline/.github/workflows/public-zh64-asr-once.yml@refs/heads/main")
        event = dict(after=head, ref="refs/heads/main", deleted=False,
                     repository=dict(private=False, full_name="jiying2007/kws-pipeline"))
        claim = dict(ref=c.claim_ref(self.digest), object=dict(sha=head, type="commit"))
        parent = dict(ref=self.request["parent_claim_ref"], object=dict(sha=self.request["parent_head_sha"], type="commit"))
        self.assertEqual(c.validate_execution(env, event, claim, parent, self.request, self.digest), (head, "123"))
        for key, value in (("GITHUB_EVENT_NAME", "pull_request"), ("GITHUB_RUN_ATTEMPT", "2"),
                           ("GITHUB_SHA", self.request["parent_head_sha"]), ("GH_TOKEN", "not-for-models")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_execution(dict(env, **{key: value}), event, claim, parent, self.request, self.digest)

    def test_parent_must_be_closed_failed_original_run(self):
        value = dict(id=self.request["parent_run_id"], run_attempt=1, head_sha=self.request["parent_head_sha"],
            status="completed", conclusion="failure", event="push", path=".github/workflows/public-zh64-asr-once.yml",
            repository=dict(full_name="jiying2007/kws-pipeline", private=False))
        c.verify_parent_run(value, self.request)
        for key, changed in (("run_attempt", 2), ("status", "in_progress"), ("head_sha", "a" * 40)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.verify_parent_run(dict(value, **{key: changed}), self.request)

    def test_arbitrary_zip_fails_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "exact original"):
            c.archive_files(b"not-the-parent", self.request)

    def test_capacity_failure_occurs_before_directory_or_docker(self):
        class Boundary:
            def host_capacity(self):
                raise ValueError("current host memory reserve unavailable")
            def container_args(self, *args):
                raise RuntimeError("must never reach Docker argv")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-created"
            with self.assertRaisesRegex(ValueError, "memory reserve"):
                c.host.launch(Boundary(), Path(tmp), Path(tmp), output, "image", "name", [], c.time.monotonic() + 900)
            self.assertFalse(output.exists())

    def test_prior_start_or_uncertain_cleanup_forbids_continuation(self):
        for extra in ("started.json", "capture/clip-000001/claim.json", "container-qwen06.json"):
            files = parent_prefix_fixture(self.request)
            files["inference/batch-3/whisper/" + extra] = b"{}"
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "no original Whisper"):
                c.validate_parent(files, self.request)
        files = parent_prefix_fixture(self.request)
        path = "inference/batch-3/sensevoice/container-qwen06.json"
        value = c.parse(files[path]); value["operations"]["cleanup"]["status"] = "failed_no_retry"
        files[path] = c.canonical(value)
        with self.assertRaisesRegex(ValueError, "cleanup completed"):
            c.validate_parent(files, self.request)

    def test_attempted_cell_cannot_be_read_as_unstarted(self):
        files = parent_prefix_fixture(self.request)
        value = c.parse(files["observations.json"])
        value["providers"]["whisper"][48]["attempt_consumed"] = True
        files["observations.json"] = c.canonical(value)
        with self.assertRaisesRegex(ValueError, "exclude every consumed"):
            c.validate_parent(files, self.request)


class NativeIdentityTests(unittest.TestCase):
    def fixture(self):
        old_source = dict(executed_head_sha="b" * 40, files=[dict(path="bindings/runtime/installed-inventory.json",
            bytes=3, sha256="c" * 64)], schema="retained-asr-source-lock-v1", source_inventory_scope="unchanged")
        config = dict(source_lock_sha256=c.sha(c.canonical(old_source)), asset_lock_sha256="d" * 64,
                      effective_config=dict(temperature=0.0, language="zh", sample_len=224))
        old = dict(model_family="whisper", model_sha256="e" * 64, native_decode_config=config,
                   native_decode_config_sha256=c.sha(c.canonical(config)))
        fresh_source = dict(old_source, executed_head_sha="a" * 40)
        fresh = copy.deepcopy(old)
        fresh["native_decode_config"]["source_lock_sha256"] = c.sha(c.canonical(fresh_source))
        fresh["native_decode_config_sha256"] = c.sha(c.canonical(fresh["native_decode_config"]))
        return old, fresh, old_source, fresh_source

    def test_true_new_head_and_only_hash_propagation_allowed(self):
        old, fresh, original, source = self.fixture()
        c.same_whisper_identity(old, fresh, original, source, "a" * 40)
        with self.assertRaises(ValueError):
            c.same_whisper_identity(old, old, original, original, "b" * 40)

    def test_decoder_model_asset_or_inventory_drift_is_rejected(self):
        for kind in ("decoder", "model", "asset", "inventory", "scope"):
            old, fresh, original, source = self.fixture()
            if kind == "decoder": fresh["native_decode_config"]["effective_config"]["temperature"] = 0.2
            if kind == "model": fresh["model_sha256"] = "f" * 64
            if kind == "asset": fresh["native_decode_config"]["asset_lock_sha256"] = "f" * 64
            if kind == "inventory":
                source = copy.deepcopy(source); source["files"][0]["sha256"] = "f" * 64
            if kind == "scope": source["source_inventory_scope"] = "looser"
            fresh["native_decode_config_sha256"] = c.sha(c.canonical(fresh["native_decode_config"]))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                c.same_whisper_identity(old, fresh, original, source, "a" * 40)


class CompositionTests(unittest.TestCase):
    def test_176_saved_rows_are_unchanged_and_parent_not_mutated(self):
        parent, fresh = composition_fixture()
        original = copy.deepcopy(parent)
        combined = c.combine(parent, fresh)
        self.assertEqual(parent, original)
        for family in ("qwen06", "sensevoice"):
            self.assertEqual(combined["providers"][family], original["providers"][family])
        self.assertEqual(combined["providers"]["whisper"][:48], original["providers"]["whisper"][:48])
        self.assertEqual(sum(len(rows) for rows in combined["providers"].values()), 192)
        self.assertFalse(combined["single_run_three_model_execution"])
        self.assertTrue(combined["parent_observations_policy_preserved"])
        self.assertFalse(combined["training_admitted"])

    def test_duplicate_new_slot_replay_and_input_changes_are_rejected(self):
        for kind in ("duplicate", "already_consumed", "different_pcm", "missing"):
            parent, fresh = composition_fixture()
            if kind == "duplicate": fresh[-1] = fresh[0]
            if kind == "already_consumed": parent["providers"]["whisper"][48]["attempt_consumed"] = True
            if kind == "different_pcm": fresh[0]["pcm_sha256"] = "0" * 64
            if kind == "missing": fresh.pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                c.combine(parent, fresh)

    def test_source_has_no_assert_or_model_import_and_only_whisper_setup(self):
        body = c.read(HERE / "continuation.py", c.MIB)
        tree = ast.parse(body)
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))
        self.assertFalse({"torch", "numpy", "scipy", "funasr", "qwen_asr", "whisper"} & set(sys.modules))
        source = body.decode()
        self.assertNotIn("setup_adapter.py", source)
        self.assertNotIn("--derive", source)
        self.assertIn('worker_command("whisper", "infer", head', source)
        self.assertIn('"--family", "whisper", "--executed-head", head', source)
        workflow = (HERE.parents[1] / ".github/workflows/public-zh64-asr-once.yml").read_text()
        actual = workflow.split("  actual-first-whisper16:", 1)[1]
        self.assertIn("contents: read", actual)
        self.assertNotIn("contents: write", actual)
        self.assertIn("--parent-artifact", actual)
        self.assertIn("--verify-parent", actual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
