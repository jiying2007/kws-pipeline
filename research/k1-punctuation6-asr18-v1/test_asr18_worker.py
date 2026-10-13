"""Stdlib fakes only: zero ML imports, downloads, network or real recognition."""
from pathlib import Path
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign_contract as contract
import native_worker as worker


def fixtures(count=3):
    clips = [{"audio_id": f"clip-{i:06d}", "wav_sha256": str(i) * 64,
              "audio_path": "audio/" + str(i) * 64 + ".wav"} for i in range(1, count + 1)]
    job = {"schema": "blind-source-screen32-job-v1", "clips": clips}
    # Deliberately different from SHA order: fixed cell order is not hash order.
    order = [row["audio_id"] for row in reversed(clips)]
    plan = {"schema": "k1-asr18-anonymous-order-metadata-proposal-v1", "execution_enabled": False,
            "job_sha256": contract.digest(contract.canonical(job)), "family_order": list(contract.FAMILIES),
            "audio_order": order}
    binding = {"schema": "k1-asr18-worker-input-binding-v1", "job_sha256": plan["job_sha256"],
        "execution_plan_sha256": contract.digest(contract.canonical(plan)),
        "screening_receipt_sha256": "a" * 64, "tts_authority_sha256": "b" * 64,
        "audio": [{"audio_id": row["audio_id"], "wav_sha256": row["wav_sha256"],
                   "pcm_sha256": "c" * 64, "audio_frames": 2} for row in reversed(clips)]}
    return job, plan, binding


class FakeSession:
    def __init__(self, events, fail=None):
        self.events, self.fail = events, fail

    def bind(self, clip):
        self.events.append(("bind_audio", clip["audio_id"]))
        return clip["audio_id"]

    def infer(self, bound, directory):
        self.events.append(("infer", bound))
        if self.fail == "infer":
            raise RuntimeError("synthetic-only failure")
        return {"synthetic": True}, {"synthetic": True}


class FakeReceipts:
    def __init__(self, events, fail=None):
        self.events, self.fail = events, fail

    def bind_predecode(self, **kwargs):
        self.events.append(("bind_predecode", kwargs["recognizer_id"]))
        return kwargs

    def emit_start(self, binding, directory, **kwargs):
        self.events.append(("durable_start", kwargs["audio_id"]))
        if self.fail == "start":
            raise OSError("synthetic uncertain fsync")
        return kwargs

    def mark_native_enter(self, attempt):
        self.events.append(("native_enter", attempt["audio_id"]))
        if self.fail == "enter":
            raise ValueError("synthetic enter failure")

    def mark_native_return(self, attempt):
        self.events.append(("native_return", attempt["audio_id"]))
        if self.fail == "return":
            raise ValueError("synthetic return failure")

    def emit_completion(self, attempt, **kwargs):
        self.events.append(("completion", attempt["audio_id"]))
        if self.fail == "completion":
            raise OSError("synthetic completion fsync failure")

    def fail_attempt(self, attempt):
        self.events.append(("failed_consumed", attempt["audio_id"]))
        if self.fail == "cleanup":
            raise OSError("synthetic terminal cleanup failure")


class WorkerTests(unittest.TestCase):
    def execute(self, root, mode="decode", fail=None, family="qwen06", change=None):
        job, plan, binding = fixtures()
        if change:
            change(job, plan, binding)
        events = []
        fake = FakeSession(events, fail)
        receipts = FakeReceipts(events, fail)
        dependencies = lambda name: {"campaign_contract": contract, "worker_receipts": receipts}[name]
        kwargs = dict(runtime=root / "runtime", inputs=root / "input", output=root,
            frozen=root / "frozen", policy_path=root / "policy.json", job=job, execution_plan=plan,
            input_binding=binding, run_id="900", head="d" * 40, policy_sha256="e" * 64)
        with mock.patch.object(worker, "_local", side_effect=dependencies), \
             mock.patch.object(worker, "_construct_native", return_value=fake) as construct, \
             mock.patch.object(worker, "_compare_frozen", side_effect=lambda *a: events.append(("compare", family))), \
             mock.patch.object(worker.signal, "setitimer"):
            try:
                result = worker._execute_family(family, mode, **kwargs)
                return result, events, construct.call_count
            except BaseException as exc:
                exc.synthetic_events = events
                exc.synthetic_constructs = construct.call_count
                raise

    def test_prepare_zero_forward_no_start_for_each_family(self):
        for family in contract.FAMILIES:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as name:
                result, events, loads = self.execute(Path(name), mode="prepare", family=family)
                self.assertEqual(loads, 1)
                self.assertEqual(events, [])
                self.assertEqual(result["audio_forward_calls"], 0)
                self.assertEqual(result["completed"], 0)
                self.assertEqual(list(Path(name).glob("clip-*")), [])
                self.assertEqual(contract.read_json(Path(name) / "prepared.json"), result)

    def test_decode_fixed_cell_order_not_sha_roster(self):
        for family in contract.FAMILIES:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as name:
                result, events, loads = self.execute(Path(name), family=family)
                self.assertEqual(result["audio_ids"], fixtures()[1]["audio_order"])
                self.assertEqual(loads, 1)
                expected = [("compare", family), ("bind_predecode", family)]
                for audio_id in fixtures()[1]["audio_order"]:
                    expected.extend((stage, audio_id) for stage in
                        ("bind_audio", "durable_start", "native_enter", "infer", "native_return", "completion"))
                self.assertEqual(events, expected)

    def test_every_fault_stops_later_audio_without_retry(self):
        for fail in ("start", "enter", "infer", "return", "completion"):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as name:
                with self.assertRaises((OSError, ValueError, RuntimeError)) as caught:
                    self.execute(Path(name), fail=fail)
                events = caught.exception.synthetic_events
                self.assertEqual([row[1] for row in events if row[0] == "durable_start"], ["clip-000003"])
                calls = [row for row in events if row[0] == "infer"]
                self.assertEqual(len(calls), 0 if fail in ("start", "enter") else 1)
                self.assertFalse(any(row[1] in ("clip-000001", "clip-000002") for row in events))

    def test_timer_cleanup_fault_is_terminal_after_one_call(self):
        events = []
        fake = FakeSession(events)
        with mock.patch.object(worker.signal, "setitimer", side_effect=(None, OSError("synthetic cleanup"))):
            with self.assertRaises(OSError):
                worker._native_call(fake, "clip-000001", Path("/synthetic"))
        self.assertEqual(events, [("infer", "clip-000001")])

    def test_disabled_release_blocks_before_constructor_and_guard(self):
        for family in contract.FAMILIES:
            for mode in ("prepare", "decode"):
                with self.subTest(family=family, mode=mode), \
                     mock.patch.object(worker, "_json", return_value=contract.read_json(HERE / "execution-release.json")), \
                     mock.patch.object(worker, "_construct_native") as construct, \
                     mock.patch.object(worker, "_verify_closed_sources") as closed:
                    with self.assertRaisesRegex(ValueError, "disabled"):
                        worker.run_family(family, mode)
                    construct.assert_not_called()
                    closed.assert_not_called()

    def test_direct_real_constructor_and_loader_also_fail_disabled(self):
        with mock.patch.object(worker, "_json", return_value=contract.read_json(HERE / "execution-release.json")), \
             mock.patch.object(worker.importlib.util, "spec_from_file_location") as importer:
            with self.assertRaisesRegex(ValueError, "disabled"):
                worker._construct_native("qwen06", Path("/runtime"), Path("/input"), Path("/output"), {}, "900", "d" * 40)
            with self.assertRaisesRegex(ValueError, "disabled"):
                worker._fixed_module("retained_worker")
            importer.assert_not_called()

    def test_current_worker_rejects_archived_v1_release_before_source_or_model_work(self):
        with mock.patch.object(worker, "_json", return_value=contract.disabled_release()) as read, \
             mock.patch.object(worker, "_construct_native") as construct, \
             mock.patch.object(worker, "_verify_closed_sources") as closed:
            with self.assertRaisesRegex(ValueError, "source-selected v2 execution release required"):
                worker.run_family("qwen06", "decode")
            self.assertEqual(read.call_count, 1)
            construct.assert_not_called()
            closed.assert_not_called()

    def test_zero_eligible_and_hidden_conditioning_fail_before_load(self):
        changes = (
            lambda j, p, b: j["clips"].clear(),
            lambda j, p, b: j.update(expected_transcript="synthetic"),
            lambda j, p, b: p.update(host_only_cell_map=[]),
            lambda j, p, b: b.update(seed=1),
            lambda j, p, b: p["audio_order"].reverse(),
        )
        for change in changes:
            with tempfile.TemporaryDirectory() as name, self.assertRaises(ValueError) as caught:
                self.execute(Path(name), change=change)
            self.assertEqual(caught.exception.synthetic_constructs, 0)

    def test_freeze_option_presence_and_bytes_both_match(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            current, frozen = root / "current", root / "frozen"
            current.mkdir(); frozen.mkdir()
            for field in worker._FROZEN_REQUIRED:
                (current / field).write_bytes(b"synthetic")
                (frozen / field).write_bytes(b"synthetic")
            worker._compare_frozen(current, frozen)
            (frozen / worker._FROZEN_OPTIONAL[0]).write_bytes(b"synthetic")
            with self.assertRaisesRegex(ValueError, "roster"):
                worker._compare_frozen(current, frozen)
            (current / worker._FROZEN_OPTIONAL[0]).write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "drift"):
                worker._compare_frozen(current, frozen)

    def test_diagnostics_redact_unknown_path_message_and_locals(self):
        sensitive = "not-to-be-retained-synthetic-secret"
        try:
            raise RuntimeError(sensitive)
        except RuntimeError as exc:
            value = worker._failure_diagnostic(exc)
        body = contract.canonical(value)
        self.assertNotIn(sensitive.encode(), body)
        self.assertNotIn(str(HERE).encode(), body)
        self.assertFalse(value["message_disclosed"])
        self.assertFalse(value["locals_disclosed"])

    def test_import_is_stdlib_only_in_fresh_isolated_process(self):
        source = ("import sys;sys.path.insert(0," + repr(str(HERE)) + ");import native_worker;"
                  "bad={'torch','numpy','transformers','funasr','whisper','qwen_asr'};"
                  "assert not (bad & set(sys.modules)); print('stdlib-only')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", source],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "stdlib-only")

    def test_foreign_cached_contract_rejected_without_registry_replacement(self):
        with tempfile.TemporaryDirectory() as name:
            fake = Path(name)
            (fake / "campaign_contract.py").write_text("marker = 'synthetic foreign module'\n")
            source = ("import sys;sys.path.insert(0," + repr(str(fake)) + ");import campaign_contract;"
                      "sys.path.insert(0," + repr(str(HERE)) + ");import native_worker;"
                      "native_worker._local('campaign_contract')")
            result = subprocess.run([sys.executable, "-I", "-B", "-c", source],
                capture_output=True, text=True, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("foreign fixed-source module cache", result.stderr)

    def test_cli_accepts_no_arbitrary_paths_or_plugins(self):
        for args in (["qwen06", "prepare", "--backend=x"], ["qwen06"], ["other", "prepare"]):
            with self.assertRaises(ValueError):
                worker.main(args)


# The integration fixture contains only invented WAVs and native-return JSON.
# It exercises the actual durable receipt/validator chain without ML packages.
from test_asr18_receipts import ReceiptFixture


class IntegratedWorkerTests(ReceiptFixture):
    def invoke(self, family, *, native_failure=False):
        import shutil
        inputs = self.root / ("input-" + family)
        (inputs / "audio").mkdir(parents=True)
        for clip in self.job["clips"]:
            shutil.copyfile(self.wavs[clip["wav_sha256"]], inputs / clip["audio_path"])
            (inputs / clip["audio_path"]).chmod(0o444)
        seen = []
        fixture = self
        class SyntheticNative:
            def bind(self, clip):
                return clip["audio_id"]
            def infer(self, audio_id, directory):
                # A genuine canonical/durable record is already on disk before
                # control reaches this synthetic stand-in for the one call.
                started = contract.read_json(directory / "started.json")
                fixture.assertEqual(started["audio_id"], audio_id)
                fixture.assertEqual(started["request_id"], "k1asr18-123-" + family + "-" + audio_id)
                fixture.assertFalse((directory / "started.json").stat().st_mode & 0o222)
                seen.append(audio_id)
                if native_failure:
                    raise TimeoutError("invented native timeout")
                row, raw = fixture.result(family)
                if family == "whisper":
                    (directory / "raw-output.json").write_bytes(contract.canonical(raw))
                return row, raw
        with mock.patch.object(worker, "_construct_native", return_value=SyntheticNative()), \
             mock.patch.object(worker.signal, "setitimer"):
            try:
                result = worker._execute_family(family, "decode", runtime=self.root / "runtime", inputs=inputs,
                    output=self.current[family], frozen=self.prepared[family], policy_path=self.policy_path,
                    job=self.job, execution_plan=self.plan, input_binding=self.input, run_id="123", head="c" * 40,
                    policy_sha256=contract.digest(contract.canonical(self.policy)))
                return result, seen
            except BaseException as exc:
                exc.synthetic_calls = seen
                raise

    def test_all_three_worker_calls_use_real_durable_native_validation(self):
        for family in contract.FAMILIES:
            result, seen = self.invoke(family)
            self.assertEqual(result["audio_ids"], self.plan["audio_order"])
            self.assertEqual(seen, self.plan["audio_order"])
            for audio_id in seen:
                completed = contract.read_json(self.current[family] / audio_id / "completed.json")
                self.assertEqual(completed["status"], "success")

    def test_actual_claim_survives_native_failure_and_no_later_directory(self):
        with self.assertRaises(TimeoutError) as caught:
            self.invoke("qwen06", native_failure=True)
        first = self.plan["audio_order"][0]
        self.assertEqual(caught.exception.synthetic_calls, [first])
        self.assertTrue((self.current["qwen06"] / first / "started.json").is_file())
        self.assertFalse((self.current["qwen06"] / first / "completed.json").exists())
        self.assertFalse(any((self.current["qwen06"] / audio_id).exists() for audio_id in self.plan["audio_order"][1:]))
        with self.assertRaisesRegex(ValueError, "restart"):
            self.bind("qwen06")


if __name__ == "__main__":
    unittest.main()
