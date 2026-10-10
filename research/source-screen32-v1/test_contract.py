#!/usr/bin/env python3
"""Fictional metadata fixtures only; no audio, models, network or installation."""
import copy
import contextlib
import types
from unittest import mock
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("screen32", HERE / "contract.py")
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)


def ledger(generated=32):
    rows = []
    for i, cell in enumerate(c.expected_plan()["cells"]):
        rows.append({"cell_id": cell["cell_id"], "status": "GENERATED" if i < generated else "NOT_RUN",
                     "attempts": int(i < generated), "audio": {
                         "wav_sha256": c.digest(["fictional-wav", i]), "pcm_sha256": c.digest(["fictional-pcm", i]),
                         "frames": 16000, "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2,
                     } if i < generated else None})
    return rows


def primary(job):
    return {name: [{"audio_id": row["audio_id"], "status": "complete", "raw_text": "你好小窝。", "quality_flags": []}
                   for row in job["clips"]] for name in c.ASR_IDS}


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.plan = c.expected_plan()

    def test_committed_freeze(self):
        self.assertEqual(len(c.validate_plan(c.decode((HERE / "plan.json").read_bytes()))), 32)
        self.assertGreaterEqual(c.verify_local_pins(), 15)

    def test_exact_cells(self):
        rows = self.plan["cells"]
        self.assertEqual(len({r["seed"] for r in rows}), 32)
        self.assertEqual(len({r["lineage_group"] for r in rows}), 4)
        self.assertEqual([r["intended_text"] for r in rows[:8]], list(c.TEXTS))
        self.assertTrue(all(r["attempts_max"] == 1 and not r["training_admitted"] for r in rows))

    def test_plan_mutations(self):
        for key, value in [("execution_enabled", True), ("max_tts_calls", 33), ("regeneration", True),
                           ("voice_cloning", True), ("reference_audio", "reference.wav"), ("paid_api", True),
                           ("max_attempts_per_cell", True), ("new_field", "unknown")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan = copy.deepcopy(self.plan); plan[key] = value; c.validate_plan(plan)

    def test_cell_mutations(self):
        for key, value in [("seed", 1), ("intended_text", "你好，小窝。"), ("instruction", "Serena"),
                           ("model_revision", "main"), ("human_gold", True), ("role", "heldout"),
                           ("training_admitted", True), ("attempts_max", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan = copy.deepcopy(self.plan); plan["cells"][0][key] = value; c.validate_plan(plan)

    def test_asr_context_and_boundary_mutations(self):
        for section, key, value in [("asr", "context", "小窝"), ("asr", "hotwords", ["小窝"]),
                                     ("asr", "human_gold", True), ("boundary", "d20", "PASS"),
                                     ("boundary", "d90", "PASS"), ("boundary", "shipping_approved", True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                plan = copy.deepcopy(self.plan); plan[section][key] = value; c.validate_plan(plan)

    def test_reordering_or_missing_cells_rejected(self):
        for rows in [self.plan["cells"][:-1], self.plan["cells"][::-1]]:
            plan = copy.deepcopy(self.plan); plan["cells"] = rows
            with self.assertRaises(ValueError): c.validate_plan(plan)

    def test_strict_json(self):
        for raw in [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b' ' * (1024 * 1024 + 1)]:
            with self.assertRaises(ValueError): c.decode(raw)

    def test_qwen_preview_is_design_only(self):
        preview = c.request_preview(self.plan, "screen32-001")
        self.assertEqual(preview["method"], "generate_voice_design")
        self.assertIn("instruct", preview["kwargs"])
        self.assertNotIn("speaker", preview["kwargs"])
        self.assertNotIn("ref_audio", preview["kwargs"])
        self.assertFalse(preview["execution_enabled"])
        self.assertEqual(preview["kwargs"]["max_new_tokens"], 120)

    def test_firered_preview_stays_blocked(self):
        preview = c.request_preview(self.plan, "screen32-017")
        self.assertIn("CUDA", preview["blocked"])
        self.assertFalse(preview["execution_enabled"])
        self.assertFalse(preview["kwargs"]["do_tn"])
        self.assertFalse(preview["kwargs"]["do_split"])

    def test_zero_call_ledger_retains_denominator(self):
        rows = ledger(0)
        c.validate_generation_ledger(self.plan, rows)
        self.assertEqual(c.blind_job(self.plan, rows)["clips"], [])
        self.assertEqual(len(rows), 32)

    def test_single_source_missing_does_not_fabricate_16(self):
        rows = ledger(16)
        self.assertEqual(len(c.blind_job(self.plan, rows)["clips"]), 16)

    def test_attempt_bool_duplicate_wav_pcm_rejected(self):
        for kind in ("attempts", "wav_sha256", "pcm_sha256"):
            rows = ledger()
            if kind == "attempts": rows[0][kind] = True
            else: rows[1]["audio"][kind] = rows[0]["audio"][kind]
            with self.subTest(kind=kind), self.assertRaises(ValueError): c.validate_generation_ledger(self.plan, rows)

    def test_after_failure_no_retry_or_later_cell(self):
        rows = ledger(); rows[0].update(status="FAILED_NO_RETRY", audio=None)
        with self.assertRaises(ValueError): c.validate_generation_ledger(self.plan, rows)
        rows = ledger(0); rows[0].update(status="FAILED_NO_RETRY", attempts=1)
        c.validate_generation_ledger(self.plan, rows)
        rows[0]["attempts"] = 2
        with self.assertRaises(ValueError): c.validate_generation_ledger(self.plan, rows)

    def test_unavailable_claims_no_audio(self):
        rows = ledger(); rows[0].update(status="NOT_RUN", attempts=0)
        with self.assertRaises(ValueError): c.validate_generation_ledger(self.plan, rows)

    def test_malformed_geometry_rejected(self):
        for key, value in [("frames", True), ("frames", 192001), ("channels", True), ("sample_rate_hz", 24000)]:
            rows = ledger(); rows[0]["audio"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): c.validate_generation_ledger(self.plan, rows)

    def test_blind_projection_does_not_leak_plan_or_order(self):
        rows = ledger(); job = c.blind_job(self.plan, rows)
        self.assertEqual(len(c.validate_blind_job(job)), 32)
        self.assertEqual([a["wav_sha256"] for a in job["clips"]], sorted(a["audio"]["wav_sha256"] for a in rows))
        self.assertNotIn("你好", json.dumps(job, ensure_ascii=False))
        self.assertNotIn("design", json.dumps(job))
        for field in ("intended_text", "voice", "hotwords", "cell_id"):
            bad = copy.deepcopy(job); bad["clips"][0][field] = "leak"
            with self.assertRaises(ValueError): c.validate_blind_job(bad)

    def test_blind_count_and_identity_rejected(self):
        job = c.blind_job(self.plan, ledger()); job["clips"].append(job["clips"][0])
        with self.assertRaises(ValueError): c.validate_blind_job(job)
        job = c.blind_job(self.plan, ledger()); job["clips"][0]["audio_path"] = "../secret.wav"
        with self.assertRaises(ValueError): c.validate_blind_job(job)

    def test_disputes_frozen_and_consensus_never_gold(self):
        job = c.blind_job(self.plan, ledger(16)); observations = primary(job)
        observations["sensevoice"][0]["raw_text"] = "你好小屋"
        observations["sensevoice"][1]["raw_text"] = "你好 小窝"
        result = c.freeze_disputes(job, c.canonical(observations), c.digest(observations))
        self.assertEqual(result["whisper_audio_ids"], ["clip-000001"])
        self.assertFalse(result["human_gold"]); self.assertFalse(result["training_admitted"])
        self.assertFalse(result["whisper_execution_enabled"])
        with self.assertRaises(ValueError): c.freeze_disputes(job, c.canonical(observations), "0" * 64)
        with self.assertRaises(ValueError): c.freeze_disputes(job, c.canonical(observations) + b"\n", c.digest(observations))

    def test_missing_asr_not_laundered_to_dispute_or_agreement(self):
        job = c.blind_job(self.plan, ledger(16)); observations = primary(job)
        observations["sensevoice"][0].update(status="failed", raw_text=None)
        result = c.freeze_disputes(job, c.canonical(observations), c.digest(observations))
        self.assertEqual(result["unresolved_audio_ids"], ["clip-000001"])
        self.assertEqual(result["whisper_audio_ids"], [])
        observations["sensevoice"].pop()
        with self.assertRaises(ValueError): c.freeze_disputes(job, c.canonical(observations), c.digest(observations))

    def test_raw_label_injection_rejected(self):
        job = c.blind_job(self.plan, ledger(16)); observations = primary(job)
        observations["sensevoice"][0]["human_gold"] = True
        with self.assertRaises(ValueError): c.freeze_disputes(job, c.canonical(observations), c.digest(observations))

    def test_qwen_asset_metadata_complete_without_weight_acquisition(self):
        lock = c.decode((HERE / "qwen-model-lock.json").read_bytes())
        self.assertEqual(len(lock["files"]), 13)
        sidecars = [row for row in lock["files"] if row["metadata_kind"] == "nonweight_sidecar"]
        weights = [row for row in lock["files"] if row["metadata_kind"] == "weight_pointer_only"]
        self.assertEqual(len(sidecars), 11)
        self.assertEqual(sum(row["size_bytes"] for row in sidecars), 4468188)
        self.assertEqual(sum(row["size_bytes"] for row in lock["files"]), 4520163832)
        self.assertEqual(len(weights), 2)
        self.assertTrue(all(row["body_sha256_verified"] is False and row["body_acquired"] is False for row in weights))
        self.assertTrue(all(row["body_sha256_verified"] is True and row["body_acquired"] is True for row in sidecars))
        self.assertEqual(lock["weight_body_bytes_acquired"], 0)
        self.assertFalse(lock["execution_ready"])
        for row in lock["files"]:
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$")
            self.assertIn(lock["revision"], row["source_url"])
            self.assertEqual(row["observed_repo_revision"], lock["revision"])
        review = c.decode((HERE / "source-review.json").read_bytes())["qwen_voicedesign"]
        self.assertTrue(review["sidecar_hashes_complete"])
        self.assertEqual(review["remaining_model_sidecars"], [])
        self.assertEqual(review["exact_model_plus_candidate_runtime_bytes"], 4956729799)

    def test_static_wheel_review_is_bound_and_not_runtime_qualification(self):
        review = c.decode((HERE / "source-review.json").read_bytes())["qwen_voicedesign"]
        source = review["exact_wheel_review"]
        lock = c.decode((HERE.parents[1] / "research/qwen6_tts/model-locks.json").read_bytes())
        members = {row["relative_path"]: row["sha256"] for row in lock["qwen_tts"]["source_lock"]}
        self.assertEqual(source["verified_python_member_count"], len(members))
        self.assertEqual(source["size_bytes"], 113529)
        self.assertEqual(source["sha256"], "11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d")
        self.assertEqual(source["model_calls"], 0)
        for row in source["reviewed_members"]:
            self.assertEqual(row["sha256"], members[row["relative_path"]])
        for name in ("archive_published", "installed", "imported"):
            self.assertIs(source[name], False)
        readiness = c.decode((HERE / "readiness.json").read_bytes())
        self.assertIs(readiness["execution_ready"], False)
        self.assertEqual(readiness["qwen_adapter"]["runtime_orchestration"], "IMPLEMENTED_UNEXECUTED")
        self.assertIsNone(readiness["actual_generation_command"])

    def test_no_model_dependencies_imported(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "qwen_tts", "qwen_asr", "funasr", "fireredtts3")))


class QwenAdapterTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("qwen_adapter_test", HERE / "qwen_adapter.py")
        self.a = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.a)
        self.calls = []
        self.real_scope_guard = self.a.verify_runtime_scope
        patcher = mock.patch.object(self.a, "verify_runtime_scope", side_effect=lambda: self.calls.append("scope"))
        self.scope = patcher.start(); self.addCleanup(patcher.stop)
        param = types.SimpleNamespace(device="cpu", dtype="torch.float32")
        attention = types.SimpleNamespace(q_proj=True, k_proj=True, v_proj=True,
                                          config=types.SimpleNamespace(_attn_implementation="eager"))
        def owner():
            return types.SimpleNamespace(training=False, parameters=lambda: iter([param]),
                                         buffers=lambda: iter([param]), named_modules=lambda: iter([("attn", attention)]))
        self.model = owner()
        self.model.tts_model_type = "voice_design"; self.model.tts_model_size = "1b7"
        self.model.tokenizer_type = "qwen3_tts_tokenizer_12hz"; self.model.speaker_encoder = None
        self.model.config = types.SimpleNamespace(talker_config=types.SimpleNamespace(spk_id={}, spk_is_dialect={}))
        self.model.speech_tokenizer = types.SimpleNamespace(model=owner())
        class FakeWave(list):
            ndim = 1
            dtype = "float32"
        self.wave_type = FakeWave
        self.wave = FakeWave([0.0, 0.25, -0.25])
        def generate(**kwargs):
            self.calls.append(("generate", kwargs))
            return [self.wave], 24000
        self.wrapper = types.SimpleNamespace(model=self.model, generate_voice_design=mock.Mock(side_effect=generate))
        self.torch = types.SimpleNamespace(manual_seed=mock.Mock(), inference_mode=contextlib.nullcontext)
        self.numpy = types.SimpleNamespace(random=types.SimpleNamespace(seed=mock.Mock()))

    def adapter(self):
        return self.a.FixedQwenAdapter(self.wrapper, torch=self.torch, numpy=self.numpy)

    def test_exact_api_seed_and_output_contract(self):
        adapter = self.adapter()
        with mock.patch.object(self.a.random, "seed") as seed:
            wave, receipt = adapter.generate_cell("screen32-001")
        seed.assert_called_once_with(101001)
        self.torch.manual_seed.assert_called_once_with(101001)
        self.numpy.random.seed.assert_called_once_with(101001)
        kwargs = self.wrapper.generate_voice_design.call_args.kwargs
        self.assertEqual(kwargs, c.request_preview(c.expected_plan(), "screen32-001")["kwargs"])
        self.assertFalse({"speaker", "ref_audio", "voice_clone_prompt"} & set(kwargs))
        self.assertIs(wave, self.wave)
        self.assertFalse(receipt["human_gold"]); self.assertFalse(receipt["training_admitted"])
        self.assertFalse(receipt["native"]["termination_verified"])
        self.assertEqual(self.calls[:2], ["scope", "scope"])
        self.assertEqual(adapter.attempted, ["screen32-001"])

    def test_wrong_variant_or_size_rejected_before_call(self):
        for name, value in [("tts_model_type", "custom_voice"), ("tts_model_size", "0b6"),
                            ("tokenizer_type", "qwen3_tts_tokenizer_25hz"), ("speaker_encoder", object())]:
            old = getattr(self.model, name); setattr(self.model, name, value)
            with self.subTest(name=name), self.assertRaises(ValueError): self.adapter()
            setattr(self.model, name, old)
        self.wrapper.generate_voice_design.assert_not_called()

    def test_non_cpu_or_precision_rejected(self):
        for device, dtype in [("cuda:0", "torch.float32"), ("cpu", "torch.bfloat16")]:
            self.model.parameters = lambda: iter([types.SimpleNamespace(device=device, dtype=dtype)])
            with self.assertRaises(ValueError): self.adapter()
        self.wrapper.generate_voice_design.assert_not_called()

    def test_codec_attention_or_training_rejected(self):
        codec = self.model.speech_tokenizer.model
        codec.training = True
        with self.assertRaises(ValueError): self.adapter()
        codec.training = False; codec.named_modules = lambda: iter([])
        with self.assertRaises(ValueError): self.adapter()

    def test_stock_voice_table_rejected(self):
        self.model.config.talker_config.spk_id = {"Serena": 1}
        with self.assertRaises(ValueError): self.adapter()

    def test_real_scope_guard_has_no_caller_bypass(self):
        with mock.patch("runtime_scope.read", return_value="0::/not-a-private-root"), self.assertRaises(RuntimeError):
            self.real_scope_guard()
        with self.assertRaises(TypeError):
            self.a.FixedQwenAdapter(self.wrapper, torch=self.torch, numpy=self.numpy, scope_check=lambda: True)
        with mock.patch.object(self.a, "verify_runtime_scope", self.real_scope_guard), mock.patch("runtime_scope.read", return_value="0::/not-a-private-root"):
            with self.assertRaises(RuntimeError): self.adapter()
        self.wrapper.generate_voice_design.assert_not_called()

    def test_live_scope_failure_precedes_attempt(self):
        adapter = self.adapter(); self.scope.side_effect = RuntimeError("mock scope failed")
        with self.assertRaises(RuntimeError): adapter.generate_cell("screen32-001")
        self.wrapper.generate_voice_design.assert_not_called()
        self.assertEqual(adapter.attempted, []); self.assertTrue(adapter.stopped)

    def test_model_exception_consumes_attempt_and_stops_source(self):
        adapter = self.adapter(); self.wrapper.generate_voice_design.side_effect = RuntimeError("mock failure")
        with self.assertRaises(RuntimeError): adapter.generate_cell("screen32-001")
        for cell in ("screen32-001", "screen32-002"):
            with self.assertRaises(ValueError): adapter.generate_cell(cell)
        self.assertEqual(adapter.attempted, ["screen32-001"])
        self.assertEqual(self.wrapper.generate_voice_design.call_count, 1)

    def test_wrong_order_or_firered_has_no_call(self):
        for cell in ("screen32-002", "screen32-017"):
            adapter = self.adapter()
            with self.assertRaises(ValueError): adapter.generate_cell(cell)
        self.wrapper.generate_voice_design.assert_not_called()

    def test_sixteen_call_ceiling_and_no_extra_voice(self):
        adapter = self.adapter()
        for i in range(1, 17): adapter.generate_cell(f"screen32-{i:03d}")
        with self.assertRaises(ValueError): adapter.generate_cell("screen32-017")
        self.assertEqual(self.wrapper.generate_voice_design.call_count, 16)
        self.assertEqual(self.torch.manual_seed.call_args_list[-1], mock.call(101016))

    def test_invalid_output_stops_without_retry(self):
        for result in [([], 24000), ([self.wave], 16000), ([self.wave], True),
                       ([self.wave_type([])], 24000), ([self.wave_type([float("nan")])], 24000),
                       ([self.wave_type([0.1] * 288001)], 24000)]:
            self.wrapper.generate_voice_design.side_effect = None
            self.wrapper.generate_voice_design.return_value = result
            adapter = self.adapter()
            with self.assertRaises(ValueError): adapter.generate_cell("screen32-001")
            self.assertTrue(adapter.stopped)
            self.assertEqual(adapter.attempted, ["screen32-001"])

    def test_signal_flags_do_not_repair_waveform(self):
        for values, flag in [([0.0], "silent"), ([1.25, -0.5], "source_peak_ge_one")]:
            waveform = self.wave_type(values)
            out, receipt = self.a.validate_output([waveform], 24000)
            self.assertIs(out, waveform); self.assertIn(flag, receipt["quality_flags"])
            self.assertEqual(list(out), values)

    def test_no_execution_entry_or_dependency_import(self):
        self.assertFalse(self.a.EXECUTION_READY)
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "qwen_tts", "transformers")))


def load_tests(loader, tests, pattern):
    for filename in ("test_hosted_run.py", "test_setup_adapter.py", "test_asr_worker.py"):
        spec = importlib.util.spec_from_file_location(filename[:-3], HERE / filename)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        tests.addTests(loader.loadTestsFromModule(module))
    return tests


if __name__ == "__main__":
    unittest.main()
