#!/usr/bin/env python3
"""Fictional native receipts and real selective orchestration; zero ML calls."""
from __future__ import annotations

from argparse import Namespace
import json
import copy
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import active_asr_selective as selective
from research_job_lock import heavy_model_slot, SUPERVISED_ENV


def observation(text="你好，小窝。", complete=True, flags=None):
    return {"result": {"text": text, "decoder_terminated": complete, "quality_flags": flags or []},
            "sha256": "b" * 64}


def fictional_load(model):
    """Shape-correct fictional metadata; no checkpoint or tensors are loaded."""
    value = {"dtype": "float32", "device": "cpu"}
    if model == "whisper":
        revision = selective.batch.read_json(selective.WHISPER_SOURCE)["source_revision"]
        value.update(checkpoint_sha256=selective.WHISPER_SHA, native_source_revision=revision,
            dimensions={"n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 768, "n_audio_head": 12,
                        "n_audio_layer": 12, "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 768,
                        "n_text_head": 12, "n_text_layer": 12})
        return value, {"checkpoint_sha256": selective.WHISPER_SHA, "source_revision": revision}
    sys.path.insert(0, str(ROOT / "research/qwen6_asr/core"))
    from asr_stage.architecture import qwen06_schema, sensevoice_schema
    spec = selective.batch.read_json(selective.MODEL_LOCK)[model]
    schema = qwen06_schema() if model == "qwen06" else sensevoice_schema()
    state = {name: {**row, "sha256": "a" * 64} for name, row in schema.items()}
    verification = {"complete_checkpoint_verification_passed": True, "verified_loaded_state": state}
    if model == "qwen06":
        clean = {key: [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        checkpoint = next(r["sha256"] for r in spec["asset_lock"]["files"] if r["filename"] == "model.safetensors")
        verification.update(schema="qwen-complete-load-receipt-v1", loading_info=clean, verified_alias_proofs=[],
            checkpoint_evidence={"checkpoint_file_sha256": checkpoint, "expected_state_schema": schema,
                "expected_state_schema_sha256": selective.batch.digest(schema), "approved_aliases": [],
                "tensors": copy.deepcopy(state)})
        value.update(loading_info=clean, effective_attention=[
            {"name": f"thinker.{prefix}.layers.{i}.self_attn", "implementation": "eager"}
            for prefix, count in (("audio_tower", 18), ("model", 28)) for i in range(count)])
    else:
        verification.update(schema="sensevoice-exact-apply-receipt-v1",
            loading_info={"missing_keys": [], "unexpected_keys": []},
            verified_loaded_state_sha256=selective.batch.digest(state),
            contract={"expected_schema": schema, "expected_schema_sha256": selective.batch.digest(schema),
                "identity_key_mapping_only": True, "checkpoint_state": copy.deepcopy(state),
                "checkpoint_state_sha256": selective.batch.digest(state)})
        value.update(frontend="kaldi-native-fbank", dither=0.0, no_vad=True, use_itn=False)
    value["loaded_state_verification"] = verification
    return value, {key: spec[key] for key in ("asset_lock_sha256", "source_lock_sha256")}


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.source = {"clip_id": "clip-000", "source_id": "bulk-000000000000000000000000",
                       "quality_uncertain": False, "quality_reasons": []}
        self.primary = {"qwen06": observation(), "sensevoice": observation("你好小窝")}

    def route(self, source=None, primary=None):
        return selective.route_item(source or self.source, "a" * 64, primary or self.primary)

    def test_policy_is_frozen_and_hash_uses_global_identity(self):
        self.assertEqual(selective.batch.read_json(selective.POLICY_PATH), selective.POLICY)
        row = self.route()
        self.assertEqual(row["sample_hash"], "b617ae80346ac9c1e5fb2e1b63d1a0a2fd479814c537019d035d772e49b3c807")
        self.assertEqual(row["whisper_state"], "not_requested")
        self.assertEqual(row["reasons"], ["two_primary_agree_not_sampled"])
        renumbered = self.route(source={**self.source, "clip_id": "clip-027"})
        self.assertEqual(row["sample_hash"], renumbered["sample_hash"])
        audited = self.route(source={**self.source, "source_id": "bulk-000000000000000000000001"})
        self.assertEqual(audited["sample_hash"], "054a426f42f8adb4e233ac2316f69243a4e4eefbabd976b6513325a169324422")
        self.assertEqual(audited["reasons"], ["deterministic_20pct"])
        self.assertEqual(audited["whisper_state"], "requested")

    def test_orthographic_fold_keeps_distinctions_and_english_boundaries(self):
        self.assertEqual(selective.normalize("你好，小 窝。"), "你好小窝")
        for a, b in (("小屋", "小窝"), ("您好", "你好"), ("the rapist", "therapist"),
                     ("we're ready", "were ready"), ("小窝", "小窝小窝")):
            self.assertNotEqual(selective.normalize(a), selective.normalize(b))

    def test_each_mandatory_condition_requests_third_even_when_hash_not_selected(self):
        cases = [({"qwen06": {"result": None, "sha256": None}, "sensevoice": observation()}, "primary_incomplete"),
                 ({"qwen06": observation(complete=False), "sensevoice": observation()}, "primary_incomplete"),
                 ({"qwen06": observation("  。"), "sensevoice": observation()}, "primary_incomplete"),
                 ({"qwen06": observation(flags=["decoding_warning"]), "sensevoice": observation()}, "primary_quality_uncertain"),
                 ({"qwen06": observation(), "sensevoice": observation("你好小屋")}, "primary_disagreement")]
        for primary, reason in cases:
            with self.subTest(reason=reason, primary=primary):
                row = self.route(primary=primary)
                self.assertEqual(row["whisper_state"], "requested")
                self.assertIn(reason, row["reasons"])
        row = self.route(source={**self.source, "quality_uncertain": True, "quality_reasons": ["SILENT_OR_TOO_LOUD"]})
        self.assertEqual(row["reasons"], ["source_quality_uncertain"])

    def test_usable_empty_cannot_fake_agreement_and_absence_has_no_hash(self):
        empty = {m: observation("") for m in selective.PRIMARY}
        self.assertEqual(self.route(primary=empty)["whisper_state"], "requested")
        with self.assertRaisesRegex(ValueError, "missing primary"):
            self.route(primary={"qwen06": {"result": None, "sha256": "b" * 64}, "sensevoice": observation()})

    def test_isolated_queue_entry_bootstraps_only_its_own_tools(self):
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / "tools/active_asr_selective.py"), "--help"],
                                capture_output=True, text=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("declare,run,audit", result.stdout)


class SelectiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runtime = self.root / "runtime"
        self.runtime.mkdir()
        selective.batch.write_once(self.runtime / "setup.json", {"status": "complete", "installed_packages": {},
            "lock_sha256": "f" * 64, "active_python": "fictional-zero-model-fixture"})
        self.input_dir = self.root / "inputs"
        self.input_dir.mkdir()
        clips, source_rows = [], []
        for index in range(4):
            cid, sid = f"clip-{index:03d}", f"bulk-{index:024x}"
            raw = struct.pack("<ffff", .1, -.2, .2, -.1)
            wav = self.input_dir / (cid + ".wav")
            # This test covers identity/orchestration, not a WAV decoder.
            wav.write_bytes(b"fictional-wave-fixture" + raw)
            pcm = self.input_dir / (cid + ".f32")
            pcm.write_bytes(raw)
            sha = selective.sha_file(pcm)
            clips.append({"id": cid, "original_member": "cells/" + sid + "/native-f32.wav",
                "original_wav": {"path": wav.name, "bytes": wav.stat().st_size, "sha256": selective.sha_file(wav)},
                "original_pcm": {"sha256": sha, "channels": 1, "dtype": "<f4", "frames": 4, "sample_rate": 24000},
                "derived_pcm": {"path": pcm.name, "sha256": sha, "channels": 1, "dtype": "<f4", "frames": 4, "sample_rate": 16000}})
            source_rows.append({"clip_id": cid, "source_id": sid, "original_wav_sha256": selective.sha_file(wav),
                "original_pcm_sha256": sha, "derived_pcm_sha256": sha,
                "quality_uncertain": index == 3, "quality_reasons": ["SILENT_OR_TOO_LOUD"] if index == 3 else []})
        self.inputs = self.input_dir / "inputs.json"
        selective.batch.write_once(self.inputs, {"schema": "active-asr-inputs-v1", "source_archive_sha256": "c" * 64,
            "created_at": "fictional", "preparation_source_sha256": "d" * 64,
            "derivation": {"fixture": "no real audio or model execution"}, "clips": clips})
        self.source_map = self.root / "source-map.json"
        selective.batch.write_once(self.source_map, {"schema": "active-asr-source-map-v2",
            "input_manifest_sha256": selective.sha_file(self.inputs), "source_archive_sha256": "c" * 64,
            "plan_sha256": "a" * 64, "rows": source_rows})
        self.output = self.root / "output"
        self.request_path = self.root / "request.json"
        self.request = selective.declare(Namespace(inputs=self.inputs, source_map=self.source_map,
            runtime=self.runtime, output=self.output, request=self.request_path, policy=selective.POLICY_PATH,
            threads=1, budget_seconds=30))
        self.calls = []
        self.divergence = True

    def fictional_worker(self, command, env, log, runtime, remaining):
        args = {command[i]: command[i + 1] for i in range(3, len(command), 2)}
        self.assertEqual(command[2], str(selective.WORKER))
        self.assertEqual(set(args), {"--model", "--runtime", "--inputs", "--output", "--attempt", "--threads", "--ids"})
        self.assertNotIn(str(self.source_map), command)
        model, ids = args["--model"], args["--ids"].split(",")
        self.calls.append((model, ids))
        inputs = selective.batch.read_json(self.inputs)
        identity = self.request["model_identities"][model]
        with heavy_model_slot(runtime / "heavy-model.lock", environment=env) as lease:
            self.assertTrue(lease.inherited and lease.supervised)
            for clip in inputs["clips"]:
                if clip["id"] not in ids:
                    continue
                text = "你好小屋" if self.divergence and model == "sensevoice" and clip["id"] == "clip-002" else "你好小窝"
                if model == "qwen06":
                    native = {"decoded_skip_special_tokens": "language Chinese<asr_text>" + text,
                              "token_ids": [10, 151645]}
                elif model == "sensevoice":
                    tagged = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>" + text
                    native = {"result": [{"text": tagged}], "token_evidence": {"decoded": tagged}}
                else:
                    native = {"text": text, "language": "zh", "effective_options": selective.batch.CONFIGS["whisper"],
                              "pre_finalize_greedy_trace": [{"completed": True}]}
                common = {"identity_sha256": selective.batch.digest({"model_identity": identity, "input": clip}),
                          "model": model, "clip_id": clip["id"], "attempt": "selective-001",
                          "input": clip, "model_identity": identity}
                selective.batch.write_once(self.output / "claims" / model / (clip["id"] + ".json"),
                    {"schema": "active-asr-call-claim-v1", **common})
                path = self.output / "native" / model / (clip["id"] + ".json")
                selective.batch.write_once(path, {"schema": "active-asr-native-output-v1", **common, "native": native})
                projected, terminated, flags = selective.native_audit.native_projection(model, native)
                selective.batch.write_once(self.output / "results" / model / (clip["id"] + ".json"),
                    {"schema": "active-asr-result-v1", **common, "status": "complete", "text": projected,
                     "decoder_terminated": terminated, "quality_flags": flags,
                     "native_output_sha256": selective.sha_file(path), "decoder_source_sha256": self.request["worker_sha256"],
                     "boundary": selective.batch.BOUNDARY})
            loaded, assets = fictional_load(model)
            selective.batch.write_once(self.output / "attempts/selective-001" / (model + "-load.json"), loaded)
            selective.batch.write_once(self.output / "attempts/selective-001" / (model + ".json"),
                {"schema": "active-asr-worker-v1", "model": model, "attempt": "selective-001", "status": "complete",
                 "identity": identity, "source_sha256": self.request["worker_sha256"], "selected": ids,
                 "reused": 0, "unresolved_prior_calls": [], "new_calls": len(ids), "boundary": selective.batch.BOUNDARY,
                 "assets": assets,
                 "heavy_lock": {"inherited": lease.inherited, "supervised": lease.supervised, "process_group": os.getpgrp()}})
        return 0

    def execute_fixture(self):
        # The actual inherited flock is exercised; a test double stands in for
        # kernel installation so this unit-test process retains its environment.
        with heavy_model_slot(self.runtime / "heavy-model.lock") as lease:
            env = lease.child_environment()
            env[SUPERVISED_ENV] = "1"
            env["KWS_RESEARCH_RECIPE_SHA256"] = selective.sha_file(self.request_path)
            with patch.dict(os.environ, env), patch.object(selective.batch.research_runtime, "check_interpreter"), \
                    patch("builtins.print"), \
                    patch.object(selective.research_no_network, "install", return_value={
                        "seccomp_installed": True, "no_new_privileges": True,
                        "environment": selective.research_no_network.ENVIRONMENT}), \
                    patch.object(selective.batch, "run_worker", side_effect=self.fictional_worker):
                return selective.run(self.request_path)

    def test_actual_controller_routes_subset_and_sealed_resume_is_read_only(self):
        final = self.execute_fixture()
        state = final["state"]
        self.assertEqual(state["status"], "complete")
        self.assertEqual(self.calls, [("qwen06", [f"clip-{x:03d}" for x in range(4)]),
                                     ("sensevoice", [f"clip-{x:03d}" for x in range(4)]),
                                     ("whisper", ["clip-001", "clip-002", "clip-003"])])
        self.assertEqual((state["expected_matrix_cells"], state["verified_matrix_cells"], state["original_recognition_calls"]), (11, 11, 11))
        self.assertEqual(state["not_requested_matrix"], [["whisper", "clip-000"]])
        before = {str(p): (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(selective.batch, "run_worker", side_effect=AssertionError("resume loaded a model")):
            self.assertEqual(selective.run(self.request_path), final)
        after = {str(p): (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_unrequested_native_call_or_forged_routing_cannot_be_sealed(self):
        self.execute_fixture()
        path = self.output / "results/whisper/clip-000.json"
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "sealed selective evidence changed"):
            selective.run(self.request_path)
        path.unlink()
        routing = selective.batch.read_json(self.output / "routing.json")
        routing["items"][0]["whisper_state"] = "requested"
        selective.save(self.output / "routing.json", routing)
        with self.assertRaisesRegex(ValueError, "routing differs"):
            selective.run(self.request_path)

    def test_missing_requested_primary_stays_incomplete_even_with_completed_whisper(self):
        self.execute_fixture()
        (self.output / "results/qwen06/clip-002.json").unlink()
        request, inputs, mapping, identities = selective.verify_request(self.request_path)
        selective.save(self.output / "routing.json", selective.routing_for(request, inputs, mapping, identities))
        state = selective.audit_state(request, inputs, mapping, identities)
        self.assertEqual(state["status"], "incomplete")
        self.assertEqual(state["verified_matrix_cells"], 10)
        self.assertTrue((self.output / "results/whisper/clip-002.json").is_file())

    def test_native_text_projection_cannot_be_replaced_by_agreeing_consumer_text(self):
        self.execute_fixture()
        path = self.output / "results/sensevoice/clip-002.json"
        result = selective.batch.read_json(path)
        result["text"] = "你好小窝"
        selective.save(path, result)
        with self.assertRaisesRegex(ValueError, "consumer projection"):
            selective.run(self.request_path)

    def test_unsupervised_and_interrupted_jobs_fail_before_worker(self):
        with patch.object(selective.batch.research_runtime, "check_interpreter"), \
                patch.object(selective.batch, "run_worker") as worker:
            with self.assertRaisesRegex(ValueError, "bounded supervised queue lease"):
                selective.run(self.request_path)
            self.assertFalse(self.output.exists())
            self.output.mkdir()
            with self.assertRaisesRegex(ValueError, "interrupted selective job"):
                selective.run(self.request_path)
            worker.assert_not_called()

    def test_source_quality_and_audio_identity_are_required_before_request(self):
        source = selective.batch.read_json(self.source_map)
        source["rows"][3]["quality_uncertain"] = False
        selective.save(self.source_map, source)
        with self.assertRaisesRegex(ValueError, "source quality"):
            selective.validate_map(self.inputs, self.source_map)
        source["rows"][3]["quality_uncertain"] = True
        source["rows"][0]["original_pcm_sha256"] = "1" * 64
        selective.save(self.source_map, source)
        with self.assertRaisesRegex(ValueError, "audio identity"):
            selective.validate_map(self.inputs, self.source_map)

    def test_source_mutation_and_noncanonical_request_fail_before_calls(self):
        self.request_path.write_text(json.dumps(self.request, ensure_ascii=False, indent=2))
        with self.assertRaisesRegex(ValueError, "canonical request"):
            selective.run(self.request_path)
        selective.save(self.request_path, self.request)
        self.request["source_files"]["tools/active_asr_worker.py"] = "1" * 64
        selective.save(self.request_path, self.request)
        with self.assertRaisesRegex(ValueError, "source/boundary changed"):
            selective.run(self.request_path)

    def test_completed_native_evidence_can_resume_only_its_interrupted_metadata_seal(self):
        self.execute_fixture()
        inventory = self.output / "audit/file-hashes.json"
        before = (inventory.read_bytes(), inventory.stat().st_ino, inventory.stat().st_mtime_ns)
        (self.output / "audit/final-summary.json").unlink()
        with patch.object(selective.batch, "run_worker", side_effect=AssertionError("audit called a model")):
            final = selective.seal_or_verify(self.request_path)
        self.assertEqual(final["state"]["status"], "complete")
        self.assertEqual(before, (inventory.read_bytes(), inventory.stat().st_ino, inventory.stat().st_mtime_ns))
        (self.output / "audit/final-summary.json").unlink()
        inventory.write_bytes(inventory.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "partial audit inventory changed"):
            selective.seal_or_verify(self.request_path)

    def test_missing_or_changed_actual_load_receipts_block_complete(self):
        self.execute_fixture()
        path = self.output / "attempts/selective-001/whisper-load.json"
        raw = path.read_bytes()
        path.unlink()
        args = selective.verify_request(self.request_path)
        state = selective.audit_state(*args)
        self.assertEqual(state["status"], "incomplete")
        self.assertTrue(any(x["scope"] == "native_model_load" for x in state["errors"]))
        value = json.loads(raw)
        value["checkpoint_sha256"] = "1" * 64
        selective.save(path, value)
        self.assertEqual(selective.audit_state(*args)["status"], "incomplete")
        for model in ("qwen06", "sensevoice"):
            loaded, assets = fictional_load(model)
            first = next(iter(loaded["loaded_state_verification"]["verified_loaded_state"]))
            loaded["loaded_state_verification"]["verified_loaded_state"][first]["sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                selective.validate_load(model, loaded, assets)

    def test_no_whisper_selected_requires_no_load_or_call(self):
        source = selective.batch.read_json(self.source_map)
        candidates = [f"bulk-{i:024x}" for i in range(100) if
                      int(selective.agreement_hash("a" * 64, f"bulk-{i:024x}"), 16) * 5 >= 2**256]
        for row, sid in zip(source["rows"], candidates):
            row.update(source_id=sid, quality_uncertain=False, quality_reasons=[])
        selective.save(self.source_map, source)
        self.request["source_map"]["sha256"] = selective.sha_file(self.source_map)
        selective.save(self.request_path, self.request)
        self.divergence = False
        final = self.execute_fixture()
        self.assertEqual(final["state"]["status"], "complete")
        self.assertEqual(final["state"]["original_recognition_calls"], 8)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((self.output / "attempts/selective-001/whisper-load.json").exists())

    def test_actual_vendor_validation_sources_are_pinned(self):
        paths = selective.source_identity()
        for name in ("qwen_loading_gate_v2.py", "checkpoint_guard/exact_apply.py", "checkpoint_guard/checkpoint_contract.py"):
            self.assertIn("research/qwen6_asr/core/vendor/" + name, paths)


if __name__ == "__main__":
    unittest.main()
