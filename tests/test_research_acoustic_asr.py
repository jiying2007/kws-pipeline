"""Audio-only and real toy-queue counterexamples; no model is loaded."""
import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_acoustic_asr as a
q, cp = a.queue, a.checkpoints


def wav(values, *, channels=1, rate=16000):
    target = io.BytesIO()
    with wave.open(target, "wb") as writer:
        writer.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(struct.pack("<" + "h" * len(values), *values))
    return target.getvalue()


def save_fixture(folder, descriptor):
    receipt = folder / "explicit-fixture-provider.json"
    q.write_json(receipt, {"schema": cp.SAVE, "status": "saved", "provider": "explicit-test-fixture",
        "object_id": "fixture-only", "provider_response": {"saved_id": "fixture-only"},
        "artifact": descriptor["capsule"], "member_manifest_sha256": descriptor["member_manifest_sha256"]})
    cp.acknowledge(folder, receipt)


def binding(root, ident="aug-fixture", *, noise=False, values=None):
    body = wav(values if values is not None else [1, -1, 2, -2] * 16)
    path = root / (ident + ".wav")
    q.write_new(path, body)
    pcm, floating, measured = a.decode_pcm16(body)
    parent = {"full_text": "现在你好小窝把灯打开", "ctc_mask": True, "ctc_weight": .5,
        "target_ids": list(range(1, 12)), "keyword_targets": [1, 0], "keyword_mask": [True, True],
        "keyword_weight": [.5, .5]}
    row = {"sample_id": ident, "audio": "audio/" + ident + ".wav", "file_sha256": q.sha(body),
        "pcm_sha256": q.sha(pcm), "samples": len(pcm) // 2, "sample_rate_hz": 16000,
        "parent_key": ["a" * 64, "parent-fixture"], "parent_file_sha256": "b" * 64,
        "kind": "recorded-noise-mix" if noise else "digital-level", "recipe": {"digital_gain": .1},
        "source_family_id": "family-fixture", "speaker_id": "speaker-fixture",
        "source_role": "development-only", "split": "development", **a.augment.inherited_labels(parent, noise)}
    return {"variant_row": row, "parent_row": parent, "variant_row_sha256": q.sha(q.canonical(row)),
        "parent_row_sha256": q.sha(q.canonical(parent)), "variant_wav": a.pin(path),
        "original_pcm16_sha256": q.sha(pcm), "decoded_float32_sha256": q.sha(floating),
        "measurements": measured}


def native_results(text):
    return {name: {"model": name, "status": "complete", "boundary": copy.deepcopy(a.batch.BOUNDARY),
                  "text": text, "decoder_terminated": True, "quality_flags": []} for name in a.batch.MODELS}


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_pcm16_float32_is_exact_including_small_values_and_extrema(self):
        values = [-32768, -32767, -1, 0, 1, 32767]
        pcm, floating, measured = a.decode_pcm16(wav(values))
        self.assertEqual(pcm, struct.pack("<6h", *values))
        self.assertEqual(struct.unpack("<6f", floating), tuple(x / 32768 for x in values))
        self.assertIs(measured["normalization"], False)
        self.assertIs(measured["rms_used_as_admission_gate"], False)

    def test_deliberate_attenuation_and_zero_are_observed_without_rms_rejection(self):
        for values in ([1, -1] * 20, [0] * 40):
            _pcm, floating, measured = a.decode_pcm16(wav(values))
            self.assertEqual(len(floating), len(values) * 4)
            self.assertLess(measured["rms"], .0001)
            self.assertEqual(measured["samples"], len(values))

    def test_wrong_sample_rate_channels_and_truncated_pcm_fail(self):
        for body in (wav([0, 1], rate=24000), wav([0, 1], channels=2), wav([1, 2])[:-2]):
            with self.assertRaises((ValueError, EOFError, wave.Error)):
                a.decode_pcm16(body)

    def test_audio_preparation_preserves_original_bytes_and_opaque_decoder_inputs(self):
        ref = binding(self.root)
        result = a.prepare_audio(self.root / "chunk", [ref["variant_row"]["sample_id"]],
                                 {ref["variant_row"]["sample_id"]: ref}, "a" * 64)
        inputs, mapping = a.selective.validate_map(result["inputs"]["path"], result["source_map"]["path"])
        clip = inputs["clips"][0]
        self.assertEqual((Path(result["inputs"]["path"]).parent / clip["original_wav"]["path"]).read_bytes(),
                         Path(ref["variant_wav"]["path"]).read_bytes())
        self.assertEqual(clip["original_pcm"]["sha256"], clip["derived_pcm"]["sha256"])
        self.assertNotIn(ref["parent_row"]["full_text"], json.dumps(inputs, ensure_ascii=False))
        self.assertNotIn("full_text", json.dumps(mapping))
        self.assertEqual(mapping["rows"][0]["quality_reasons"],
                         ["derived-acoustics-not-yet-independently-observed", "digital-level"])
        primary = {m: {"sha256": "a" * 64, "result": native_results(ref["parent_row"]["full_text"])[m]}
                   for m in a.selective.PRIMARY}
        route = a.selective.route_item(mapping["rows"][0], "a" * 64, primary)
        self.assertEqual((route["whisper_state"], route["reasons"]), ("requested", ["source_quality_uncertain"]))
        derived = Path(result["inputs"]["path"]).parent / clip["derived_pcm"]["path"]
        derived.write_bytes(b"x" * derived.stat().st_size)
        with self.assertRaisesRegex(ValueError, "audio identity changed"):
            a.batch.verify_inputs(result["inputs"]["path"])

    def test_same_keyword_but_changed_context_is_a_fulltext_disagreement(self):
        ref = binding(self.root)
        results = native_results(ref["parent_row"]["full_text"])
        results["whisper"]["text"] = "现在你好小窝把门打开"
        view = a.observe_variant(ref, results)
        self.assertEqual(view["conclusion"], "full-transcript-disagreement")
        self.assertIs(view["quarantine_suggested"], True)
        self.assertEqual({tuple(r["keyword_occurrences"].values()) for r in view["observations"].values()}, {(1, 0)})

    def test_agreement_does_not_promote_noise_to_full_ctc_or_human_truth(self):
        ref = binding(self.root, noise=True)
        before = copy.deepcopy(ref)
        result = a.observe_variant(ref, native_results("现在，你好小窝，把灯打开。"))
        self.assertEqual(result["conclusion"], "unanimous-parent-fulltext-match")
        self.assertIs(result["quarantine_suggested"], False)
        self.assertIs(result["inherited_variant_labels"]["ctc_mask"], False)
        self.assertEqual(result["inherited_variant_labels"]["ctc_weight"], 0)
        self.assertIsNone(result["inherited_variant_labels"]["target_ids"])
        self.assertIs(result["human_gold"], False)
        self.assertIs(result["new_loss_weights_created"], False)
        self.assertEqual(ref, before)
        self.assertEqual((result["split"], result["source_family_id"], result["speaker_id"]),
                         ("development", "family-fixture", "speaker-fixture"))

    def test_empty_quality_and_nontermination_keep_native_evidence_uncertain(self):
        ref = binding(self.root)
        result = a.observe_variant(ref, native_results(""))
        self.assertEqual(result["conclusion"], "all-empty-native-transcripts")
        for key, value in (("decoder_terminated", False), ("quality_flags", ["repeat_suspected"])):
            native = native_results(ref["parent_row"]["full_text"]); native["qwen06"][key] = value
            result = a.observe_variant(ref, native)
            self.assertEqual(result["conclusion"], "native-quality-or-termination-uncertain")
            self.assertIs(result["quarantine_suggested"], True)

    def test_peer_or_target_text_boundary_and_missing_engine_are_rejected(self):
        ref = binding(self.root)
        native = native_results(ref["parent_row"]["full_text"])
        native["qwen06"]["boundary"]["target_text_received"] = True
        with self.assertRaisesRegex(ValueError, "NATIVE_RESULT_BOUNDARY"):
            a.observe_variant(ref, native)
        del native["qwen06"]
        with self.assertRaisesRegex(ValueError, "ALL_THREE_OBSERVATIONS"):
            a.observe_variant(ref, native)

    def test_duplicate_package_declaration_is_not_a_fresh_retry_budget(self):
        stage_path = self.root / "stage.json"; q.write_json(stage_path, {"explicit_fixture": True})
        stage = {"runtime": str(self.root), "declaration_registration": str(self.root / "registrations/a.json")}
        plan = {"variant_manifest": a.pin(stage_path)}
        a.register_declaration(stage_path, stage, plan)
        with self.assertRaisesRegex(ValueError, "ALREADY_DECLARED"):
            a.register_declaration(stage_path, stage, plan)


class QueueDefinitionTests(unittest.TestCase):
    def test_exact124_scope_uses_native_recipe_and_bounded_16_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = [f"aug-{n:03d}" for n in range(124)]
            stage = {"root": str(root), "chunks": [ids[n:n+8] for n in range(0, 124, 8)],
                     "asr_python": sys.executable, "asr_requests": []}
            for index in range(16):
                request = root / (str(index) + ".json")
                q.write_json(request, {"schema": "active-asr-selective-request-v2", "explicit_fixture": index})
                stage["asr_requests"].append(a.pin(request))
            definition = a.queue_definition(root / "stage.json", stage)
            self.assertEqual((len(definition["jobs"]), definition["global_budget_minutes"]), (16, 160))
            self.assertEqual([len(x) for x in stage["chunks"]], [8] * 15 + [4])
            for index, job in enumerate(definition["jobs"]):
                self.assertEqual(job["recipe"], stage["asr_requests"][index]["path"])
                self.assertEqual(job["recipe_sha256"], stage["asr_requests"][index]["sha256"])
                self.assertEqual(job["timeout_minutes"], 10)
                self.assertEqual(job["depends_on"], [a.job_id(index - 1)] if index else [])
                command = q.expanded_argv(root / "queue", job)
                self.assertNotIn("{recipe}", command)
                self.assertEqual(command[command.index("--chunk-index") + 1], str(index))


class PrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lock = self.root / "asr/heavy-model.lock"; self.lock.parent.mkdir()
        context_root = self.root / "context"; context_root.mkdir()
        handoff_root = self.root / "handoff"; handoff_root.mkdir()
        def metadata(path, value):
            path.parent.mkdir(parents=True, exist_ok=True); q.write_json(path, value); return a.pin(path)
        context_validator = self.root / "context-owner/tools/research_context_v3_campaign.py"
        context_validator.parent.mkdir(parents=True); q.write_new(context_validator, b"# explicit old owner identity fixture\n")
        handoff_validator = self.root / "handoff-owner/tools/research_context_v3_metadata_handoff.py"
        handoff_validator.parent.mkdir(parents=True); q.write_new(handoff_validator, b"# explicit handoff identity fixture\n")
        original_stage = metadata(self.root / "original-stage.json", {"explicit_original_stage_fixture": True})
        driver = {"executable": sys.executable}
        campaign = context_root / "campaign.json"
        q.write_json(campaign, {"root": str(context_root), "stage": original_stage, "driver_runtime": driver,
            "asr_runtime": str(self.lock.parent),
            "pipeline_files": {"tools/research_context_v3_campaign.py": q.file_record(context_validator)}})
        selection = self.root / "selection.json"; q.write_json(selection, {"explicit_pre_asr_fixture": True})
        jobs = [f"generate-{n:03d}" for n in range(1, 10)]
        status = {"schema": a.HANDOFF, "completed_chunks": jobs, "driver_active": False,
            "budget": {"failed_jobs": [], "unresolved_jobs": [], "actual_total_is_complete": True,
                       "overrun_nanoseconds": 0, "new_overrun_nanoseconds": 0}}
        self.gate = cp.Gate(max_wait_seconds=0, on_wait=save_fixture)
        failure = metadata(self.root / "old-nonmodel-failure.json", {"explicit_nonmodel_failure_fixture": True})
        failure_folder = self.root / "old-failure-checkpoint"
        self.gate.checkpoint(failure_folder, {"explicit_no_model_fixture": True, "failure": failure}, [failure["path"]], intent=False)
        readers = {}
        for revision in ("context-v3-r1", "context-v3-metadata-r2"):
            source = self.root / revision / "automation/explicit_reader_fixture.py"
            source.parent.mkdir(parents=True); q.write_new(source, b"# explicit pinned reader fixture\n")
            readers[revision] = {"revision": revision, "root": str(self.root / revision),
                "source_files_sha256": {"automation/explicit_reader_fixture.py": q.file_record(source)["sha256"]}}
        handoff = handoff_root / "handoff.json"
        q.write_json(handoff, {"schema": a.HANDOFF, "root": str(handoff_root), "parent_campaign": a.pin(campaign),
            "original_validator": a.pin(context_validator), "original_stage": original_stage,
            "driver_runtime": driver, "heavy_lock": str(self.lock),
            "pipeline_files": {"tools/research_context_v3_metadata_handoff.py": q.file_record(handoff_validator)},
            "original_completed_chunks": jobs[:4], "continuation_chunks": jobs[4:], "read_only_generation_jobs": jobs[4:5],
            "new_generation_jobs": jobs[5:], "automatic_retries": 0, "model_concurrency": 1,
            "initial_status": metadata(self.root / "initial-status.json", {"explicit_initial_status_fixture": True}),
            "failure_receipt": failure, "failure_checkpoint": str(failure_folder),
            "original_generation_results": {ident: metadata(self.root / "original-results" / (ident + ".json"),
                {"explicit_terminal_fixture": ident}) for ident in jobs[:5]}})
        self.stage = {"context_campaign": a.pin(campaign), "context_validator": a.pin(context_validator),
                      "context_handoff": a.pin(handoff), "handoff_validator": a.pin(handoff_validator),
                      "ingestion_selection": a.pin(selection), "heavy_lock": str(self.lock)}
        status["handoff"] = a.pin(handoff)
        closures, packages = [], []
        for index, ident in enumerate(jobs):
            base = context_root if index < 4 else handoff_root
            revision = "context-v3-r1" if index < 4 else "context-v3-metadata-r2"
            complete = base / "chunks" / ident / "complete.json"
            complete.parent.mkdir(parents=True); q.write_json(complete, {"explicit_no_model_fixture": ident})
            folder = base / "checkpoints" / ident / "chunk-complete"
            self.gate.checkpoint(folder,
                {"campaign": a.pin(campaign), "phase": "chunk-complete", "job_id": ident,
                 "completion": a.pin(complete)}, [campaign, complete], intent=False)
            closures.append({"job_id": ident, "reader_revision": revision,
                             "completion": a.pin(complete), "checkpoint": a.phase_metadata(folder)})
            manifest = metadata(complete.parent / "weak-view/MANIFEST.json", {"explicit_weak_fixture": ident})
            packages.append({"path": str(Path(manifest["path"]).parent), "manifest_sha256": manifest["sha256"], "reader": readers[revision]})
        final = metadata(handoff_root / "complete.json", {"explicit_handoff_complete_fixture": True})
        final_folder = handoff_root / "checkpoints/handoff-complete"
        self.gate.checkpoint(final_folder, {"phase": "handoff-complete", "handoff": a.pin(handoff), "completion": final},
                             [handoff, final["path"]], intent=False)
        self.context_closure = {"status": status, "campaign": a.pin(campaign), "handoff": a.pin(handoff),
            "handoff_validator": a.pin(handoff_validator), "completion": final,
            "completion_checkpoint": a.phase_metadata(final_folder), "closures": closures,
            "packages": packages, "readers": readers, "new_model_calls": 0}
        ingestion_root = self.root / "ingestion"; ingestion_root.mkdir()
        self.prepared = ingestion_root / "prepared"; self.prepared.mkdir()
        self.preparation_request = {"schema": a.INGESTION_PREP, "campaign": a.pin(campaign),
            "campaign_validator": a.pin(context_validator), "handoff": a.pin(handoff),
            "handoff_validator": a.pin(handoff_validator), "selection": a.pin(selection)}
        q.write_json(self.prepared / "request.json", self.preparation_request)
        q.write_json(self.prepared / "campaign-closure.json", self.context_closure)
        q.write_json(self.prepared / "MANIFEST.json", {name: q.file_record(self.prepared / name)
            for name in ("request.json", "campaign-closure.json")})
        owner_root = self.root / "owner-source/training"; owner_root.mkdir(parents=True)
        self.validator = owner_root / "research_context_v3_consume.py"
        # This explicitly replaces only the separately tested large upstream
        # consumer. The fresh process still authenticates all actual fixture
        # capsule ZIPs/acks and the immutable prepared closure remains bound.
        code = ("import json,sys\nfrom pathlib import Path\nsys.path.insert(0," + repr(str(ROOT / "tools")) + ")\n"
            "import research_d2_checkpoints as cp\n"
            "def load_stage(path):\n    return {'explicit_no_model_fixture': True}\n"
            "def validate_request(value):\n    return value\n"
            "def handoff_closure(request):\n"
            "    value=json.loads(Path(" + repr(str(self.prepared / "campaign-closure.json")) + ").read_bytes())\n"
            "    for item in [x['checkpoint'] for x in value['closures']]+[value['completion_checkpoint']]:\n"
            "        cp.verify_ack(Path(item['capsule.json']['path']).parent)\n"
            "        if any(cp.pin(p['path'])!=p for p in item.values()): raise ValueError('fixture closure drift')\n"
            "    return value\n")
        q.write_new(self.validator, code.encode())
        self.stage["ingestion_validator"] = a.pin(self.validator)
        original_recipe = ingestion_root / "recipe.json"
        q.write_json(original_recipe, {"schema": "kws-context-v3-ingestion-job-v1",
            "selection_sha256": self.stage["ingestion_selection"]["sha256"], "optimizer_steps": 25,
            "prepared": str(self.prepared), "prepared_manifest_sha256": q.file_record(self.prepared / "MANIFEST.json")["sha256"],
            "explicit_no_model_fixture": True})
        request = {"schema": q.SCHEMA, "queue_id": "explicit-prerequisite-fixture", "global_budget_minutes": 10,
            "jobs": [{"job_id": "context-v3-ingestion", "recipe": str(original_recipe),
                "recipe_sha256": q.file_record(original_recipe)["sha256"],
                "argv": [sys.executable, "-I", "-S", "-B", "-c",
                         "from pathlib import Path;import sys;(Path(sys.argv[1])/'proof.json').write_text('{}');raise SystemExit(1 if Path(sys.argv[2]).exists() else 0)",
                         "{output_dir}", str(ingestion_root / "explicit-fail-marker")],
                "cwd": str(ingestion_root), "depends_on": [], "timeout_minutes": 10, "outputs": ["proof.json"]}]}
        q.write_json(ingestion_root / "queue-request.json", request)
        q.create(ingestion_root / "queue", ingestion_root / "queue-request.json",
                 ROOT / "configs/research/active-plan-20261105.json", self.lock)
        self.ingestion = {"schema": "kws-context-v3-ingestion-stage-v1", "root": str(ingestion_root),
            "heavy_lock": str(self.lock), "source_hashes": {"training/research_context_v3_consume.py": q.file_record(self.validator)["sha256"]},
            "automatic_retries": 0, "worker_seconds": 120, "queue_minutes": 10,
            "pending_intent_restart_allowed": False, "runtime": {"executable": sys.executable},
            "queue": a.pin(ingestion_root / "queue/queue.json"), "request": a.pin(ingestion_root / "queue/request.json"),
            "active_plan": a.pin(ingestion_root / "queue/active-plan.json"),
            "recipe": a.pin(ingestion_root / "queue/recipes/context-v3-ingestion.json"), "original_recipe": a.pin(original_recipe)}
        self.ingestion_path = ingestion_root / "stage.json"; q.write_json(self.ingestion_path, self.ingestion)

    def close_ingestion(self, *, readback=True, failed=False):
        root = Path(self.ingestion["root"])
        if failed: q.write_new(root / "explicit-fail-marker", b"explicit fixture failure\n")
        report, code = q.run(root / "queue", max_jobs=1)
        self.assertEqual((report["status"], code), ("failed_or_blocked", 1) if failed else ("completed", 0))
        common = {"stage": a.pin(self.ingestion_path), "recipe": self.ingestion["recipe"],
            "queue": self.ingestion["queue"], "heavy_lock": str(self.lock),
            "queue_result": a.pin(root / "queue/jobs/context-v3-ingestion/result.json")}
        sources = [self.ingestion_path, self.validator, *cp.regular_tree(root / "queue")]
        self.gate.checkpoint(root / "checkpoints/training-terminal",
            {**common, "phase": "training-terminal", "terminal_state": "failed" if failed else "completed"}, sources, intent=False)
        if readback and not failed:
            path = root / "readback.json"; q.write_json(path, {"explicit_no_model_fixture": True})
            self.gate.checkpoint(root / "checkpoints/readback-complete",
                {**common, "phase": "readback-complete", "readback": a.pin(path), "failure": None}, sources + [path], intent=False)

    def test_all_nine_closures_and_actual_ingestion_terminal_readback_metadata_are_retained(self):
        self.close_ingestion()
        result = a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)
        self.assertEqual((len(result["context_closures"]), result["ingestion_terminal_state"], result["new_model_calls"]),
                         (9, "completed", 0))
        paths = set(a.pinned_metadata_paths(result))
        for folder in (self.root / "context/checkpoints/generate-004/chunk-complete",
                       self.root / "handoff/checkpoints/generate-009/chunk-complete",
                       self.root / "handoff/checkpoints/handoff-complete",
                       self.root / "ingestion/checkpoints/training-terminal",
                       self.root / "ingestion/checkpoints/readback-complete"):
            self.assertTrue({folder / n for n in ("capsule.json", "ack.json", "provider-save.json", "phase-marker.json")} <= paths)
        self.assertIn(self.root / "ingestion/readback.json", paths)
        self.assertIn(self.root / "handoff/chunks/generate-009/complete.json", paths)
        self.assertIn(self.prepared / "MANIFEST.json", paths)
        self.assertIn(self.prepared / "request.json", paths)
        self.assertIn(self.prepared / "campaign-closure.json", paths)
        self.assertIn(self.validator, paths)
        self.assertEqual(result["schema"], "kws-acoustic-asr-prerequisites-v2")
        self.assertFalse(any(p.suffix in (".zip", ".wav", ".f32", ".pt") for p in paths))

    def test_pending_ingestion_blocks_acoustic_launch_even_when_context_is_closed(self):
        with self.assertRaisesRegex(ValueError, "INGESTION_STILL_ACTIVE"):
            a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)

    def test_completed_training_without_durable_readback_still_blocks(self):
        self.close_ingestion(readback=False)
        with self.assertRaises((ValueError, FileNotFoundError)):
            a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)

    def test_one_unclosed_context_chunk_blocks_even_if_training_completed(self):
        self.close_ingestion()
        (self.root / "context/checkpoints/generate-004/chunk-complete/ack.json").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)

    def test_missing_final_handoff_ack_blocks_even_with_all_chunks_and_training_closed(self):
        self.close_ingestion()
        (self.root / "handoff/checkpoints/handoff-complete/ack.json").unlink()
        with self.assertRaisesRegex(ValueError, "MIXED_INGESTION_CLOSURE_VERIFICATION_FAILED"):
            a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)

    def test_prepared_request_or_closure_cannot_change_after_recipe_pin(self):
        self.close_ingestion()
        path = self.prepared / "request.json"
        value = a.read(path); value["schema"] = "kws-context-v3-ingestion-preparation-v1"
        path.write_bytes(q.canonical(value))
        with self.assertRaisesRegex(ValueError, "INGESTION_PREPARATION_METADATA_CHANGED"):
            a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)

    def test_same_bytes_at_a_new_handoff_validator_path_are_not_the_prepared_binding(self):
        self.close_ingestion()
        original = Path(self.stage["handoff_validator"]["path"])
        replacement = self.root / "replacement-handoff/tools" / original.name
        replacement.parent.mkdir(parents=True); replacement.write_bytes(original.read_bytes())
        changed = {**self.stage, "handoff_validator": a.pin(replacement)}
        with self.assertRaisesRegex(ValueError, "INGESTION_HANDOFF_CHANGED"):
            a.prerequisite_readback(changed, self.ingestion_path, self.validator)

    def test_durably_failed_ingestion_is_stopped_and_never_requires_a_retry(self):
        self.close_ingestion(failed=True)
        result = a.prerequisite_readback(self.stage, self.ingestion_path, self.validator)
        self.assertEqual(result["ingestion_terminal_state"], "failed")
        self.assertIsNone(result["ingestion_readback_checkpoint"])
        self.assertEqual(result["ingestion_readback_files"], {})
        self.assertEqual(result["new_model_calls"], 0)
        self.assertEqual(a.queue.result_for(Path(self.ingestion["root"]) / "queue",
            q.load_queue(Path(self.ingestion["root"]) / "queue")[2]["jobs"][0],
            q.load_queue(Path(self.ingestion["root"]) / "queue")[3])["status"], "failed")

    def test_unrelated_ingestion_source_cannot_replace_predeclared_validator(self):
        self.close_ingestion()
        replacement = self.root / "replacement/training/research_context_v3_consume.py"
        replacement.parent.mkdir(parents=True); replacement.write_bytes(self.validator.read_bytes())
        with self.assertRaisesRegex(ValueError, "PREDECLARED_INGESTION_SOURCE_CHANGED"):
            a.prerequisite_readback(self.stage, self.ingestion_path, replacement)


class QueueOwnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage_path = self.root / "stage.json"
        self.stage = {"root": str(self.root), "chunks": [["aug-fixture-one"], ["aug-fixture-two"]],
            "heavy_lock": str(self.root / "heavy-model.lock"), "asr_requests": []}
        q.write_json(self.root / "prerequisites.json", {"explicit_no_model_fixture": True})
        jobs = []
        for index in range(2):
            ident = a.job_id(index)
            folder = self.root / "chunks" / ident; folder.mkdir(parents=True)
            native = folder / "asr-request.json"; q.write_json(native, {"explicit_no_model_fixture": index})
            self.stage["asr_requests"].append(a.pin(native))
            recipe = self.root / "recipes" / (ident + ".json"); recipe.parent.mkdir(exist_ok=True)
            q.write_json(recipe, {"explicit_owner_fixture": index})
            marker = self.root / (ident + ".called-once.json")
            code = """import json,os,sys
from pathlib import Path
output=Path(sys.argv[1]);marker=Path(sys.argv[2])
sys.path.insert(0,sys.argv[3])
import research_acoustic_asr as a
stage_path=Path(sys.argv[4]);stage=a.read(stage_path)
# Only the large upstream input/prerequisite verifier is replaced. Actual
# run_job, queue environment, inherited lease and live intent verification run.
a.load_stage=lambda path:(stage,{},None,None)
a.verify_prerequisites=lambda stage:None
def explicit_no_model_native(request_path):
    a.require(os.environ['KWS_RESEARCH_RECIPE_SHA256']==a.queue.file_record(request_path)['sha256'],'fixture_native_recipe_mismatch')
    a.require(os.environ['KWS_RESEARCH_RECIPE']==str(request_path),'fixture_native_recipe_path_mismatch')
    value={'owner':os.environ['KWS_ACOUSTIC_ASR_OWNER'],'intent':os.environ['KWS_ACOUSTIC_ASR_INTENT']}
    with marker.open('x') as stream:json.dump(value,stream)
    (output/'proof.json').write_text(json.dumps(value))
    return {'state':{'status':'complete'}}
a.selective.run=explicit_no_model_native
raise SystemExit(a.run_job(stage_path,int(sys.argv[5]),Path(sys.argv[6])))
"""
            jobs.append({"job_id": ident, "recipe": str(native), "recipe_sha256": q.file_record(native)["sha256"],
                "argv": [sys.executable, "-I", "-S", "-B", "-c", code, "{output_dir}", str(marker),
                         str(ROOT / "tools"), str(self.stage_path), str(index), "{recipe}"],
                "cwd": str(self.root), "depends_on": [a.job_id(index-1)] if index else [],
                "timeout_minutes": .1, "outputs": ["proof.json"]})
        request = {"schema": q.SCHEMA, "queue_id": "explicit-acoustic-owner-fixture", "global_budget_minutes": 1,
                   "jobs": jobs}
        q.write_json(self.root / "queue-request.json", request)
        q.create(self.root / "queue", self.root / "queue-request.json", ROOT / "configs/research/active-plan-20261105.json",
                 Path(self.stage["heavy_lock"]))
        q.write_json(self.stage_path, self.stage)
        loaded = (self.stage, {"explicit_fixture": True}, request, "a" * 64)
        for name, options in (("load_stage", {"return_value": loaded}), ("verify_prerequisites", {"return_value": None}),
                              ("JOBS", {"new": 2})):
            patcher = mock.patch.object(a, name, **options); patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch.object(a, "checkpoint_sources", side_effect=self.sources)
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch.object(a, "save_or_verify_view", side_effect=self.view)
        patcher.start(); self.addCleanup(patcher.stop)
        patcher = mock.patch.object(a, "aggregate", side_effect=self.aggregate)
        patcher.start(); self.addCleanup(patcher.stop)
        self.gate = cp.Gate(max_wait_seconds=0, on_wait=save_fixture)

    def sources(self, stage, index):
        root = Path(stage["root"])
        return [self.stage_path, root / "prerequisites.json", *cp.regular_tree(root / "recipes"),
            root / "queue/queue.json", root / "queue/request.json", root / "queue/active-plan.json",
            *cp.regular_tree(root / "queue/recipes"), *cp.regular_tree(root / "queue/jobs" / a.job_id(index)),
            *cp.regular_tree(root / "chunks" / a.job_id(index))]

    def view(self, stage, _plan, index):
        cp.verify_ack(self.root / "checkpoints" / a.job_id(index) / "asr-complete")
        value = {"observed_rows": 1, "asr_calls": 0, "by_conclusion": {"explicit_no_model_fixture": 1}}
        path = self.root / "chunks" / a.job_id(index) / "validation.json"
        if not path.exists(): q.write_json(path, value)
        return value, a.pin(path)

    def aggregate(self, stage, _plan):
        for index in range(2): a.verify_chunk_closure(stage, index)
        folder = self.root / "validation"
        if not folder.exists():
            folder.mkdir(); q.write_json(folder / "MANIFEST.json", {"explicit_no_model_fixture": True})
        return {"explicit_no_model_fixture": True}, a.pin(folder / "MANIFEST.json")

    def test_real_queues_require_every_native_and_view_closure_before_next_chunk(self):
        calls, real_run = [], q.run
        def launch(*args, **kwargs):
            if calls: a.verify_chunk_closure(self.stage, 0)
            calls.append(1)
            return real_run(*args, **kwargs)
        with mock.patch.object(q, "run", side_effect=launch):
            result, code = a.run(self.stage_path, gate=self.gate)
        self.assertEqual((code, result["status"], len(calls)), (0, "completed", 2))
        for index in range(2):
            marker = q.loads((self.root / (a.job_id(index) + ".called-once.json")).read_bytes())
            self.assertEqual(marker["owner"], self.gate.owner)
            a.verify_chunk_closure(self.stage, index)
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            a.run(self.stage_path, gate=cp.Gate(max_wait_seconds=0, on_wait=save_fixture))
            execute.assert_not_called()
        self.assertNotIn(a.OWNER_ENV, os.environ)

    def test_no_ack_means_no_claim_or_model_launch(self):
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            with self.assertRaisesRegex(ValueError, "SAVE_NOT_ACKNOWLEDGED"):
                a.run(self.stage_path, gate=cp.Gate(max_wait_seconds=0))
            execute.assert_not_called()
        self.assertFalse((self.root / "queue/jobs/observe-001/claim.json").exists())

    def test_pending_saved_intent_cannot_grant_a_new_owner_launch(self):
        a.checkpoint(self.gate, self.stage_path, self.stage, 0, "before-asr")
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            with self.assertRaisesRegex(ValueError, "AMBIGUOUS_INTENT_AFTER_RESTART"):
                a.run(self.stage_path, gate=cp.Gate(max_wait_seconds=0, on_wait=save_fixture))
            execute.assert_not_called()

    def test_orphan_receipt_is_durably_closed_before_halt_without_relaunch(self):
        root, _, request, qsha = q.load_queue(self.root / "queue")
        q.write_json(root / "jobs/observe-001/claim.json", {"schema": "kws-local-research-job-claim-v1",
            **q.job_binding(request["jobs"][0], qsha), "attempt": 1, "automatic_retry": False,
            "explicit_orphan_fixture": True})
        with mock.patch.object(q, "run", wraps=q.run) as execute:
            result, code = a.run(self.stage_path, gate=self.gate)
            execute.assert_not_called()
        self.assertEqual((result["status"], code), ("unknown", 1))
        folder = self.root / "checkpoints/observe-001/asr-terminal"
        cp.verify_ack(folder)
        self.assertEqual(cp.verify_capsule(folder)["context"]["native_job_result"],
                         a.pin(root / "jobs/observe-001/result.json"))
        self.assertFalse((root / "jobs/observe-002/claim.json").exists())

    def test_view_failure_keeps_native_terminal_durable_and_next_chunk_unclaimed(self):
        with mock.patch.object(a, "save_or_verify_view", side_effect=ValueError("explicit view counterexample")):
            with self.assertRaisesRegex(ValueError, "explicit view counterexample"):
                a.run(self.stage_path, gate=self.gate)
        cp.verify_ack(self.root / "checkpoints/observe-001/asr-complete")
        self.assertFalse((self.root / "queue/jobs/observe-002/claim.json").exists())

    def test_busy_slot_keeps_live_gate_until_single_claim(self):
        calls, real_run = [], q.run
        def launch(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                report = q.inspect(args[0]); report["status"] = "waiting_for_heavy_model_slot"
                return report, 75
            return real_run(*args, **kwargs)
        with mock.patch.object(q, "run", side_effect=launch), mock.patch.object(a.time, "sleep"):
            _result, code = a.run(self.stage_path, gate=self.gate)
        self.assertEqual((code, len(calls)), (0, 3))
        for index in range(2):
            self.assertEqual(q.loads((self.root / "queue/jobs" / a.job_id(index) / "claim.json").read_bytes())["attempt"], 1)

    def test_status_never_counts_an_empty_ack_as_durable(self):
        folder = self.root / "checkpoints/observe-001/chunk-complete"
        folder.mkdir(parents=True); (folder / "ack.json").write_bytes(b"{}")
        with self.assertRaises((ValueError, KeyError, FileNotFoundError)):
            a.status(self.stage_path)

    def test_worker_preserves_exact_native_recipe_binding_and_receipt_exit_code(self):
        recipe = self.root / "queue/recipes/observe-001.json"
        environment = {"KWS_RESEARCH_RECIPE_SHA256": q.file_record(recipe)["sha256"],
            "KWS_RESEARCH_RECIPE": str(recipe), "KWS_RESEARCH_JOB_ID": "observe-001",
            "KWS_RESEARCH_OUTPUT_DIR": str(self.root / "queue/jobs/observe-001/outputs")}
        with (mock.patch.dict(os.environ, environment), mock.patch.object(a, "verify_intent") as intent,
              mock.patch.object(a.batch, "heavy_model_slot") as slot,
              mock.patch.object(a.selective, "run", return_value={"state": {"status": "complete"}}) as native):
            slot.return_value.__enter__.return_value = SimpleNamespace(supervised=True, inherited=True)
            self.assertEqual(a.run_job(self.stage_path, 0, recipe), 0)
            native.assert_called_once_with(recipe)
            intent.assert_called_once_with(self.stage_path, self.stage, 0)
            native.return_value = {"state": {"status": "incomplete"}}
            self.assertEqual(a.run_job(self.stage_path, 0, recipe), 1)
            os.environ["KWS_RESEARCH_OUTPUT_DIR"] = str(self.root / "wrong")
            with self.assertRaisesRegex(ValueError, "WORKER_SCOPE"):
                a.run_job(self.stage_path, 0, recipe)


if __name__ == "__main__":
    unittest.main()
