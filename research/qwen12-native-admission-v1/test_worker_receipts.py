#!/usr/bin/env python3
"""Invented fixtures only: no model imports, downloads or real inference.

All three resolver start/completion/transcript chains below come from the actual
prospective emit functions. TrustedFixture only supplies invented trust root,
WAV, native configs and dependencies; its old manufactured envelopes are removed.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
DEFAULT_BASE = next((parent for parent in Path(__file__).resolve().parents
    if (parent / "tools/asr_native_contracts.py").is_file()), HERE)
BASE = Path(os.environ.get("KWS_NATIVE_BASE", str(DEFAULT_BASE)))
SOURCE = HERE / "source" if (HERE / "source").is_dir() else BASE
for folder in (BASE / "tools", BASE / "tests", BASE / "research/qwen6_asr", BASE / "research/qwen6_asr/core"):
    sys.path.insert(0, str(folder))
sys.path.insert(0, str(SOURCE / "research/qwen12-native-admission-v1"))
import campaign_contract as campaign
spec = importlib.util.spec_from_file_location("prospective_worker_receipts", SOURCE / "research/qwen12-native-admission-v1/worker_receipts.py")
w = importlib.util.module_from_spec(spec); sys.modules[spec.name] = w; spec.loader.exec_module(w)
from asr_resolver_fixtures import TrustedFixture
from asr_native_fixtures import rebind_native_fixture
from asr_stage.decoding import qwen_transcript
from sense_parser import corrected_sense_extract
import asr_source_resolver as resolver
from speech_asr_consensus import validate_recognizer_policy, admission_from_consensus


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="invented-native-worker-receipts-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixture = f = TrustedFixture(self.root)
        f.use_native_v2()
        # Only this synthetic receipt fixture replaces one permitted source in
        # the in-memory pins. Production constants and exclusion remain strict;
        # unpatched original-cohort negatives are covered in FixedSlotTests.
        sources=list(campaign.ORIGINAL_SOURCES)
        sources[1]=('clip-000002',w.digest(f.audio),w.digest(b"\xe8\x03"*160),160)
        patcher=patch.object(campaign,'ORIGINAL_SOURCES',tuple(sources));patcher.start();self.addCleanup(patcher.stop)
        self.job=campaign.original_job()
        for field,value in (('ORIGINAL_JOB_SHA256',w.digest(w.canonical(self.job))),
                            ('ORIGINAL_JOB_BYTES',len(w.canonical(self.job)))):
            patcher=patch.object(campaign,field,value);patcher.start();self.addCleanup(patcher.stop)
        self.prepared = {}; self.current = {}; self.bindings = {}
        source_id = "qwen12-" + w.digest(f.audio)
        for value in (f.record, f.documents["rights.json"], f.documents["source-policy.json"], *f.record["observations"]):
            value["source_id"] = source_id
        f.record["source_provenance"]["rights_evidence_sha256"] = w.digest(w.canonical(f.documents["rights.json"]))
        f.documents["source-policy.json"]["source_provenance"] = copy.deepcopy(f.record["source_provenance"])
        for index, family in enumerate(w.FAMILIES):
            prefix = f"asr-{index}"
            code = f.documents[prefix + "/source-lock.json"]
            model = f.documents[prefix + "/model-lock.json"]
            config = f.documents[prefix + "/native-config.json"]
            code["workflow_run_id"] = "1"
            config["source_lock_sha256"] = w.digest(w.canonical(code))
            capture = f.native_captures[index]
            capture["source_lock_sha256"] = config["source_lock_sha256"]
            rebind_native_fixture(config, capture)
            frozen, current = self.root / "frozen" / family, self.root / "current" / family
            frozen.mkdir(parents=True); current.mkdir(parents=True)
            self.prepared[family] = frozen; self.current[family] = current
            for name, value in zip(w.FILES, (code, model, config)):
                (frozen / name).write_bytes(w.canonical(value)); (frozen / name).chmod(0o444)
                (current / name).write_bytes(w.canonical(value))
            for name in ("started.json", "receipt.json", "raw-output.json"):
                del f.documents[prefix + "/" + name]
        self.policy = w.build_policy(self.prepared)
        f.record["recognizer_policy"] = copy.deepcopy(self.policy)
        self.policy_sha = w.digest(w.canonical(self.policy))
        f.record["recognizer_policy_sha256"] = self.policy_sha
        self.policy_path = self.root / "policy.json"
        self.policy_path.write_bytes(w.canonical(self.policy)); self.policy_path.chmod(0o444)
        f.source.chmod(0o444)
        for index, family in enumerate(w.FAMILIES):
            observation = f.record["observations"][index]
            observation["request_id"] = "qwen12-1-"+family+"-clip-000002"
            observation["recognizer_id"] = family
            observation["recognizer_policy_sha256"] = self.policy_sha
            observation["recognizer_identity_sha256"] = w.digest(w.canonical(self.policy["recognizers"][index]))
            refs = f.record["evidence_bindings"]["recognizers"][index]
            refs["recognizer_id"] = family
            refs["receipt"] = f.ref(f"asr-{index}/completed.json")
            refs["raw_output"] = f.ref(f"asr-{index}/raw-transcript.json")
            self.bindings[family] = self.bind(family)

    def bind(self, family):
        return w.bind_predecode(recognizer_id=family, policy_path=self.policy_path,
            expected_policy_sha256=self.policy_sha, frozen_dir=self.prepared[family],
            current_dir=self.current[family], run_id="1", executed_head_sha="a" * 40, original_job=self.job)

    def start(self, family="qwen06", name=None):
        directory = self.root / (name or "call-" + family)
        directory.mkdir()
        index = w.FAMILIES.index(family)
        return w.emit_start(self.bindings[family], directory, wav_path=self.fixture.source,
            expected_file_sha256=w.digest(self.fixture.audio),
            request_id=self.fixture.record["observations"][index]["request_id"],audio_id="clip-000002")

    def result(self, family):
        index = w.FAMILIES.index(family)
        text = self.fixture.record["observations"][index]["actual_text"]
        capture = copy.deepcopy(self.fixture.native_captures[index])
        termination = capture["termination"]
        if family == "qwen06":
            decoded = "language Chinese<asr_text>" + text
            row, metadata = qwen_transcript(decoded, termination["generated_token_ids"])
            raw = {"decoded_skip_special_tokens": decoded, "decoded_with_special_tokens": decoded + "<|im_end|>",
                "token_ids": termination["generated_token_ids"], "metadata": metadata,
                "native_v2_validated": True, "native_capture": capture}
        elif family == "sensevoice":
            decoded = "<|zh|><|NEUTRAL|><|Speech|><|woitn|>" + text
            row = corrected_sense_extract(decoded, termination["lexical_token_ids"])
            metadata = row.pop("metadata")
            raw = {"raw_result": ([{"key": "invented-blind-clip-id", "text": decoded}], {}),
                "token_evidence": {"token_ids": termination["raw_token_ids"], "decoded": decoded},
                "token_binding": {key: copy.deepcopy(termination[key]) for key in
                    ("raw_token_ids", "metadata_token_ids", "lexical_token_ids", "metadata_boundary_verified", "raw_redecode_matches_capture")},
                "metadata": metadata, "native_v2_validated": True, "native_capture": capture}
            raw["token_binding"].update(raw_decoded_sha256=w.digest(decoded.encode()),
                parser_contract_sha256=self.policy["recognizers"][index]["native_decode_config"]["effective_config"]["parser_contract_sha256"])
        else:
            row = None
            raw = {"raw_text": text, "native_capture": capture, "decoder_completion_validated": True,
                "full_input_processed": True, "returned_token_ids": termination["generated_token_ids"][:-1]}
        return row, raw

    def finish(self, family="qwen06", attempt=None, result=None):
        attempt = attempt or self.start(family)
        row, raw = result or self.result(family)
        if family == "whisper":
            (attempt.directory / "raw-output.json").write_bytes(w.canonical(raw))
        return attempt, w.emit_completion(attempt, row=row, raw=raw)

    def emitted_fixture(self):
        f = self.fixture
        for index, family in enumerate(w.FAMILIES):
            attempt, receipt = self.finish(family)
            for path in attempt.directory.iterdir():
                # Preserve exact actual emission bytes, not reconstructed envelopes.
                f.extra_members[f"asr-{index}/{path.name}"] = path.read_bytes()
            f.record["observations"][index]["raw_output_sha256"] = receipt["raw_output_sha256"]
        f.rebuild()
        return f

    def test_actual_three_family_emissions_resolve_unchanged(self):
        f = self.emitted_fixture()
        with patch.object(resolver, "_REPO_ROOT", f.root), patch.object(resolver, "_SOURCE_ROOTS", (f.trust_root,)):
            context = resolver.resolve_source_context(f.record, evidence_dir=f.store)
            proof = resolver.verify_source_context(context, f.record)
            self.assertEqual(proof["source_identity"]["pcm_sha256"], f.record["pcm_sha256"])
            admission = admission_from_consensus(f.record, "f" * 64, verification_context=context)
            self.assertTrue(admission["ctc_training_allowed"])
            self.assertEqual(admission["consensus_record"]["actual_text"], "你好小窝")
        for index, family in enumerate(w.FAMILIES):
            raw = f.extra_members[f"asr-{index}/raw-transcript.json"]
            transcript = json.loads(raw)
            native = transcript["native_output"]
            self.assertEqual(native["sha256"], w.digest(f.extra_members[f"asr-{index}/" + native["path"]]))
            self.assertEqual(native["bytes"], len(f.extra_members[f"asr-{index}/" + native["path"]]))
            self.assertEqual(transcript["actual_text"], f.record["observations"][index]["actual_text"])
            row = transcript["native_row"]
            if family == "whisper":
                self.assertIsNone(row)
            else:
                row_bytes = f.extra_members[f"asr-{index}/" + row["path"]]
                self.assertEqual(row["sha256"], w.digest(row_bytes))
                self.assertEqual(row["bytes"], len(row_bytes))
                self.assertEqual(json.loads(row_bytes)["raw_text"], transcript["actual_text"])

    def test_start_all_common_fields_and_readonly(self):
        attempt = self.start(); start = json.loads(attempt.started_bytes)
        self.assertEqual(start["schema"], "retained-asr-started-v1")
        self.assertEqual(start["source_id"], "qwen12-" + w.digest(self.fixture.audio))
        self.assertEqual(start["audio_frames"], 160)
        self.assertEqual(start["pcm_sha256"], w.digest(b"\xe8\x03" * 160))
        self.assertFalse((attempt.directory / "started.json").stat().st_mode & 0o222)
        self.assertFalse((attempt.directory / "completed.json").exists())
        self.assertEqual(start["recognizer_policy_sha256"], self.policy_sha)
        self.assertEqual(start["run_id"], "1"); self.assertEqual(start["executed_head_sha"], "a" * 40)

    def test_duplicate_start_different_directory_rejected(self):
        self.start()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.start(name="duplicate")

    def test_duplicate_start_existing_files_fresh_binding_rejected(self):
        attempt = self.start(); self.bindings["qwen06"] = self.bind("qwen06")
        with self.assertRaisesRegex(ValueError, "existing attempt"):
            w.emit_start(self.bindings["qwen06"], attempt.directory, wav_path=self.fixture.source,
                expected_file_sha256=w.digest(self.fixture.audio),request_id="qwen12-1-qwen06-clip-000002",audio_id="clip-000002")

    def test_duplicate_completion_rejected(self):
        attempt, _ = self.finish()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            w.emit_completion(attempt, row=self.result("qwen06")[0], raw=self.result("qwen06")[1])

    def test_native_incomplete_preserves_raw_no_completion_no_retry(self):
        attempt = self.start(); row, raw = self.result("qwen06")
        raw["native_v2_validated"] = False
        raw["native_capture"]["termination"]["generated_token_ids"] = [17, 18]
        with self.assertRaises(ValueError): w.emit_completion(attempt, row=row, raw=raw)
        self.assertEqual((attempt.directory / "raw-native-output.json").read_bytes(), w.canonical(raw))
        self.assertFalse((attempt.directory / "completed.json").exists())
        self.assertFalse((attempt.directory / "raw-transcript.json").exists())
        with self.assertRaisesRegex(ValueError, "duplicate"):
            w.emit_completion(attempt, row=self.result("qwen06")[0], raw=self.result("qwen06")[1])

    def test_false_adapter_flag_rejected_even_valid_capture(self):
        for family in ("qwen06", "sensevoice", "whisper"):
            with self.subTest(family=family):
                attempt = self.start(family); row, raw = self.result(family)
                raw["decoder_completion_validated" if family == "whisper" else "native_v2_validated"] = False
                with self.assertRaises(ValueError): w.emit_completion(attempt, row=row, raw=raw)
                self.assertFalse((attempt.directory / "completed.json").exists())

    def test_fullpolicy_hash_drift_before_bind(self):
        with self.assertRaisesRegex(ValueError, "full-policy hash"):
            w.bind_predecode(recognizer_id="qwen06", policy_path=self.policy_path,
                expected_policy_sha256="0" * 64, frozen_dir=self.prepared["qwen06"],
                current_dir=self.current["qwen06"], run_id="1", executed_head_sha="a" * 40,original_job=self.job)

    def mutate(self, path, value):
        path.chmod(0o644); path.write_bytes(w.canonical(value)); path.chmod(0o444)

    def test_fullpolicy_drift_between_start_completion(self):
        attempt = self.start(); policy = copy.deepcopy(self.policy)
        policy["recognizers"][2]["model_revision"] = "changed"
        self.mutate(self.policy_path, policy)
        with self.assertRaisesRegex(ValueError, "full-policy bytes"):
            w.emit_completion(attempt, row=self.result("qwen06")[0], raw=self.result("qwen06")[1])
        self.assertFalse((attempt.directory / "completed.json").exists())

    def test_missing_policy_rejected(self):
        self.policy_path.unlink()
        with self.assertRaises(FileNotFoundError): self.bind("qwen06")

    def test_writable_policy_rejected(self):
        self.policy_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "read-only"): self.bind("qwen06")

    def test_unfrozen_policy_rejected(self):
        policy = copy.deepcopy(self.policy); policy["frozen_before_recognition"] = False
        self.mutate(self.policy_path, policy); self.policy_sha = w.digest(w.canonical(policy))
        with self.assertRaisesRegex(ValueError, "frozen"): self.bind("qwen06")

    def test_policy_missing_family_rejected(self):
        policy = copy.deepcopy(self.policy); policy["recognizers"].pop()
        self.mutate(self.policy_path, policy); self.policy_sha = w.digest(w.canonical(policy))
        with self.assertRaisesRegex(ValueError, "three"): self.bind("qwen06")

    def test_target_field_in_policy_rejected(self):
        policy = copy.deepcopy(self.policy); policy["intended_text"] = "invented target"
        self.mutate(self.policy_path, policy); self.policy_sha = w.digest(w.canonical(policy))
        with self.assertRaisesRegex(ValueError, "public"): self.bind("qwen06")

    def test_current_config_lock_drift_before_bind(self):
        for filename in w.FILES:
            with self.subTest(filename=filename):
                original = (self.current["qwen06"] / filename).read_bytes()
                (self.current["qwen06"] / filename).write_bytes(original + b"\n")
                with self.assertRaises(ValueError): self.bind("qwen06")
                (self.current["qwen06"] / filename).write_bytes(original)

    def test_current_config_drift_after_start(self):
        attempt = self.start()
        (self.current["qwen06"] / "native-config.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "configuration or locks drift"):
            w.emit_completion(attempt, row=self.result("qwen06")[0], raw=self.result("qwen06")[1])
        self.assertFalse((attempt.directory / "completed.json").exists())

    def test_frozen_config_writable_rejected(self):
        (self.prepared["qwen06"] / "native-config.json").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "read-only"): self.bind("qwen06")

    def test_real_lineage_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "lineage"):
            w.bind_predecode(recognizer_id="qwen06", policy_path=self.policy_path,
                expected_policy_sha256=self.policy_sha, frozen_dir=self.prepared["qwen06"],
                current_dir=self.current["qwen06"], run_id="2", executed_head_sha="a" * 40,original_job=self.job)

    def test_raw_row_text_mismatch_rejected(self):
        for family in ("qwen06", "sensevoice"):
            with self.subTest(family=family):
                attempt = self.start(family); row, raw = self.result(family)
                row["raw_text"] = "invented substituted transcript"
                with self.assertRaisesRegex(ValueError, "transcript mismatch"):
                    w.emit_completion(attempt, row=row, raw=raw)
                self.assertFalse((attempt.directory / "completed.json").exists())

    def test_sense_rich_raw_text_mismatch_rejected(self):
        attempt = self.start("sensevoice"); row, raw = self.result("sensevoice")
        raw["raw_result"][0][0]["text"] += "invented added text"
        with self.assertRaisesRegex(ValueError, "association"):
            w.emit_completion(attempt, row=row, raw=raw)

    def test_capture_input_coverage_mismatch_rejected(self):
        attempt = self.start(); row, raw = self.result("qwen06")
        raw["native_capture"]["input"].update(audio_frames=159, processed_end_frame=159)
        with self.assertRaisesRegex(ValueError, "coverage"):
            w.emit_completion(attempt, row=row, raw=raw)
        self.assertFalse((attempt.directory / "completed.json").exists())

    def test_whisper_original_output_preserved(self):
        attempt, receipt = self.finish("whisper")
        raw_bytes = w.canonical(self.result("whisper")[1])
        self.assertEqual((attempt.directory / "raw-output.json").read_bytes(), raw_bytes)
        self.assertEqual((attempt.directory / "raw-native-output.json").read_bytes(), raw_bytes)
        self.assertNotEqual(receipt["raw_output_sha256"], w.digest(raw_bytes))

    def test_whisper_raw_text_substitution_rejected(self):
        attempt = self.start("whisper"); row, raw = self.result("whisper")
        (attempt.directory / "raw-output.json").write_bytes(w.canonical(raw))
        raw["raw_text"] += "invented substitution"
        with self.assertRaisesRegex(ValueError, "original Whisper raw output"):
            w.emit_completion(attempt, row=row, raw=raw)
        self.assertFalse((attempt.directory / "completed.json").exists())

    def test_mutated_started_rejected(self):
        attempt = self.start(); value = json.loads(attempt.started_bytes); value["request_id"] = "other"
        self.mutate(attempt.directory / "started.json", value)
        with self.assertRaisesRegex(ValueError, "started receipt drift"):
            w.emit_completion(attempt, row=self.result("qwen06")[0], raw=self.result("qwen06")[1])

    def test_writable_audio_and_audio_hash_drift_rejected(self):
        self.fixture.source.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "read-only"): self.start()
        self.fixture.source.write_bytes(self.fixture.audio + b"changed")
        self.fixture.source.chmod(0o444)
        with self.assertRaisesRegex(ValueError, "WAV hash"): self.start(name="other-call")

    def test_emitted_receipt_tamper_rejected_by_unchanged_resolver(self):
        f = self.emitted_fixture()
        member = "asr-0/completed.json"
        receipt = json.loads(f.extra_members[member]); receipt["recognizer_policy_sha256"] = "0" * 64
        f.extra_members[member] = w.canonical(receipt); f.rebuild()
        with patch.object(resolver, "_REPO_ROOT", f.root), patch.object(resolver, "_SOURCE_ROOTS", (f.trust_root,)):
            with self.assertRaisesRegex(ValueError, "completion"):
                resolver.resolve_source_context(f.record, evidence_dir=f.store)

    def test_historical_slot_writer_denies_fresh_paths_requests_and_aliases(self):
        old=campaign.historical_slot()
        measured={'file_sha256':old['wav_sha256'],'pcm_sha256':old['pcm_sha256'],
                  'audio_frames':old['audio_frames']}
        for index,(audio_id,source_id,request_id,pcm) in enumerate((
                ('clip-000001',old['source_id'],'qwen12-1-qwen06-clip-000001',old['pcm_sha256']),
                ('clip-000002',old['source_id'],'qwen12-1-qwen06-clip-000002',old['pcm_sha256']),
                ('clip-000002','renamed-source','fresh-request',old['pcm_sha256']),
                ('clip-000002',old['source_id'],'qwen12-1-qwen06-clip-000002','0'*64))):
            directory=self.root/f'old-slot-new-path-{index}';directory.mkdir()
            binding=self.bind('qwen06')
            with patch.object(w,'_wave',return_value={**measured,'pcm_sha256':pcm}):
                with self.subTest(audio_id=audio_id,source_id=source_id,request_id=request_id),                     self.assertRaisesRegex(ValueError,'historical consumed'):
                    w.emit_start(binding,directory,wav_path=self.fixture.source,
                        expected_file_sha256=old['wav_sha256'],request_id=request_id,
                        audio_id=audio_id,source_id=source_id)
            self.assertEqual(list(directory.iterdir()),[])
            self.assertFalse(binding._requests);self.assertFalse(binding._sources)

    def test_writer_original_identity_frames_alias_and_request_drift_reject(self):
        base={'file_sha256':w.digest(self.fixture.audio),'pcm_sha256':w.digest(b"\xe8\x03"*160),
              'audio_frames':160}
        cases=[('file_sha256','0'*64),('pcm_sha256','0'*64),('audio_frames',159),
               ('audio_frames',160.0),('audio_frames',True),('audio_id','clip-000003'),
               ('source_id','renamed-source'),('request_id','new-request')]
        for index,(field,value) in enumerate(cases):
            directory=self.root/f'identity-drift-{index}';directory.mkdir()
            measured=dict(base);kwargs={'audio_id':'clip-000002','source_id':'qwen12-'+base['file_sha256'],
                                      'request_id':'qwen12-1-qwen06-clip-000002'}
            (measured if field in measured else kwargs)[field]=value
            with patch.object(w,'_wave',return_value=measured):
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    w.emit_start(self.bind('qwen06'),directory,wav_path=self.fixture.source,
                        expected_file_sha256=base['file_sha256'],**kwargs)
            self.assertFalse((directory/'started.json').exists())

    def test_binding_original_job_snapshot_is_immutable_and_exact(self):
        for rows in (self.job['clips'][1:],self.job['clips'][::-1],self.job['clips']+[self.job['clips'][0]]):
            bad=copy.deepcopy(self.job);bad['clips']=rows
            with self.assertRaisesRegex(ValueError,'original anonymous job'):
                w.bind_predecode(recognizer_id='qwen06',policy_path=self.policy_path,
                    expected_policy_sha256=self.policy_sha,frozen_dir=self.prepared['qwen06'],
                    current_dir=self.current['qwen06'],run_id='1',executed_head_sha='a'*40,original_job=bad)
        snapshot=self.bindings['qwen06'].original_job_bytes
        self.job['clips'][0]['audio_id']='renamed'
        self.assertEqual(self.bindings['qwen06'].original_job_bytes,snapshot)
        self.assertEqual(json.loads(snapshot)['clips'][0]['audio_id'],'clip-000001')
        self.start()  # Existing binding was not weakened by caller mutation.

    def test_build_policy_without_all_prepare_outputs_rejected(self):
        with self.assertRaisesRegex(ValueError, "three"):
            w.build_policy({family: path for family, path in self.prepared.items() if family != "whisper"})
        self.assertEqual(set(validate_recognizer_policy(self.policy)), set(w.FAMILIES))


if __name__ == "__main__":
    unittest.main()
