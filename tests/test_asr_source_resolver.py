#!/usr/bin/env python3
"""Pure stdlib invented evidence tests. No real speech, model, or ASR is run."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "training")]
import asr_source_resolver as resolver
from asr_resolver_fixtures import trusted_fixture, encoded, sha
from speech_asr_consensus import (admission_from_consensus, validate_consensus_admission,
    validate_consensus_record, require_independent_reference_labels)
from speech_label_admission import canonical_sha256

PRODUCTION_ROOTS = resolver._SOURCE_ROOTS


class SourceResolverMetadataTests(unittest.TestCase):
    def resolve(self, fixture):
        with patch.object(resolver, "_SOURCE_ROOTS", (fixture.trust_root,)):
            return fixture.admission()

    @staticmethod
    def labeled_source(f):
        record = f.record
        row = {"schema_version": 1, "evidence_class": "speech-like-synthetic-recording-v1",
               **copy.deepcopy(record["source_provenance"]), "audio": str(f.source),
               "source_id": record["source_id"], "source_family_id": record["source_family_id"],
               "file_sha256": record["file_sha256"], "pcm_sha256": record["pcm_sha256"],
               "actual_text": record["actual_text"], "tokens": record["actual_tokens"],
               "kind": record["kind"], "keyword_id": record["keyword_id"],
               "provider_kind": "offline-tts", "provider_name": "invented", "provider_version": "1",
               "voice_id": "invented-resolver-voice", "generation_config_sha256": "a" * 64,
               "label_provenance_sha256": "c" * 64, "admission": f.admission()}
        return row

    def test_native_base_materializer_external_loader_and_trainer_reverify(self):
        from materialize_speech_like_base_index import materialize
        from external_base_dataset import load_external_base_bundle
        from test_reviewed_corpus_admission import AdmissionTests
        from test_training_admission import LabelEvidenceAncestryMetadataTests
        human = AdmissionTests(); human.setUp(); self.addCleanup(human.doCleanups)
        config = human.bundle()
        with trusted_fixture() as f:
            f.use_native_v2()
            with patch.object(resolver, "_SOURCE_ROOTS", (f.trust_root,)):
                source = self.labeled_source(f)
                manifest = f.root / "labeled.jsonl"; manifest.write_text(json.dumps(source) + "\n")
                rows, summary = materialize(manifest=manifest, split="train",
                    tokens_path=ROOT / "keywords/tokens.example.txt", keywords_path=ROOT / "keywords/zh_cn_example.tsv",
                    portable_audio_dir=f.root / "portable" / "audio", path_base=f.root / "portable", evidence_dir=f.store)
                self.assertEqual(rows[0]["label_origin"], "machine-consensus")
                index, summary_path = f.root / "portable/index.jsonl", f.root / "portable/summary.json"
                index.write_text(json.dumps(rows[0]) + "\n"); summary_path.write_text(json.dumps(summary))
                config["generator"]["external_base_dataset"]["train"] = {
                    "index": str(index), "summary": str(summary_path),
                    "index_sha256": sha(index.read_bytes()), "summary_sha256": sha(summary_path.read_bytes())}
                loaded, _ = load_external_base_bundle(human.root / "config.json", config, evidence_dir=f.store)
                self.assertEqual(sum(row.get("label_origin") == "machine-consensus" for row in loaded), 1)
                train_manifest, _ = LabelEvidenceAncestryMetadataTests.machine_manifest(f)
                receipt = LabelEvidenceAncestryMetadataTests.train_fixture(train_manifest, f.store)
                self.assertEqual(receipt["manifests"][0]["label_evidence"]["machine_consensus_rows"], 1)
                with self.assertRaises(ValueError):
                    load_external_base_bundle(human.root / "config.json", config, evidence_dir=f.root / "missing")

    def test_portable_materializer_validates_all_rows_before_any_copy(self):
        from materialize_speech_like_base_index import materialize
        with trusted_fixture() as f:
            row = self.labeled_source(f)
            manifest = f.root / "labeled.jsonl"
            manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            destination = f.root / "portable/audio"
            with self.assertRaises(ValueError):
                materialize(manifest=manifest, split="train", tokens_path=ROOT / "keywords/tokens.example.txt",
                    keywords_path=ROOT / "keywords/zh_cn_example.tsv", portable_audio_dir=destination,
                    path_base=destination.parent, evidence_dir=f.store)
            self.assertFalse(destination.exists())

    def test_portable_copy_race_rejects_and_never_returns_success(self):
        import materialize_speech_like_base_index as base
        from test_reviewed_corpus_admission import AdmissionTests
        from speech_label_admission import admission_from_reviews
        for mutation in ("source", "target", "earlier-target"):
            with self.subTest(mutation=mutation), trusted_fixture() as f:
                rows = [self.labeled_source(f)]
                human = AdmissionTests(); human.setUp(); self.addCleanup(human.doCleanups); human.write()
                human_row = copy.deepcopy(human.rows[1])
                human_row.update(audio=str(human.generated / "generalization-search/fixture.wav"),
                    actual_text=human_row["text"], kind="positive", keyword_id=1,
                    label_provenance_sha256="e" * 64,
                    admission=admission_from_reviews([human.reviews[1]], "f" * 64))
                rows.append(human_row)
                manifest = f.root / "labeled.jsonl"
                manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
                destination = f.root / "portable/audio"; original_copy = base.shutil.copyfileobj
                copied = []
                def racing_copy(source, output):
                    if mutation == "source" and not copied:
                        Path(source.name).write_bytes(b"different source")
                    original_copy(source, output); output.flush(); copied.append(Path(output.name))
                    if mutation == "target": Path(output.name).write_bytes(b"different target")
                    elif mutation == "earlier-target" and len(copied) == 2:
                        copied[0].write_bytes(b"late target change")
                with patch.object(base.shutil, "copyfileobj", side_effect=racing_copy), self.assertRaises(ValueError):
                    base.materialize(manifest=manifest, split="train", tokens_path=ROOT / "keywords/tokens.example.txt",
                        keywords_path=ROOT / "keywords/zh_cn_example.tsv", portable_audio_dir=destination,
                        path_base=destination.parent, evidence_dir=f.store)
                self.assertFalse((destination.parent / "summary.json").exists())

    def test_production_has_zero_train_roots_and_screen_sources_stay_rejected(self):
        self.assertEqual([root for root in PRODUCTION_ROOTS if root.allowed_role == "train"], [])
        with trusted_fixture() as f:
            for root in PRODUCTION_ROOTS:
                f.record["evidence_bindings"]["root_id"] = root.root_id
                with patch.object(resolver, "_SOURCE_ROOTS", PRODUCTION_ROOTS), \
                     patch.object(resolver, "_REPO_ROOT", ROOT), self.assertRaisesRegex(ValueError, "screen-only"):
                    f.admission()

    def test_actual_resolver_positive_and_fresh_consumer(self):
        with trusted_fixture() as f:
            admission = f.admission()
            self.assertTrue(admission["ctc_training_allowed"])
            validate_consensus_admission(admission, split="train", frames=160, for_training=True, evidence_dir=f.store)
            with self.assertRaises(ValueError):
                require_independent_reference_labels([{"admission": admission}])

    def test_native_three_families_and_private_conflict_refs_use_actual_resolver(self):
        with trusted_fixture() as f:
            f.use_native_v2()
            f.add_raw_observation("你好小屋", "uncertain", original="invented private: 屋 may differ".encode())
            admission = self.resolve(f)
            self.assertTrue(admission["ctc_training_allowed"])
            self.assertTrue(admission["consensus_record"]["human_conflict"])
            self.assertEqual(admission["consensus_record"]["actual_text"], "你好小窝")
            serialized = json.dumps(admission, ensure_ascii=False)
            self.assertNotIn("invented private", serialized)
            self.assertNotIn("你好小屋", serialized)
            self.assertNotIn("acoustic_complete", serialized)
            self.assertEqual(f.documents["human-observation-1.json"]["observed_text"], "你好小屋")
            # Later clear observation clears only current conflict; history remains.
            f.add_raw_observation("你好小窝", "clear")
            admission = self.resolve(f)
            self.assertFalse(admission["consensus_record"]["human_conflict"])
            self.assertTrue(admission["consensus_record"]["historical_human_conflict"])

    def test_unknown_raw_note_needs_no_fabricated_transcript_or_review_fields(self):
        with trusted_fixture() as f:
            f.use_native_v2(); f.add_raw_observation()
            admission = self.resolve(f)
            self.assertTrue(admission["ctc_training_allowed"])
            self.assertTrue(admission["consensus_record"]["human_conflict"])
            note = f.documents["human-observation-1.json"]
            self.assertIsNone(note["observed_text"])
            self.assertNotIn("actual_tokens", note)
            self.assertNotIn("verdict", note)

    @staticmethod
    def human_review(f):
        from speech_label_admission import TRANSCRIPT_POLICY
        record = f.record
        return {"schema_version": 2, "evidence_class": "speech-like-audio-review-v2",
            **{key: record[key] for key in ("source_id", "file_sha256", "pcm_sha256", "source_family_id", "intended_text")},
            "reviewer_id": "invented-metadata-listener", "review_origin": "human", "review_revision": 1,
            "supersedes_review_sha256": None, "actual_text": "屋", "actual_tokens": ["unknown"],
            "transcript_policy": TRANSCRIPT_POLICY, "acoustic_complete": False, "speech_present": True,
            "kind": "negative", "keyword_id": None, "verdict": "rejected", "allowed_purpose": "ctc-training-only"}

    def test_typed_private_review_keeps_original_revision_and_disagreement_without_veto(self):
        with trusted_fixture() as f:
            f.use_native_v2(); first = self.human_review(f)
            f.add_raw_observation(original=encoded(first), original_format="human-review-v2")
            admission = self.resolve(f)
            self.assertTrue(admission["ctc_training_allowed"])
            self.assertTrue(admission["consensus_record"]["human_conflict"])
            self.assertNotIn(first["reviewer_id"], json.dumps(admission))
            second = {**first, "review_revision": 2, "supersedes_review_sha256": canonical_sha256(first),
                "verdict": "accepted", "actual_text": f.record["actual_text"], "actual_tokens": f.record["actual_tokens"],
                "acoustic_complete": True, "kind": "positive", "keyword_id": 1}
            f.add_raw_observation(original=encoded(second), original_format="human-review-v2")
            admission = self.resolve(f)
            self.assertFalse(admission["consensus_record"]["human_conflict"])
            self.assertTrue(admission["consensus_record"]["historical_human_conflict"])
            # The same private bytes retain the original actual text unchanged.
            self.assertEqual(json.loads(f.extra_members["private-note-1.txt"])["actual_text"], "屋")

    def test_typed_review_cannot_be_hidden_as_opaque_note_or_inline_body(self):
        with trusted_fixture() as f:
            f.use_native_v2(); first = self.human_review(f)
            f.add_raw_observation(original=encoded(first))
            with self.assertRaisesRegex(ValueError, "downgraded"): self.resolve(f)
            f.record["human_review_history"] = [first]
            with self.assertRaisesRegex(ValueError, "private source-bound references"):
                validate_consensus_record(f.record)

    def test_typed_review_explicit_use_ban_still_vetoes(self):
        from speech_asr_consensus import human_conflicts
        with trusted_fixture() as f:
            f.use_native_v2(); first = self.human_review(f)
            first["training_use_prohibited"] = True
            with self.assertRaisesRegex(ValueError, "restriction"):
                f.add_raw_observation(original=encoded(first), original_format="human-review-v2")
            # Bypass fixture convenience recomputation only to reach the actual
            # resolver's independent original-byte check, never its validator.
            f.record.update(human_conflict=True, historical_human_conflict=True)
            f.rebuild()
            with self.assertRaisesRegex(ValueError, "restriction"): self.resolve(f)

    def test_human_history_cannot_be_omitted_or_conflict_erased(self):
        for mutation in ("omit", "erase", "rewrite", "original"):
            with self.subTest(mutation=mutation), trusted_fixture() as f:
                f.use_native_v2(); f.add_raw_observation("屋", "uncertain")
                if mutation == "omit": f.record["human_observation_history"] = []
                elif mutation == "erase": f.record["human_conflict"] = False
                elif mutation == "rewrite": f.documents["human-observation-1.json"]["observed_text"] = "窝"; f.rebuild()
                else: f.extra_members["private-note-1.txt"] = b"changed note"; f.rebuild()
                with self.assertRaises(ValueError): self.resolve(f)

    def test_explicit_training_ban_overrides_native_consensus(self):
        with trusted_fixture() as f:
            f.use_native_v2(); f.add_raw_observation("屋", "uncertain")
            note = f.documents["human-observation-1.json"]
            note["training_use_prohibited"] = True
            ref = f.record["human_observation_history"][0]
            ref["observation_sha256"] = canonical_sha256(note)
            policy = f.documents["source-policy.json"]
            policy["human_observation_history"] = copy.deepcopy(f.record["human_observation_history"])
            policy["human_observation_history_sha256"] = canonical_sha256(policy["human_observation_history"])
            f.rebuild()
            with self.assertRaisesRegex(ValueError, "restriction"): self.resolve(f)

    def test_native_early_policy_rejects_family_aliases_and_fake_legacy_flags(self):
        from speech_asr_consensus import validate_recognizer_policy
        with trusted_fixture() as f:
            f.use_native_v2()
            policy = copy.deepcopy(f.record["recognizer_policy"])
            policy["recognizers"][2]["model_family"] = "another-whisper-family-alias"
            with self.assertRaisesRegex(ValueError, "family"):
                validate_recognizer_policy(policy)
            for field in ("acoustic_complete", "speech_present", "coverage_complete", "transcript_complete"):
                record = copy.deepcopy(f.record); record["observations"][0][field] = True
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "manufacture"):
                    validate_consensus_record(record)

    def test_native_missing_capture_and_unknown_whisper_termination_reject(self):
        for mutation in ("missing", "termination", "input"):
            with self.subTest(mutation=mutation), trusted_fixture() as f:
                f.use_native_v2()
                receipt = f.documents["asr-2/receipt.json"]
                if mutation == "missing": receipt.pop("native_capture")
                elif mutation == "termination": receipt["native_capture"]["termination"]["stop_reason"] = "unknown"
                else: receipt["native_capture"]["input"]["audio_frames"] = 159
                f.rebuild()
                with self.assertRaises(ValueError): self.resolve(f)

    def test_native_hash_dependencies_must_resolve_and_match_loaded_inventory(self):
        for mutation in ("missing", "changed", "unloaded", "generation"):
            with self.subTest(mutation=mutation), trusted_fixture() as f:
                f.use_native_v2()
                ref = f.record["evidence_bindings"]["recognizers"][1]["native_dependencies"]
                if mutation == "missing": ref.pop("tokenizer_sha256")
                elif mutation == "changed": f.extra_members[ref["tokenizer_sha256"]["member"]] = b"changed"
                elif mutation == "unloaded":
                    # Same dependency bytes copied elsewhere cannot impersonate a loaded asset.
                    old = ref["tokenizer_sha256"]["member"]
                    f.extra_members["other.json"] = f.extra_members[old]
                    ref["tokenizer_sha256"]["member"] = "other.json"
                else:
                    key = f.record["evidence_bindings"]["recognizers"][0]["native_dependencies"]["resolved_generation_config_sha256"]["member"]
                    value = json.loads(f.extra_members[key]); value["num_beams"] = True
                    f.extra_members[key] = encoded(value)
                f.rebuild()
                with self.assertRaises(ValueError): self.resolve(f)

    def test_self_authored_report_or_forged_context_never_grants_authority(self):
        with trusted_fixture() as f:
            pending = admission_from_consensus(f.record, "f" * 64)
            self.assertFalse(pending["ctc_training_allowed"])
            for context in ({"verified": True}, object()):
                with self.assertRaises(ValueError):
                    admission_from_consensus(f.record, "f" * 64, verification_context=context)
            with self.assertRaises(ValueError): resolver.VerifiedSourceContext()
            pending.update(ctc_training_allowed=True, status="VERIFIED_SOURCE_TRAIN_LABEL")
            with self.assertRaises(ValueError):
                validate_consensus_admission(pending, split="train", for_training=True, evidence_dir=f.store)

    def test_context_cannot_cross_records_or_outlive_batch(self):
        with trusted_fixture() as f:
            with resolver.source_verification_session(f.store) as session:
                context = resolver.resolve_source_context(f.record, session=session)
                changed = copy.deepcopy(f.record); changed["intended_text"] = "other intent"
                with self.assertRaisesRegex(ValueError, "different record"):
                    resolver.verify_source_context(context, changed)
            with self.assertRaisesRegex(ValueError, "closed batch"):
                resolver.verify_source_context(context, f.record)

    def test_batch_snapshot_reuses_bytes_but_end_rehash_detects_changes(self):
        with trusted_fixture() as f:
            with patch.object(resolver, "_read_bounded", wraps=resolver._read_bounded) as reader:
                with resolver.source_verification_session(f.store) as session:
                    for _ in range(4):
                        context = resolver.resolve_source_context(f.record, session=session)
                        resolver.verify_source_context(context, f.record)
                    self.assertEqual(reader.call_count, 2)  # provenance + archive, not per row
                self.assertEqual(reader.call_count, 4)
            with self.assertRaisesRegex(ValueError, "changed"):
                with resolver.source_verification_session(f.store) as session:
                    resolver.resolve_source_context(f.record, session=session)
                    data = f.archive.read_bytes(); f.archive.unlink(); f.archive.write_bytes(data + b"changed")

    def test_standalone_context_rechecks_changed_files(self):
        with trusted_fixture() as f:
            context = resolver.resolve_source_context(f.record, evidence_dir=f.store)
            f.archive.write_bytes(f.archive.read_bytes() + b"changed")
            with self.assertRaises(ValueError): resolver.verify_source_context(context, f.record)

    def test_trusted_policy_cannot_launder_restricted_role_or_nested_bool_int(self):
        for field, value in (("split", "test"), ("role", "source_screen_only"),
                             ("source_screen_only", True), ("training_use_allowed", False),
                             ("review_origin", "human"), ("synthetic", 1), ("loaded_bytes", True)):
            with self.subTest(field=field), trusted_fixture() as f:
                policy = f.documents["source-policy.json"]
                if field == "synthetic": policy["source_provenance"][field] = value
                elif field == "loaded_bytes": f.documents["asr-0/receipt.json"]["loaded_model_assets"][0]["bytes"] = value
                else: policy[field] = value
                f.rebuild()
                with self.assertRaises(ValueError): self.resolve(f)

    def test_provenance_requires_actual_execution_identity(self):
        from dataclasses import replace
        for field, value in (("executed_head_sha", None), ("run_id", True), ("run_id", 0)):
            with self.subTest(field=field, value=value), trusted_fixture() as f:
                f.provenance[field] = value
                f.provenance_path.write_bytes(encoded(f.provenance))
                f.trust_root = replace(f.trust_root, provenance_sha256=sha(encoded(f.provenance)))
                with self.assertRaisesRegex(ValueError, "actual run"):
                    self.resolve(f)

    def test_unsafe_member_paths_and_archive_inventory_fail(self):
        for mutation in ("path", "inventory", "size", "zip"):
            with self.subTest(mutation=mutation), trusted_fixture() as f:
                if mutation == "path": f.record["evidence_bindings"]["source_audio"]["member"] = "../source.wav"
                elif mutation == "inventory":
                    f.provenance["archives"]["capture"]["exact_members"].pop("source.wav")
                elif mutation == "size": f.provenance["archives"]["capture"]["exact_members"]["source.wav"]["bytes"] = 1
                else: f.archive.write_bytes(f.archive.read_bytes()[:-7])
                if mutation in ("inventory", "size"):
                    f.provenance_path.write_bytes(encoded(f.provenance))
                    from dataclasses import replace
                    f.trust_root = replace(f.trust_root, provenance_sha256=sha(encoded(f.provenance)))
                with self.assertRaises((ValueError, OSError)): self.resolve(f)

    def test_bounded_reader_rejects_symlink_and_reads_at_most_limit_plus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / "small"; path.write_bytes(b"a")
            (root / "link").symlink_to(path)
            with self.assertRaises(ValueError): resolver._read_bounded(root, "link", 4)
            original = os.fstat; calls = []
            def grow_after_first_stat(fd):
                result = original(fd); calls.append(fd)
                if len(calls) == 1: path.write_bytes(b"x" * 10)
                return result
            with patch.object(resolver.os, "fstat", side_effect=grow_after_first_stat), \
                 self.assertRaisesRegex(ValueError, "size bound"):
                resolver._read_bounded(root, "small", 4)

    def test_source_pcm_and_complete_triplet_are_required(self):
        for mutation in ("pcm", "two", "partial", "family", "blank", "oov"):
            with self.subTest(mutation=mutation), trusted_fixture() as f:
                if mutation == "pcm": f.record["pcm_sha256"] = "0" * 64
                elif mutation == "two": f.record["observations"].pop()
                elif mutation == "partial": f.record["observations"][0]["end_frame"] = 159
                elif mutation == "family":
                    policy = f.record["recognizer_policy"]["recognizers"]
                    policy[2]["model_family"] = policy[0]["model_family"]
                else:
                    for observation in f.record["observations"]: observation["actual_text"] = "" if mutation == "blank" else "屋"
                    f.record["actual_text"] = "" if mutation == "blank" else "屋"
                    f.record["actual_tokens"] = []
                with self.assertRaises(ValueError): self.resolve(f)


if __name__ == "__main__":
    unittest.main()
