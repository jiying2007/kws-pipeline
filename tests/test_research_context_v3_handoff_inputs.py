#!/usr/bin/env python3
"""Typed mixed-reader projections; fictional PCM, no model execution."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tests")]
import research_context_v3_handoff_inputs as handoff
import research_context_v3_inputs as original
import research_d1_vocab as vocab
from test_research_context_v3_inputs import observed_row, ordinary_row, TOKENS

DATA = os.environ.get("KWS_CONTEXT_METADATA_DATA_ROOT")


def relation():
    return {"schema": handoff.CONTAINER_POLICY, "relation": "peak-timestamp-plus-one-second",
        "returned_wav": {"bytes": 96080, "sha256": "a" * 64},
        "bulk_generation_wav": {"bytes": 96080, "sha256": "b" * 64},
        "native_array": {"bytes": 96128, "sha256": "c" * 64},
        "native_float32_pcm_sha256": "d" * 64, "native_frames": 24000, "native_rate": 24000,
        "bulk_peak_timestamp": 1001, "returned_peak_timestamp": 1000, "differing_byte_offsets": [60],
        "all_other_container_bytes_equal": True, "original_native_array_to_pcm_join_verified": True,
        "signal_samples_changed": False, "original_files_rewritten": False,
        "technical_gate_changed": False, "generation_outcome_changed": False}


def metadata_row(base=None):
    row = observed_row() if base is None else copy.deepcopy(base)
    row.update(consumer="context-v3-metadata-verified-full-transcript-v1",
        source_adapter_schema="kws-active-context-v3-metadata-native-source-v1",
        native_container_relation=relation(), original_pcm_sha256="d" * 64, original_wav_sha256="a" * 64)
    return row


class MetadataProjectionTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = vocab.make_vocabulary(TOKENS.read_bytes())

    def test_full_observed_text_and_both_container_hashes_are_preserved(self):
        row = metadata_row(); before = copy.deepcopy(row)
        projected = handoff.project_metadata_row(row, self.vocabulary)
        self.assertEqual(row, before)
        self.assertEqual(projected["consumer"], handoff.METADATA_ROW_SCHEMA)
        self.assertEqual(projected["native_container_relation"], before["native_container_relation"])
        self.assertEqual(projected["source_adapter_schema"], before["source_adapter_schema"])
        self.assertEqual(projected["target_ids"], [self.vocabulary["token_to_id"][c] for c in row["full_text"]])
        self.assertGreater(len(projected["target_ids"]), 4)
        self.assertIn(2600, projected["target_ids"])
        self.assertFalse(projected["keyword_loss_used"])

    def test_ordinary_speech_keeps_complete_nonblank_targets(self):
        row = metadata_row(ordinary_row())
        projected = handoff.project_metadata_row(row, self.vocabulary)
        self.assertEqual(projected["keyword_targets"], [0, 0])
        self.assertEqual(len(projected["target_ids"]), len(row["full_text"]))
        self.assertNotIn(0, projected["target_ids"])
        self.assertTrue(projected["ctc_mask"])

    def test_uncertain_failed_native_oov_and_geometry_keep_zero_loss(self):
        uncertain = metadata_row(ordinary_row())
        uncertain.update(label_status="uncertain", ctc_mask=False, ctc_weight=0,
            keyword_targets=[None, None], keyword_mask=[False, False], keyword_weight=[0, 0])
        failed = copy.deepcopy(uncertain)
        failed.update(generation_outcome="failed-native", full_text=None, native_container_relation=None)
        oov = metadata_row(ordinary_row("请把灯打开🧭"))
        short = metadata_row(observed_row("你好小窝窝窝")); short["samples"] = 400 + 320 * 6
        for row in (uncertain, failed, oov, short):
            with self.subTest(outcome=row["generation_outcome"], text=row["full_text"]):
                projected = handoff.project_metadata_row(row, self.vocabulary)
                self.assertEqual(projected["full_text"], row["full_text"])
                self.assertIsNone(projected["target_ids"])
                self.assertFalse(projected["ctc_mask"])
                self.assertEqual(projected["keyword_targets"], [None, None])
                self.assertEqual(projected["keyword_weight"], [0, 0])

    def test_readers_and_projections_cannot_cast_each_others_schema(self):
        with self.assertRaisesRegex(ValueError, "ROW_SCOPE"):
            handoff.project_metadata_row(observed_row(), self.vocabulary)
        with self.assertRaisesRegex(ValueError, "ROW_SCOPE"):
            original.project_row(metadata_row(), self.vocabulary)
        row = metadata_row(); row.pop("native_container_relation")
        with self.assertRaisesRegex(ValueError, "RELATION_REQUIRED"):
            handoff.project_metadata_row(row, self.vocabulary)

    def test_changed_timestamp_direction_extent_hash_or_signal_claim_is_rejected(self):
        for key, value in (("bulk_peak_timestamp", 1002), ("bulk_peak_timestamp", 999),
                           ("bulk_peak_timestamp", True), ("differing_byte_offsets", [64]),
                           ("differing_byte_offsets", []), ("differing_byte_offsets", [61, 60]),
                           ("signal_samples_changed", True), ("technical_gate_changed", True),
                           ("native_float32_pcm_sha256", "e" * 64)):
            row = metadata_row(); row["native_container_relation"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                handoff.project_metadata_row(row, self.vocabulary)
        row = metadata_row()
        row["native_container_relation"]["bulk_generation_wav"] = row["native_container_relation"]["returned_wav"]
        with self.assertRaises(ValueError): handoff.project_metadata_row(row, self.vocabulary)

    def test_byte_identical_container_is_explicit_and_failed_native_has_no_relation(self):
        row = metadata_row(); value = row["native_container_relation"]
        value.update(relation="byte-identical", bulk_generation_wav=copy.deepcopy(value["returned_wav"]),
                     bulk_peak_timestamp=None, returned_peak_timestamp=None, differing_byte_offsets=[])
        self.assertTrue(handoff.project_metadata_row(row, self.vocabulary)["ctc_mask"])
        with self.assertRaisesRegex(ValueError, "FAILED_NATIVE"):
            handoff.check_relation(relation(), "failed-native", "d" * 64, "a" * 64)

    def test_keyword_or_vocabulary_changes_cannot_pass_on_container_agreement(self):
        row = metadata_row(); row["keyword_targets"] = [0, 0]
        with self.assertRaises(ValueError): handoff.project_metadata_row(row, self.vocabulary)
        row = metadata_row(); row["requested_vocabulary_sha256"] = "0" * 64
        with self.assertRaises(ValueError): handoff.project_metadata_row(row, self.vocabulary)


class GlobalIdentityTests(unittest.TestCase):
    def test_family_speaker_native_derived_reference_and_parent_aliases_do_not_cross_splits(self):
        fields = ("source_family_id", "speaker_id", "pcm_sha256", "wav_sha256",
                  "parent_source_sha256", "reference_pcm_sha256", "reference_wav_sha256")
        for field in fields:
            with self.subTest(field=field):
                ledger = handoff.GlobalSources()
                ledger.add({"role": "train-candidate", field: "same-alias"})
                with self.assertRaisesRegex(ValueError, "SPLIT_LEAK"):
                    ledger.add({"role": "development-only", field: "same-alias"})
        ledger = handoff.GlobalSources()
        ledger.add({"role": "train-candidate", "reference_pcm_sha256": "same"})
        with self.assertRaisesRegex(ValueError, "SPLIT_LEAK"):
            ledger.add({"role": "development-only", "pcm_sha256": "same"})

    def test_cell_and_original_native_pcm_are_global_even_within_one_split(self):
        for field in ("source_id", "original_pcm_sha256"):
            ledger = handoff.GlobalSources()
            ledger.add({"role": "train-candidate", field: "same"})
            with self.assertRaisesRegex(ValueError, "DUPLICATE"):
                ledger.add({"role": "train-candidate", field: "same"})

    def test_missing_alias_is_not_a_shared_identity_and_same_role_aliases_are_allowed(self):
        ledger = handoff.GlobalSources()
        for role in ("train-candidate", "development-only"):
            ledger.add({"role": role})
        for _ in range(2): ledger.add({"role": "train-candidate", "speaker_id": "same"})
        with self.assertRaisesRegex(ValueError, "ROLE"):
            ledger.add({"role": "independent-test"})


class DeclarationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.reader = self.root / "reader"; self.package = self.root / "package"
        self.reader.mkdir(); self.package.mkdir()
        (self.package / "MANIFEST.json").write_bytes(b"{}\n")
        for name in handoff.READERS["context-v3-metadata-r2"]["files"]:
            path = self.reader / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# explicit source-roster-only fixture; never imported\n")

    def declaration(self, revision="context-v3-r1"):
        return {"path": str(self.package), "manifest_sha256": handoff.sha((self.package / "MANIFEST.json").read_bytes()),
            "reader": {"revision": revision, "root": str(self.reader), "source_files_sha256": {
                name: handoff.sha((self.reader / name).read_bytes()) for name in handoff.READERS[revision]["files"]}}}

    def test_complete_original14_or_metadata15_roster_is_required(self):
        for revision, count in (("context-v3-r1", 14), ("context-v3-metadata-r2", 15)):
            declaration = self.declaration(revision)
            self.assertEqual(len(handoff.validate_declaration(declaration)["files"]), count)
            declaration["reader"]["source_files_sha256"].pop(next(iter(declaration["reader"]["source_files_sha256"])))
            with self.assertRaisesRegex(ValueError, "ROSTER"):
                handoff.validate_declaration(declaration)

    def test_changed_reader_hash_and_package_hash_are_rejected_before_import(self):
        declaration = self.declaration()
        declaration["reader"]["source_files_sha256"][original.DATA_FILES[0]] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SOURCE_CHANGED"):
            handoff.validate_declaration(declaration)
        declaration = self.declaration(); declaration["manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "PACKAGE_PIN_CHANGED"):
            handoff.validate_declaration(declaration)

    def test_revision_unknown_fields_and_symlink_aliases_are_not_inferred(self):
        declaration = self.declaration(); declaration["reader"]["revision"] = "latest"
        with self.assertRaisesRegex(ValueError, "READER_DECLARATION"):
            handoff.validate_declaration(declaration)
        declaration = self.declaration(); declaration["implicit_reader"] = True
        with self.assertRaisesRegex(ValueError, "PACKAGE_DECLARATION"):
            handoff.validate_declaration(declaration)
        alias = self.root / "alias"; alias.symlink_to(self.package, target_is_directory=True)
        declaration = self.declaration(); declaration["path"] = str(alias)
        with self.assertRaisesRegex(ValueError, "PATH_ALIAS"):
            handoff.validate_declaration(declaration)

    def test_duplicate_json_and_nonfinite_numbers_fail_closed(self):
        for body in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.assertRaises(ValueError): handoff.loads(body)

    def test_cli_will_not_overwrite_existing_projection(self):
        output = self.root / "existing.json"; output.write_bytes(b"original\n")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / handoff.PIPELINE_FILES[0]),
            "project", "--declaration", str(self.root / "absent.json"), "--output", str(output)],
            capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"CREATE_ONLY_OUTPUT", result.stderr)
        self.assertEqual(output.read_bytes(), b"original\n")


def build_real_fixtures(data_root, output):
    """Fresh fixture process only: create fictional sources before all receipts."""
    sys.path.insert(0, str(Path(data_root) / "automation"))
    import test_active_context_v3_weak_view as old_fixture
    import test_active_context_v3_metadata_weak_view as metadata_fixture
    new_case = metadata_fixture.ReplayTests(); new_case.setUp()
    fixture = new_case.f
    all_clips = copy.deepcopy(fixture.manifest["clips"])
    weak = old_fixture.weak
    official = TOKENS.read_bytes()
    weak.FSMN_TOKENS_SHA = handoff.sha(official)

    def subset(fixture, indices, reader, name):
        clips = [copy.deepcopy(all_clips[i]) for i in indices]
        fixture.archive.unlink()
        with zipfile.ZipFile(fixture.archive, "x") as archive:
            for clip in clips:
                archive.writestr(clip["original_member"], (fixture.inputs / clip["original_wav"]["path"]).read_bytes())
        fixture.manifest["clips"] = clips
        fixture.manifest["source_archive_sha256"] = weak.sha(fixture.archive.read_bytes())
        (fixture.inputs / "inputs.json").write_bytes(weak.canonical(fixture.manifest) + b"\n")
        fixture.input_sha = weak.sha((fixture.inputs / "inputs.json").read_bytes())
        fixture.mapping = reader.build_source_map(fixture.inputs / "inputs.json", fixture.input_sha,
                                                   fixture.archive, **fixture.options())
        fixture.map_path.write_bytes(weak.canonical(fixture.mapping) + b"\n")
        shutil.rmtree(fixture.asr); fixture.asr.mkdir()
        for clip, index in zip(clips, indices):
            text = (fixture.observed_text if index == fixture.changed_text_index else
                    fixture.missing_keyword_text if index == fixture.missing_keyword_index else
                    fixture.plan["cells"][index]["prompt"])
            for model in weak.MODELS: fixture.asr_fixture(clip, model, text)
        fixture.selective()
        fixture.tokens.write_bytes(official)
        destination = output / name
        reader.create_view(fixture.inputs / "inputs.json", fixture.input_sha, fixture.archive, fixture.asr,
            destination, fsmn_tokens=fixture.tokens, selective_source_map=fixture.map_path,
            selective_policy=fixture.policy_path,
            selective_audit_sha256=weak.sha((fixture.asr / "audit/final-summary.json").read_bytes()), **fixture.options())
        return destination

    try:
        first = subset(fixture, [4, 5, 6, 7], old_fixture.context, "r1")
        second = subset(fixture, [0, 1, 2, 3], metadata_fixture.context, "r2")
        vocabulary = vocab.make_vocabulary(official)
        path = output / "vocabulary.json"; path.write_bytes(vocab.canonical(vocabulary) + b"\n")
        declarations = []
        for package, revision in ((first, "context-v3-r1"), (second, "context-v3-metadata-r2")):
            declarations.append({"path": str(package), "manifest_sha256": handoff.sha((package / "MANIFEST.json").read_bytes()),
                "reader": {"revision": revision, "root": str(Path(data_root).absolute()), "source_files_sha256": {
                    name: handoff.sha((Path(data_root) / name).read_bytes()) for name in handoff.READERS[revision]["files"]}}})
        options = {"expected_plan_sha256": fixture.plan_sha, "vocabulary_path": str(path),
                   "vocabulary_sha256": handoff.sha(path.read_bytes()), "original_tokens_path": str(TOKENS)}
        expected, _ = original.verified_projection([{key: declarations[0][key] for key in ("path", "manifest_sha256")}],
            old_fixture.context, declarations[0]["reader"]["source_files_sha256"], **options)
        handoff.write_new(output / "expected-r1.json", expected)
        handoff.write_new(output / "declaration.json", {"packages": declarations, **options})
    finally:
        new_case.doCleanups()


@unittest.skipUnless(DATA, "set KWS_CONTEXT_METADATA_DATA_ROOT for original/additive native fixture replay")
class RealReaderIntegrationTests(unittest.TestCase):
    def test_fresh_mixed_replay_preserves_original_rows_and_checks_cross_reader_aliases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subprocess.run([sys.executable, "-I", "-S", "-B", str(Path(__file__).absolute()),
                "--build-fixtures", str(Path(DATA).absolute()), str(root)], capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            declaration = handoff.loads((root / "declaration.json").read_bytes())
            before = {str(p): handoff.sha(p.read_bytes()) for name in ("r1", "r2") for p in (root / name).rglob("*") if p.is_file()}
            poison = types.ModuleType("active_asr_weak_view"); poison.__file__ = "/explicit-wrong-reader/active_asr_weak_view.py"
            with mock.patch.dict(sys.modules, {"active_asr_weak_view": poison}):
                rows, report = handoff.aggregate(**declaration)
                self.assertIs(sys.modules["active_asr_weak_view"], poison)
            old_rows = [row for row in rows if row["consumer"] == original.ROW_SCHEMA]
            self.assertEqual(old_rows, handoff.loads((root / "expected-r1.json").read_bytes()))
            self.assertEqual(len(rows), 8)
            self.assertEqual(report["row_consumer_schemas"], {original.ROW_SCHEMA: 4, handoff.METADATA_ROW_SCHEMA: 4})
            self.assertEqual(report["reader_revision_rows"], {"context-v3-r1": 4, "context-v3-metadata-r2": 4})
            self.assertEqual(len(report["metadata_reconciled_cell_ids"]), 1)
            self.assertEqual((report["planned_rows"], len(report["unobserved_cell_ids"])), (72, 64))
            self.assertTrue(report["fresh_process_per_package"])
            self.assertTrue(report["cross_reader_reference_aliases_checked"])
            self.assertTrue(all(row["keyword_weight"] == [0, 0] for row in rows if not row["ctc_mask"]))
            self.assertEqual(handoff.aggregate(**declaration), (rows, report))
            self.assertEqual(before, {str(p): handoff.sha(p.read_bytes()) for name in ("r1", "r2") for p in (root / name).rglob("*") if p.is_file()})
            wrong = copy.deepcopy(declaration)
            wrong["packages"][1]["reader"]["revision"] = "context-v3-r1"
            wrong["packages"][1]["reader"]["source_files_sha256"] = declaration["packages"][0]["reader"]["source_files_sha256"]
            with self.assertRaisesRegex(ValueError, "PACKAGE_REPLAY_FAILED"):
                handoff.aggregate(**wrong)
            second = next(row for row in rows if row["consumer"] == handoff.METADATA_ROW_SCHEMA)
            for field, value in (("original_pcm_sha256", second["original_pcm_sha256"]), ("source_id", second["source_id"])):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "DUPLICATE"):
                    handoff.aggregate(**declaration, prior_sources=[{"role": second["source_role"], field: value}])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--build-fixtures":
        build_real_fixtures(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        unittest.main()
