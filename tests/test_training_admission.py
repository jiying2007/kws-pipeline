"""Only synthetic PCM and invented test receipts; no listening or optimization."""
import ast
import copy
import json
import hashlib
import pathlib
import struct
import sys
import tempfile
import unittest
import wave

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "tools")]
from audit_dataset import write_lineage_sidecar
from speech_label_admission import admission_from_reviews, canonical_sha256
from training_admission import (verify_training_manifests, require_promotable_admission,
                                inherit_warm_start_admission, require_qualification_admission)


class AdmissionTests(unittest.TestCase):
    def test_reproducibility_smoke_explicitly_declares_fixture_lane(self):
        import ast
        tree = ast.parse((ROOT / "training/reproducibility_smoke.py").read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "train_command"
                               for target in node.targets)
                       and isinstance(node.value, ast.List)]
        values = [node.value for node in assignments[0].value.elts
                  if isinstance(node, ast.Constant)]
        self.assertIn("--synthetic-contract-test-only", values)
        self.assertLess(values.index("--"), values.index("--synthetic-contract-test-only"))

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.wav = self.root / "fixture.wav"
        pcm = struct.pack("<160h", *range(160))
        with wave.open(str(self.wav), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(pcm)
        wav_hash = hashlib.sha256(self.wav.read_bytes()).hexdigest()
        pcm_hash = hashlib.sha256(pcm).hexdigest()
        self.manifest = self.root / "train.tsv"
        self.manifest.write_text("fixture.wav\t1 2 3 4\n")
        self.tokens = {"<blk>": 0, "ni3": 1, "hao3": 2, "xiao3": 3, "wo1": 4}
        names = ["ni3", "hao3", "xiao3", "wo1"]
        review = dict(schema_version=2, evidence_class="speech-like-audio-review-v2",
                      source_id="invented-fixture", source_family_id="fixture-family",
                      reviewer_id="invented-unit-test-not-a-listener", review_revision=1,
                      supersedes_review_sha256=None, review_origin="human",
                      allowed_purpose="ctc-training-only", verdict="accepted",
                      acoustic_complete=True, speech_present=True, kind="positive", keyword_id=1,
                      transcript_policy="xiaowo-four-syllable-transcript-v1",
                      intended_text="你好小窝", actual_text="你好小窝", actual_tokens=names,
                      file_sha256=wav_hash, pcm_sha256=pcm_hash)
        self.row = dict(path="fixture.wav", source_path="fixture.wav", wav_sha256=wav_hash,
                        source_wav_sha256=wav_hash, source_pcm_sha256=pcm_hash,
                        source_family_id="fixture-family", kind="positive", keyword_id=1,
                        tokens=names, target_ids=[1, 2, 3, 4], actual_text="你好小窝",
                        speech_like_provenance={"source_id": "invented-fixture",
                                                "source_family_id": "fixture-family"},
                        admission=admission_from_reviews([review], canonical_sha256([review])))

    def validate(self, row=None):
        write_lineage_sidecar(self.manifest, [self.row if row is None else row])
        return verify_training_manifests([self.manifest], self.tokens)

    def test_bound_fixture_and_unknown_authenticity(self):
        result = self.validate()
        self.assertEqual(result["reviewed_rows"], 1)
        self.assertFalse(result["listener_authenticity_verified"])
        require_promotable_admission(result)

    def test_missing_receipt_rejected(self):
        with self.assertRaises(ValueError):
            verify_training_manifests([self.manifest], self.tokens)
        row = copy.deepcopy(self.row)
        del row["admission"]
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_receipt_target_completeness_and_pcm_changes_rejected(self):
        for change in ("text", "tokens", "complete", "pcm", "purpose"):
            row = copy.deepcopy(self.row)
            if change == "text": row["actual_text"] = "小窝小窝"
            if change == "tokens": row["tokens"] = ["xiao3", "wo1"]
            if change == "complete": row["admission"]["review_history"][0]["acoustic_complete"] = False
            if change == "pcm": row["source_pcm_sha256"] = "0" * 64
            if change == "purpose": row["admission"]["ctc_training_allowed"] = False
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(row)

    def test_explicit_fixture_lane_is_not_promotable(self):
        result = verify_training_manifests([self.manifest], self.tokens, diagnostic=True)
        self.assertFalse(result["promotion_allowed"])
        with self.assertRaises(ValueError): require_promotable_admission(result)
        with self.assertRaises(ValueError): require_promotable_admission(None)
        result = self.validate()
        result["promotion_allowed"] = False
        with self.assertRaises(ValueError): require_promotable_admission(result)


    def test_flattened_lineage_must_match_reviewed_source(self):
        for field in ("source_family_id", "source_id", "speaker_id", "session_id"):
            row = copy.deepcopy(self.row)
            row[field] = "different-unreviewed-identity"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(row)
        row = copy.deepcopy(self.row)
        del row["source_family_id"]
        with self.assertRaises(ValueError):
            self.validate(row)

    def test_canonical_numeric_targets_required(self):
        for item in (True, 1.0, "1"):
            row = copy.deepcopy(self.row)
            row["target_ids"][0] = item
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.validate(row)

    def test_reviewed_jsonl_consumes_bound_numeric_ids(self):
        # Extract the exact parser functions without importing Torch or running
        # main(), a model, an optimizer, or data generation.
        tree = ast.parse((ROOT / "training/train_ctc.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"parse_token_ids", "manifest_rows"}]
        namespace = {"json": json, "pathlib": pathlib, "IDENTITY_FIELDS": ()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "train_ctc.py", "exec"), namespace)
        manifest = self.root / "reviewed.jsonl"
        manifest.write_text(json.dumps(self.row) + "\n")
        result = verify_training_manifests([manifest], self.tokens)
        require_promotable_admission(result)
        consumed = namespace["manifest_rows"](manifest)
        self.assertEqual(consumed[0]["tokens"], [1, 2, 3, 4])
        self.assertEqual(consumed[0]["audio"], "fixture.wav")
        fixture = {"audio": "fixture.wav", "tokens": [1, 2, 3, 4]}
        manifest.write_text(json.dumps(fixture) + "\n")
        self.assertEqual(namespace["manifest_rows"](manifest)[0]["tokens"], [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            verify_training_manifests([manifest], self.tokens)

    def test_historical_fixture_and_descendant_taint_cannot_promote(self):
        reviewed = self.validate()
        fixture = verify_training_manifests([self.manifest], self.tokens, diagnostic=True)
        for ancestor in (None, fixture):
            child = copy.deepcopy(reviewed)
            inherit_warm_start_admission(child, ancestor)
            self.assertFalse(child["promotion_allowed"])
            with self.assertRaises(ValueError):
                require_promotable_admission(child)
            grandchild = copy.deepcopy(reviewed)
            inherit_warm_start_admission(grandchild, child)
            with self.assertRaises(ValueError):
                require_promotable_admission(grandchild)
            # A later reviewed ancestor must not clear an existing taint.
            inherit_warm_start_admission(child, reviewed)
            self.assertFalse(child["promotion_allowed"])
            child["promotion_allowed"] = True
            with self.assertRaises(ValueError):
                require_promotable_admission(child)
        clean = copy.deepcopy(reviewed)
        inherit_warm_start_admission(clean, reviewed)
        require_promotable_admission(clean)

    def test_no_unsupported_listener_authentication_or_missing_manifest_name(self):
        reviewed = self.validate()
        for state in (True, None, "false"):
            value = copy.deepcopy(reviewed)
            value["listener_authenticity_verified"] = state
            with self.subTest(state=state), self.assertRaises(ValueError):
                require_promotable_admission(value)
        value = copy.deepcopy(reviewed)
        del value["manifests"][0]["name"]
        with self.assertRaises(ValueError):
            require_promotable_admission(value)

class QualificationAdmissionMetadataTests(unittest.TestCase):
    """Only schema dictionaries and text hashes; no model, training, or CPU run."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.binding = {"name": "train.tsv", "sha256": "1" * 64, "lineage_sha256": "2" * 64}
        self.references = "3" * 64
        self.admission = {
            "policy": "reviewed-source-training-consumption-v1",
            "purpose": "reviewed-ctc-training", "promotion_allowed": True,
            "reviewed_rows": 1, "listener_authenticity_verified": False,
            "manifests": [copy.deepcopy(self.binding)],
        }
        self.training = {
            "admission": self.admission,
            "manifests": [{key: self.binding[key] for key in ("name", "sha256")}],
        }
        self.selected = [copy.deepcopy(self.binding)]
        self.audit = {
            "schema_version": 3, "sha256": "4" * 64,
            "required_metadata": ["speaker_id", "session_id", "source_id"],
            "audited_manifest_sha256s": [self.binding["sha256"], self.references],
            "manifest_bindings": [copy.deepcopy(self.binding),
                {"name": "references.jsonl", "sha256": self.references,
                 "lineage_sha256": self.references}],
        }

    def check(self):
        return require_qualification_admission(self.training, self.selected, self.audit, self.references)

    def raw_audit(self):
        return {
            "schema_version": 3, "clean": True,
            "identity_policy": {"required_metadata": self.audit["required_metadata"]},
            "splits": {
                "train": {"manifest": "/private/corpus/train.tsv",
                          "manifest_sha256": self.binding["sha256"],
                          "lineage_sidecar_sha256": self.binding["lineage_sha256"]},
                "qualification": {"manifest": "/private/held-out/references.jsonl",
                                  "manifest_sha256": self.references},
            },
        }

    def normalize_audit(self, value):
        from qualification_manifest import validate_dataset_audit
        path = self.root / "audit.json"
        path.write_text(json.dumps(value))
        return validate_dataset_audit(path, [self.binding["sha256"], self.references])

    def test_reviewed_tsv_and_jsonl_bindings_accept(self):
        self.assertIs(self.check(), self.admission)
        for rows in (self.training["manifests"], self.admission["manifests"], self.selected,
                     self.audit["manifest_bindings"]):
            rows[0]["name"] = "train.jsonl"
            if "lineage_sha256" in rows[0]:
                rows[0]["lineage_sha256"] = self.binding["sha256"]
        self.assertIs(self.check(), self.admission)
        for rows in (self.admission["manifests"], self.selected, self.audit["manifest_bindings"]):
            rows[0]["lineage_sha256"] = "5" * 64
        with self.assertRaisesRegex(ValueError, "JSONL lineage"):
            self.check()

    def test_audit_retains_actual_sidecar_and_jsonl_pairs(self):
        normalized = self.normalize_audit(self.raw_audit())
        self.assertCountEqual(normalized["manifest_bindings"], self.audit["manifest_bindings"])
        self.audit = normalized
        self.check()
        for replacement in (None, "5" * 64):
            raw = self.raw_audit()
            if replacement is None:
                del raw["splits"]["train"]["lineage_sidecar_sha256"]
            else:
                raw["splits"]["train"]["lineage_sidecar_sha256"] = replacement
            self.audit = self.normalize_audit(raw)
            with self.subTest(sidecar=replacement), self.assertRaisesRegex(ValueError, "reviewed training lineage"):
                self.check()

    def test_audit_rejects_invalid_hashes_and_missing_reference_coverage(self):
        for field in ("manifest_sha256", "lineage_sidecar_sha256"):
            for invalid in (None, 17, "short", "A" * 64):
                raw = self.raw_audit()
                raw["splits"]["train"][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(ValueError):
                    self.normalize_audit(raw)
        raw = self.raw_audit()
        del raw["splits"]["qualification"]
        with self.assertRaisesRegex(ValueError, "every selected"):
            self.normalize_audit(raw)
        raw = self.raw_audit()
        raw["splits"]["qualification"]["lineage_sidecar_sha256"] = "6" * 64
        with self.assertRaisesRegex(ValueError, "JSONL lineage"):
            self.normalize_audit(raw)

    def test_selected_tsv_lineage_bytes_are_rehashed(self):
        from qualification_manifest import training_manifest_artifact
        from audit_dataset import lineage_path
        manifest = self.root / "train.tsv"
        manifest.write_text("invented.wav\t1 2\n")
        sidecar = lineage_path(manifest)
        sidecar.write_text('{"schema_only": "before"}\n')
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        before = training_manifest_artifact(manifest, digest)
        self.training["manifests"] = [{"name": before["name"], "sha256": before["sha256"]}]
        self.admission["manifests"] = [copy.deepcopy(before)]
        self.audit["manifest_bindings"][0] = copy.deepcopy(before)
        self.audit["audited_manifest_sha256s"][0] = digest
        self.selected = [before]
        self.check()
        sidecar.write_text('{"schema_only": "after"}\n')
        self.selected = [training_manifest_artifact(manifest, digest)]
        self.assertEqual(self.selected[0]["sha256"], before["sha256"])
        with self.assertRaisesRegex(ValueError, "selected training lineage"):
            self.check()
        sidecar.unlink()
        with self.assertRaises(FileNotFoundError):
            training_manifest_artifact(manifest, digest)
        jsonl = self.root / "train.jsonl"
        jsonl.write_text('{"schema_only": true}\n')
        digest = hashlib.sha256(jsonl.read_bytes()).hexdigest()
        self.assertEqual(training_manifest_artifact(jsonl, digest)["lineage_sha256"], digest)

    def test_manifest_names_hashes_multiplicity_and_audit_coverage_are_bound(self):
        for target in ("admission", "recorded", "selected", "audited"):
            for change in ("name", "sha256", "duplicate"):
                saved = copy.deepcopy((self.training, self.selected, self.audit))
                rows = {"admission": self.training["admission"]["manifests"],
                        "recorded": self.training["manifests"], "selected": self.selected,
                        "audited": self.audit["manifest_bindings"]}[target]
                if change == "duplicate":
                    rows.append(copy.deepcopy(rows[0]))
                else:
                    rows[0][change] = "other.tsv" if change == "name" else "7" * 64
                with self.subTest(target=target, change=change), self.assertRaises(ValueError):
                    self.check()
                self.training, self.selected, self.audit = saved
        # Two identical selections need two matching audited bindings.
        for rows in (self.training["admission"]["manifests"], self.training["manifests"], self.selected):
            rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "reviewed training lineage"):
            self.check()
        self.audit["manifest_bindings"].append(copy.deepcopy(self.audit["manifest_bindings"][0]))
        self.audit["audited_manifest_sha256s"].append(self.binding["sha256"])
        self.check()
        # Bare hash lists cannot replace the separately named lineage bindings.
        self.audit["manifest_bindings"][0]["name"] = "different.tsv"
        with self.assertRaisesRegex(ValueError, "reviewed training lineage"):
            self.check()

    def test_binding_fields_reject_missing_or_noncanonical_values(self):
        for target in ("selected", "audited"):
            for field, invalid in (("name", ""), ("name", "nested/train.tsv"),
                                   ("sha256", True), ("sha256", "F" * 64),
                                   ("lineage_sha256", "not-a-hash"),
                                   ("lineage_sha256", None)):
                rows = self.selected if target == "selected" else self.audit["manifest_bindings"]
                before = copy.deepcopy(rows[0])
                rows[0][field] = invalid
                with self.subTest(target=target, field=field, invalid=invalid), self.assertRaises(ValueError):
                    self.check()
                rows[0] = before

    def test_missing_or_inconsistent_normalized_audit_rejects(self):
        for field in ("manifest_bindings", "audited_manifest_sha256s"):
            saved = copy.deepcopy(self.audit)
            del self.audit[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check()
            self.audit = saved
        self.audit["manifest_bindings"] = self.audit["manifest_bindings"][:1]
        self.audit["audited_manifest_sha256s"] = self.audit["audited_manifest_sha256s"][:1]
        with self.assertRaisesRegex(ValueError, "every selected"):
            self.check()

    def gate_prefix(self):
        # Only sections inspected before the admission check. No resource/CPU
        # evidence is fabricated or evaluated by these metadata tests.
        names = ("model", "model_provenance", "model_checkpoint", "training_tokens",
                 "dataset_audit", "keyword_pack", "tokens", "config", "eval_runner",
                 "references", "detections", "board_runner", "board_audio",
                 "evidence_collector", "evidence_raw", "attestation_verification", "evidence")
        artifacts = {name: {"name": name, "sha256": "8" * 64, "bytes": 1} for name in names}
        artifacts["dataset_audit"]["sha256"] = self.audit["sha256"]
        artifacts["references"]["sha256"] = self.references
        artifacts["training_manifests"] = [dict(row, bytes=1) for row in self.selected]
        artifacts["raw_evidence"] = [{"name": "raw", "sha256": "9" * 64, "bytes": 1}]
        lineage = {key: "8" * 64 for key in ("provenance_sha256", "model_sha256", "tokens_sha256",
                    "checkpoint_sha256", "training_tokens_sha256")}
        lineage.update(token_bytes_identical_to_training=True, frontend_spec_version=2,
                       frontend_kind=0, frontend_name="logmel", training=self.training, quantization={})
        return {"schema_version": 3, "source_sha": "a" * 40, "sku": "schema-only",
                "runtime": {"model_abi": 2, "keyword_pack_abi": 3, "sample_rate_hz": 16000,
                            "frame_length_samples": 400, "frame_hop_samples": 320,
                            "frontend_kind": 0, "frontend_name": "logmel"},
                "vocabulary": {"fingerprint": "0x" + "a" * 16, "sha256": "8" * 64},
                "model_lineage": lineage, "artifacts": artifacts, "dataset_audit": self.audit,
                "evaluation": {}, "board": {}, "evidence": {}}

    def producer_admission_check(self):
        # Execute the producer's actual metadata-check statement without invoking
        # its CLI, model reader, evidence metrics, or any qualification CPU work.
        tree = ast.parse((ROOT / "tools/qualification_manifest.py").read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        checks = [node for node in main.body if isinstance(node, ast.Expr)
                  and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                  and node.value.func.id == "require_qualification_admission"]
        self.assertEqual(len(checks), 1)
        namespace = {"require_qualification_admission": require_qualification_admission,
                     "model_lineage": {"training": self.training},
                     "training_manifest_artifacts": self.selected, "dataset_audit": self.audit,
                     "hashes": {"references": self.references}}
        exec(compile(ast.Module(body=checks, type_ignores=[]), "qualification_manifest.py", "exec"), namespace)

    def test_current_producer_and_independent_gate_reject_unreviewed_admission(self):
        from unittest import mock
        from qualification_gate import validate_manifest

        class PastAdmission(Exception):
            pass

        self.producer_admission_check()
        with mock.patch("qualification_gate.validate_corpus", side_effect=PastAdmission), \
                self.assertRaises(PastAdmission):
            validate_manifest(self.gate_prefix())
        reviewed = copy.deepcopy(self.admission)
        cases = [None, {}, dict(reviewed, purpose="synthetic-contract-test-only"),
                 dict(reviewed, promotion_allowed=False),
                 dict(reviewed, ancestor_admission="diagnostic-or-unverified")]
        for admission in cases:
            self.training["admission"] = admission
            with self.subTest(admission=admission), self.assertRaises(ValueError):
                self.producer_admission_check()
            with self.subTest(gate_admission=admission), self.assertRaises(ValueError):
                validate_manifest(self.gate_prefix())
        del self.training["admission"]
        with self.assertRaisesRegex(ValueError, "reviewed training admission"):
            self.producer_admission_check()
        with self.assertRaisesRegex(ValueError, "reviewed training admission"):
            validate_manifest(self.gate_prefix())

    def test_gate_rechecks_audited_and_selected_lineage(self):
        from qualification_gate import validate_manifest
        for rows in (self.selected, self.audit["manifest_bindings"]):
            original = rows[0]["lineage_sha256"]
            rows[0]["lineage_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "lineage"):
                validate_manifest(self.gate_prefix())
            rows[0]["lineage_sha256"] = original

    def test_generic_provenance_reader_preserves_present_admission_and_absence(self):
        from corpus_identity import corpus_digest
        from model_provenance import validate_model_provenance
        records = [{"recording": "schema-only", "manifest": "train.tsv", "path": "unused.wav",
                    "file_sha256": "a" * 64, "pcm_sha256": "b" * 64,
                    "frames": 160, "duration_s": 0.01}]
        digest = corpus_digest(records)
        training = dict(self.training, corpus_identity={"schema_version": 1,
                        "corpus_sha256": digest, "recordings": records}, examples=1,
                        seed=1, epochs=1, batch_size=1, learning_rate=0.01,
                        optimizer="schema-only", weight_decay=0.0, grad_clip_norm=1.0)
        stats = {"scale": 0.1, "max_abs_error": 0.0, "rmse": 0.0, "signal_rms": 1.0, "snr_db": 1.0}
        provenance = {"schema_version": 3,
            "model": {"sha256": "c" * 64, "bytes": 1, "abi": 2, "feature_dim": 1,
                      "hidden_dim": 1, "vocab_size": 2, "vocab_fingerprint": "0x0000000000000001",
                      "frontend_spec_version": 2, "frontend_kind": 0, "frontend_name": "logmel"},
            "checkpoint": {"name": "unused.pt", "sha256": "d" * 64},
            "tokens": {"sha256": "e" * 64, "checkpoint_sha256": "e" * 64,
                       "byte_identical_to_training": True}, "training": training,
            "quantization": {"scheme": "symmetric-int8-per-matrix", "in_proj": stats,
                             "rec_proj": stats, "out_proj": stats}}
        path = self.root / "provenance.json"
        def read():
            path.write_text(json.dumps(provenance))
            return validate_model_provenance(path, model_sha256="c" * 64, model_bytes=1,
                    feature_dim=1, hidden_dim=1, vocab_size=2, vocab_fingerprint=1,
                    tokens_sha256="e" * 64, checkpoint_sha256="d" * 64,
                    training_tokens_sha256="e" * 64, training_manifest_sha256s=["1" * 64],
                    training_corpus_sha256=digest)
        for admission in (self.admission, None, {"purpose": "synthetic-contract-test-only"},
                          dict(self.admission, ancestor_admission="diagnostic-or-unverified")):
            training["admission"] = admission
            self.assertEqual(read()["training"]["admission"], admission)
        del training["admission"]
        self.assertNotIn("admission", read()["training"])


if __name__ == "__main__":
    unittest.main()
