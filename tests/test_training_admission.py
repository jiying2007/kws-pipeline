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
                                inherit_warm_start_admission)


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


if __name__ == "__main__":
    unittest.main()
