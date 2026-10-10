"""Retained public bytes and explicitly invented review declarations only."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import screen32_human_review as review


class Screen32HumanReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = review.build_packet(ROOT)

    def document(self, text="你好小窝", exposure="EXPOSED"):
        document = review.receipt_template(self.packet)
        document.update(reviewer_id="invented-fixture-listener", source_reference="invented-test-only")
        document["rows"] = document["rows"][:1]
        row = document["rows"][0]
        row["actual_text"] = text
        row["acoustic_completeness"] = "COMPLETE"
        row["review"].update(status="clean", label_origin="human_listening", listened_to_audio=True,
                             listened_entire_clip=True, independent_human=True, complete=True,
                             hypothesis_exposure=exposure)
        return document

    def test_packet_is_label_free_all16_and_delivered_order(self):
        encoded = json.dumps(self.packet, ensure_ascii=False)
        for forbidden in ("intended_text", "raw_text", "asr_results", "你好", "小窝", "小屋", "小吴"):
            self.assertNotIn(forbidden, encoded)
        self.assertEqual(self.packet["denominator"], 16)
        self.assertEqual([r["audio_file"] for r in self.packet["rows"]], [f"{n:02d}.wav" for n in range(1, 17)])
        self.assertEqual(self.packet["rows"][0]["wav_sha256"], "9040a0927a11d788b16410be3daf2c3ddb70e4c4bc2c23920eac3a824e3800c0")
        self.assertNotEqual(self.packet["binding"]["original_blind_job_raw_sha256"],
                            self.packet["binding"]["blind_job_canonical_sha256"])

    def test_checked_in_pending_material_is_reproducible(self):
        directory = HERE / "screen32-review"
        self.assertEqual(review.load(directory / "listening-packet.json"), self.packet)
        self.assertEqual(review.load(directory / "pending-receipts.json"), review.receipt_template(self.packet))

    def test_no_receipt_cannot_promote_machine_agreement(self):
        result = review.source_report(ROOT, [])
        self.assertEqual((result["denominator"], result["pending_rows"], result["complete_human_actual_rows"]), (16, 16, 0))
        for row in result["rows"]:
            self.assertIsNone(row["actual_text"])
            self.assertIsNone(row["label_diagnostics"])
            self.assertIsNone(row["actual_expected_keywords"])
            self.assertIsNone(row["ctc_target"])
            self.assertEqual(row["acoustic_completeness"], "UNKNOWN")
            self.assertFalse(row["training_admitted"])

    def test_pending_template_is_not_human_truth(self):
        result = review.source_report(ROOT, [review.receipt_template(self.packet)])
        self.assertEqual(result["complete_human_actual_rows"], 0)
        self.assertEqual(result["pending_rows"], 16)

    def test_exposed_direct_human_words_accepted_but_no_accuracy_claim(self):
        result = review.source_report(ROOT, [self.document()])
        first = result["rows"][0]
        self.assertEqual((result["submitted_rows"], result["pending_rows"], result["complete_human_actual_rows"]), (1, 15, 1))
        self.assertEqual(first["actual_expected_keywords"], ["K1"])
        self.assertEqual(first["review"]["hypothesis_exposure"], "EXPOSED")
        self.assertEqual(first["reviewer_declared_acoustic_completeness"], "COMPLETE")
        self.assertEqual(first["acoustic_completeness"], "UNKNOWN")
        self.assertEqual(first["label_diagnostics"]["acoustic_completeness"], "UNKNOWN")
        self.assertFalse(first["independent_accuracy_qualified"])
        self.assertIsNone(first["ctc_target"])
        self.assertEqual(first["split"], "UNASSIGNED")
        self.assertEqual(first["lineage_status"], "unverified")

    def test_all16_are_preserved_if_everyone_reviews(self):
        document = self.document()
        first = document["rows"][0]
        document["rows"] = [dict(copy.deepcopy(row), **{k: copy.deepcopy(first[k]) for k in
                             ("actual_text", "partial_text", "review", "acoustic_completeness")})
                            for row in self.packet["rows"]]
        result = review.source_report(ROOT, [document])
        self.assertEqual((result["denominator"], result["complete_human_actual_rows"]), (16, 16))
        self.assertFalse(result["training_admitted"])

    def test_unknown_or_false_human_declarations_do_not_promote(self):
        for field in ("independent_human", "complete", "listened_entire_clip"):
            for value in (None, False):
                document = self.document()
                document["rows"][0]["review"][field] = value
                result = review.source_report(ROOT, [document])
                self.assertEqual(result["complete_human_actual_rows"], 0)

    def test_unattributed_complete_receipt_remains_unqualified(self):
        for field in ("reviewer_id", "source_reference"):
            document = self.document()
            document[field] = None
            result = review.source_report(ROOT, [document])
            self.assertEqual(result["complete_human_actual_rows"], 0)
            self.assertEqual(result["unqualified_submitted_rows"], 1)
            self.assertFalse(result["rows"][0]["review_provenance_complete"])
            self.assertFalse(result["rows"][0]["reviewer_identity_authenticated"])
            self.assertFalse(result["rows"][0]["label_diagnostics"]["original_complete_label_eligible"])

    def test_partial_receipt_can_keep_identity_fields_null(self):
        document = self.document()
        document.update(reviewer_id=None, source_reference=None)
        document["rows"][0].update(actual_text=None, partial_text="你", acoustic_completeness="UNKNOWN")
        document["rows"][0]["review"].update(status="incomplete", complete=False)
        result = review.source_report(ROOT, [document])
        self.assertEqual(result["submitted_rows"], 1)
        self.assertEqual(result["complete_human_actual_rows"], 0)

    def test_unknown_or_incomplete_acoustics_cannot_qualify_original_complete(self):
        for state in ("UNKNOWN", "INCOMPLETE"):
            document = self.document()
            document["rows"][0]["acoustic_completeness"] = state
            row = review.source_report(ROOT, [document])["rows"][0]
            self.assertFalse(row["human_actual_review_complete"])
            self.assertFalse(row["label_diagnostics"]["original_complete_label_eligible"])
            self.assertEqual(row["reviewer_declared_acoustic_completeness"], state)

    def test_clean_receipt_with_partial_text_rejects(self):
        document = self.document()
        document["rows"][0]["partial_text"] = "uncertain"
        with self.assertRaisesRegex(ValueError, "partial text"):
            review.source_report(ROOT, [document])

    def test_pcm_readback_must_match_retained_input(self):
        original = review.wave.Wave_read.readframes
        def changed(audio, count):
            data = original(audio, count)
            return bytes([data[0] ^ 1]) + data[1:] if data else data
        with patch.object(review.wave.Wave_read, "readframes", changed), self.assertRaisesRegex(ValueError, "PCM digest"):
            review.build_packet(ROOT)

    def test_latest_partial_replaces_older_complete_without_losing_history(self):
        first = self.document()
        second = copy.deepcopy(first)
        second.update(revision=2, supersedes_receipt_sha256=review.digest(review.canonical(first)))
        row = second["rows"][0]
        row.update(actual_text=None, partial_text="你", acoustic_completeness="INCOMPLETE")
        row["review"].update(status="incomplete", complete=False)
        result = review.source_report(ROOT, [first, second])
        self.assertEqual(result["complete_human_actual_rows"], 0)
        self.assertIsNone(result["rows"][0]["actual_text"])
        self.assertEqual(result["rows"][0]["partial_text"], "你")
        self.assertEqual(len(result["rows"][0]["review_history"]), 2)

    def test_stale_missing_reordered_and_duplicate_document_history_reject(self):
        first = self.document()
        second = copy.deepcopy(first)
        second.update(revision=2, supersedes_receipt_sha256=review.digest(review.canonical(first)))
        for history in ([second], [second, first], [first, first]):
            with self.assertRaisesRegex(ValueError, "history"):
                review.source_report(ROOT, history)

    def test_oov_words_preserved_and_never_folded_or_blanked(self):
        for text, oov in (("小屋", ["屋"]), ("小吴", ["吴"]), ("你好：", []), ("你好—", ["—"])):
            row = review.source_report(ROOT, [self.document(text)])["rows"][0]
            self.assertEqual(row["actual_text"], text)
            self.assertEqual(row["production_oov_characters"], oov)
            self.assertIsNone(row["ctc_target"])
            self.assertFalse(row["automatic_relabel"])
            self.assertFalse(row["silence_or_blank_target_inferred"])
            self.assertFalse(row["training_admitted"])

    def test_actual_words_win_over_plan_and_machine(self):
        row = review.source_report(ROOT, [self.document("你好")])["rows"][0]
        self.assertEqual(row["actual_expected_keywords"], [])
        self.assertFalse(row["label_diagnostics"]["plan_matches_human_actual"])
        self.assertFalse(row["label_diagnostics"]["original_complete_label_eligible"])

    def test_blank_null_inaudible_ambiguous_are_not_known_negative(self):
        for status, actual in (("inaudible", None), ("ambiguous", ""), ("incomplete", "你")):
            document = self.document()
            document["rows"][0]["review"].update(status=status, complete=False)
            document["rows"][0]["actual_text"] = actual
            row = review.source_report(ROOT, [document])["rows"][0]
            self.assertIsNone(row["actual_expected_keywords"])
            self.assertIsNone(row["ctc_target"])
        for text in (None, "", "！"):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                review.source_report(ROOT, [self.document(text)])

    def test_swapped_or_changed_binding_fields_reject(self):
        for field, value in (("review_id", "02"), ("audio_file", "02.wav"), ("wav_sha256", "0" * 64),
                             ("pcm_sha256", "0" * 64), ("decoder_input_binding_sha256", "0" * 64),
                             ("representation", "native_float"), ("sample_rate_hz", True)):
            document = self.document()
            document["rows"][0][field] = value
            with self.assertRaisesRegex(ValueError, "binding"):
                review.source_report(ROOT, [document])

    def test_stale_packet_and_unknown_or_duplicate_row_reject(self):
        for mutation in (lambda d: d.update(packet_sha256="0" * 64),
                         lambda d: d["rows"][0].update(review_id="99"),
                         lambda d: d["rows"].append(copy.deepcopy(d["rows"][0]))):
            document = self.document()
            mutation(document)
            with self.assertRaises(ValueError):
                review.source_report(ROOT, [document])

    def test_authority_fields_and_machine_intent_origin_reject(self):
        for origin in (None, "asr", "intended_text", "generated"):
            document = self.document()
            document["rows"][0]["review"]["label_origin"] = origin
            with self.assertRaises(ValueError):
                review.source_report(ROOT, [document])
        for extra in ("intended_text", "asr_results", "ctc_target", "training_admitted", "human_gold"):
            document = self.document()
            document["rows"][0][extra] = True
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                review.source_report(ROOT, [document])

    def test_machine_cannot_fill_pending_row(self):
        document = review.receipt_template(self.packet)
        document["rows"][0]["actual_text"] = "你好小窝"
        with self.assertRaisesRegex(ValueError, "pending"):
            review.source_report(ROOT, [document])

    def test_exact_booleans_and_bounded_text_required(self):
        for field in ("independent_human", "listened_to_audio", "complete"):
            document = self.document()
            document["rows"][0]["review"][field] = 1
            with self.assertRaises(ValueError):
                review.source_report(ROOT, [document])
        with self.assertRaisesRegex(ValueError, "bounded"):
            review.source_report(ROOT, [self.document("a" * 8193)])

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for text in ('{"rows":[],"rows":[]}', '{"x":NaN}', '{"x":Infinity}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    review.load(path)

    def test_input_and_frozen_source_mutations_rejected(self):
        read = Path.read_bytes
        for relative in (review.BASE / "plan.json", review.BASE / "execution-freeze.json", review.BASE / "contract.py",
                         review.BASE / review.GEN / "generation/tts/screen32-001.wav",
                         review.BASE / review.ASR / "sensevoice/terminal-outcomes.json"):
            target = ROOT / relative
            def changed(path, target=target):
                return read(path) + b" " if path == target else read(path)
            with patch.object(Path, "read_bytes", changed), self.assertRaises(ValueError):
                review.build_packet(ROOT)

    def test_errors_and_reports_do_not_mutate_supplied_evidence(self):
        document = self.document()
        before = copy.deepcopy(document)
        review.source_report(ROOT, [document])
        self.assertEqual(document, before)
        document["rows"][0]["wav_sha256"] = "bad"
        before = copy.deepcopy(document)
        with self.assertRaises(ValueError):
            review.source_report(ROOT, [document])
        self.assertEqual(document, before)

    def test_cli_no_overwrite_and_failure_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("preserve")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(review.main(["packet", "--output", str(output)]), 2)
            self.assertEqual(output.read_text(), "preserve")
            bad = Path(directory) / "bad.json"
            bad.write_text('{"schema":"wrong"}')
            absent = Path(directory) / "absent.json"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(review.main(["report", "--receipts", str(bad), "--output", str(absent)]), 2)
            self.assertFalse(absent.exists())


if __name__ == "__main__":
    unittest.main()
