"""Retained public bytes and explicitly invented review declarations only."""
import copy
import contextlib
from collections import Counter
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

    def test_one_verified_byte_snapshot_per_public_report_call(self):
        read = Path.read_bytes
        for reporter, arguments in ((review.build_packet, (ROOT,)),
                                    (review.source_report, (ROOT, [])),
                                    (review.screening_report, (ROOT, []))):
            with self.subTest(reporter=reporter.__name__):
                reads = Counter()
                def counted(path):
                    reads[path] += 1
                    return read(path)
                with patch.object(Path, "read_bytes", counted), \
                     patch.object(review, "verified_inputs", wraps=review.verified_inputs) as verify, \
                     patch.object(review, "parse_json", wraps=review.parse_json) as parse:
                    reporter(*arguments)
                self.assertEqual(verify.call_count, 1)
                self.assertEqual(parse.call_count, len(review.PINS))
                self.assertTrue(reads)
                self.assertTrue(all(count == 1 for count in reads.values()), dict(reads))
                self.assertEqual(sum(path.suffix == ".wav" for path in reads), 16)

    def test_json_and_pcm_decode_the_hashed_bytes_without_reopening(self):
        opened = review.wave.open
        def byte_stream_only(source, mode):
            self.assertIsInstance(source, io.BytesIO)
            return opened(source, mode)
        with patch.object(Path, "read_text", side_effect=AssertionError("verified JSON reopened")), \
             patch.object(review.wave, "open", byte_stream_only):
            self.assertEqual(review.build_packet(ROOT), self.packet)

    def test_snapshot_is_local_to_call_and_next_call_revalidates(self):
        read = Path.read_bytes
        for relative in (review.BASE / "plan.json", review.BASE / review.GEN / "generation/tts/screen32-001.wav"):
            with self.subTest(relative=relative):
                target = ROOT / relative
                calls = 0
                def changed_after_first_read(path):
                    nonlocal calls
                    raw = read(path)
                    if path == target:
                        calls += 1
                        return raw if calls == 1 else raw + b" "
                    return raw
                with patch.object(Path, "read_bytes", changed_after_first_read):
                    self.assertEqual(review.build_packet(ROOT), self.packet)
                    with self.assertRaises(ValueError):
                        review.build_packet(ROOT)
                self.assertEqual(calls, 2)

    def test_historical_review_requires_exact_archive_without_live_fallback(self):
        read = Path.read_bytes
        original = "research/experiment_quality_guards/quality_gates.py"
        archive = ROOT / review.HISTORICAL_SOURCE_PATHS[original]
        def corrupted(path):
            return read(path) + b" " if path == archive else read(path)
        with patch.object(Path, "read_bytes", corrupted), self.assertRaisesRegex(ValueError, "frozen source changed"):
            review.build_packet(ROOT)
        def absent(path):
            if path == archive:
                raise FileNotFoundError(path)
            return read(path)
        with patch.object(Path, "read_bytes", absent), self.assertRaises(FileNotFoundError):
            review.build_packet(ROOT)
        # A historical report neither executes nor authenticates the new live
        # source. The runtime admission guard separately remains live-path-bound.
        def changed_live(path):
            return read(path) + b" " if path == ROOT / original else read(path)
        with patch.object(Path, "read_bytes", changed_live):
            self.assertEqual(review.build_packet(ROOT), self.packet)

    def test_pending_reports_preserve_exact_preoptimization_canonical_bytes(self):
        expected = {review.source_report: "7891a59df8dd2bdb97be331174a2fe1c354ec9eb8fa5d307b59e58ac74712c10",
                    review.screening_report: "5ff67b095870b070c122c6e01a20e0abf542ac54260cac37da8faffdafd3250d"}
        for reporter, digest in expected.items():
            self.assertEqual(review.digest(review.canonical(reporter(ROOT, []))), digest)

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



class Screen32WeakScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packet = review.build_packet(ROOT)
        cls.values = review.verified_inputs(ROOT)

    @staticmethod
    def observation(text="虚构甲", status="complete", flags=None):
        return {"raw_text": text, "status": status, "quality_flags": flags or []}

    def decision(self, actual="虚构甲", complete=True, observations=None):
        if observations is None:
            observations = [self.observation("虚构，甲。"), self.observation()]
        return review.weak_screening(actual, complete, observations)

    def synthetic_values(self):
        values = copy.deepcopy(self.values)
        for rows in values[review.ASR + "/primary-raw.json"].values():
            for row in rows:
                row.update(self.observation())
        return values

    def document(self):
        document = review.receipt_template(self.packet)
        document.update(reviewer_id="invented-screening-listener", source_reference="synthetic-test-only")
        document["rows"] = document["rows"][:3]
        for row in document["rows"]:
            row.update(actual_text="虚构甲", acoustic_completeness="COMPLETE")
            row["review"].update(status="clean", label_origin="human_listening", listened_to_audio=True,
                                 listened_entire_clip=True, independent_human=True, complete=True,
                                 hypothesis_exposure="EXPOSED")
        document["rows"][1].update(actual_text=None, partial_text="虚构", acoustic_completeness="UNKNOWN")
        document["rows"][1]["review"].update(status="ambiguous", complete=False)
        document["rows"][2]["actual_text"] = "虚构乙"
        return document

    def test_all_three_consensus_categories_are_weak_and_preserve_words(self):
        for actual, complete, category in (
                ("虚构甲", True, "human_and_both_asr_agree"),
                (None, False, "dual_asr_agree_human_uncertain"),
                ("虚构乙", True, "human_vs_dual_asr_disagreement")):
            with self.subTest(category=category):
                result = self.decision(actual, complete)
                self.assertEqual(result["category"], category)
                self.assertTrue(result["dual_asr_agreement"])
                self.assertEqual(result["screening_candidate_text"], "虚构甲")
                self.assertEqual(result["screening_evidence_strength"], "weak_machine_consensus")
                self.assertEqual(result["asr_results"][0]["raw_text"], "虚构，甲。")
                self.assertEqual(result["machine_normalized_texts"], ["虚构甲", "虚构甲"])

    def test_uncertain_words_matching_consensus_do_not_become_complete(self):
        result = self.decision("虚构甲", False)
        self.assertEqual(result["category"], "dual_asr_agree_human_uncertain")
        self.assertFalse(result["human_actual_review_complete"])

    def test_disagreeing_asr_never_selects_human_or_majority_candidate(self):
        result = self.decision(observations=[self.observation(), self.observation("虚构乙")])
        self.assertEqual(result["category"], "dual_asr_disagree")
        self.assertFalse(result["dual_asr_agreement"])
        self.assertIsNone(result["screening_candidate_text"])

    def test_missing_failed_empty_and_punctuation_only_are_unresolved(self):
        bad = [None, {}, self.observation(None), self.observation(""), self.observation("，。 "),
               self.observation(status="failed"), self.observation(status="pending")]
        for observation in bad:
            for observations in ([observation, observation], [self.observation(), observation]):
                with self.subTest(observations=observations):
                    result = self.decision(observations=observations)
                    self.assertEqual(result["category"], "dual_asr_missing_incomplete_or_flagged")
                    self.assertFalse(result["dual_asr_agreement"])
                    self.assertIsNone(result["screening_candidate_text"])

    def test_quality_flags_block_even_identical_complete_asr(self):
        observations = [self.observation(flags=["synthetic-quality-warning"]), self.observation()]
        result = self.decision(observations=observations)
        self.assertEqual(result["category"], "dual_asr_missing_incomplete_or_flagged")
        self.assertEqual(result["asr_quality_flags"], ["synthetic-quality-warning"])
        self.assertEqual(result["asr_results"], observations)
        self.assertFalse(result["dual_asr_agreement"])
        self.assertIsNone(result["screening_candidate_text"])

    def test_comparison_does_not_fold_characters_or_case(self):
        for first, second in (("Ａ", "A"), ("A", "a"), ("虚构甲", "虚构乙")):
            result = self.decision(observations=[self.observation(first), self.observation(second)])
            self.assertEqual(result["category"], "dual_asr_disagree")
            self.assertIsNone(result["screening_candidate_text"])

    def test_invalid_types_and_observation_counts_reject(self):
        for observations in ([], [self.observation()], [self.observation()] * 3, {},
                             [True, self.observation()], [self.observation(123), self.observation()],
                             [self.observation(flags="flag"), self.observation()]):
            with self.assertRaises(ValueError):
                self.decision(observations=observations)
        for complete in (None, 1, "true"):
            with self.assertRaises(ValueError):
                self.decision(complete=complete)
        for actual in (None, "", "！"):
            with self.assertRaisesRegex(ValueError, "nonempty"):
                self.decision(actual=actual)

    def test_no_case_grants_gold_or_any_admission(self):
        for observations in ([self.observation(), self.observation()],
                             [self.observation(), self.observation("虚构乙")], [None, None]):
            result = self.decision(observations=observations)
            for field in ("human_gold", "machine_consensus_overwrites_actual_text", "automatic_relabel",
                          "silence_or_blank_target_inferred", "training_admitted", "shipping_approved",
                          "independent_accuracy_qualified"):
                self.assertIs(result[field], False)
            self.assertIsNone(result["ctc_target"])
            self.assertIsNone(result["final_adjudicated_text"])
            self.assertEqual(result["acoustic_completeness"], "UNKNOWN")

    def test_input_observations_are_not_mutated_or_aliased(self):
        observations = [self.observation(), self.observation()]
        before = copy.deepcopy(observations)
        result = self.decision(observations=observations)
        self.assertEqual(observations, before)
        result["asr_results"][0]["raw_text"] = "changed"
        self.assertEqual(observations, before)

    def test_screening_partitions_all16_without_changing_original_report(self):
        values = self.synthetic_values()
        clips = review.keyed(values[review.GEN + "/blind/job.json"]["clips"], "wav_sha256")
        saved = values[review.ASR + "/primary-raw.json"]
        for index, change in ((3, {"raw_text": "虚构乙"}), (4, {"status": "failed"})):
            opaque = clips[self.packet["rows"][index]["wav_sha256"]]["audio_id"]
            review.keyed(saved["sensevoice"], "audio_id")[opaque].update(change)
        document = self.document()
        before = copy.deepcopy(document)
        with patch.object(review, "verified_inputs", return_value=values):
            original = review.source_report(ROOT, [document])
            result = review.screening_report(ROOT, [document])
        self.assertEqual(document, before)
        self.assertEqual(result["source_review_report_canonical_sha256"], review.digest(review.canonical(original)))
        self.assertEqual(result["partition_counts"], {
            "human_and_both_asr_agree": 1, "dual_asr_agree_human_uncertain": 12,
            "human_vs_dual_asr_disagreement": 1, "dual_asr_disagree": 1,
            "dual_asr_missing_incomplete_or_flagged": 1})
        self.assertEqual(result["denominator"], 16)
        self.assertEqual(result["dual_asr_agreement_weak_candidates"], 14)
        self.assertEqual(result["asr_observation_order"], ["qwen06", "sensevoice"])
        identities = [identity for group in result["partition_rows"].values() for identity in group]
        self.assertEqual(sorted(identities), [f"{n:02d}" for n in range(1, 17)])
        for old, new in zip(original["rows"], result["rows"]):
            self.assertEqual(old, {key: value for key, value in new.items() if key != "screening"})
        self.assertEqual(result["rows"][1]["partial_text"], "虚构")
        self.assertEqual(result["rows"][2]["actual_text"], "虚构乙")
        self.assertEqual(result["rows"][2]["screening"]["screening_candidate_text"], "虚构甲")

    def test_no_human_receipt_remains_unknown_despite_weak_machine_candidates(self):
        with patch.object(review, "verified_inputs", return_value=self.synthetic_values()):
            result = review.screening_report(ROOT, [])
        self.assertEqual((result["pending_rows"], result["complete_human_actual_rows"]), (16, 0))
        self.assertEqual(result["partition_counts"]["dual_asr_agree_human_uncertain"], 16)
        for row in result["rows"]:
            self.assertIsNone(row["actual_text"])
            self.assertIsNone(row["partial_text"])
            self.assertIsNone(row["actual_expected_keywords"])
            self.assertIsNone(row["label_diagnostics"])
            self.assertFalse(row["human_actual_review_complete"])
            self.assertEqual(row["acoustic_completeness"], "UNKNOWN")
            self.assertIsNone(row["ctc_target"])
            self.assertFalse(row["training_admitted"])
        for field in ("training_admitted", "shipping_approved", "independent_accuracy_qualified",
                      "human_gold", "new_run_authorized", "publication_authorized"):
            self.assertIs(result[field], False)
        self.assertEqual(result["new_model_calls"], 0)

    def test_latest_pending_supersedes_clean_consensus_without_losing_history(self):
        first = self.document()
        second = review.receipt_template(self.packet)
        second.update(revision=2, supersedes_receipt_sha256=review.digest(review.canonical(first)))
        second["rows"] = second["rows"][:1]
        with patch.object(review, "verified_inputs", return_value=self.synthetic_values()):
            result = review.screening_report(ROOT, [first, second])
        row = result["rows"][0]
        self.assertIsNone(row["actual_text"])
        self.assertFalse(row["human_actual_review_complete"])
        self.assertEqual(row["screening"]["category"], "dual_asr_agree_human_uncertain")
        self.assertEqual(row["review_history"][0]["row"]["actual_text"], "虚构甲")
        self.assertEqual(len(row["review_history"]), 2)

    def test_oov_agreement_never_changes_production_words_or_policy(self):
        document = self.document()
        with patch.object(review, "verified_inputs", return_value=self.synthetic_values()):
            row = review.screening_report(ROOT, [document])["rows"][0]
        self.assertEqual(row["screening"]["category"], "human_and_both_asr_agree")
        self.assertEqual(row["actual_text"], "虚构甲")
        self.assertEqual(row["production_oov_characters"], sorted("虚构甲"))
        self.assertEqual(row["production_transcript_policy"], "xiaowo-four-syllable-transcript-v1")
        self.assertIsNone(row["ctc_target"])
        self.assertFalse(row["training_admitted"])

    def test_screening_cli_is_reproducible_private_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "screening.json"
            receipt = Path(directory) / "invented-receipt.json"
            document = self.document()
            receipt.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
            with patch.object(review, "verified_inputs", return_value=self.synthetic_values()):
                expected = review.screening_report(ROOT, [document])
                self.assertEqual(review.main(["screening", "--receipts", str(receipt), "--output", str(output)]), 0)
                self.assertEqual(review.load(output), expected)
                before = output.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(review.main(["screening", "--output", str(output)]), 2)
                self.assertEqual(output.read_bytes(), before)
            self.assertEqual(review.load(receipt), document)

    def test_invalid_receipts_fail_before_screening_output(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "invalid.json"
            receipt.write_text('{"schema":"wrong"}')
            output = Path(directory) / "absent.json"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(review.main(["screening", "--receipts", str(receipt), "--output", str(output)]), 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
