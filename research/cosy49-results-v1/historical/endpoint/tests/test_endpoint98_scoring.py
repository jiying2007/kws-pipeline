"""Invented observations only. Never launches compiler/native/model/decoder work."""
import ast
import copy
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import score_endpoint as scorer
import compare_endpoint as comparison
import analyze_saved_pair as analysis


def toy_trace():
    manifest = json.loads((ROOT / "metadata/manifest.json").read_text())
    geometry = json.loads((ROOT / "metadata/geometry.json").read_text())
    bindings = dict(protocol_sha256="1"*64, model_sha256="2"*64, library_sha256="3"*64,
                    decoder_config_sha256="4"*64, manifest_sha256=scorer.FROZEN_MANIFEST_SHA,
                    geometry_sha256=scorer.FROZEN_GEOMETRY_SHA)
    records = [{"kind": "run_start", "schema": scorer.RAW_SCHEMA, **bindings},
               {"kind": "model_loaded", "weights_bytes": 1565280, "state_bytes": 333704,
                "wall_ns_since_run_start": 0, "cpu_ns_since_run_start": 0}]
    tick = 0
    for row, geo in zip(manifest["rows"], geometry["rows"]):
        identity = {k: row[k] for k in ("recording", "frames", "wav_sha256", "pcm_sha256")}
        records.append({"kind": "clip_start", **identity})
        count = 0
        for i, call in enumerate(geo["callback_plan"]):
            count += call["call_samples"]
            tick += 1
            active_rows = bool(call["selected_rows"])
            rec = {"kind": "callback", "recording": row["recording"], **call,
                   "valid": int(active_rows), "state": 0, "keyword": 0,
                   "start_frame": -1 if active_rows else 0, "end_frame": -1 if active_rows else 0,
                   "score": 0, "decoder_rows_decoded": call["selected_rows"],
                   "logits": [[1, 0, 0, 0, 0, 0] for _ in range(call["selected_rows"])],
                   "callback_entry_wall_ns_since_run_start": tick,
                   "callback_entry_cpu_ns_since_run_start": tick}
            feed = {"kind": "feed", "recording": row["recording"], "feed_index": i,
                    "input_samples": call["call_samples"], "cumulative_samples": count,
                    "callbacks_delta": int(call["phase"] == "feed"),
                    "service_wall_ns_including_callback_output": 0,
                    "service_cpu_ns_including_callback_output": 0}
            if call["phase"] == "finish":
                records.append(feed)
            records.append(rec)
            if call["phase"] == "feed":
                records.append(feed)
        records.append({"kind": "finish", "recording": row["recording"], "finish_calls": 1,
                        "callbacks_delta": geo["tail_feeds"], "service_wall_ns_including_callback_output": 0,
                        "service_cpu_ns_including_callback_output": 0})
        records.append({"kind": "clip_end", **identity, **{k: geo[k] for k in
                        ("feed_calls", "finish_calls", "callbacks", "fbank_rows", "model_rows")},
                        "decoder_rows_decoded": geo["model_rows"], "event_count": 0, "complete": True})
    records.append({"kind": "run_end", "complete": True, **geometry["totals"],
                    "decoder_rows_decoded": geometry["totals"]["model_rows"], "event_count": 0,
                    "wall_ns": tick+1, "process_cpu_ns": tick+1, "maxrss_kib": 1000,
                    "minor_faults": 0, "major_faults": 0, "block_input_ops": 0, "block_output_ops": 0})
    return records, bindings


def event(keyword, index=0):
    return {"keyword": keyword, "start_frame": 60*index, "end_frame": 60*index+6,
            "available_samples": 4800*(index+1), "score": 0.5}


def row(report, alias=None, group=None):
    return next(r for r in report["streams"] if
                (r.get("alias") == alias if alias is not None else r["group"] == group))


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.bindings = toy_trace()
        cls.report = cls.score(cls.raw)

    @classmethod
    def score(cls, records):
        return scorer.score(records, ROOT / "metadata/manifest.json", ROOT / "metadata/geometry.json", cls.bindings)

    def bad(self, edit):
        records = copy.deepcopy(self.raw)
        edit(records)
        with self.assertRaises(scorer.ValidationError):
            self.score(records)

    def callback(self, records):
        return next(r for r in records if r["kind"] == "callback")

    def test_complete_invented_exact98(self):
        self.assertEqual(len(self.report["streams"]), 98)
        self.assertEqual(self.report["observed_totals"]["model_rows"], 51829)
        self.assertEqual(self.report["aggregate"]["activation_events"], 0)
        self.assertFalse(self.report["qualification"])

    def test_old76_schema_rejected(self):
        self.bad(lambda records: records[0].update(schema="a20-endpoint76-paired-raw-v1"))

    def test_missing_finish_rejected(self):
        self.bad(lambda records: records.pop(next(i for i, r in enumerate(records) if r["kind"] == "finish")))

    def test_truncated_trace_rejected(self):
        self.bad(lambda records: records.pop())

    def test_wrong_model_binding_rejected(self):
        self.bad(lambda records: records[0].update(model_sha256="5"*64))

    def test_missing_source_rejected(self):
        self.bad(lambda records: records[2].update(recording="invented-wrong-source"))

    def test_nonfinite_logits_rejected(self):
        self.bad(lambda records: self.callback(records)["logits"][0].__setitem__(0, float("nan")))

    def test_callback_geometry_rejected(self):
        self.bad(lambda records: self.callback(records).update(decoder_total_frames=0))

    def test_invalid_empty_callback_rejected(self):
        self.bad(lambda records: next(r for r in records if r["kind"] == "callback" and
                                      r["selected_rows"] == 0).update(state=1))

    def test_decode_endpoint_must_have_entered_decoder(self):
        self.bad(lambda records: self.callback(records).update(state=1, keyword=1, start_frame=0,
                  end_frame=6, score=0.5, decoder_rows_decoded=1))

    def test_feed_size_rejected(self):
        self.bad(lambda records: next(r for r in records if r["kind"] == "feed").update(input_samples=1))

    def test_bad_final_total_rejected(self):
        self.bad(lambda records: records[-1].update(clips=76))

    def test_nonmonotonic_timing_rejected(self):
        self.bad(lambda records: records[-1].update(wall_ns=0))

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(scorer.ValidationError):
            scorer._parse_json('{"invented":1,"invented":2}', "toy")

    def test_raw_cap_and_truncated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "invented.jsonl"
            with path.open("wb") as output:
                output.truncate(14*1024**2+1)
            with self.assertRaisesRegex(scorer.ValidationError, "size cap"):
                scorer._load_raw(path)
            path.write_text('{}')
            with self.assertRaisesRegex(scorer.ValidationError, "final newline"):
                scorer._load_raw(path)

    def test_metadata_hashes_and_approved_gate_are_bound(self):
        for name, digest in (("manifest.json", scorer.FROZEN_MANIFEST_SHA),
                             ("geometry.json", scorer.FROZEN_GEOMETRY_SHA)):
            self.assertEqual(hashlib.sha256((ROOT/"metadata"/name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((ROOT/"src/candidate_control.py").read_bytes()).hexdigest(),
                         comparison.GATE_SHA256)


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw, bindings = toy_trace()
        cls.original = scorer.score(raw, ROOT/"metadata/manifest.json", ROOT/"metadata/geometry.json", bindings)
        for r in cls.original["streams"]:
            if r["group"] == "Qwen20" and r["kind"] == "positive":
                r["events"] = [event(r["keyword_id"])]
            if r["group"] == "D20":
                target = r["target_ids"]
                r["events"] = [event(k) for k, kw in enumerate(([1, 2, 3, 4], [3, 4, 3, 4]), 1)
                               if any(target[i:i+4] == kw for i in range(len(target)-3))]
            if r["group"] == "Cosy22":
                r["events"] = [event(k) for k in {"A": [1], "B": [2], "H": [1], "N": [1], "O": [2]}.get(r["alias"], [])]
        # All event observations below are invented. No historical acoustic trace is scored.
        cls.historical = {"qwen_frozen_scoring_baseline": {"clips": copy.deepcopy([
            r for r in cls.original["streams"] if r["group"] == "Qwen20"])}, "groups": {}}
        for group in ("Qwen20", "Cosy12", "N0", "DEMAND", "FLEURS20"):
            cls.historical["groups"][group] = {"rows": [
                {"recording": r["recording"], "native_events" if group == "Cosy12" else "events": copy.deepcopy(r["events"])}
                for r in cls.original["streams"] if r["group"] == group]}

    def setUp(self):
        self.before = copy.deepcopy(self.original)
        self.after = copy.deepcopy(self.original)
        frozen = comparison.frozen_document
        self.patch = patch.object(comparison, "frozen_document", side_effect=lambda name, digest:
                                 self.historical if name == "FROZEN-HISTORICAL-EVIDENCE.json" else frozen(name, digest))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def compare(self):
        return comparison.compare_pair(self.before, self.after)

    def gain(self):
        row(self.after, alias="C")["events"] = [event(1)]

    def test_each_single_gain_suffices_without_qwen_improvement(self):
        for alias in ("C", "H", "N"):
            with self.subTest(alias=alias):
                self.after = copy.deepcopy(self.original)
                row(self.after, alias=alias)["events"] = [event(1)] if alias == "C" else []
                result = self.compare()
                self.assertEqual(result["status"], "EXPOSED_DEVELOPMENT_SIGNAL_ONLY")
                self.assertFalse(result["old76_improvement_required"])
                self.assertFalse(result["quality_pass"])
                self.assertFalse(result["next_stage_authorized"])

    def test_no_targeted_gain_stops(self):
        self.assertEqual(self.compare()["status"], "STOP_NO_TARGETED_ENDPOINT_GAIN")

    def test_all_prior_correct_cosy_keywords_protected(self):
        for alias in ("A", "B", "O"):
            with self.subTest(alias=alias):
                self.after = copy.deepcopy(self.original)
                self.gain()
                row(self.after, alias=alias)["events"] = []
                self.assertEqual(self.compare()["status"], "STOP_COSY_REGRESSION")

    def test_wrong_and_repeated_cosy_events_are_not_hidden(self):
        for event_list in ([event(2)], [event(1), event(1, 1)], [event(1), event(2, 1)]):
            self.after = copy.deepcopy(self.original)
            self.gain()
            row(self.after, alias="A")["events"] = event_list
            self.assertEqual(self.compare()["status"], "STOP_COSY_REGRESSION")

    def test_d_and_i_short_words_remain_nonwake(self):
        for alias, text in (("D", "小窝"), ("I", "小屋")):
            self.after = copy.deepcopy(self.original)
            self.gain()
            item = row(self.after, alias=alias)
            self.assertEqual(item["text"], text)
            item["events"] = [event(1)]
            self.assertEqual(self.compare()["status"], "STOP_COSY_REGRESSION")

    def test_w_is_diagnostic_only_and_q_t_excluded(self):
        self.gain()
        row(self.after, alias="W")["events"] = [event(1), event(2, 1)]
        result = self.compare()
        self.assertEqual(result["status"], "EXPOSED_DEVELOPMENT_SIGNAL_ONLY")
        self.assertIn("final-syllable cutoff uncertain", result["cosy22_endpoint_gate"]["W_interpretation"])
        self.assertFalse(set("QT") & {r.get("alias") for r in self.before["streams"]})

    def test_cosy_baseline_drift_visible_and_stops(self):
        row(self.before, alias="H")["events"] = []
        self.gain()
        result = self.compare()
        self.assertEqual(result["status"], "BASELINE_DRIFT")
        self.assertEqual(result["cosy22_endpoint_gate"]["status"], "STOP_BASELINE_DRIFT")

    def test_qwen_no_regression_guards(self):
        self.gain()
        item = row(self.after, group="Qwen20")
        item["events"] = []
        self.assertEqual(self.compare()["status"], "OLD76_REGRESSION")
        for values in ([event(item["keyword_id"]), event(item["keyword_id"], 1)],
                       [event(item["keyword_id"]), event(3-item["keyword_id"], 1)]):
            item["events"] = values
            self.assertEqual(self.compare()["status"], "OLD76_REGRESSION")

    def test_d20_and_n0_guards(self):
        for group in ("D20", "N0"):
            self.after = copy.deepcopy(self.original)
            self.gain()
            item = row(self.after, group=group)
            item["events"] = [] if item["events"] else [event(1)]
            self.assertEqual(self.compare()["status"], "OLD76_REGRESSION")

    def test_all_descriptive_groups_changes_require_unadjudicated_review(self):
        self.gain()
        for group in ("Cosy12", "DEMAND", "FLEURS20"):
            row(self.after, group=group)["events"] = [event(1)]
        result = self.compare()
        self.assertEqual(result["status"], "REVIEW_REQUIRED_UNADJUDICATED_CHANGES")
        self.assertEqual(len(result["unadjudicated_behavior_changes"]), 3)
        self.assertTrue(all(r["adjudication"] == "UNADJUDICATED" for r in result["unadjudicated_behavior_changes"]))
        self.assertIsNone(result["false_activation_rate_per_hour"])

    def test_greedy_changes_are_descriptive_and_do_not_supply_gain(self):
        row(self.after, alias="C")["greedy_ids"] = [1, 2, 3, 4]
        self.assertEqual(self.compare()["status"], "STOP_NO_TARGETED_ENDPOINT_GAIN")

    def test_historical_native_drift_explicit(self):
        self.gain()
        row(self.before, group="N0")["events"] = [event(1)]
        result = self.compare()
        self.assertEqual(result["status"], "BASELINE_DRIFT")
        self.assertTrue(result["original_historical_drift"]["N0"]["discrete_drift"])
        self.assertIsNone(result["original_historical_drift"]["D20"]["discrete_drift"])

    def test_missing_trace_or_mapping_cannot_be_negative(self):
        row(self.after, alias="C")["complete"] = False
        with self.assertRaises(ValueError):
            self.compare()
        self.after = copy.deepcopy(self.original)
        row(self.after, alias="D")["text"] = "小窝小窝"
        with self.assertRaises(ValueError):
            self.compare()


class SavedAnalysisTests(unittest.TestCase):
    def saved_fixture(self, base):
        def write(name, data):
            path = base/name
            path.write_text(json.dumps(data))
            return {"path": name, "bytes": path.stat().st_size, "sha256": analysis.sha(path)}
        bundles = [write("toy-" + arm + ".json", {"payload": {"sha256": str(i+1)*64},
                                                 "library": {"sha256": str(i+3)*64}})
                   for i, (arm, _) in enumerate(analysis.ARMS)]
        release = {"schema": "a20-endpoint98-paired-release-v1", "protocol_sha256": "5"*64,
                   "decoder_config_sha256": "6"*64,
                   **{key: ref for (_, key), ref in zip(analysis.ARMS, bundles)}}
        files = [write("release-used.json", release)]
        files += [write(arm+".raw.jsonl", {"invented": arm}) for arm, _ in analysis.ARMS]
        summary = {"schema": "a20-endpoint98-pair-acquisition-v1",
                   "status": "PAIR_ACQUIRED_PENDING_PRIVATE_READBACK_AND_INDEPENDENT_SAVED_AUDIT",
                   "results": [{"arm": arm, "status": "COMPLETE_ACQUISITION_PENDING_TRACE_AUDIT"}
                               for arm, _ in analysis.ARMS], "files": files}
        write("pair-summary.json", summary)
        write("receipt.json", {"schema": "a20-endpoint98-raw-private-readback-v1", "status": "PASS",
                               "pair_summary_sha256": analysis.sha(base/"pair-summary.json")})
        return summary

    def test_caps_and_arm_identity(self):
        self.assertEqual(analysis.SAVED_FILE_CAP, 32*1024**2)
        self.assertEqual([a for a, _ in analysis.ARMS], ["original_A20", "candidate_cosy49_step300"])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)/"invented.json"
            with path.open("wb") as output:
                output.truncate(analysis.SAVED_FILE_CAP+1)
            with self.assertRaisesRegex(RuntimeError, "cap"):
                analysis.bounded_load(path)

    def test_failed_acquisition_cannot_write_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            (base/"pair-summary.json").write_text(json.dumps({"schema": "a20-endpoint98-pair-acquisition-v1", "status": "FAILED_NO_RETRY"}))
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                analysis.analyze(base, base/"absent-receipt.json", base/"output")
            self.assertFalse((base/"output").exists())

    def test_saved_analysis_invented_positive_path_binds_order_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            self.saved_fixture(base)
            with patch.object(analysis, "file_ref", side_effect=lambda ref: base/ref["path"]), \
                 patch.object(analysis, "score", side_effect=[{"toy": "original"}, {"toy": "candidate"}]) as score, \
                 patch.object(analysis, "compare_pair", return_value={"status": "INVENTED_TEST_ONLY"}), \
                 patch("subprocess.Popen", side_effect=AssertionError("launch forbidden")):
                result = analysis.analyze(base, base/"receipt.json", base/"output")
            self.assertEqual([call.args[0].name for call in score.call_args_list],
                             [arm+".raw.jsonl" for arm, _ in analysis.ARMS])
            self.assertEqual(result["new_model_or_decoder_calls"], 0)
            self.assertFalse(result["next_stage_authorized"])
            identity = json.loads((base/"output/identity.json").read_text())
            self.assertEqual(identity["independent_saved_audit_status"], "PENDING")
            for ref in identity["files"]:
                path = base/"output"/ref["path"]
                self.assertEqual(path.stat().st_size, ref["bytes"])
                self.assertEqual(analysis.sha(path), ref["sha256"])

    def test_reversed_arm_summary_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            summary = self.saved_fixture(base)
            summary["results"].reverse()
            (base/"pair-summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(RuntimeError, "arm order"):
                analysis.analyze(base, base/"receipt.json", base/"output")
            self.assertFalse((base/"output").exists())

    def test_saved_evidence_drift_fails_before_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            self.saved_fixture(base)
            (base/"original_A20.raw.jsonl").write_text("invented changed evidence")
            with self.assertRaisesRegex(RuntimeError, "evidence drift"):
                analysis.analyze(base, base/"receipt.json", base/"output")
            self.assertFalse((base/"output").exists())

    def test_analysis_modules_have_no_launcher_calls(self):
        forbidden = {"Popen", "run", "system", "execv", "execve", "spawn", "ctypes", "CDLL"}
        for filename in ("score_endpoint.py", "compare_endpoint.py", "analyze_saved_pair.py"):
            tree = ast.parse((ROOT/"src"/filename).read_text())
            for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
                name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
                self.assertNotIn(name, forbidden, filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
