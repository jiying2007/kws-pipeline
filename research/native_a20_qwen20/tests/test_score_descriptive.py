"""Pure-data toy tests: no audio/model/frontend/decoder execution."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "src" / "score_descriptive.py"
SPEC = importlib.util.spec_from_file_location("score_descriptive", SOURCE)
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(activations=None, instrumented=True):
    """20 fake hashes and tiny invented traces, preserving only the label grid."""
    activations = activations or {}
    manifest = {"rows": []}
    geometry = {"rows": []}
    for label, (text, kind, keyword) in SCORER.LABELS.items():
        for voice_key, (voice, split) in SCORER.VOICES.items():
            recording = f"qwen3-{label}-{voice_key}"
            manifest["rows"].append({
                "recording": recording, "declared_text": text, "kind": kind,
                "keyword_id": keyword, "voice": voice, "historical_split": split,
                "historical_exposed": True, "sample_rate_hz": 16000,
                "frames": 14720, "duration_s": 0.92, "verified": True,
                "wav_sha256": digest("toy wav " + recording),
                "pcm_sha256": digest("toy pcm " + recording),
            })
            plan = []
            for index, available, call_samples, fbank, rows, center in (
                    (0, 4800, 4800, 28, 9, 0), (1, 9600, 4800, 30, 10, 27),
                    (2, 14400, 4800, 30, 10, 57), (3, 14720, 320, 0, 0, 87)):
                plan.append({"call_index": index, "available_samples": available,
                             "call_samples": call_samples, "fbank_rows": fbank,
                             "selected_rows": rows,
                             "centers": [center + 3 * row for row in range(rows)],
                             "phase": "finish" if index == 3 else "feed"})
            geometry["rows"].append({"recording": recording, "frames": 14720,
                                     "feed_calls": 4, "callbacks": 4, "fbank_rows": 88,
                                     "model_rows": 29, "callback_plan": plan})
    geometry["totals"] = {key: sum(row[key] for row in geometry["rows"])
                          for key in SCORER.COUNT_FIELDS}
    bindings = {key: digest("toy " + key) for key in SCORER.HASH_KEYS
                if key not in ("manifest_sha256", "geometry_sha256")}
    header = {"kind": "run_start", "schema": SCORER.RAW_SCHEMA, **bindings,
              "manifest_sha256": hashlib.sha256(SCORER.canonical_json_bytes(manifest)).hexdigest(),
              "geometry_sha256": hashlib.sha256(SCORER.canonical_json_bytes(geometry)).hexdigest()}
    raw = [header]
    total_decoded = total_events = 0
    for row, geom in zip(manifest["rows"], geometry["rows"]):
        name = row["recording"]
        identity = {key: row[key] for key in ("recording", "frames", "wav_sha256", "pcm_sha256")}
        raw.append({"kind": "clip_start", **identity})
        model_rows = decoded_rows = event_count = 0
        for plan in geom["callback_plan"]:
            index = plan["call_index"]
            if index == 3 and instrumented:
                raw.append({"kind": "feed", "recording": name, "feed_index": 3,
                            "input_samples": 320, "cumulative_samples": 14720, "callbacks_delta": 0})
            keyword = activations.get((name, index), 0)
            rows = plan["selected_rows"]
            start = model_rows * 3
            decoded = 4 if keyword else rows
            model_rows += rows
            decoded_rows += decoded
            event_count += int(keyword > 0)
            call = {"kind": "callback", "recording": name, **plan,
                    "valid": int(rows > 0), "state": int(keyword > 0), "keyword": keyword,
                    "start_frame": start if keyword else (-1 if rows else 0),
                    "end_frame": start + 9 if keyword else (-1 if rows else 0),
                    "score": 0.8 if keyword else 0, "decoder_rows_decoded": decoded,
                    "decoder_total_frames": model_rows * 3,
                    "logits": [[0.1, -0.2, 0.3, 0.4, -0.5, 0.6] for _ in range(rows)]}
            raw.append(call)
            if instrumented:
                if index < 3:
                    raw.append({"kind": "feed", "recording": name, "feed_index": index,
                                "input_samples": 4800, "cumulative_samples": 4800 * (index + 1),
                                "callbacks_delta": 1})
                else:
                    raw.append({"kind": "finish", "recording": name, "finish_calls": 1,
                                "callbacks_delta": 1})
        raw.append({"kind": "clip_end", **identity, "feed_calls": 4, "finish_calls": 1,
                    "callbacks": 4, "fbank_rows": 88, "model_rows": 29,
                    "decoder_rows_decoded": decoded_rows, "event_count": event_count, "complete": True})
        total_decoded += decoded_rows
        total_events += event_count
    raw.append({"kind": "run_end", "complete": True, "clips": 20,
                **geometry["totals"], "decoder_rows_decoded": total_decoded, "event_count": total_events})
    return raw, manifest, geometry, bindings


def first(raw, kind):
    return next(row for row in raw if row["kind"] == kind)


class DescriptiveScoreTests(unittest.TestCase):
    def run_score(self, data):
        return SCORER.score(*data)

    def assert_bad(self, data, expression):
        with self.assertRaisesRegex(SCORER.ValidationError, expression):
            self.run_score(data)

    def test_all_empty_are_explicit_and_complete(self):
        report = self.run_score(fixture())
        self.assertEqual(len(report["clips"]), 20)
        self.assertTrue(all(clip["events"] == [] for clip in report["clips"]))
        self.assertEqual(report["aggregate"]["positive_miss_clips"], 10)
        self.assertEqual(report["aggregate"]["confusable_triggered_clips"], 0)
        self.assertTrue(all(clip["feed_finish_records_validated"] for clip in report["clips"]))

    def test_wrong_keyword_is_preserved_and_does_not_count_as_hit(self):
        report = self.run_score(fixture({("qwen3-kw1-dylan", 0): 2}))
        clip = report["clips"][0]
        self.assertEqual(clip["descriptive_outcome"], "miss")
        self.assertEqual(clip["events"][0]["keyword"], 2)
        self.assertEqual(report["aggregate"]["positive_wrong_keyword_events"], 1)
        self.assertEqual(report["aggregate"]["positive_hit_clips"], 0)

    def test_repeated_correct_events_are_not_deduplicated(self):
        report = self.run_score(fixture({("qwen3-kw1-dylan", 0): 1,
                                         ("qwen3-kw1-dylan", 2): 1}))
        self.assertEqual(len(report["clips"][0]["events"]), 2)
        self.assertEqual(report["aggregate"]["positive_matching_events"], 2)
        self.assertEqual(report["aggregate"]["positive_hit_clips"], 1)
        self.assertEqual(report["aggregate"]["repeated_activation_events"], 1)
        self.assertEqual(report["aggregate"]["positive_repeated_matching_events"], 1)

    def test_wrong_then_right_retains_both(self):
        report = self.run_score(fixture({("qwen3-kw1-dylan", 0): 2,
                                         ("qwen3-kw1-dylan", 2): 1}))
        self.assertEqual(report["clips"][0]["descriptive_outcome"], "hit")
        self.assertEqual([e["keyword"] for e in report["clips"][0]["events"]], [2, 1])
        self.assertEqual(report["aggregate"]["positive_repeated_matching_events"], 0)

    def test_confusable_events_and_original_groups(self):
        report = self.run_score(fixture({("qwen3-repeat-nihao-serena", 0): 1,
                                         ("qwen3-repeat-nihao-serena", 2): 2}))
        self.assertEqual(report["aggregate"]["confusable_triggered_clips"], 1)
        self.assertEqual(report["aggregate"]["confusable_event_count"], 2)
        self.assertEqual(report["by"]["historical_split"]["development_a"]["confusable_event_count"], 2)
        self.assertEqual(report["by"]["voice"]["Serena"]["confusable_event_count"], 2)
        self.assertEqual(report["by"]["keyword_id"]["null"]["confusable_event_count"], 2)

    def test_available_audio_time_is_not_decoder_endpoint(self):
        report = self.run_score(fixture({("qwen3-kw1-dylan", 0): 1}))
        event = report["clips"][0]["events"][0]
        self.assertEqual(event["available_audio_s"], 4800 / 16000)
        self.assertEqual(event["end_frame"], 9)
        self.assertNotEqual(event["available_audio_s"], event["end_frame"] * 0.01)
        self.assertIn("not a word endpoint", report["event_time_semantics"])

    def test_absent_feed_instrumentation_fails(self):
        self.assert_bad(fixture(instrumented=False), "mandatory feed/finish instrumentation")

    def test_missing_clip_cannot_be_a_miss(self):
        data = fixture()
        data[0][:] = [r for r in data[0] if r.get("recording") != "qwen3-kw1-dylan"]
        self.assert_bad(data, "recording/order mismatch")

    def test_duplicate_clip_fails(self):
        data = fixture()
        data[0].insert(2, copy.deepcopy(data[0][1]))
        self.assert_bad(data, "unexpected record kind")

    def test_duplicate_callback_fails(self):
        data = fixture()
        data[0].insert(3, copy.deepcopy(data[0][2]))
        self.assert_bad(data, "geometry mismatch")

    def test_missing_callback_fails(self):
        data = fixture()
        del data[0][2]
        self.assert_bad(data, "callback delta mismatch")

    def test_missing_run_end_fails(self):
        data = fixture()
        data[0].pop()
        self.assert_bad(data, "missing, duplicate or trailing")

    def test_trailing_records_fail(self):
        data = fixture()
        data[0].append(copy.deepcopy(data[0][-1]))
        self.assert_bad(data, "missing, duplicate or trailing")

    def test_missing_finish_fails(self):
        data = fixture()
        data[0].remove(first(data[0], "finish"))
        self.assert_bad(data, "mandatory feed/finish instrumentation")

    def test_duplicate_finish_fails(self):
        data = fixture()
        record = first(data[0], "finish")
        data[0].insert(data[0].index(record) + 1, copy.deepcopy(record))
        self.assert_bad(data, "record after finish")

    def test_bad_finish_count_fails(self):
        for kind in ("finish", "clip_end"):
            with self.subTest(kind=kind):
                data = fixture()
                first(data[0], kind)["finish_calls"] = 2
                self.assert_bad(data, "finish_calls must be one")

    def test_incomplete_clip_fails(self):
        data = fixture()
        first(data[0], "clip_end")["complete"] = False
        self.assert_bad(data, "clip incomplete")

    def test_inconsistent_or_unknown_label_fails(self):
        for field, value in (("declared_text", "unexpected"), ("keyword_id", 2),
                             ("kind", "confusable"), ("historical_split", "held_out"),
                             ("historical_exposed", False), ("voice", "Someone")):
            with self.subTest(field=field):
                data = fixture()
                data[1]["rows"][0][field] = value
                self.assert_bad(data, "manifest .*: (inconsistent|historical_exposed)")

    def test_contradictory_raw_label_fails(self):
        data = fixture()
        first(data[0], "clip_start")["declared_text"] = "小窝小窝"
        self.assert_bad(data, "label metadata mismatch")

    def test_input_hash_mismatch_fails(self):
        for kind in ("clip_start", "clip_end"):
            data = fixture()
            first(data[0], kind)["wav_sha256"] = "f" * 64
            self.assert_bad(data, "input hash mismatch")

    def test_all_run_hash_bindings_checked(self):
        for key in SCORER.HASH_KEYS:
            data = fixture()
            data[0][0][key] = "e" * 64
            self.assert_bad(data, "run_start hash mismatch")

    def test_all_zero_hashes_fail_even_when_consistent(self):
        data = fixture()
        data[0][0]["model_sha256"] = "0" * 64
        data[3]["model_sha256"] = "0" * 64
        self.assert_bad(data, "all-zero hash")
        data = fixture()
        data[1]["rows"][0]["wav_sha256"] = "0" * 64
        self.assert_bad(data, "all-zero hash")

    def test_model_loaded_once_at_start_is_allowed(self):
        data = fixture()
        load = {"kind": "model_loaded", "model_load_us": 12.5,
                "state_bytes": 1000, "weights_bytes": 400}
        data[0].insert(1, load)
        report = self.run_score(data)
        self.assertEqual(report["model_load_record"], load)
        data[0].insert(1, copy.deepcopy(load))
        self.assert_bad(data, "missing clip_start")

    def test_late_model_loaded_record_fails(self):
        data = fixture()
        data[0].insert(2, {"kind": "model_loaded", "recording": "qwen3-kw1-dylan"})
        self.assert_bad(data, "unexpected record kind")

    def test_independent_bindings_required(self):
        data = fixture()
        with self.assertRaisesRegex(SCORER.ValidationError, "bindings are required"):
            SCORER.score(*data[:3])

    def test_geometry_mismatch_fails(self):
        data = fixture()
        first(data[0], "callback")["available_samples"] += 1
        self.assert_bad(data, "geometry mismatch")

    def test_wrong_decoder_clock_fails(self):
        data = fixture()
        first(data[0], "callback")["decoder_total_frames"] += 3
        self.assert_bad(data, "decoder clock mismatch")

    def test_early_break_without_activation_fails(self):
        data = fixture()
        first(data[0], "callback")["decoder_rows_decoded"] = 2
        self.assert_bad(data, "invalid inactive callback result")

    def test_valid_early_activation_keeps_full_model_clock(self):
        data = fixture({("qwen3-kw1-dylan", 1): 1})
        # Four decoded rows produce four keyword tokens, while all ten model
        # rows advance the clock. This toy obeys the actual frame constraints.
        report = self.run_score(data)
        event = report["clips"][0]["events"][0]
        self.assertEqual(event["decoder_total_frames"], 57)
        self.assertEqual(event["decoder_rows_decoded"], 4)
        self.assertEqual(len(event["logits"]), 10)

    def test_impossible_short_activation_duration_fails(self):
        for duration in (0, 3):
            with self.subTest(duration=duration):
                data = fixture({("qwen3-kw1-dylan", 0): 1})
                first(data[0], "callback")["end_frame"] = duration
                self.assert_bad(data, "duration violates fixed decoder gate")

    def test_impossible_long_activation_duration_fails(self):
        # Isolate the saved-result gate at a later synthetic callback clock.
        plan = {"call_index": 9, "available_samples": 44800, "call_samples": 1600,
                "fbank_rows": 10, "selected_rows": 3, "centers": [252, 255, 258]}
        call = {**plan, "logits": [[0.0] * 6 for _ in range(3)], "valid": 1,
                "state": 1, "keyword": 1, "start_frame": 0, "end_frame": 252,
                "score": 0.8, "decoder_rows_decoded": 1, "decoder_total_frames": 261}
        with self.assertRaisesRegex(SCORER.ValidationError, "duration violates fixed decoder gate"):
            SCORER._validate_callback(call, plan, 84, "toy callback")

    def test_cooldown_violation_fails(self):
        # End frames 9 and 36 are individually plausible but only 27 apart.
        data = fixture({("qwen3-kw1-dylan", 0): 1, ("qwen3-kw1-dylan", 1): 1})
        self.assert_bad(data, "cooldown of 50 frames")

    def test_skipped_and_reset_frames_cannot_support_later_event(self):
        for start in (0, 12):
            with self.subTest(start=start):
                data = fixture({("qwen3-kw1-dylan", 0): 1, ("qwen3-kw1-dylan", 2): 1})
                second = next(r for r in data[0] if r.get("recording") == "qwen3-kw1-dylan"
                              and r["kind"] == "callback" and r["call_index"] == 2)
                # Frame 0 was cleared by the first activation; frame 12 was in
                # that activation's skipped tail and was never decoded at all.
                second["start_frame"] = start
                self.assert_bad(data, "not in actually decoded centers since reset")

    def test_event_cannot_reach_undecoded_current_tail(self):
        data = fixture({("qwen3-kw1-dylan", 0): 1})
        first(data[0], "callback")["end_frame"] = 12
        self.assert_bad(data, "invalid activation frame interval")

    def test_end_event_counts_match_preserved_activations(self):
        for kind in ("clip_end", "run_end"):
            with self.subTest(kind=kind):
                data = fixture({("qwen3-kw1-dylan", 0): 1})
                first(data[0], kind)["event_count"] = 0
                self.assert_bad(data, "event_count mismatch")

    def test_zero_row_callback_must_be_all_zero(self):
        data = fixture()
        zero = next(r for r in data[0] if r["kind"] == "callback" and r["selected_rows"] == 0)
        zero["valid"] = 1
        self.assert_bad(data, "invalid empty callback result")

    def test_nonfinite_logit_or_diagnostic_fails(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            for field in ("logits", "service_time_us"):
                with self.subTest(value=value, field=field):
                    data = fixture()
                    call = first(data[0], "callback")
                    if field == "logits":
                        call["logits"][0][0] = value
                    else:
                        call[field] = value
                    self.assert_bad(data, "finite number")

    def test_wrong_logits_shape_fails(self):
        data = fixture()
        first(data[0], "callback")["logits"][0].pop()
        self.assert_bad(data, "six classes")

    def test_unknown_keyword_or_boolean_count_fails(self):
        data = fixture()
        first(data[0], "callback")["keyword"] = 3
        self.assert_bad(data, "unknown decoder keyword")
        data = fixture()
        first(data[0], "callback")["valid"] = True
        self.assert_bad(data, "invalid valid")
        data = fixture()
        first(data[0], "callback")["centers"][0] = False
        self.assert_bad(data, "expected integer")

    def test_corrupt_feed_cumulative_samples_fails(self):
        data = fixture()
        first(data[0], "feed")["cumulative_samples"] += 1
        self.assert_bad(data, "feed sample count mismatch")

    def test_feed_size_must_match_fixed_protocol(self):
        data = fixture()
        feed = first(data[0], "feed")
        feed.update(input_samples=4799, cumulative_samples=4799)
        self.assert_bad(data, "fixed 4800-sample chunk protocol")

    def test_wrong_totals_fail(self):
        data = fixture()
        data[0][-1]["decoder_rows_decoded"] -= 1
        self.assert_bad(data, "decoded total mismatch")
        data = fixture()
        data[0][-1]["finish_calls"] = 21
        self.assert_bad(data, "finish total mismatch")

    def test_missing_manifest_row_and_duplicate_geometry_row_fail(self):
        data = fixture()
        data[1]["rows"].pop()
        self.assert_bad(data, "exactly the 20 bounded clips")
        data = fixture()
        data[2]["rows"][-1] = copy.deepcopy(data[2]["rows"][0])
        self.assert_bad(data, "duplicate recording")

    def test_unverified_input_and_contradictory_receipt_fail(self):
        data = fixture()
        data[1]["rows"][0]["verified"] = False
        self.assert_bad(data, "input is not verified")
        data = fixture()
        data[1]["rows"][0]["review_receipt"] = {"intended_text": "小窝小窝"}
        self.assert_bad(data, "inconsistent review receipt")

    def test_json_file_hashes_and_malformed_eof(self):
        raw, manifest, geometry, bindings = fixture()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            mp, gp, rp = directory / "m.json", directory / "g.json", directory / "raw.jsonl"
            mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            gp.write_text(json.dumps(geometry, indent=2), encoding="utf-8")
            raw[0]["manifest_sha256"] = hashlib.sha256(mp.read_bytes()).hexdigest()
            raw[0]["geometry_sha256"] = hashlib.sha256(gp.read_bytes()).hexdigest()
            data = "".join(json.dumps(row) + "\n" for row in raw)
            rp.write_text(data, encoding="utf-8")
            self.assertEqual(SCORER.score(rp, mp, gp, bindings)["aggregate"]["clips"], 20)
            rp.write_text(data.rstrip("\n"), encoding="utf-8")
            with self.assertRaisesRegex(SCORER.ValidationError, "truncated EOF"):
                SCORER.score(rp, mp, gp, bindings)
            rp.write_text(data[:-20] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SCORER.ValidationError, "malformed JSON"):
                SCORER.score(rp, mp, gp, bindings)

    def test_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            path.write_text('{"kind":"run_start","kind":"run_end"}\n', encoding="utf-8")
            data = fixture()
            with self.assertRaisesRegex(SCORER.ValidationError, "duplicate JSON object key"):
                SCORER.score(path, *data[1:])

    def test_no_quality_rate_or_latency_metrics(self):
        report = self.run_score(fixture())
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key.lower(), {"frr", "far", "latency", "qualified", "accuracy"})
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        walk(report)


if __name__ == "__main__":
    unittest.main()
