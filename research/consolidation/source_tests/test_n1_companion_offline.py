"""Partial N1 synthetic coverage; never historical evidence or PCM reproduction.

The original source files are unchanged.  A temporary copy of the N1 scorer is
bound to invented metadata, then exercised with invented logits and events.
No saved audio, acquisition input, model, network, native library, subprocess,
or original verify_saved.py execution is involved.  These tests cannot attest
the original PCM, human-adjudication bytes, saved observations, or PR485 result.
"""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[3] / "research/n1-wholeclip-leading-v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class N1CompanionOfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_before = {p: p.read_bytes() for p in
                             (SOURCE / "src/geometry.py", SOURCE / "src/score_endpoint.py")}
        cls.geometry = load("n1_consolidation_geometry", SOURCE / "src/geometry.py")
        cls.tmp = tempfile.TemporaryDirectory(prefix="n1-synthetic-only-")
        cls.addClassCleanup(cls.tmp.cleanup)
        root = Path(cls.tmp.name)
        (root / "src").mkdir()
        (root / "metadata").mkdir()
        cls.manifest = {"rows": [{"recording": "N1", "frames": 52578,
            "wav_sha256": digest(b"invented wav identity, no audio"),
            "pcm_sha256": digest(b"invented PCM identity, no audio"),
            "label": 1, "domain": "INVENTED_TEST_ONLY", "declared_text": "你好小窝屋"}]}
        row = dict(recording="N1", **cls.geometry.geometry(52578))
        keys = ("frames", "feed_calls", "callbacks", "fbank_rows", "model_rows")
        cls.geom = {"rows": [row], "totals": {k: row[k] for k in keys}}
        cls.geom["totals"].update(clips=1, finish_calls=1, decoder_input_rows=row["model_rows"])
        for name, value in (("manifest.json", cls.manifest), ("geometry.json", cls.geom)):
            (root / "metadata" / name).write_bytes(encoded(value))
        path = root / "src/score_endpoint.py"
        path.write_bytes(cls.source_before[SOURCE / "src/score_endpoint.py"])
        cls.scorer = load("n1_consolidation_synthetic_endpoint", path)
        # Only the in-memory test module is rebound to invented metadata.
        cls.scorer.FROZEN_MANIFEST_SHA = digest(encoded(cls.manifest))
        cls.scorer.FROZEN_GEOMETRY_SHA = digest(encoded(cls.geom))
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        for path, before in cls.source_before.items():
            if path.read_bytes() != before:
                raise AssertionError("Original N1 source changed")

    def fixture(self, events=None):
        events = events or {}
        s = self.scorer
        bindings = {key: digest(("INVENTED " + key).encode()) for key in s.HASH_KEYS
                    if key not in ("manifest_sha256", "geometry_sha256")}
        row, geo = self.manifest["rows"][0], self.geom["rows"][0]
        identity = {key: row[key] for key in ("recording", "frames", "wav_sha256", "pcm_sha256")}
        records = [{"kind": "run_start", "schema": s.RAW_SCHEMA, **bindings,
                    "manifest_sha256": s.FROZEN_MANIFEST_SHA, "geometry_sha256": s.FROZEN_GEOMETRY_SHA},
                   {"kind": "model_loaded", "weights_bytes": 1565280, "state_bytes": 333704,
                    "wall_ns_since_run_start": 0, "cpu_ns_since_run_start": 0},
                   {"kind": "clip_start", **identity}]
        model_rows = decoded = 0
        for plan in geo["callback_plan"]:
            index, rows = plan["call_index"], plan["selected_rows"]
            keyword = events.get(index, 0)
            searched = 4 if keyword else rows
            start = model_rows * 3
            model_rows += rows
            decoded += searched
            callback = {"kind": "callback", "recording": "N1", **plan,
                "valid": int(bool(rows)), "state": int(bool(keyword)), "keyword": keyword,
                "start_frame": start if keyword else (-1 if rows else 0),
                "end_frame": start + 9 if keyword else (-1 if rows else 0),
                "score": 0.75 if keyword else 0, "decoder_rows_decoded": searched,
                "logits": [[1, 0, 0, 0, 0, 0] for _ in range(rows)],
                "callback_entry_wall_ns_since_run_start": index + 1,
                "callback_entry_cpu_ns_since_run_start": index + 1}
            feed = {"kind": "feed", "recording": "N1", "feed_index": index,
                "input_samples": plan["call_samples"], "cumulative_samples": plan["available_samples"],
                "callbacks_delta": int(plan["phase"] == "feed"),
                "service_wall_ns_including_callback_output": 0,
                "service_cpu_ns_including_callback_output": 0}
            records.extend([feed, callback] if plan["phase"] == "finish" else [callback, feed])
        records += [{"kind": "finish", "recording": "N1", "finish_calls": 1,
                     "callbacks_delta": geo["tail_feeds"],
                     "service_wall_ns_including_callback_output": 0,
                     "service_cpu_ns_including_callback_output": 0},
                    {"kind": "clip_end", **identity, **{key: geo[key] for key in s.COUNT_FIELDS},
                     "finish_calls": 1, "decoder_rows_decoded": decoded, "event_count": len(events), "complete": True},
                    {"kind": "run_end", "complete": True, **self.geom["totals"],
                     "decoder_rows_decoded": decoded, "event_count": len(events),
                     "wall_ns": 100, "process_cpu_ns": 100, "maxrss_kib": 1,
                     "minor_faults": 0, "major_faults": 0, "block_input_ops": 0, "block_output_ops": 0}]
        return records, bindings

    def score(self, records, bindings):
        return self.scorer.score(records, self.root / "metadata/manifest.json",
                                 self.root / "metadata/geometry.json", bindings)

    def test_modules_are_stdlib_source_without_execution_imports(self):
        forbidden = {"subprocess", "ctypes", "socket", "urllib", "requests", "torch", "numpy", "scipy"}
        for source in self.source_before.values():
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    self.assertTrue(forbidden.isdisjoint(n.name.split(".")[0] for n in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or "").split(".")[0], forbidden)

    def test_five_whole_feed_prefix_is_first_covering_44_rows(self):
        self.assertEqual(self.geometry.geometry(19200)["model_rows"], 39)
        self.assertEqual(self.geometry.geometry(24000)["model_rows"], 49)

    def test_complete_n1_integer_geometry(self):
        row = self.geometry.geometry(52578)
        self.assertEqual((row["feed_calls"], row["tail_samples"], row["fbank_rows"], row["model_rows"]),
                         (11, 4578, 327, 109))
        centers = [center for plan in row["callback_plan"] for center in plan["centers"]]
        self.assertEqual(centers, list(range(0, 327, 3)))
        self.assertEqual(row["callback_plan"][-1]["phase"], "finish")

    def test_geometry_exact_feed_has_no_synthetic_tail(self):
        row = self.geometry.geometry(4800)
        self.assertEqual((row["full_feeds"], row["tail_feeds"], row["callbacks"]), (1, 0, 1))
        self.assertEqual(row["callback_plan"][0]["phase"], "feed")

    def test_complete_invented_miss_stays_unqualified(self):
        result = self.score(*self.fixture())
        self.assertFalse(result["qualification"])
        self.assertEqual(result["aggregate"]["activation_events"], 0)
        counts = result["streams"][0]["whole_clip_counts"]
        self.assertEqual((counts["expected_K1_occurrences"], counts["missing_K1"], counts["event_total"]), (1, 1, 0))
        self.assertFalse(counts["temporal_matching_claim"])

    def test_pre_source_event_remains_pre_source_not_true_word_match(self):
        stream = self.score(*self.fixture({0: 1}))["streams"][0]
        self.assertEqual(stream["whole_clip_counts"]["K1_pre_source_availability_events"], 1)
        self.assertEqual(stream["events"][0]["availability_region"], "artificial_lead")
        self.assertEqual(stream["events"][0]["source_relative_available_samples"], -19200)
        self.assertFalse(stream["word_label_observation"]["acoustic_word_end_available"])

    def test_original_and_tail_availability_regions_remain_distinct(self):
        stream = self.score(*self.fixture({5: 1, 10: 1}))["streams"][0]
        self.assertEqual([row["availability_region"] for row in stream["events"]], ["original_PCM", "artificial_tail"])
        self.assertEqual(stream["whole_clip_counts"]["duplicate_K1_events"], 1)
        self.assertFalse(stream["whole_clip_counts"]["temporal_matching_claim"])

    def test_wrong_keyword_is_retained(self):
        counts = self.score(*self.fixture({5: 2}))["streams"][0]["whole_clip_counts"]
        self.assertEqual((counts["missing_K1"], counts["K2_events"], counts["other_keyword_events"]), (1, 1, 1))

    def test_malformed_invented_traces_fail_closed(self):
        def first(rows, kind):
            return next(row for row in rows if row["kind"] == kind)
        changes = {
            "missing_final": lambda rows: rows.pop(),
            "missing_model_load": lambda rows: rows.pop(1),
            "missing_callback": lambda rows: rows.remove(first(rows, "callback")),
            "missing_finish": lambda rows: rows.remove(first(rows, "finish")),
            "nonfinite_logit": lambda rows: first(rows, "callback")["logits"][0].__setitem__(0, float("nan")),
            "wrong_classes": lambda rows: first(rows, "callback")["logits"][0].pop(),
            "wrong_center": lambda rows: first(rows, "callback")["centers"].__setitem__(0, 1),
            "wrong_identity": lambda rows: first(rows, "clip_start").update(pcm_sha256="9" * 64),
            "missing_timing": lambda rows: first(rows, "callback").pop("callback_entry_wall_ns_since_run_start"),
            "wrong_clock": lambda rows: first(rows, "callback").update(decoder_total_frames=999),
            "short_feed": lambda rows: first(rows, "feed").update(input_samples=4799),
            "incomplete": lambda rows: first(rows, "clip_end").update(complete=False),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                rows, bindings = self.fixture()
                rows = copy.deepcopy(rows)
                change(rows)
                with self.assertRaises(self.scorer.ValidationError):
                    self.score(rows, bindings)

    def test_duplicate_json_and_nonfinite_numbers_are_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(self.scorer.ValidationError):
                self.scorer._parse_json(raw, "invented")

    def test_boolean_and_all_zero_identity_are_not_valid_numbers_or_hashes(self):
        with self.assertRaises(self.scorer.ValidationError):
            self.scorer._int(True, "invented")
        with self.assertRaises(self.scorer.ValidationError):
            self.scorer._hash("0" * 64, "invented")

    def test_truncated_raw_jsonl_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invented.jsonl"
            path.write_bytes(b'{"kind":"invented"}')
            with self.assertRaisesRegex(self.scorer.ValidationError, "final newline"):
                self.scorer._load_raw(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
