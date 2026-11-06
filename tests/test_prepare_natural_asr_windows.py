#!/usr/bin/env python3
"""Synthetic slicing/identity counterexamples; no real audio, network or models."""
from __future__ import annotations

import copy
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import prepare_natural_asr_windows as n


def waveform(values, rate=16000, channels=1):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(channels); writer.setsampwidth(2); writer.setframerate(rate)
        writer.writeframes(struct.pack("<" + "h" * len(values), *values))
    return stream.getvalue()


class NaturalWindowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.audio = self.root / "audio"; self.audio.mkdir()
        self.values = [-32768, -1000, -1, 0, 1, 32767, 2, -2,
                       100, -100, 256, -256, 4096, -4096, 9, -9] * 4
        self.parent = self.make_parent("session-a", self.values)
        self.value = {"schema": n.DECLARATION, "status": "declared_not_materialized",
            "selection_reason": "Fixed synthetic boundary fixture; no acoustic selection",
            "parents": [self.parent], "windows": self.windows("session-a")}
        self.plan = self.root / "declaration.json"
        self.output = self.root / "bundle"
        self.persist()

    def make_parent(self, ident, values):
        raw = waveform(values)
        (self.audio / (ident + ".wav")).write_bytes(raw)
        return {"parent_id": ident, "audio_path": ident + ".wav", "wav_sha256": n.q.sha(raw),
            "pcm_sha256": n.q.sha(struct.pack("<" + "h" * len(values), *values)),
            "sample_rate_hz": 16000, "sample_count": len(values), "dataset_id": "fixture",
            "source_partition": "validation", "session_id": ident, "source_family_id": "family-a",
            "split": "development", "provenance": {"source_archive_sha256": None,
                "source_member": None, "source_wav_sha256": None, "channel_index": None,
                "source_audit_sha256": None, "annotation_sha256": None}}

    @staticmethod
    def windows(ident):
        return [{"window_id": ident + "-left", "parent_id": ident, "start_sample": 0, "end_sample": 8},
                {"window_id": ident + "-right", "parent_id": ident, "start_sample": 4, "end_sample": 12}]

    def persist(self):
        self.plan.write_bytes(n.q.canonical(self.value))
        self.plan_sha = n.q.sha(self.plan.read_bytes())

    def prepare(self):
        self.persist()
        return n.prepare(self.plan, self.plan_sha, self.audio, self.output)

    def test_exact_half_open_slices_pcm_scale_and_existing_asr_contract(self):
        before = (self.audio / self.parent["audio_path"]).read_bytes()
        result = self.prepare()
        inputs, mapping = n.selective.validate_map(self.output / "inputs.json", self.output / "source-map.json")
        self.assertEqual([x["id"] for x in inputs["clips"]], ["clip-000", "clip-001"])
        self.assertNotIn("session-a", json.dumps(inputs))
        self.assertNotIn("family-a", json.dumps(inputs))
        self.assertNotIn("development", json.dumps(inputs))
        with zipfile.ZipFile(self.output / "window-audio.zip") as archive:
            self.assertEqual(archive.namelist(), ["original/clip-000.wav", "original/clip-001.wav"])
            for clip, window in zip(inputs["clips"], self.value["windows"]):
                expected = self.values[window["start_sample"]:window["end_sample"]]
                raw = (self.output / clip["original_wav"]["path"]).read_bytes()
                with wave.open(io.BytesIO(raw), "rb") as reader:
                    self.assertEqual(reader.readframes(100), struct.pack("<8h", *expected))
                floating = (self.output / clip["derived_pcm"]["path"]).read_bytes()
                self.assertEqual(struct.unpack("<8f", floating), tuple(x / 32768 for x in expected))
                self.assertEqual(archive.read(clip["original_member"]), raw)
                self.assertEqual(clip["original_pcm"]["dtype"], "<i2")
        self.assertTrue(all(row["quality_uncertain"] for row in mapping["rows"]))
        agreed_fixture = {model: {"sha256": "a" * 64, "result": {
            "text": "合成测试文本", "decoder_terminated": True, "quality_flags": []}}
            for model in n.selective.PRIMARY}
        for row in mapping["rows"]:
            route = n.selective.route_item(row, mapping["plan_sha256"], agreed_fixture)
            self.assertEqual((route["whisper_state"], route["reasons"]),
                             ("requested", ["source_quality_uncertain"]))
        self.assertEqual(result["coverage"]["window_input_samples"], 16)
        self.assertEqual(result["coverage"]["unique_parent_samples"], 12)
        self.assertEqual(result["coverage"]["overlap_samples"], 4)
        self.assertEqual(result["boundary"]["target_presence"], "unknown")
        self.assertIs(result["boundary"]["duration_is_negative_exposure"], False)
        self.assertEqual(result["boundary"]["new_model_calls"], 0)
        self.assertEqual((self.audio / self.parent["audio_path"]).read_bytes(), before)

    def test_nested_unsorted_and_adjacent_windows_use_interval_union(self):
        spans = [(10, 15), (0, 5), (3, 12), (20, 22), (21, 24), (12, 13)]
        self.value["windows"] = [{"window_id": f"window-{i}", "parent_id": "session-a",
                                  "start_sample": a, "end_sample": b} for i, (a, b) in enumerate(spans)]
        n.validate_declaration(self.value)
        result = n.coverage(self.value)
        self.assertEqual(result["by_parent"][0]["union_sample_ranges"], [[0, 15], [20, 24]])
        self.assertEqual((result["window_input_samples"], result["unique_parent_samples"], result["overlap_samples"]),
                         (25, 19, 6))

    def test_distinct_parent_timelines_are_not_merged(self):
        self.value["parents"].append(self.make_parent("session-b", list(reversed(self.values))))
        self.value["windows"].extend(self.windows("session-b"))
        result = self.prepare()["coverage"]
        self.assertEqual((result["window_input_samples"], result["unique_parent_samples"], result["overlap_samples"]),
                         (32, 24, 8))
        self.assertEqual(len(result["by_parent"]), 2)

    def test_full_parent_change_outside_selected_windows_is_rejected(self):
        changed = self.values.copy(); changed[-1] += 1
        (self.audio / self.parent["audio_path"]).write_bytes(waveform(changed))
        with self.assertRaisesRegex(ValueError, "PARENT_WAV_HASH_CHANGED"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_wrong_parent_wav_pcm_count_or_rate_is_rejected_before_output(self):
        for key, value in (("wav_sha256", "0" * 64), ("pcm_sha256", "0" * 64),
                           ("sample_count", len(self.values) + 1), ("sample_rate_hz", 24000)):
            with self.subTest(key=key):
                prior = self.parent[key]; self.parent[key] = value
                with self.assertRaises(ValueError):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.parent[key] = prior

    def test_integer_nonempty_in_range_and_bounded_windows_required(self):
        for a, b in ((-1, 4), (0, 65), (4, 4), (8, 4), (True, 8), (0, False), (0.0, 8), (0, 8.0)):
            with self.subTest(bounds=(a, b)):
                changed = copy.deepcopy(self.value)
                changed["windows"][0].update(start_sample=a, end_sample=b)
                with self.assertRaisesRegex(ValueError, "INTEGER_HALF_OPEN_BOUNDS"):
                    n.validate_declaration(changed)
        self.value["parents"][0]["sample_count"] = n.MAX_WINDOW_SAMPLES + 1
        self.value["windows"][0]["end_sample"] = n.MAX_WINDOW_SAMPLES + 1
        with self.assertRaisesRegex(ValueError, "INTEGER_HALF_OPEN_BOUNDS"):
            n.validate_declaration(self.value)

    def test_duplicate_window_range_and_unknown_parent_are_rejected(self):
        changed = copy.deepcopy(self.value)
        changed["windows"][1].update(start_sample=0, end_sample=8)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_WINDOW_RANGE"):
            n.validate_declaration(changed)
        self.value["windows"][0]["parent_id"] = "missing"
        with self.assertRaisesRegex(ValueError, "WINDOW_IDENTITY"):
            n.validate_declaration(self.value)

    def test_same_family_cannot_cross_splits_even_across_datasets(self):
        parent = self.make_parent("session-b", list(reversed(self.values)))
        parent.update(split="train", dataset_id="second-dataset")
        self.value["parents"].append(parent); self.value["windows"].extend(self.windows("session-b"))
        with self.assertRaisesRegex(ValueError, "SOURCE_FAMILY_SPLIT_OVERLAP"):
            n.validate_declaration(self.value)

    def test_parent_pcm_alias_and_second_channel_of_same_session_are_rejected(self):
        parent = self.make_parent("session-b", list(reversed(self.values)))
        self.value["parents"].append(parent); self.value["windows"].extend(self.windows("session-b"))
        for key, value in (("pcm_sha256", self.parent["pcm_sha256"]), ("session_id", "session-a"),
                           ("audio_path", self.parent["audio_path"])):
            with self.subTest(key=key):
                prior = parent[key]; parent[key] = value
                with self.assertRaisesRegex(ValueError, "DUPLICATE_PARENT_PCM_PATH_OR_SESSION"):
                    n.validate_declaration(self.value)
                parent[key] = prior

    def test_test_partition_and_label_fields_are_rejected(self):
        for target, key, value in (("parent", "split", "test"), ("parent", "source_partition", "Test"),
                                   ("parent", "expected", "negative"), ("window", "confirmed", True),
                                   ("top", "accepted_negative_duration_s", 1)):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.value)
                row = changed["parents"][0] if target == "parent" else changed["windows"][0] if target == "window" else changed
                row[key] = value
                with self.assertRaises(ValueError):
                    n.validate_declaration(changed)

    def test_missing_parent_leaves_declaration_unmaterialized(self):
        (self.audio / self.parent["audio_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(n.read_declaration(self.plan)[1]["status"], "declared_not_materialized")

    def test_path_traversal_symlinks_and_hardlinks_are_rejected(self):
        for path in ("../session-a.wav", "/session-a.wav", "./session-a.wav", "dir//sample.wav", "dir\\sample.wav"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                n.relative(path)
        original = self.audio / self.parent["audio_path"]
        moved = self.audio / "actual.wav"; original.rename(moved); original.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "PATH_ALIAS"):
            self.prepare()
        original.unlink(); os.link(moved, original)
        with self.assertRaisesRegex(ValueError, "BOUNDED_REGULAR_FILE_REQUIRED"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_create_only_and_read_only_verification(self):
        first = self.prepare()
        before = {p.relative_to(self.output).as_posix(): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(n.verify(self.output, self.audio, self.plan_sha), first)
        with self.assertRaisesRegex(ValueError, "CREATE_ONLY_OUTPUT_REQUIRED"):
            self.prepare()
        after = {p.relative_to(self.output).as_posix(): p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_tampered_audio_bindings_counts_archive_and_producer_are_rejected(self):
        self.prepare()
        for name in ("original/clip-000.wav", "pcm16k/clip-000.f32", "source-map.json",
                     "natural-window-bindings.json", "summary.json", "window-audio.zip", "preparation-source.py"):
            with self.subTest(name=name):
                path = self.output / name; original = path.read_bytes()
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaisesRegex(ValueError, "BUNDLE_FILE_CHANGED"):
                    n.verify(self.output, self.audio, self.plan_sha)
                path.write_bytes(original)
        (self.output / "extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "BUNDLE_FILE_ROSTER_CHANGED"):
            n.verify(self.output, self.audio, self.plan_sha)

    def test_declaration_hash_and_duplicate_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "DECLARATION_HASH_CHANGED"):
            n.prepare(self.plan, "0" * 64, self.audio, self.output)
        raw = self.plan.read_bytes().replace(b'"status":', b'"status":"declared_not_materialized","status":', 1)
        self.plan.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "DUPLICATE_JSON_KEY"):
            n.prepare(self.plan, n.q.sha(raw), self.audio, self.output)
        self.assertFalse(self.output.exists())

    def test_window_source_identity_survives_clip_order_and_alias_changes(self):
        self.prepare()
        first = json.loads((self.output / "source-map.json").read_text())["rows"]
        self.value["windows"].reverse()
        self.value["windows"][0]["window_id"] = "renamed-window"
        self.output = self.root / "bundle-reordered"
        self.prepare()
        second = json.loads((self.output / "source-map.json").read_text())["rows"]
        self.assertEqual([x["source_id"] for x in first], [x["source_id"] for x in reversed(second)])

    def test_malformed_truncated_or_wrong_pcm_formats_fail(self):
        good = waveform(self.values)
        for raw in (good[:-1], good + b"\0", waveform(self.values, rate=24000),
                    waveform(self.values, channels=2), waveform([])):
            with self.subTest(length=len(raw)), self.assertRaises(ValueError):
                n.parent_pcm(raw)

    def test_repository_template_matches_retained_original_source_receipt(self):
        _raw, template = n.read_declaration(ROOT / "configs/research/natural-alimeeting-eval-tail2-v1.json")
        audit_path = ROOT / "research/alimeeting-continuous-intake-v1/audit/audit.json"
        audit = json.loads(audit_path.read_text())
        parent = template["parents"][0]
        row = next(row for row in audit["rows"] if row["session_id"] == parent["session_id"])
        self.assertEqual(parent["wav_sha256"], row["wav_sha256"])
        self.assertEqual(parent["pcm_sha256"], row["pcm_sha256"])
        self.assertEqual(parent["sample_count"], row["sample_count"])
        self.assertEqual(parent["provenance"]["source_audit_sha256"], n.q.sha(audit_path.read_bytes()))
        self.assertEqual(parent["provenance"]["annotation_sha256"], row["textgrid"]["sha256"])
        self.assertEqual(template["windows"][-1]["end_sample"], row["sample_count"])
        counts = n.coverage(template)
        self.assertEqual((counts["window_input_samples"], counts["unique_parent_samples"], counts["overlap_samples"]),
                         (926352, 846352, 80000))
        self.assertEqual((counts["window_input_seconds"], counts["unique_parent_seconds"]), (57.897, 52.897))
        self.assertEqual(template["status"], "declared_not_materialized")


if __name__ == "__main__":
    unittest.main()
