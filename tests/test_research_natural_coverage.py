#!/usr/bin/env python3
"""Synthetic coverage/selection counterexamples; verifier doubles are explicit.

The fixtures create tiny PCM parents and real natural input packages. Native
recognition replay belongs to the separately tested data verifier; no ASR,
trained checkpoint or real-label accuracy is claimed by these tests.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import research_natural_coverage as c


def parent(ident="session-a", count=16000 * 1000, split="development", family=None):
    return {"parent_id": ident, "audio_path": ident + ".wav",
        "wav_sha256": c.q.sha((ident + ":wav").encode()), "pcm_sha256": c.q.sha(ident.encode()),
        "sample_rate_hz": 16000, "sample_count": count, "dataset_id": "fixture",
        "source_partition": "validation", "session_id": ident,
        "source_family_id": family or ident, "split": split,
        "provenance": {"source_archive_sha256": None, "source_member": None, "source_wav_sha256": None,
                       "channel_index": None, "source_audit_sha256": None, "annotation_sha256": None}}


def declaration(p, spans):
    return {"schema": c.natural.DECLARATION, "status": "declared_not_materialized",
        "selection_reason": "Invented metadata fixture; no recognition outputs used",
        "parents": [p], "windows": [{"window_id": f"{p['parent_id']}-fixture-{i}",
            "parent_id": p["parent_id"], "start_sample": a, "end_sample": b} for i, (a, b) in enumerate(spans)]}


class PlanningTests(unittest.TestCase):
    def test_fixed_formula_retains_roles_and_exact_eight_parent_coverage(self):
        values = []
        for i in range(8):
            p = parent(f"session-{i}", count=16000 * (1000 + i))
            values.append(declaration(p, [(p["sample_count"] - 480000, p["sample_count"])]))
        before = copy.deepcopy(values)
        report = c.fixed_plan(values)
        self.assertEqual(values, before)
        self.assertEqual(len(report["declarations"]), 16)
        self.assertEqual(len(report["combined_declaration"]["windows"]), 64)
        self.assertEqual([report["coverage"][key] for key in
            ("window_input_seconds", "unique_parent_seconds", "overlap_seconds")], [1920, 1760, 160])
        for shard in report["declarations"]:
            self.assertEqual(len(shard["windows"]), 4)
            self.assertEqual(shard["parents"][0]["split"], "development")
        first = report["combined_declaration"]["windows"][:8]
        for index, (a, b) in enumerate(zip(first[::2], first[1::2]), 1):
            anchor = (values[0]["parents"][0]["sample_count"] - 880000) * index // 5
            self.assertEqual((a["start_sample"], a["end_sample"], b["start_sample"], b["end_sample"]),
                             (anchor, anchor + 480000, anchor + 400000, anchor + 880000))
        self.assertIs(report["selection_uses_model_outputs"], False)
        self.assertEqual(report["boundary"]["accepted_negative_seconds"], 0)
        self.assertEqual(report["boundary"]["new_model_calls"], 0)

    def test_parent_and_declaration_order_does_not_choose_new_samples(self):
        a, b = parent("a"), parent("b")
        da = declaration(a, [(a["sample_count"] - 10, a["sample_count"])])
        db = declaration(b, [(b["sample_count"] - 10, b["sample_count"])])
        self.assertEqual(c.fixed_plan([da, db]), c.fixed_plan([db, da]))

    def test_intersection_is_rejected_without_adaptive_relocation(self):
        p = parent()
        anchor = (p["sample_count"] - c.PAIR) // 5
        for span in ((anchor + 1, anchor + 2), (anchor + c.PAIR - 1, anchor + c.PAIR)):
            with self.subTest(span=span), self.assertRaisesRegex(ValueError, "INTERSECTS_PRIOR"):
                c.fixed_plan([declaration(p, [span])])

    def test_half_open_adjacent_prior_interval_is_not_overlap(self):
        p = parent()
        anchor = (p["sample_count"] - c.PAIR) // 5
        result = c.fixed_plan([declaration(p, [(anchor - 10, anchor)])])
        self.assertEqual(result["coverage"]["unique_parent_seconds"], 220)

    def test_short_parent_or_overlapping_fixed_groups_is_rejected(self):
        for count, reason in ((c.PAIR, "PARENT_TOO_SHORT"), (c.PAIR + 100, "GROUPS_OVERLAP")):
            p = parent(count=count)
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, reason):
                c.fixed_plan([declaration(p, [(count - 1, count)])])

    def test_cross_declaration_alias_channel_and_split_conflicts(self):
        a = parent()
        first = declaration(a, [(a["sample_count"] - 2, a["sample_count"])])
        cases = []
        alias = copy.deepcopy(a); alias.update(parent_id="alias", audio_path="alias.wav")
        cases.append((alias, "PARENT_ALIAS"))
        channel = parent("second"); channel.update(session_id=a["session_id"])
        channel["provenance"]["channel_index"] = 1
        cases.append((channel, "SECOND_CHANNEL"))
        split = parent("train-side", split="train", family=a["source_family_id"])
        cases.append((split, "SOURCE_FAMILY_SPLIT_CONFLICT"))
        for other, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                c.fixed_plan([first, declaration(other, [(other["sample_count"] - 2, other["sample_count"])])])

    def test_current_frozen_tail_roster_has_exact_64_nonoverlapping_new_windows(self):
        names = ("natural-alimeeting-eval-tail2-v1.json", "natural-alimeeting-eval-tail7-shard-a-v1.json",
                 "natural-alimeeting-eval-tail7-shard-b-v1.json")
        values = [c.q.loads((ROOT / "configs/research" / name).read_bytes()) for name in names]
        result = c.fixed_plan(values)
        self.assertEqual(len(result["declarations"]), 16)
        self.assertEqual(result["coverage"]["unique_parent_samples"], 1760 * 16000)
        self.assertEqual({p["split"] for p in result["combined_declaration"]["parents"]}, {"development"})
        self.assertEqual(sum(len(x["windows"]) for x in values), 9)

    def test_create_once_plan_and_external_declaration_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); p = parent()
            path = root / "prior.json"
            path.write_bytes(c.q.canonical(declaration(p, [(p["sample_count"] - 10, p["sample_count"])])))
            digest = c.q.sha(path.read_bytes()); output = root / "planned"
            with self.assertRaisesRegex(ValueError, "FILE_HASH_CHANGED"):
                c.write_plan([path], ["0" * 64], output)
            self.assertFalse(output.exists())
            report = c.write_plan([path], [digest], output)
            for name, identity in report["files"].items():
                self.assertEqual(c.q.file_record(output / name), identity)
            with self.assertRaisesRegex(ValueError, "NEW_PRIVATE_OUTPUT"):
                c.write_plan([path], [digest], output)


class CoverageTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.audio = self.root / "audio"; self.audio.mkdir()
        pcm = struct.pack("<64h", *list(range(-32, 32)))
        raw = c.natural.wav_bytes(pcm); (self.audio / "session-a.wav").write_bytes(raw)
        self.parent = parent(count=64)
        self.parent.update(wav_sha256=c.q.sha(raw), pcm_sha256=c.q.sha(pcm))
        self.first = self.make_bundle("first", self.parent, [(0, 8)])
        self.second = self.make_bundle("second", self.parent, [(4, 12)])

    def make_bundle(self, ident, source, spans):
        value = declaration(source, spans)
        path = self.root / (ident + ".json"); path.write_bytes(c.q.canonical(value))
        digest = c.q.sha(path.read_bytes()); bundle = self.root / ident
        c.natural.prepare(path, digest, self.audio, bundle)
        return {"bundle": str(bundle), "audio_root": str(self.audio), "declaration_sha256": digest}

    def request(self, prepared=None):
        return {"schema": c.REQUEST, "prepared": prepared or [self.first, self.second], "views": [], "verifier": None}

    def view_double(self, prepared, *, quorum=True, usable=True):
        batches, rows = [], []
        for index, entry in enumerate(prepared):
            bundle = Path(entry["bundle"])
            raw = (bundle / "natural-window-bindings.json").read_bytes(); binding = c.q.loads(raw)
            batch = f"batch-{index}"
            batches.append({"batch_id": batch, "status": "complete", "bindings_sha256": c.q.sha(raw),
                "declaration_sha256": entry["declaration_sha256"],
                "input_manifest_sha256": binding["inputs_manifest_sha256"],
                "asr_request_sha256": "a" * 64, "final_audit_sha256": "b" * 64,
                "inventory_sha256": "c" * 64, "requested_native_results": 3 * len(binding["rows"])})
            rows.extend({**row, "batch_id": batch, "sample_rate_hz": 16000,
                "observations": [{"model": model, "status": "complete", "requested": True, "usable": usable}
                                 for model in ("qwen06", "sensevoice", "whisper")],
                "decision": {"full_text_quorum": quorum, "ctc_mask": quorum}}
                for row in binding["rows"])
        return {"schema": c.VIEW, "batches": batches, "rows": rows,
                "boundary": {"accepted_negative_seconds": 0, "human_gold": False}}

    def run_with_verifier_double(self, request, reports):
        """Explicitly invent only data-verifier output, never native evidence."""
        request["views"] = [{"path": str(self.root / f"view-{i}"), "manifest_sha256": c.q.sha(str(i).encode())}
                            for i in range(len(reports))]
        request["verifier"] = {"explicit_test_double": True}
        calls = []
        def verify(path, *, expected_manifest_sha256):
            index = len(calls)
            self.assertEqual(expected_manifest_sha256, request["views"][index]["manifest_sha256"])
            calls.append(path)
            return reports[index]
        verifier = type("VerifierDouble", (), {"verify_view": staticmethod(verify)})()
        with mock.patch.object(c, "load_verifier", return_value=verifier), \
             mock.patch.object(c, "source_checks"), \
             mock.patch.object(c.natural.selective, "seal_or_verify", side_effect=AssertionError("seal writer called")):
            result = c.summarize(request)
        self.assertEqual(len(calls), len(reports))
        return result

    def test_prepared_union_does_not_claim_observation_or_negative_exposure(self):
        before = {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        result = c.summarize(self.request())
        self.assertEqual((result["prepared"]["window_input_samples"], result["prepared"]["unique_parent_samples"],
                          result["prepared"]["overlap_samples"]), (16, 12, 4))
        self.assertEqual(result["observed"]["unique_parent_samples"], 0)
        self.assertEqual(len(result["prepared_without_complete_observation"]), 2)
        self.assertEqual(result["boundary"]["accepted_negative_seconds"], 0)
        self.assertIs(result["boundary"]["natural_far_available"], False)
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns)
                                 for p in self.root.rglob("*") if p.is_file()})

    def test_unusable_but_complete_native_returns_count_only_observed_coverage(self):
        request = self.request()
        report = self.view_double(request["prepared"], quorum=False, usable=False)
        result = self.run_with_verifier_double(request, [report])
        self.assertEqual(result["observed"]["unique_parent_samples"], 12)
        self.assertEqual(result["weak_fulltext_quorum"]["unique_parent_samples"], 0)
        self.assertEqual(result["prepared_without_complete_observation"], [])

    def test_quorum_has_its_own_union_and_is_never_a_far_denominator(self):
        request = self.request(); report = self.view_double(request["prepared"])
        report["rows"][1]["decision"] = {"full_text_quorum": False, "ctc_mask": False}
        result = self.run_with_verifier_double(request, [report])
        self.assertEqual(result["observed"]["unique_parent_samples"], 12)
        self.assertEqual(result["weak_fulltext_quorum"]["unique_parent_samples"], 8)
        self.assertEqual(result["boundary"]["accepted_negative_seconds"], 0)

    def test_claim_or_incomplete_or_unrequested_cells_cannot_be_observed(self):
        changes = [lambda r: r["batches"][0].update(status="claimed"),
                   lambda r: r["rows"][0]["observations"][0].update(status="incomplete"),
                   lambda r: r["rows"][0]["observations"][0].update(requested=False),
                   lambda r: r["rows"][0]["observations"][0].update(model="qwen"),
                   lambda r: r["rows"][0]["observations"].pop(),
                   lambda r: r["batches"][0].update(final_audit_sha256=None)]
        for change in changes:
            request = self.request(); report = self.view_double(request["prepared"]); change(report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_with_verifier_double(request, [report])

    def test_alias_split_bounds_pcm_and_opaque_clip_rebinding_are_rejected(self):
        for field, value in (("split", "train"), ("session_id", "alias"), ("source_family_id", "new-family"),
                             ("start_sample", 1), ("sample_rate_hz", 16000.0),
                             ("parent_pcm_sha256", "0" * 64), ("clip_id", "clip-999")):
            request = self.request(); report = self.view_double(request["prepared"])
            report["rows"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ROW_SOURCE_BINDING"):
                self.run_with_verifier_double(request, [report])

    def test_partial_batch_cannot_claim_complete_observed_coverage(self):
        entry = self.make_bundle("two-rows", self.parent, [(16, 24), (20, 28)])
        request = self.request([entry]); report = self.view_double([entry]); report["rows"].pop()
        with self.assertRaisesRegex(ValueError, "BATCH_ROSTER_OR_COUNT"):
            self.run_with_verifier_double(request, [report])

    def test_duplicate_prepared_and_observed_sources_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PREPARED_PACKAGE"):
            c.summarize(self.request([self.first, self.first]))
        alias = self.make_bundle("duplicate-window", self.parent, [(0, 8)])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_PREPARED_WINDOW|DUPLICATE_PREPARED_PACKAGE"):
            c.summarize(self.request([self.first, alias]))
        request = self.request(); report = self.view_double(request["prepared"])
        with self.assertRaisesRegex(ValueError, "DUPLICATE_OBSERVED_WINDOW"):
            self.run_with_verifier_double(request, [report, copy.deepcopy(report)])

    def test_changed_parent_outside_window_is_still_rejected(self):
        pcm = list(range(-32, 32)); pcm[-1] += 1
        (self.audio / "session-a.wav").write_bytes(c.natural.wav_bytes(struct.pack("<64h", *pcm)))
        with self.assertRaisesRegex(ValueError, "PARENT_WAV_HASH_CHANGED"):
            c.summarize(self.request())

    def test_cross_bundle_second_channel_and_family_split_are_rejected(self):
        for split, session, reason in (("train", "session-b", "SOURCE_FAMILY_SPLIT_CONFLICT"),
                                       ("development", "session-a", "SECOND_CHANNEL")):
            p = parent("session-b", count=64, split=split, family=self.parent["source_family_id"])
            pcm = struct.pack("<64h", *list(reversed(range(-32, 32))))
            raw = c.natural.wav_bytes(pcm); (self.audio / p["audio_path"]).write_bytes(raw)
            p.update(wav_sha256=c.q.sha(raw), pcm_sha256=c.q.sha(pcm), session_id=session)
            entry = self.make_bundle("other-" + split, p, [(0, 8)])
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                c.summarize(self.request([self.first, entry]))

    def test_source_pin_is_checked_before_verifier_code_is_loaded(self):
        folder = self.root / "fake-data/automation"; folder.mkdir(parents=True)
        path = folder / Path(c.VERIFIER).name
        path.write_text("raise RuntimeError('must not execute an unpinned verifier')\n")
        with self.assertRaisesRegex(ValueError, "VERIFIER_SOURCE_CHANGED"):
            c.load_verifier({"path": str(path), "source_files_sha256": {c.VERIFIER: "0" * 64}})

    def test_cached_foreign_automation_namespace_or_module_is_rejected(self):
        folder = self.root / "fake-data/automation"; folder.mkdir(parents=True)
        path = folder / Path(c.VERIFIER).name
        path.write_text("raise RuntimeError('mixed namespace must be rejected first')\n")
        source = {"path": str(path), "source_files_sha256": {c.VERIFIER: c.q.sha(path.read_bytes())}}
        namespace = types.ModuleType("automation")
        namespace.__path__ = [str(self.root / "other-data/automation")]
        module = types.ModuleType("automation.active_asr_weak_view")
        module.__file__ = str(self.root / "other-data/automation/active_asr_weak_view.py")
        for name, cached in (("automation", namespace), ("automation.active_asr_weak_view", module)):
            with self.subTest(name=name), mock.patch.dict(sys.modules, {name: cached}), \
                    self.assertRaisesRegex(ValueError, "MIXED_DATA_MODULES"):
                c.load_verifier(source)


if __name__ == "__main__":
    unittest.main()
