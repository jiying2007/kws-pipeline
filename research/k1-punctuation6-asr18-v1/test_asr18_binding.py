"""Offline synthetic WAV and public-receipt checks, never real screening approval."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import asr18_metadata as m
import input_binding as b
import raw_data_reader as r
from test_asr18_raw_data_reader import synthetic_authority, synthetic_complete_archive, FakeOpener


def riff(chunks):
    body = b"WAVE" + b"".join(tag + struct.pack("<I", len(raw)) + raw + (b"\0" if len(raw) % 2 else b"") for tag, raw in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def wav(index, *, native=False, silent=False):
    if native:
        fmt = struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)
        samples = struct.pack("<f", 0 if silent else (index + 1) / 64) * 3840
    else:
        fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
        samples = struct.pack("<h", 0 if silent else index + 1) * 2560
    return riff([(b"fmt ", fmt), (b"data", samples)])


def members_fixture(*, silent_index=None):
    members, cells = {}, []
    for index, cell_id in enumerate(m.CELLS):
        silent = index == silent_index
        raw, pcm = wav(index, native=True), wav(index, silent=silent)
        members[f"raw/{cell_id}.wav"], members[f"derived/{cell_id}.wav"] = raw, pcm
        native, derived = b._wave(raw, native=True), b._wave(pcm, native=False)
        observation = {**native, **{k: derived[k] for k in ("wav_sha256", "pcm_sha256", "frames")},
            "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2, "derivative_peak": .125,
            "quality_flags": ["derived_pcm_silent"] if silent else []}
        eos = {"capture_calls": 1, "iterations": 3, "sequence_length": 3, "last_token_ids": [2150],
            "eos": 2150, "ended_with_eos": True, "hit_token_cap_without_eos": False, "earlier_eos_seen": False}
        cells.append({"cell_id": cell_id, "status": "complete", "attempts": 1, "observation": observation, "eos": eos})
    members["outcomes.json"] = m.canonical({"status": "completed_six_once", "consumed_attempts": 6, "cells": cells})
    return members


def history():
    return [{"wav_sha256": r.sha(b"private-old-wav"), "pcm_sha256": r.sha(b"private-old-pcm")}]


def setup_binding(*, rows=None, old=None):
    authority = synthetic_authority()[0]
    rows = rows or b.measure_complete_rows(members_fixture())
    policy = r.sha(b"synthetic-reviewed-policy")
    authority_hash = m.digest(authority)
    raw = b.make_screening_receipt(rows, history() if old is None else old, authority,
        expected_authority_sha256=authority_hash, screening_policy_sha256=policy)
    kwargs = {"expected_receipt_sha256": r.sha(raw), "actual_authority": authority, "measured_rows": rows,
        "expected_policy_sha256": policy, "expected_authority_sha256": authority_hash}
    return raw, kwargs


class BindingTests(unittest.TestCase):
    def test_six_measured_rows_exact_native_pcm_frame_identity(self):
        rows = b.measure_complete_rows(members_fixture())
        self.assertEqual([row["cell_id"] for row in rows], list(m.CELLS))
        self.assertTrue(all(row["source_frames"] == 3840 and row["frames"] == 2560 and row["generation_iterations"] == 3 for row in rows))
        self.assertEqual(len({row["wav_sha256"] for row in rows}), 6)

    def test_incomplete_or_retried_tts_cannot_cross_barrier(self):
        for change in (lambda o: o.update(status="stopped_no_retry"), lambda o: o["cells"].pop(),
                       lambda o: o["cells"][1].update(status="failed_no_retry"), lambda o: o["cells"][1].update(attempts=2),
                       lambda o: o["cells"][1].update(attempts=True)):
            members = members_fixture(); outcomes = json.loads(members["outcomes.json"]); change(outcomes)
            members["outcomes.json"] = m.canonical(outcomes)
            with self.assertRaises(ValueError): b.measure_complete_rows(members)

    def test_nonnatural_eos_codec_and_iteration_one_rejected(self):
        for change in (lambda e: e.update(earlier_eos_seen=True), lambda e: e.update(hit_token_cap_without_eos=True),
                       lambda e: e.update(iterations=1, sequence_length=1), lambda e: e.update(iterations=4, sequence_length=4),
                       lambda e: e.update(ended_with_eos=False), lambda e: e.update(capture_calls=True)):
            members = members_fixture(); outcomes = json.loads(members["outcomes.json"]); change(outcomes["cells"][0]["eos"])
            members["outcomes.json"] = m.canonical(outcomes)
            with self.assertRaises(ValueError): b.measure_complete_rows(members)

    def test_wav_pcm_and_quality_tampering_rejected(self):
        for key, value in (("wav_sha256", "a" * 64), ("pcm_sha256", "b" * 64), ("frames", 20), ("raw_float_sha256", "d" * 64),
                           ("quality_flags", ["silent"]), ("sample_rate_hz", True)):
            members = members_fixture(); outcomes = json.loads(members["outcomes.json"])
            outcomes["cells"][0]["observation"][key] = value; members["outcomes.json"] = m.canonical(outcomes)
            with self.assertRaises(ValueError): b.measure_complete_rows(members)

    def test_pcm_silence_recomputed_not_trusted_from_receipt(self):
        members = members_fixture(silent_index=0)
        self.assertEqual(b.measure_complete_rows(members)[0]["quality_flags"], ["derived_pcm_silent"])
        outcomes = json.loads(members["outcomes.json"]); outcomes["cells"][0]["observation"]["quality_flags"] = []
        members["outcomes.json"] = m.canonical(outcomes)
        with self.assertRaises(ValueError): b.measure_complete_rows(members)

    def test_wave_unknown_chunks_trailer_nan_or_bad_geometry(self):
        samples = struct.pack("<f", float("nan")) * 3840
        bad = riff([(b"fmt ", struct.pack("<HHIIHH", 3, 1, 24000, 96000, 4, 32)), (b"data", samples)])
        for raw, native in ((bad, True), (wav(0) + b"x", False), (wav(0)[:-1], False), (wav(0), True)):
            with self.assertRaises(ValueError): b._wave(raw, native=native)

    def test_receipt_public_allowlist_never_exports_history(self):
        raw, kwargs = setup_binding()
        body = raw.decode()
        for forbidden in (history()[0]["wav_sha256"], history()[0]["pcm_sha256"], m.digest(history()),
                          "historical", "inventory", "match", "coverage", "private", "voice", "seed", "text"):
            self.assertNotIn(forbidden, body)
        out = b.verify_screening_receipt(raw, **kwargs)
        self.assertFalse(out["execution_enabled"])
        self.assertEqual(len(m.initial_ledger(out["projection"])), 18)

    def test_shape_valid_receipt_without_expected_hash_never_approves_history(self):
        raw, kwargs = setup_binding()
        for value in (None, "a" * 64):
            with self.assertRaises(ValueError): b.verify_screening_receipt(raw, **{**kwargs, "expected_receipt_sha256": value})

    def test_receipt_omission_extra_private_key_or_changed_disposition_rejected(self):
        raw, kwargs = setup_binding()
        for change in (lambda v: v["cells"].pop(), lambda v: v.update(private_history="forbidden"),
                       lambda v: v["cells"][0].update(disposition="not_run_duplicate"),
                       lambda v: v["cells"][0].update(private_path="/private")):
            receipt = json.loads(raw); change(receipt)
            with self.assertRaises(ValueError): b.verify_screening_receipt(m.canonical(receipt), **kwargs)

    def test_changed_new_identity_source_or_policy_rejected(self):
        raw, kwargs = setup_binding()
        altered = deepcopy(kwargs); altered["measured_rows"][0]["pcm_sha256"] = "e" * 64
        with self.assertRaises(ValueError): b.verify_screening_receipt(raw, **altered)
        for key in ("expected_policy_sha256", "expected_authority_sha256"):
            with self.assertRaises(ValueError): b.verify_screening_receipt(raw, **{**kwargs, key: "a" * 64})
        altered = deepcopy(kwargs); altered["actual_authority"]["input"]["run_id"] += 1
        altered["expected_authority_sha256"] = m.digest(altered["actual_authority"])
        with self.assertRaises(ValueError): b.verify_screening_receipt(raw, **altered)

    def test_historical_duplicate_cannot_be_promoted_by_blind_shape(self):
        rows = b.measure_complete_rows(members_fixture())
        old = history() + [{"wav_sha256": rows[0]["wav_sha256"], "pcm_sha256": r.sha(b"different-old-pcm")}]
        raw, kwargs = setup_binding(rows=rows, old=old)
        verified = b.verify_screening_receipt(raw, **kwargs)
        self.assertEqual(verified["projection"]["cells"][0]["disposition"], "not_run_duplicate")
        receipt = json.loads(raw); receipt["cells"][0]["disposition"] = "eligible"
        changed = m.canonical(receipt)
        # Structurally eligible can never replace the separately frozen hash.
        with self.assertRaises(ValueError): b.verify_screening_receipt(changed, **kwargs)

    def test_symmetric_mixed_transitive_duplicates_and_quality_precedence(self):
        rows = b.measure_complete_rows(members_fixture(silent_index=0))
        rows[1]["wav_sha256"] = rows[0]["wav_sha256"]
        rows[2]["pcm_sha256"] = rows[1]["pcm_sha256"]
        raw, kwargs = setup_binding(rows=rows)
        verified = b.verify_screening_receipt(raw, **kwargs)
        self.assertEqual([row["disposition"] for row in verified["projection"]["cells"][:3]], ["not_run_duplicate"] * 3)
        self.assertEqual(verified["blind"]["eligible_cells"], 3)
        ledger = m.initial_ledger(verified["projection"])
        self.assertEqual(len(ledger), 18)
        self.assertEqual(sum(row["status"] == "not_run_duplicate" for row in ledger), 9)

    def test_all_excluded_blocks_model_setup_and_keeps_six_receipt_rows(self):
        rows = b.measure_complete_rows(members_fixture())
        old = [{"wav_sha256": row["wav_sha256"], "pcm_sha256": row["pcm_sha256"]} for row in rows]
        raw, kwargs = setup_binding(rows=rows, old=old)
        self.assertEqual(len(json.loads(raw)["cells"]), 6)
        with self.assertRaisesRegex(ValueError, "zero eligible"): b.verify_screening_receipt(raw, **kwargs)

    def test_opaque_binding_and_order_are_only_new_audio_hashes(self):
        raw, kwargs = setup_binding()
        verified = b.verify_screening_receipt(raw, **kwargs)
        worker = verified["worker_input_binding"]
        self.assertEqual([row["audio_id"] for row in worker["audio"]], verified["blind"]["execution_plan"]["audio_order"])
        self.assertEqual(worker["job_sha256"], m.digest(verified["blind"]["job"]))
        for forbidden in ("k1p6-", '"cell_id"', "private", "voice", "seed", "historical", '"text"'):
            self.assertNotIn(forbidden, m.canonical(worker).decode())
        self.assertEqual([row["wav_sha256"] for row in verified["blind"]["job"]["clips"]], sorted(row["wav_sha256"] for row in kwargs["measured_rows"]))

    def test_input_files_only_job_and_exact_audio_no_receipts_or_map(self):
        members = members_fixture()
        raw, kwargs = setup_binding(rows=b.measure_complete_rows(members))
        verified = b.verify_screening_receipt(raw, **kwargs)
        audio = {row["wav_sha256"]: members[f'derived/{row["cell_id"]}.wav'] for row in kwargs["measured_rows"]}
        material = b.worker_input_files(verified, audio)
        self.assertEqual(len(material["input_files"]), 7)
        self.assertTrue(all(name == "job.json" or name.startswith("audio/") for name in material["input_files"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            staged = b.stage_worker_inputs(root, verified, audio)
            self.assertEqual({p.name for p in root.iterdir()}, {"job.json", "audio"})
            self.assertEqual(len(list((root / "audio").iterdir())), 6)
            self.assertNotIn("opaque", {p.name for p in root.iterdir()})
            self.assertFalse(staged["execution_enabled"])
            with self.assertRaises(FileExistsError): b.stage_worker_inputs(root, verified, audio)

    def test_extra_or_rebound_audio_is_rejected(self):
        members = members_fixture(); raw, kwargs = setup_binding(rows=b.measure_complete_rows(members))
        verified = b.verify_screening_receipt(raw, **kwargs)
        audio = {row["wav_sha256"]: members[f'derived/{row["cell_id"]}.wav'] for row in kwargs["measured_rows"]}
        altered = dict(audio); altered[next(iter(altered))] = wav(5)
        with self.assertRaises(ValueError): b.worker_input_files(verified, altered)
        altered = dict(audio); altered["a" * 64] = b"private-extra"
        with self.assertRaises(ValueError): b.worker_input_files(verified, altered)

    def test_mutated_opaque_binding_cannot_leak_or_rebind_input(self):
        members = members_fixture(); raw, kwargs = setup_binding(rows=b.measure_complete_rows(members))
        verified = b.verify_screening_receipt(raw, **kwargs)
        audio = {row["wav_sha256"]: members[f'derived/{row["cell_id"]}.wav'] for row in kwargs["measured_rows"]}
        for change in (lambda value: value["worker_input_binding"].update(private_history="forbidden"),
                       lambda value: value["worker_input_binding"]["audio"][0].update(audio_frames=1),
                       lambda value: value["worker_input_binding"].update(screening_receipt_sha256="e" * 64)):
            changed = deepcopy(verified); change(changed)
            with self.assertRaises(ValueError): b.worker_input_files(changed, audio)


    def test_full_offline_zip_reader_receipt_and_worker_projection(self):
        payload, authority = synthetic_complete_archive()
        authority, manifest, chunks = synthetic_authority(payload)
        import io, zipfile
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            authority["input"]["member_manifest_sha256"] = r.sha(archive.read(r.MEMBER_MANIFEST))
        authority_hash = m.digest(authority)
        acquired = r.RawDataReader(_opener=FakeOpener([manifest, *chunks])).read_complete(authority,
            expected_authority_sha256=authority_hash)
        members = r.verify_complete_package(acquired["archive"], authority,
            expected_authority_sha256=authority_hash,
            expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"])
        rows = b.measure_complete_rows(members)
        policy = r.sha(b"independently-reviewed-synthetic-policy")
        receipt = b.make_screening_receipt(rows, history(), authority,
            expected_authority_sha256=authority_hash, screening_policy_sha256=policy)
        bound = b.bind_complete_input_package(acquired["archive"], authority, receipt,
            expected_authority_sha256=authority_hash,
            expected_source_freeze_sha256=authority["input"]["source_freeze_sha256"],
            expected_receipt_sha256=r.sha(receipt), expected_policy_sha256=policy)
        material = b.worker_input_files(bound["verification"], bound["eligible_audio"])
        self.assertEqual(len(material["input_files"]), 7)
        self.assertEqual(len(m.initial_ledger(bound["verification"]["projection"])), 18)
        self.assertFalse(bound["verification"]["execution_enabled"])


    def test_complete_package_verification_precedes_receipt_or_worker_staging(self):
        raw, kwargs = setup_binding()
        with mock.patch.object(r, "verify_complete_package", side_effect=ValueError("semantic failure")) as verify, \
             mock.patch.object(b, "measure_complete_rows") as measure:
            with self.assertRaisesRegex(ValueError, "semantic failure"):
                b.bind_complete_input_package(b"synthetic", kwargs["actual_authority"], raw,
                    expected_authority_sha256=kwargs["expected_authority_sha256"],
                    expected_source_freeze_sha256=kwargs["actual_authority"]["input"]["source_freeze_sha256"],
                    expected_receipt_sha256=kwargs["expected_receipt_sha256"], expected_policy_sha256=kwargs["expected_policy_sha256"])
            verify.assert_called_once(); measure.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
