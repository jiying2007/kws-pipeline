"""Stdlib fake interrupted snapshots only; no inference, network, or runtime setup."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

from contract import canonical, cells
from test_tts_semantics import archive
from test_tts_worker import float_wav, pcm_wav, RESULTS, riff
from tts_common import sha
import tts_finalize as finalize
from tts_worker import new_outcomes


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="k1-finalize-fake-", dir=RESULTS)
        self.addCleanup(self.temp.cleanup)
        self.runtime = Path(self.temp.name)
        self.output = self.runtime / "publication-members"
        self.output.mkdir()

    def put(self, name, raw):
        path = self.output / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(raw)

    def populate(self, members):
        for name, raw in members.items():
            self.put(name, raw)

    def read(self, name="outcomes.json"):
        return json.loads((self.output / name).read_bytes())

    def update(self, fn):
        value = self.read(); fn(value)
        self.put("outcomes.json", canonical(value))

    def run_finalize(self):
        return finalize.finalize_after_exit(self.output, self.runtime, worker_exited=True)

    def snapshot(self):
        return {path.relative_to(self.runtime).as_posix(): path.read_bytes()
                for path in self.runtime.rglob("*") if path.is_file()}

    def stale(self, index=0, status="not_run"):
        members = archive(index, True)
        out = json.loads(members["outcomes.json"])
        row = out["cells"][index]
        if status == "not_run":
            out["cells"][index] = new_outcomes()["cells"][index]
            out["ledger"][index] = new_outcomes()["ledger"][index]
            out["consumed_attempts"] = index
        else:
            row["status"] = "started"; row.pop("error_class")
            out["ledger"][index]["status"] = "started"
        out["status"] = "started"
        members["outcomes.json"] = canonical(out)
        self.populate(members)
        return members

    def assert_rejected_without_mutation(self):
        before = self.snapshot()
        with self.assertRaises((ValueError, OSError)):
            self.run_finalize()
        self.assertEqual(self.snapshot(), before)

    def test_no_worker_start_retains_six_unstarted_and_explicit_absent_originals(self):
        result = self.run_finalize()
        self.assertEqual(result["cells"], new_outcomes()["cells"])
        self.assertEqual(result["consumed_attempts"], 0)
        self.assertEqual(result["status"], "not_run")
        self.assertIsNone(result["finalization"]["original_outcomes"])
        self.assertIsNone(result["finalization"]["original_eos"])
        self.assertEqual(self.read("eos.json")["rows"], [])

    def test_requires_explicit_confirmed_worker_exit(self):
        for value in (False, 1, None, "yes"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                finalize.finalize_after_exit(self.output, self.runtime, worker_exited=value)
        self.assertEqual(self.snapshot(), {})

    def test_only_fixed_private_sibling_roots(self):
        wrong = self.runtime / "different"; wrong.mkdir()
        with self.assertRaises(ValueError):
            finalize.finalize_after_exit(wrong, self.runtime, worker_exited=True)
        with self.assertRaises(ValueError):
            finalize.finalize_after_exit(self.output, self.output, worker_exited=True)

    def test_complete_six_is_byte_identical_noop(self):
        self.populate(archive())
        before = self.snapshot()
        self.run_finalize()
        self.assertEqual(self.snapshot(), before)

    def test_ordinary_failure_without_invalid_audio_is_noop(self):
        self.populate(archive(2, True, True))
        before = self.snapshot(); result = self.run_finalize()
        self.assertEqual(self.snapshot(), before)
        self.assertNotIn("finalization", result)

    def test_durable_claim_before_atomic_outcome_update_is_consumed_failure(self):
        original = self.stale()
        result = self.run_finalize()
        self.assertEqual([row["status"] for row in result["cells"]], ["failed_no_retry"] + ["not_run"] * 5)
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertEqual(result["consumed_attempts"], 1)
        self.assertEqual(result["finalization"]["reconciled_cells"], [{"cell_id": "k1p6-001", "previous_status": "not_run"}])
        self.assertEqual((self.output / "claims/k1p6-001.started.json").read_bytes(), original["claims/k1p6-001.started.json"])
        for name in ("outcomes.json", "eos.json"):
            self.assertEqual((self.runtime / "finalization-private" / ("original-" + name)).read_bytes(), original[name])
        self.assertEqual(result["cells"][0]["observation"], None)

    def test_started_last_cell_stays_failure_even_with_valid_pair(self):
        original = self.stale(5, "started")
        self.put("raw/k1p6-006.wav", float_wav([0.125] * 3840))
        self.put("derived/k1p6-006.wav", pcm_wav(2560))
        result = self.run_finalize()
        self.assertEqual(result["cells"][:5], json.loads(original["outcomes.json"])["cells"][:5])
        self.assertEqual(result["cells"][5]["status"], "failed_no_retry")
        self.assertEqual(result["consumed_attempts"], 6)
        self.assertEqual(result["finalization"]["quarantined_audio"], [])
        self.assertTrue((self.output / "derived/k1p6-006.wav").exists())

    def test_missing_outcomes_can_recover_exactly_one_original_claim(self):
        members = archive(0, True)
        del members["outcomes.json"]; del members["eos.json"]
        self.populate(members)
        result = self.run_finalize()
        self.assertEqual(result["consumed_attempts"], 1)
        self.assertIsNone(result["finalization"]["original_outcomes"])

    def test_never_infers_final_global_success_from_six_saved_complete_cells(self):
        self.populate(archive())
        self.update(lambda out: out.update(status="started"))
        original_cells = self.read()["cells"]
        result = self.run_finalize()
        self.assertEqual(result["status"], "stopped_no_retry")
        self.assertEqual(result["cells"], original_cells)
        self.assertEqual(result["finalization"]["reconciled_cells"], [])
        self.assertFalse(result["finalization"]["completion_inferred"])

    def test_invalid_raw_is_private_hash_only_evidence(self):
        self.stale()
        partial = b"RIFF\x00private-sentinel-not-a-wav"
        self.put("raw/k1p6-001.wav", partial)
        result = self.run_finalize()
        self.assertFalse((self.output / "raw/k1p6-001.wav").exists())
        self.assertEqual((self.runtime / "finalization-private/raw/k1p6-001.wav").read_bytes(), partial)
        self.assertNotIn(b"private-sentinel", canonical(result))
        self.assertEqual(result["finalization"]["quarantined_audio"], [{"path": "raw/k1p6-001.wav",
            "bytes": len(partial), "sha256": sha(self.runtime / "finalization-private/raw/k1p6-001.wav"), "reason": "invalid_waveform"}])

    def test_invalid_raw_and_valid_derivative_both_quarantined_without_orphan(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"RIFF")
        self.put("derived/k1p6-001.wav", pcm_wav())
        result = self.run_finalize()
        self.assertEqual({row["path"]: row["reason"] for row in result["finalization"]["quarantined_audio"]},
                         {"raw/k1p6-001.wav": "invalid_waveform", "derived/k1p6-001.wav": "native_unavailable"})

    def test_derivative_missing_native_is_preserved_privately(self):
        self.stale()
        self.put("derived/k1p6-001.wav", pcm_wav())
        result = self.run_finalize()
        self.assertEqual(result["finalization"]["quarantined_audio"][0]["reason"], "native_unavailable")

    def test_invalid_wave_chunk_order_catches_wave_error(self):
        self.stale()
        import struct
        self.put("derived/k1p6-001.wav", riff([(b"data", b"\0" * 10),
            (b"fmt ", struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16))]))
        result = self.run_finalize()
        self.assertEqual(result["finalization"]["quarantined_audio"][0]["reason"], "invalid_waveform")

    def test_mismatched_partial_pair_only_quarantines_derivative(self):
        self.stale()
        self.put("raw/k1p6-001.wav", float_wav([0.125] * 3840))
        self.put("derived/k1p6-001.wav", pcm_wav(2561))
        result = self.run_finalize()
        self.assertEqual(result["finalization"]["quarantined_audio"][0]["reason"], "pair_geometry_mismatch")
        self.assertTrue((self.output / "raw/k1p6-001.wav").is_file())

    def test_oversized_invalid_wave_has_truthful_length_not_wave_cap(self):
        self.stale()
        raw = b"x" * (12 * 24000 * 4 + 257)
        self.put("raw/k1p6-001.wav", raw)
        result = self.run_finalize()
        self.assertEqual(result["finalization"]["quarantined_audio"][0]["bytes"], len(raw))

    def test_empty_partial_wave_is_not_ignored(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"")
        result = self.run_finalize()
        self.assertEqual(result["finalization"]["quarantined_audio"][0]["bytes"], 0)

    def test_unfinished_atomic_saves_are_never_promoted(self):
        self.stale()
        fake = archive()["outcomes.json"]
        self.put("outcomes.json.tmp", fake)
        self.put("eos.json.tmp", b"private-sentinel-partial-json")
        result = self.run_finalize()
        self.assertEqual(result["consumed_attempts"], 1)
        self.assertEqual([row["path"] for row in result["finalization"]["preserved_temporaries"]],
                         ["eos.json.tmp", "outcomes.json.tmp"])
        self.assertEqual((self.runtime / "finalization-private/outcomes.json.tmp").read_bytes(), fake)
        self.assertNotIn(b"private-sentinel", canonical(result))
        self.assertFalse((self.output / "outcomes.json.tmp").exists())

    def test_malformed_claim_is_not_reclassified_as_not_run(self):
        self.stale()
        self.put("claims/k1p6-001.started.json", b'{"partial":')
        self.assert_rejected_without_mutation()

    def test_noncanonical_claim_is_not_rewritten(self):
        self.stale()
        name = "claims/k1p6-001.started.json"
        self.put(name, (self.output / name).read_bytes() + b"\n")
        self.assert_rejected_without_mutation()

    def test_claim_without_complete_prefix_fails_closed(self):
        self.populate(archive(0))
        self.put("claims/k1p6-002.started.json", archive()["claims/k1p6-002.started.json"])
        self.assert_rejected_without_mutation()

    def test_multiple_unrecorded_claims_cannot_infer_completed_prefix(self):
        self.stale()
        self.put("claims/k1p6-002.started.json", archive()["claims/k1p6-002.started.json"])
        self.assert_rejected_without_mutation()

    def test_unknown_claim_and_unstarted_audio_rejected(self):
        self.stale()
        self.put("claims/unrecognized.started.json", b"{}")
        self.assert_rejected_without_mutation()

    def test_unstarted_audio_rejected(self):
        self.populate(archive(0))
        self.put("raw/k1p6-006.wav", b"broken")
        self.assert_rejected_without_mutation()

    def test_saved_claim_receipt_mismatch_rejected(self):
        self.stale(status="started")
        self.update(lambda out: out["cells"][0]["claim"].update(sha256="0" * 64))
        self.assert_rejected_without_mutation()

    def test_saved_incomplete_observation_is_not_silently_removed(self):
        self.stale(status="started")
        self.update(lambda out: out["cells"][0].update(observation={"frames": 12}))
        self.assert_rejected_without_mutation()

    def test_complete_invalid_wave_cannot_be_converted_to_failure(self):
        self.populate(archive())
        self.put("raw/k1p6-001.wav", b"RIFF")
        self.assert_rejected_without_mutation()

    def test_complete_wave_hash_mismatch_cannot_be_repaired(self):
        self.populate(archive())
        self.put("raw/k1p6-001.wav", float_wav([0.25] * 3840))
        self.assert_rejected_without_mutation()

    def test_missing_complete_eos_fails_closed(self):
        members = archive(); del members["eos.json"]
        self.populate(members)
        self.assert_rejected_without_mutation()

    def test_corrupt_outcomes_fails_closed(self):
        self.put("outcomes.json", b'{"unfinished":')
        self.assert_rejected_without_mutation()

    def test_overlong_temporary_rejected_before_quarantine(self):
        self.stale()
        self.put("outcomes.json.tmp", b"x" * (finalize.MAX_JSON + 1))
        self.put("raw/k1p6-001.wav", b"RIFF")
        self.assert_rejected_without_mutation()

    def test_aggregate_audio_read_cap_not_per_file_multiplied(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"x" * (finalize.MAX_AUDIO // 2 + 1))
        self.put("derived/k1p6-001.wav", b"x" * (finalize.MAX_AUDIO // 2))
        self.assert_rejected_without_mutation()

    def test_existing_private_quarantine_never_overwritten(self):
        self.stale()
        private = self.runtime / "finalization-private"; private.mkdir()
        (private / "existing").write_bytes(b"keep")
        self.assert_rejected_without_mutation()

    def test_alias_claim_rejected(self):
        self.populate(archive(0))
        target = self.runtime / "elsewhere.json"
        target.write_bytes(archive()["claims/k1p6-001.started.json"])
        (self.output / "claims").mkdir()
        (self.output / "claims/k1p6-001.started.json").symlink_to(target)
        self.assert_rejected_without_mutation()

    def test_hardlinked_audio_rejected(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"RIFF")
        os.link(self.output / "raw/k1p6-001.wav", self.runtime / "alias.wav")
        self.assert_rejected_without_mutation()

    def test_idempotent_repeat_does_not_create_claims_or_rewrite_evidence(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"partial")
        first = self.run_finalize(); before = self.snapshot()
        self.assertEqual(self.run_finalize(), first)
        self.assertEqual(self.snapshot(), before)

    def test_repeat_requires_original_private_bytes(self):
        self.stale()
        self.run_finalize()
        (self.runtime / "finalization-private/original-outcomes.json").write_bytes(b"altered")
        self.assert_rejected_without_mutation()

    def test_quarantined_wav_cannot_reappear_as_new_valid_output(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"partial")
        self.run_finalize()
        self.put("raw/k1p6-001.wav", float_wav([0.125] * 3840))
        self.assert_rejected_without_mutation()

    def test_strict_finalization_shape_paths_enums_and_counts(self):
        self.stale()
        self.put("raw/k1p6-001.wav", b"partial")
        self.put("outcomes.json.tmp", b"partial")
        result = self.run_finalize()
        validators = [
            lambda value: value.update(unexpected_private_note="sentinel"),
            lambda value: value.update(schema="other"),
            lambda value: value.update(calls_resumed=0),
            lambda value: value.update(completion_inferred=True),
            lambda value: value["reconciled_cells"][0].update(previous_status="complete"),
            lambda value: value["reconciled_cells"].append(deepcopy(value["reconciled_cells"][0])),
            lambda value: value["reconciled_cells"][0].update(cell_id="k1p6-006"),
            lambda value: value["quarantined_audio"][0].update(path="../private.wav"),
            lambda value: value["quarantined_audio"][0].update(reason="private-text"),
            lambda value: value["quarantined_audio"][0].update(reason="native_unavailable"),
            lambda value: value["quarantined_audio"][0].update(bytes=True),
            lambda value: value["quarantined_audio"][0].update(bytes=finalize.MAX_AUDIO + 1),
            lambda value: value["quarantined_audio"][0].update(sha256="NOT A HASH"),
            lambda value: value["preserved_temporaries"][0].update(path="private.log"),
            lambda value: value["original_outcomes"].update(raw_text="sentinel"),
        ]
        for mutate in validators:
            value = deepcopy(result["finalization"]); mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, TypeError)):
                finalize.validate_finalization(value, result["cells"])
        self.assertTrue(finalize.validate_finalization(result["finalization"], result["cells"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
