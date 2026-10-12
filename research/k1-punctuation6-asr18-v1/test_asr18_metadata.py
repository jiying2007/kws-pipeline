"""Synthetic metadata-only tests. No real identities, WAVs, history or runtime."""
from copy import deepcopy
import ast
import hashlib
from pathlib import Path
import unittest

import asr18_metadata as m


def fake_hash(label):
    return hashlib.sha256(("SYNTHETIC-TEST-NOT-AUDIO-" + label).encode()).hexdigest()


def rows():
    return [{"cell_id": cell, "status": "complete", "attempts": 1,
        "wav_sha256": fake_hash(cell + "-wav"), "pcm_sha256": fake_hash(cell + "-pcm"),
        "raw_wav_sha256": fake_hash(cell + "-raw-wav"), "raw_float_sha256": fake_hash(cell + "-float"),
        "source_frames": 13440, "frames": 8960, "sample_rate_hz": 16000, "channels": 1,
        "sample_width_bytes": 2, "eos_verified": True, "generation_iterations": 8,
        "quality_flags": []} for cell in m.CELLS]


def history():
    return [{"wav_sha256": fake_hash("private-fixture-wav"), "pcm_sha256": fake_hash("private-fixture-pcm")}]


def projection():
    return m.screen_metadata(rows(), history())


class MetadataTests(unittest.TestCase):
    def test_disabled_proposal_unknowns_and_fixed_eighteen(self):
        value = m.proposed_preparation()
        m.validate_disabled_preparation(value)
        self.assertEqual(len(value["slots"]), 18)
        self.assertTrue(all(v is None for k, v in value.items() if k.startswith("actual_")))
        for key in ("execution_enabled", "production_claim_writer_implemented", "reader_implemented", "publisher_implemented"):
            changed = deepcopy(value)
            changed[key] = True
            with self.assertRaises(ValueError):
                m.validate_disabled_preparation(changed)
        with self.assertRaises(ValueError):
            m.authorize_execution({"enabled": True, "independently_reviewed": True})

    def test_all_six_complete_once_and_order_required(self):
        for alteration in (lambda r: r.pop(), lambda r: r.reverse(),
                           lambda r: r[2].update(status="failed_no_retry"),
                           lambda r: r[2].update(attempts=2), lambda r: r[2].update(attempts=True)):
            value = rows()
            alteration(value)
            with self.assertRaises(ValueError):
                m.screen_metadata(value, history())

    def test_new_unknown_hashes_do_not_bind(self):
        for key in ("wav_sha256", "pcm_sha256", "raw_wav_sha256", "raw_float_sha256"):
            for unknown in (None, "UNKNOWN", "", False):
                value = rows()
                value[1][key] = unknown
                with self.assertRaises(ValueError):
                    m.screen_metadata(value, history())

    def test_strict_duration_geometry_eos_and_token_limit(self):
        for update in ({"source_frames": 288000, "frames": 192000}, {"frames": 1601},
                       {"channels": True}, {"sample_rate_hz": 24000}, {"eos_verified": 1},
                       {"generation_iterations": 120}, {"generation_iterations": True}):
            value = rows()
            value[0].update(update)
            with self.assertRaises(ValueError):
                m.screen_metadata(value, history())

    def test_exact_eos_codec_geometry_all_allowed_iterations(self):
        for iterations in range(2, 120):
            value = rows()
            source_frames = (iterations - 1) * 1920
            value[0].update(generation_iterations=iterations, source_frames=source_frames,
                            frames=(2 * source_frames + 2) // 3)
            out = m.screen_metadata(value, history())
            self.assertEqual(out["cells"][0]["disposition"], "eligible")

    def test_eos_first_iteration_cannot_have_nonempty_source(self):
        for source_frames in (1, 1920, 13440):
            value = rows()
            value[0].update(generation_iterations=1, source_frames=source_frames,
                            frames=(2 * source_frames + 2) // 3)
            with self.assertRaises(ValueError):
                m.screen_metadata(value, history())

    def test_resample_geometry_alone_does_not_establish_codec_geometry(self):
        for source_frames in (2400, 13439, 13441, 15360):
            value = rows()
            value[0].update(generation_iterations=8, source_frames=source_frames,
                            frames=(2 * source_frames + 2) // 3)
            with self.assertRaises(ValueError):
                m.screen_metadata(value, history())

    def test_private_history_required_but_never_authorized(self):
        with self.assertRaises(ValueError):
            m.screen_metadata(rows(), [])
        value = history()
        value[0]["coverage_approved"] = True
        with self.assertRaises(ValueError):
            m.screen_metadata(rows(), value)
        self.assertFalse(projection()["execution_enabled"])

    def test_symmetric_wav_duplicate_group(self):
        value = rows()
        value[1]["wav_sha256"] = value[0]["wav_sha256"]
        value[5]["wav_sha256"] = value[0]["wav_sha256"]
        out = m.screen_metadata(value, history())
        self.assertEqual([r["cell_id"] for r in out["cells"] if r["disposition"] == "not_run_duplicate"],
                         [m.CELLS[0], m.CELLS[1], m.CELLS[5]])

    def test_mixed_transitive_collision_members_all_quarantined(self):
        value = rows()
        value[1]["wav_sha256"] = value[0]["wav_sha256"]
        value[2]["pcm_sha256"] = value[1]["pcm_sha256"]
        out = m.screen_metadata(value, history())
        self.assertEqual([r["disposition"] for r in out["cells"][:3]], ["not_run_duplicate"] * 3)

    def test_historical_wav_or_pcm_duplicate_without_details(self):
        value = rows()
        old = history()
        old.extend([{"wav_sha256": value[0]["wav_sha256"], "pcm_sha256": fake_hash("unshared-private-a")},
                    {"wav_sha256": fake_hash("unshared-private-b"), "pcm_sha256": value[3]["pcm_sha256"]}])
        out = m.screen_metadata(value, old)
        self.assertEqual([r["cell_id"] for r in out["cells"] if r["disposition"] == "not_run_duplicate"],
                         [m.CELLS[0], m.CELLS[3]])
        body = m.canonical(out).decode()
        for private in (old[0]["wav_sha256"], old[0]["pcm_sha256"], old[1]["pcm_sha256"], old[2]["wav_sha256"], m.digest(old)):
            self.assertNotIn(private, body)
        for forbidden in ("historical_matches", "same_batch_matches", "historical_inventory", "private-fixture"):
            self.assertNotIn(forbidden, body)

    def test_derived_pcm_silent_quarantine_and_duplicate_precedence(self):
        value = rows()
        value[0]["quality_flags"] = ["derived_pcm_silent"]
        out = m.screen_metadata(value, history())
        self.assertEqual(out["cells"][0]["disposition"], "not_run_quality")
        value[1]["pcm_sha256"] = value[0]["pcm_sha256"]
        out = m.screen_metadata(value, history())
        self.assertEqual(out["cells"][0]["disposition"], "not_run_duplicate")
        self.assertEqual(out["cells"][1]["disposition"], "not_run_duplicate")

    def test_unknown_repeated_unordered_quality_flags_rejected(self):
        for flags in (["low_snr"], ["silent", "silent"], ["derived_pcm_silent", "silent"], [True]):
            value = rows()
            value[0]["quality_flags"] = flags
            with self.assertRaises(ValueError):
                m.screen_metadata(value, history())

    def test_blind_hash_order_and_fixed_cell_execution_order_are_separate(self):
        value = rows()
        hashes = sorted(r["wav_sha256"] for r in value)
        for row, sha in zip(value, reversed(hashes)):
            row["wav_sha256"] = sha
        out = m.screen_metadata(value, history())
        blind = m.blind_metadata(out)
        self.assertEqual([r["wav_sha256"] for r in blind["job"]["clips"]], hashes)
        self.assertEqual(blind["execution_plan"]["audio_order"], [f"clip-{i:06d}" for i in range(6, 0, -1)])
        self.assertEqual([r["cell_id"] for r in blind["host_only_cell_map"]], list(m.CELLS))
        m.validate_blind_metadata(out, blind)
        changed = deepcopy(blind)
        changed["execution_plan"]["audio_order"].reverse()
        with self.assertRaises(ValueError):
            m.validate_blind_metadata(out, changed)

    def test_blind_projection_has_only_audio_fields_no_tts_or_history(self):
        blind = m.blind_metadata(projection())
        self.assertEqual(set(blind["job"]), {"schema", "clips"})
        for row in blind["job"]["clips"]:
            self.assertEqual(set(row), {"audio_id", "audio_path", "wav_sha256"})
        body = m.canonical({"job": blind["job"], "execution_plan": blind["execution_plan"]}).decode()
        for forbidden in ('"voice"', '"text"', '"seed"', '"condition"', "k1p6-", "raw_float", "historical", '"target"'):
            self.assertNotIn(forbidden, body)

    def test_anonymous_order_omissions_duplicates_and_unknown_ids_rejected(self):
        out = projection()
        for alter in (lambda a: a.pop(), lambda a: a.__setitem__(1, a[0]),
                      lambda a: a.__setitem__(1, "clip-999999")):
            blind = m.blind_metadata(out)
            alter(blind["execution_plan"]["audio_order"])
            with self.assertRaises(ValueError):
                m.validate_blind_metadata(out, blind)

    def test_cell_mapping_swaps_duplicates_and_omissions_rejected(self):
        out = projection()
        for alter in (lambda a: a.pop(), lambda a: a.__setitem__(1, a[0]), lambda a: a.reverse(),
                      lambda a: a[0].update(audio_id=a[1]["audio_id"])):
            blind = m.blind_metadata(out)
            alter(blind["host_only_cell_map"])
            with self.assertRaises(ValueError):
                m.validate_blind_metadata(out, blind)

    def test_blind_job_missing_duplicate_misbound_audio_and_family_order_rejected(self):
        out = projection()
        for alter in (lambda a: a["job"]["clips"].pop(),
                      lambda a: a["job"]["clips"].__setitem__(1, a["job"]["clips"][0]),
                      lambda a: a["job"]["clips"][0].update(audio_path="audio/other.wav"),
                      lambda a: a["execution_plan"]["family_order"].reverse(),
                      lambda a: a["execution_plan"].update(job_sha256=fake_hash("wrong-job"))):
            blind = m.blind_metadata(out)
            alter(blind)
            with self.assertRaises(ValueError):
                m.validate_blind_metadata(out, blind)

    def test_public_projection_rejects_extra_history_and_target_keys(self):
        for key in ("historical_matches", "historical_inventory_sha256", "private_path", "voice", "text"):
            value = projection()
            value["cells"][0][key] = "disallowed"
            with self.assertRaises(ValueError):
                m.blind_metadata(value)

    def test_quality_and_collision_cannot_be_promoted_eligible(self):
        value = rows()
        value[0]["quality_flags"] = ["derived_pcm_silent"]
        value[2]["wav_sha256"] = value[1]["wav_sha256"]
        out = m.screen_metadata(value, history())
        for index in (0, 1, 2):
            tampered = deepcopy(out)
            tampered["cells"][index]["disposition"] = "eligible"
            with self.assertRaises(ValueError):
                m.blind_metadata(tampered)

    def test_empty_eligible_has_no_release_and_eighteen_exclusions(self):
        value = rows()
        for row in value:
            row["quality_flags"] = ["derived_pcm_silent"]
        out = m.screen_metadata(value, history())
        blind = m.blind_metadata(out)
        self.assertEqual(blind["job"]["clips"], [])
        self.assertEqual(blind["max_new_asr_calls"], 0)
        self.assertFalse(blind["release_available"])
        ledger = m.initial_ledger(out)
        self.assertEqual(len(ledger), 18)
        self.assertTrue(all(r["status"] == "not_run_quality" and r["attempts"] == 0 for r in ledger))
        self.assertEqual(m.validate_ledger(out, ledger)["consumed"], 0)

    def test_eighteen_fixed_family_major_ledger_and_complete_prefix(self):
        out = projection()
        ledger = m.initial_ledger(out)
        self.assertEqual([r["id"] for r in ledger], [cell + "-" + family for family in m.FAMILIES for cell in m.CELLS])
        for prefix in range(19):
            candidate = deepcopy(ledger)
            for row in candidate[:prefix]:
                row.update(status="complete", attempts=1)
            self.assertEqual(m.validate_ledger(out, candidate)["consumed"], prefix)

    def test_first_not_run_started_failure_stops_later_eligible(self):
        out = projection()
        for status, attempts in (("not_run", 0), ("started", 1), ("failed_no_retry", 1)):
            ledger = m.initial_ledger(out)
            ledger[0].update(status=status, attempts=attempts)
            m.validate_ledger(out, ledger)
            ledger[1].update(status="complete", attempts=1)
            with self.assertRaises(ValueError):
                m.validate_ledger(out, ledger)

    def test_exclusions_skip_without_breaking_prefix_but_never_run(self):
        value = rows()
        value[0]["quality_flags"] = ["derived_pcm_silent"]
        out = m.screen_metadata(value, history())
        ledger = m.initial_ledger(out)
        ledger[1].update(status="complete", attempts=1)
        self.assertEqual(m.validate_ledger(out, ledger)["consumed"], 1)
        for index in (0, 6, 12):
            changed = deepcopy(ledger)
            changed[index].update(status="complete", attempts=1)
            with self.assertRaises(ValueError):
                m.validate_ledger(out, changed)

    def test_denominator_order_duplicate_and_boolean_tampering_rejected(self):
        out = projection()
        for alteration in (lambda a: a.pop(), lambda a: a.reverse(), lambda a: a.__setitem__(1, a[0]),
                           lambda a: a[0].update(status="complete", attempts=True),
                           lambda a: a[0].update(status="complete", attempts=2)):
            ledger = m.initial_ledger(out)
            alteration(ledger)
            with self.assertRaises(ValueError):
                m.validate_ledger(out, ledger)

    def test_future_source_authority_remains_shape_only(self):
        value = {"repository": "jiying2007/kws-data", "data_commit": fake_hash("synthetic-data")[:40],
            "source_commit": fake_hash("synthetic-source")[:40], "run_id": 123, "run_attempt": 1,
            "job_id": 456, "job_key": "k1-tts", "source_freeze_sha256": fake_hash("source-freeze"),
            "archive_sha256": fake_hash("archive"), "archive_bytes": 1024,
            "outer_manifest_sha256": fake_hash("outer-manifest"), "member_manifest_sha256": fake_hash("member-manifest")}
        self.assertFalse(m.validate_source_authority_shape(value)["execution_enabled"])
        for key in ("data_commit", "source_commit", "run_id", "job_id", "source_freeze_sha256", "archive_sha256"):
            changed = deepcopy(value)
            changed[key] = None
            with self.assertRaises(ValueError):
                m.validate_source_authority_shape(changed)
        for update in ({"job_key": "writer-fixture"}, {"run_attempt": 2}, {"run_id": True}, {"repository": "jiying2007/kws-pipeline"}):
            changed = {**value, **update}
            with self.assertRaises(ValueError):
                m.validate_source_authority_shape(changed)

    def test_maximum_complete_package_and_manifest_are_counted(self):
        out = m.reader_budget_shape(m.MAX_ARCHIVE, m.MAX_MANIFEST, [m.MAX_CHUNK] * 48)
        self.assertEqual(out["gets"], 49)
        self.assertEqual(out["response_body_bytes"], m.MAX_ARCHIVE + m.MAX_MANIFEST)
        self.assertEqual(out["response_body_reservation_bytes"], 16 * 1024 ** 2)
        self.assertEqual(out["extra_source_gets"], 0)

    def test_body_roster_rejects_missing_duplicate_oversize_chunks(self):
        for values in ((m.MAX_ARCHIVE + 1, 1, [m.MAX_CHUNK] * 48 + [1]),
                       (m.MAX_ARCHIVE, m.MAX_MANIFEST + 1, [m.MAX_CHUNK] * 48),
                       (m.MAX_ARCHIVE, 1, [m.MAX_CHUNK] * 47), (2, 1, [1, 1]), (1, 1, [True])):
            with self.assertRaises(ValueError):
                m.reader_budget_shape(*values)

    def test_prototype_import_and_side_effect_surface(self):
        tree = ast.parse(Path(m.__file__).read_text())
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imported, {"hashlib", "json", "re"})
        self.assertFalse(any(isinstance(node, ast.ImportFrom) for node in ast.walk(tree)))
        forbidden = {"open", "exec", "eval", "compile", "__import__", "getattr", "setattr"}
        called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertFalse(called & forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
