"""Metadata-only negative tests. No third-party model imports or network calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

from contract import (OLD_CANDIDATES, SHA, canonical, cells, expected_coverage, expected_plan,
                      initial_ledger, slots, validate_coverage, validate_ledger,
                      validate_plan, validate_tts_barrier_metadata)

HERE = Path(__file__).resolve().parent


def fixture():
    rows = []
    for i, cell in enumerate(cells()):
        rows.append({**{k: cell[k] for k in ("cell_id", "voice", "text", "seed")},
                     "attempts": 1, "status": "complete", "wav_sha256": f"{i+1:064x}",
                     "pcm_sha256": f"{i+101:064x}", "raw_wav_sha256": f"{i+201:064x}",
                     "raw_float_sha256": f"{i+301:064x}", "source_frames": 24000,
                     "frames": 16000, "sample_rate_hz": 16000, "channels": 1,
                     "sample_width_bytes": 2, "eos_verified": True,
                     "generation_iterations": 20, "quality_flags": []})
    history = [{"source_id": "fixture-only-old", "wav_sha256": "a"*64, "pcm_sha256": "b"*64}]
    return rows, history


class FixedContractTests(unittest.TestCase):
    def test_exact_plan_and_serialized_files(self):
        self.assertEqual(validate_plan(expected_plan())["asr"]["new_attempts_max"], 18)
        self.assertEqual(len(cells()), 6)
        self.assertEqual(len(slots()), 18)
        self.assertEqual(json.loads((HERE/"plan.proposed.json").read_bytes()), expected_plan())
        self.assertEqual(json.loads((HERE/"coverage.proposed.json").read_bytes()), expected_coverage())

    def test_paired_seed_and_utf8_text(self):
        rows = cells()
        self.assertEqual([r["seed"] for r in rows], [1012001,1012001,1012002,1012002,1012003,1012003])
        self.assertEqual([r["text"].encode() for r in rows], ["你好小窝".encode(), "你好，小窝".encode()]*3)

    def test_mutated_science_or_execution_rejected(self):
        mutations = [("execution_enabled", True), ("status", "enabled")]
        for key, value in mutations:
            with self.subTest(key=key):
                p = expected_plan(); p[key] = value
                with self.assertRaises(ValueError): validate_plan(p)
        for field, value in (("voice", "Ryan"), ("text", "你好,小窝"), ("seed", 1337), ("attempts_max", 2)):
            with self.subTest(field=field):
                p = expected_plan(); p["tts"]["cells"][0][field] = value
                with self.assertRaises(ValueError): validate_plan(p)
        p = expected_plan(); p["tts"]["recipe"]["instruct"] = "say wo"
        with self.assertRaises(ValueError): validate_plan(p)
        p = expected_plan(); p["tts"]["recipe"]["do_sample"] = 1
        with self.assertRaises(ValueError): validate_plan(p)
        p = expected_plan(); p["asr"]["new_attempts_max"] = 19
        with self.assertRaises(ValueError): validate_plan(p)

    def test_old_candidates_exact_sha_and_no_old_k1(self):
        self.assertEqual({r[0] for r in OLD_CANDIDATES}, {"clip-000002","clip-000003","clip-000006","clip-000007","clip-000008","clip-000012"})
        self.assertTrue(all(SHA.fullmatch(r[1]) for r in OLD_CANDIDATES))
        self.assertNotIn("old-k1", {r[2] for r in OLD_CANDIDATES})

    def test_coverage_cannot_refill_original_contract(self):
        c = expected_coverage(); validate_coverage(c)
        c["historical_contract"]["old_twelve_status"] = "MET"
        with self.assertRaises(ValueError): validate_coverage(c)
        c = expected_coverage(); c["minimum_support"][0]["eligible_members"] = ["qwen12-old"]
        with self.assertRaises(ValueError): validate_coverage(c)

    def test_ledger_initial_and_all_complete(self):
        for phase, n in (("tts",6),("asr",18)):
            ledger = initial_ledger(phase)
            self.assertEqual(validate_ledger(phase,ledger), {"denominator":n,"consumed":0,"complete":0})
            for row in ledger: row.update(status="complete",attempts=1)
            self.assertEqual(validate_ledger(phase,ledger)["consumed"], n)

    def test_failed_consumed_and_later_not_run(self):
        ledger = initial_ledger("tts")
        ledger[0].update(status="failed_no_retry",attempts=1)
        self.assertEqual(validate_ledger("tts",ledger)["consumed"], 1)
        ledger[1].update(status="complete",attempts=1)
        with self.assertRaises(ValueError): validate_ledger("tts",ledger)

    def test_ledger_rejects_excess_alias_and_bool(self):
        for phase in ("tts", "asr"):
            for mutation in ("extra", "oldref", "retry", "boolean", "reorder"):
                with self.subTest(phase=phase,mutation=mutation):
                    rows = initial_ledger(phase)
                    if mutation == "extra": rows.append(deepcopy(rows[0]))
                    if mutation == "oldref": rows[0]["id"] = "clip-000001-qwen06"
                    if mutation == "retry": rows[0].update(status="complete",attempts=2)
                    if mutation == "boolean": rows[0].update(status="complete",attempts=True)
                    if mutation == "reorder": rows[0],rows[1] = rows[1],rows[0]
                    with self.assertRaises(ValueError): validate_ledger(phase,rows)

    def test_barrier_all_six_and_actual_geometry(self):
        rows, history = fixture()
        out = validate_tts_barrier_metadata(rows,history)
        self.assertEqual(out["new_asr_slots_max"],18)
        self.assertEqual(out["metadata_status"],"shape_only_not_execution_or_admission")
        for mutator in (lambda x:x.pop(), lambda x:x[0].update(frames=1),
                        lambda x:x[0].update(eos_verified=False),
                        lambda x:x[0].update(generation_iterations=120),
                        lambda x:x[0].update(source_frames=288000,frames=192000),
                        lambda x:x[0].update(attempts=True),
                        lambda x:x[0].update(status="failed_no_retry")):
            mutated = deepcopy(rows); mutator(mutated)
            with self.assertRaises(ValueError): validate_tts_barrier_metadata(mutated,history)
        with self.assertRaises(ValueError): validate_tts_barrier_metadata(rows,[])

    def test_duplicate_entire_group_is_symmetric(self):
        rows, history = fixture()
        rows[1]["pcm_sha256"] = rows[0]["pcm_sha256"]
        rows[4]["wav_sha256"] = rows[0]["wav_sha256"]
        result = validate_tts_barrier_metadata(rows,history)
        self.assertEqual([r["cell_id"] for r in result["screening"] if r["disposition"] == "not_run_duplicate"], ["k1p6-001","k1p6-002","k1p6-005"])
        self.assertEqual(result["new_asr_slots_max"],9)

    def test_historical_pcm_and_wav_collision_skip(self):
        rows, history = fixture()
        rows[0]["pcm_sha256"] = history[0]["pcm_sha256"]
        rows[3]["wav_sha256"] = history[0]["wav_sha256"]
        result = validate_tts_barrier_metadata(rows,history)
        self.assertEqual(result["new_asr_slots_max"],12)
        self.assertEqual(result["screening"][0]["historical_matches"],["fixture-only-old"])

    def test_quality_quarantine_and_skipped_denominator(self):
        rows, history = fixture(); rows[0]["quality_flags"] = ["silent"]
        rows[1]["pcm_sha256"] = history[0]["pcm_sha256"]
        screen = validate_tts_barrier_metadata(rows,history)["screening"]
        dispositions = {r["cell_id"]:r["disposition"] for r in screen}
        ledger = initial_ledger("asr")
        for row, slot in zip(ledger,slots()):
            disposition = dispositions[slot["cell_id"]]
            row.update(status=disposition if disposition != "eligible" else "complete",
                       attempts=0 if disposition != "eligible" else 1)
        self.assertEqual(validate_ledger("asr",ledger,screen),{"denominator":18,"consumed":12,"complete":12})
        with self.assertRaises(ValueError): validate_ledger("asr",ledger)
        ledger[0].update(status="complete",attempts=1)
        with self.assertRaises(ValueError): validate_ledger("asr",ledger,screen)

    def test_entry_unconditionally_disabled(self):
        for flags in ([], ["--execute"], ["--release", "execution-release.disabled.json"]):
            p = subprocess.run([sys.executable,"-B",str(HERE/"entry.py"),*flags],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn("is disabled",p.stderr)

    def test_import_does_not_load_inference_dependencies(self):
        for name in ("torch","numpy","qwen_tts","transformers","funasr","whisper"):
            self.assertNotIn(name,sys.modules)

    def test_disabled_release_has_no_real_execution_identity(self):
        release = json.loads((HERE/"execution-release.disabled.json").read_bytes())
        for name in ("tts_enabled", "asr_enabled", "training_enabled"):
            self.assertIs(release[name],False)
        for name in ("source_commit", "source_tree", "tts_run_id", "tts_source_ref", "tts_data_commit", "asr_source_ref", "asr_run_id", "historical_audio_comparison_inventory_sha256"):
            self.assertIsNone(release[name])

    def test_resources_and_destination_do_not_silently_expand(self):
        plan = expected_plan()
        self.assertEqual(plan["resources"]["tts"]["maximum_audio_seconds_all_cells"],72)
        self.assertEqual(plan["resources"]["asr_proposal"]["maximum_cumulative_audio_seconds"],216)
        self.assertEqual(plan["resources"]["asr_proposal"]["new_complete_data_package_response_body_reservation_bytes"],16*1024**2)
        self.assertLess(plan["resources"]["asr_proposal"]["proposed_mixed_scope_acquisition_reservation_bytes"],6*1024**3)
        self.assertEqual(plan["publication"]["audio_repository"],"jiying2007/kws-data")
        self.assertEqual(plan["publication"]["evidence_repository"],"jiying2007/kws-pipeline")
        self.assertEqual(plan["publication"]["transport_bounds"]["compressed_archive_bytes"],12*1024**2)
        self.assertFalse(plan["resources"]["actions_artifact_storage"])
        self.assertIsNone(plan["tts"]["instruction"])
        self.assertIsNone(plan["tts"]["phoneme_interface"])

    def test_derived_exact_pcm_silence_policy_parent_binding(self):
        plan=expected_plan(); change=json.loads((HERE/"policy-amendment.json").read_bytes())
        self.assertEqual(plan["derivation"]["parent_plan_sha256"],"a799de29be9e41c283982942ed6eb2256e951d864c12104deb152bdcd4d1f229")
        self.assertEqual(plan["derivation"]["parent_contract_sha256"],"aa860a9baa2b0aceb29d5a75f2693dec8c2bb150b68c60f49a68a08e8be017e0")
        self.assertEqual(change["exact_plan_additions"],{k:plan[k] for k in ("derivation","quality_policy")})
        self.assertEqual(change["exact_contract_behavior_delta"]["quality_flags_allowed_added"],["derived_pcm_silent"])
        self.assertEqual(plan["quality_policy"]["ordered_flags"],["silent","source_peak_ge_one","normalized_peak_ge_one","derived_pcm_silent"])

    def test_derived_pcm_silence_excludes_three_slots_without_changing_denominator(self):
        rows,history=fixture();rows[0]["quality_flags"]=["derived_pcm_silent"]
        result=validate_tts_barrier_metadata(rows,history);screen=result["screening"]
        self.assertEqual(screen[0]["disposition"],"not_run_quality")
        self.assertEqual(result["new_asr_slots_max"],15)
        ledger=initial_ledger("asr")
        for slot,row in zip(slots(),ledger):
            if slot["cell_id"]==rows[0]["cell_id"]:row["status"]="not_run_quality"
        self.assertEqual(validate_ledger("asr",ledger,screen),{"denominator":18,"consumed":0,"complete":0})
        self.assertEqual([r["status"] for r in rows],["complete"]*6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
