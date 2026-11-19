"""V5 single English word; standard-library checks, no inference or network."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/"configs/research/english-v5-mero-only.json"
HERE=Path(__file__).resolve().parent


class SingleEnglishTarget(unittest.TestCase):
    def test_exclusive_runtime_target_and_ipa(self):
        p=json.loads(PROFILE.read_bytes())
        self.assertEqual(p["schema"],"english-v5-mero-only-pipeline-v1")
        self.assertEqual(p["active_keyword_count"],1)
        self.assertEqual(p["selected_english_target"]["id"],"hey_mero")
        self.assertEqual(p["selected_english_target"]["text"],"Hey Mero")
        self.assertEqual(p["selected_english_target"]["ipa"],"/heɪ ˈmɛroʊ/")
        self.assertEqual(p["selected_english_target"]["keyword_event_id"],1)
        self.assertEqual(p["ctc"],{
            "0":"<blank>","1":"HEY","2":"MEH","3":"ROH"})
        self.assertEqual(p["target_ctc_sequence"],[1,2,3])
        self.assertEqual(p["restrictions"]["new_english_positive_keywords"],
                         ["hey_mero"])
        self.assertTrue(p["historical"]["never_remap_other_word_to_background"])

    def test_target_state_and_gates_are_not_fake_claims(self):
        p=json.loads(PROFILE.read_bytes())
        self.assertIsNone(p["data_source"]["new_weights_sha256"])
        self.assertIsNone(p["data_source"]["training_run_id"])
        self.assertEqual(p["architecture"]["primary"],"causal_fsmn_ctc")
        self.assertEqual(p["architecture"]["ctc_classes"],4)
        self.assertEqual(p["architecture"]["left_memory"],4)
        self.assertEqual(p["architecture"]["features"]["center"],False)
        self.assertEqual(p["quality_targets"]["mero_frr_max"],0.05)
        self.assertEqual(p["quality_targets"]["far_per_hour_max"],0.1)
        self.assertIn("ssc305",p["gates"]["l3"])
        self.assertEqual(p["status"],
             "research_selected_single_target_not_device_qualified")

    def test_public_c_ctc_interface_only_has_one_keyword(self):
        h=(HERE/"mero_ctc.h").read_text()
        c=(HERE/"mero_ctc.c").read_text()
        self.assertIn("#define EN_V5_MERO_TOKEN_COUNT 4u",h)
        self.assertIn("#define EN_V5_MERO_KEYWORD_ID 1u",h)
        self.assertIn("EN_V5_NONFINITE",h)
        self.assertIn("en_v5_mero_reset",c)
        self.assertIn("en_v5_mero_discontinuity",c)
        self.assertIn("available_after_sample",c)
        self.assertNotIn("Vosen",h+c)
        self.assertNotIn("Vozen",h+c)

    def test_english_v5_workflow_is_exactly_registered(self):
        manifest=json.loads((ROOT/"research/consolidation/source-retention-2026-10-07.json").read_bytes())
        workflow=".github/workflows/english-v5-mero-contract.yml"
        self.assertEqual(manifest["allowed_new_active_workflows"].count(workflow),1)
        identities=[row for row in manifest["current_added_active_workflows"]
                    if row["path"]==workflow]
        self.assertEqual(len(identities),1)
        record=identities[0]
        raw=(ROOT/workflow).read_bytes()
        self.assertEqual(record["mode"],"100644")
        self.assertEqual(record["bytes"],len(raw))
        self.assertEqual(record["sha256"],hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["git_blob_sha1"],
            hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\\0"+raw).hexdigest())
        self.assertIn("english-v5-mero-contract.yml",
            (ROOT/"tools/verify_research_sources.py").read_text())

    def test_old_research_safely_isolated_and_new_doc_authority(self):
        old=(ROOT/"research/english_stream_integration/ekws_research_stream.h").read_text()
        self.assertIn("EKWS_CLASSES 3u",old)
        self.assertIn("1 Hey Vosen, 2 Hey Mero",old)
        readme=(ROOT/"README.md").read_text()
        self.assertIn("**2026-11-19 English V5.0 active target:**",readme)
        self.assertIn("docs/ENGLISH_V5_MERO_ONLY.md",readme)
        self.assertIn("Hey Mero",readme[:400])
        self.assertIn("V5.0", (ROOT/"docs/ENGLISH_V5_MERO_ONLY.md").read_text())


if __name__=="__main__":
    unittest.main()
