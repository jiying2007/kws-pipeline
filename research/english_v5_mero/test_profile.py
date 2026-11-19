"""V5 single English word; standard-library checks, no inference or network."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import re
import unittest
import subprocess
import tempfile
import shutil
import importlib.util

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

    def test_frozen_43k_original_weight_pack_contract(self):
        spec=importlib.util.spec_from_file_location(
            "mero43_packer_contract",HERE/"pack_mero43.py")
        model=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        proof=model.source_check()
        self.assertEqual(proof["status"],"SOURCE_ONLY_NO_NEW_MODEL")
        self.assertEqual(proof["model_parameters"],43876)
        self.assertEqual(proof["pack_bytes"],216960)
        self.assertEqual(proof["artifact_id"],14514228406)
        self.assertEqual(proof["optimizer_updates"],0)
        self.assertFalse(proof["calibration_or_locked_pcm_opened"])

    def test_43k_end_to_end_c11_synthetic_streaming_contract(self):
        # Historical V5 CTC is reused; the new code supplies actual FFT512 /
        # logMel40 and four-block trained-model geometry without any C heap.
        sources=[HERE/"mero43_stream.c",HERE/"mero_ctc.c",
                 HERE/"test_mero43_stream.c"]
        with tempfile.TemporaryDirectory(prefix="mero-v5-43k-") as td:
            for compiler in ("gcc","clang"):
                self.assertIsNotNone(shutil.which(compiler))
                executable=Path(td)/("mero43-"+compiler)
                command=[compiler,"-std=c11","-Wall","-Wextra","-Werror",
                    "-pedantic","-O2","-ffp-contract=off",
                    "-I",str(HERE),*(str(p) for p in sources),
                    "-lm","-o",str(executable)]
                subprocess.run(command,check=True,capture_output=True,
                               text=True,timeout=45)
                tested=subprocess.run([str(executable)],check=True,
                               capture_output=True,text=True,timeout=30)
                self.assertIn(
                    "ENGLISH_V5_MERO43_PCM_MEL_FSMN_CTC_STREAM_CONTRACT_PASS",
                    tested.stdout)


    def test_standalone_43k_sdk_build_install_and_consumption(self):
        """Real host GCC/Clang installed target, not a mock or SSC305 board."""
        import os
        import sys
        if sys.flags.optimize:
            self.skipTest("single actual GCC/Clang package build in normal mode")
        self.assertIsNotNone(shutil.which("cmake"))
        self.assertIsNotNone(shutil.which("ctest"))
        for compiler in ("gcc","clang"):
            with tempfile.TemporaryDirectory(prefix="heym43-install-"+compiler) as temp:
                root=Path(temp)
                build=root/"build"
                prefix=root/"install"
                consumer=root/"consumer"
                consumer.mkdir()
                env=os.environ.copy()
                env["CC"]=compiler
                for command in (
                    ["cmake","-S",str(HERE),"-B",str(build),
                     "-DCMAKE_BUILD_TYPE=Release",
                     "-DHEY_MERO43_BUILD_TESTS=ON",
                     "-DHEY_MERO43_BUILD_HOST_TOOL=ON",
                     "-DCMAKE_INSTALL_PREFIX="+str(prefix)],
                    ["cmake","--build",str(build),"--parallel","2"],
                    ["ctest","--test-dir",str(build),"--output-on-failure"],
                    ["cmake","--install",str(build)]):
                    subprocess.run(command,check=True,env=env,
                                   capture_output=True,text=True,timeout=100)
                self.assertTrue((prefix/"lib/libheym43_stream.a").is_file())
                self.assertTrue((prefix/"include/heym43/mero43_stream.h").is_file())
                self.assertTrue((prefix/"include/heym43/mero_ctc.h").is_file())
                self.assertTrue((prefix/"lib/cmake/HeyMero43/HeyMero43Targets.cmake").is_file())
                consumer.joinpath("CMakeLists.txt").write_text(
                    "cmake_minimum_required(VERSION 3.18)\n"
                    "project(mero43_consumer C)\n"
                    "find_package(HeyMero43 0.1 CONFIG REQUIRED)\n"
                    "add_executable(consumer main.c)\n"
                    "target_link_libraries(consumer PRIVATE HeyMero43::stream)\n")
                consumer.joinpath("main.c").write_text(
                    '#include "mero43_stream.h"\n'
                    "int main(void) {return en_v5_m43_model_bytes()==0; }\n")
                for command in (
                    ["cmake","-S",str(consumer),"-B",str(root/"consumer-build"),
                     "-DCMAKE_PREFIX_PATH="+str(prefix)],
                    ["cmake","--build",str(root/"consumer-build"),"--parallel","2"],
                    [str(root/"consumer-build"/"consumer")]):
                    subprocess.run(command,check=True,env=env,
                                   capture_output=True,text=True,timeout=100)

    def test_v6_frozen_model_provenance_and_fresh_qa_not_promotion(self):
        """V6 source-only model identity and independent-quality boundary."""
        q=json.loads((HERE/"V6_QUALITY_EVIDENCE.json").read_bytes())
        self.assertEqual(q["schema"],"english-v6-mero-frozen-quality-evidence-v1")
        self.assertEqual(q["active_keyword"]["text"],"Hey Mero")
        self.assertEqual(q["active_keyword"]["ipa"],"/heɪ ˈmɛroʊ/")
        self.assertEqual(q["active_keyword"]["count"],1)
        self.assertEqual(set(q["models"]),{"B0","B1"})
        self.assertEqual(q["models"]["B0"]["parameter_count"],43876)
        self.assertEqual(q["models"]["B1"]["parameter_count"],43876)
        self.assertEqual(q["models"]["B0"]["artifact_id"],14514228406)
        self.assertEqual(q["models"]["B1"]["artifact_id"],14539983171)
        self.assertEqual(q["models"]["B1"]["pt_sha256"],"20d0c4632224cf332460493695c0dd21927a80ed13034a9531f5d5230d2f7e05")
        self.assertTrue(q["models"]["B0"]["c11_proven"])
        self.assertFalse(q["models"]["B1"]["c11_proven"])
        self.assertEqual(q["new_prospective_source"]["run_id"],44554345276)
        self.assertEqual(q["new_prospective_source"]["artifact_id"],14543254675)
        self.assertEqual(q["new_prospective_source"]["original_full_pcm16_wav_count"],32)
        self.assertEqual(q["new_prospective_source"]["independent_piper_speaker_indices"],list(range(488,496)))
        self.assertEqual(q["new_prospective_source"]["quality_admission"],"PENDING_INDEPENDENT_QA")
        self.assertIsNone(q["new_prospective_source"]["admitted_positive_count"])
        self.assertIsNone(q["new_prospective_source"]["admitted_ordinary_count"])
        self.assertTrue(q["new_prospective_source"]["nearphone_unknown"])
        self.assertEqual(q["old_development"]["status"],"EXPOSED_NOT_INDEPENDENT")
        self.assertEqual(q["software"]["sdk_pr"],842)
        self.assertEqual(q["software"]["physical_ssc305"],"NOT_RUN")
        self.assertEqual(q["promotion"]["independent_b0_b1_comparison"],"NOT_RUN")
        self.assertFalse(q["promotion"]["product_promotion_allowed"])
        self.assertFalse(q["restrictions"]["no_work_codex"] is False)

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
            hashlib.sha1(b"blob "+str(len(raw)).encode()+bytes([0])+raw).hexdigest())
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
