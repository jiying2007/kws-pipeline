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
                # CTest must execute both real tests in Release/-DNDEBUG;
                # merely seeing an executable with exit status 0 is not enough.
                inventory=subprocess.run(
                    ["ctest","--test-dir",str(build),"-N"],
                    check=True,env=env,capture_output=True,text=True,timeout=30)
                self.assertIn("Total Tests: 2",inventory.stdout)
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
        self.assertEqual(q["new_prospective_source"]["quality_admission"],"PASS_SYNTHETIC_MACHINE_QA_ONLY")
        self.assertEqual(q["new_prospective_source"]["admitted_positive_count"],6)
        self.assertEqual(q["new_prospective_source"]["admitted_ordinary_count"],8)
        self.assertTrue(q["new_prospective_source"]["nearphone_unknown"])
        self.assertEqual(q["old_development"]["status"],"EXPOSED_NOT_INDEPENDENT")
        self.assertEqual(q["software"]["sdk_pr"],842)
        self.assertEqual(q["software"]["physical_ssc305"],"NOT_RUN")
        self.assertEqual(q["promotion"]["independent_b0_b1_comparison"],
                         "OBSERVED_RESEARCH_DIAGNOSTIC_INCONCLUSIVE")
        paired=q["paired_development_comparison"]
        self.assertEqual(paired["workflow_run_id"],44640947580)
        self.assertEqual(paired["artifact_id"],14585918123)
        self.assertEqual(paired["machine_qualified_original_positive"],6)
        self.assertEqual(paired["machine_qualified_original_ordinary"],8)
        self.assertEqual(paired["unqualified_unknown_excluded"],18)
        self.assertEqual((paired["original"]["B0_positive_hits"],
                          paired["original"]["B1_positive_hits"]),(5,6))
        self.assertEqual((paired["half_gain"]["B0_positive_hits"],
                          paired["half_gain"]["B1_positive_hits"]),(1,1))
        self.assertEqual((paired["snr10"]["B0_positive_hits"],
                          paired["snr10"]["B1_positive_hits"]),(5,5))
        self.assertEqual(sum(paired["original"][k] for k in
                   ("B0_ordinary_false_or_extra","B1_ordinary_false_or_extra")),0)
        self.assertTrue(paired["derived_conditions_are_not_independent_denominators"])
        self.assertFalse(paired["model_selection_allowed"])
        self.assertFalse(paired["product_promotion_allowed"])
        self.assertFalse(q["promotion"]["product_promotion_allowed"])
        self.assertTrue(q["restrictions"]["no_work_codex"])

    def test_v6_saved_original_ctc_forensics_pins_and_no_promotion(self):
        """Saved true 14-clip PCM/logit finding; no model is executed in CI."""
        evidence=json.loads((HERE/"V6_QUALITY_EVIDENCE.json").read_bytes())
        result=evidence["acoustic_forensics"]
        self.assertEqual(result["status"],"SAVED_EVIDENCE_ONLY_NO_NEW_FORWARDS")
        self.assertEqual(result["paired_run_id"],44640947580)
        self.assertEqual(result["paired_result_artifact_id"],14585918123)
        self.assertEqual(result["positive_original_count"],6)
        self.assertEqual(result["ordinary_original_count"],8)
        self.assertEqual(result["positive_source_groups"],4)
        self.assertEqual(result["total_source_groups_in_comparison"],8)
        self.assertEqual(result["unqualified_excluded_count"],18)
        self.assertEqual(result["initial_25ms_model_frame_argmax_hey"],
                         {"B0":{"positive":6,"ordinary":8},
                          "B1":{"positive":6,"ordinary":8}})
        self.assertEqual(result["half_gain_positive_misses_without_meh"],
                         {"B0":5,"B1":5})
        self.assertLess(result["max_first_25ms_to_whole_rms_ratio"],0.05)
        self.assertFalse(result["half_gain_frame_logits_preserved"])
        self.assertFalse(result["new_model_promotion_allowed"])
        self.assertFalse(evidence["promotion"]["product_promotion_allowed"])
        module=importlib.util.spec_from_file_location(
            "en_v6_saved_forensics",HERE/"audit_saved_v6_ctc.py")
        audit=importlib.util.module_from_spec(module)
        module.loader.exec_module(audit)
        self.assertEqual(audit.SOURCE_SHA,
                         result["original_source_zip_sha256"])
        self.assertEqual(audit.PAIR_SHA,
                         result["paired_result_zip_sha256"])
        # Pure invented frame values: independent greedy CTC collapse sanity.
        labels,collapsed=audit.greedy([[0.,1.,0.,0.],
                                       [0.,1.,0.,0.],
                                       [2.,0.,0.,0.],
                                       [0.,0.,3.,0.],
                                       [0.,0.,0.,4.]])
        self.assertEqual(labels,[1,1,0,2,3])
        self.assertEqual(collapsed,[1,2,3])
        with self.assertRaises(ValueError):
            audit.npy(b"invalid")
        doc=(ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text()
        self.assertIn("14/14",doc)
        self.assertIn("5/6",doc)
        self.assertIn("without performing any additional model forward",doc)

    def test_v6_r2_trained_b2_research_nonpromotion_and_suffix_followup(self):
        q=json.loads((HERE/"V6_QUALITY_EVIDENCE.json").read_bytes())
        r=q["experimental_followup"]
        b=r["B2"]
        self.assertEqual(b["status"],"ACTUAL_TRAINED_RESEARCH_NO_PROMOTION")
        self.assertEqual(b["run_id"],44648437304)
        self.assertEqual(b["artifact_id"],14589303986)
        self.assertEqual(b["actual_new_optimizer_updates"],256)
        self.assertEqual(b["train_source_groups"],35)
        self.assertEqual(b["train_only_probe_count"],12)
        self.assertEqual(b["train_only_initial_argmax_hey"],12)
        self.assertEqual(b["train_only_prefix_blank_argmax_fraction"],0.991071)
        self.assertEqual(b["c11_numeric_parity"],"NOT_RUN")
        self.assertFalse(b["product_promotion_allowed"])
        e=r["reproduced_exposed_dev_diagnostic"]
        self.assertEqual(e["positive_originals"],6)
        self.assertEqual(e["ordinary_originals"],8)
        self.assertEqual(e["original_positive_hits"],{"B0":5,"B1":6,"B2":6})
        self.assertEqual(e["half_gain_positive_hits"],{"B0":1,"B1":1,"B2":3})
        self.assertEqual(e["half_gain_ordinary_false_events"],{"B0":0,"B1":0,"B2":1})
        self.assertEqual(e["one_second_prefix_new_hey_after_first_0p8s_positive"],
                         {"B0":0,"B1":0,"B2":0})
        self.assertTrue(e["derived_conditions_not_independent"])
        self.assertFalse(e["model_selection_allowed"])
        b3=r["B3"]
        self.assertEqual(b3["status"],"ACTUAL_TRAINED_SUFFIX_CTC_RESEARCH_ONLY")
        self.assertEqual(b3["protected_main_run"],44651053897)
        self.assertEqual(b3["actual_new_optimizer_updates"],256)
        self.assertEqual(b3["artifact_id"],14590093798)
        self.assertEqual(b3["actual_pt_sha256"],"682e2689e82aa91d850516c06b2be61efc64731cb1dee3b70a7d2a8e01925a46")
        self.assertEqual(b3["actual_pt_member"],"models/mero43-onset256.pt")
        self.assertEqual(b3["in_sample_first_frame_blank"],12)
        self.assertEqual(b3["in_sample_first_frame_hey"],0)
        self.assertEqual(b3["fresh_independent_eval"],"NOT_RUN")
        self.assertEqual(b3["c11_parity"],"PASS_REAL_ORIGINAL_PCM_28_CASES_PT_GCC_CLANG")
        self.assertEqual(b3["arm32_qemu_parity"],"PASS_REAL_CORTEX_A32_QEMU_28_CASES")
        parity=b3["b3_software_parity"]
        self.assertEqual(parity["actions_run"],44653612830)
        self.assertEqual(parity["run_main_sha"],"acbd6b2bab42374d2125e93537e040bc32b9c1f7")
        self.assertEqual(parity["artifact_id"],14593110591)
        self.assertEqual(parity["artifact_sha256"],
                         "f288798c50a85cce0a2fef22b7006d71237f53afec7dadbdbbe0e610fd424036")
        self.assertEqual(parity["original_audio_count"],14)
        self.assertEqual(parity["total_case_count"],28)
        self.assertEqual(parity["observed_max_abs_logit"],0.000091552734375)
        self.assertTrue(parity["full_ctc_token_and_event_clock_parity"])
        self.assertTrue(parity["reused_previously_exposed_synthetic_dev_only"])
        self.assertFalse(parity["product_promotion_allowed"])
        self.assertEqual(parity["physical_ssc305"],"NOT_RUN")
        self.assertEqual(parity["real_far_per_hour"],"NOT_MEASURED")
        self.assertIn("B3 PyTorch", (ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text())
        self.assertIn("software numerical parity", (ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text())
        self.assertFalse(b3["product_promotion_allowed"])
        continuous=r["B3_continuous_exposed_dev"]
        self.assertEqual(continuous["actions_run"],44658362723)
        self.assertEqual(continuous["artifact_id"],14594762757)
        self.assertEqual(continuous["artifact_zip_sha256"],
                         "045c022a0d57a8750d289da10d307ceb50ad5e512120f7f31209bd0df62151c1")
        self.assertEqual(continuous["main_sha"],"4995bfe0813f4bd6b65fa75a9c6764b14d64bd4e")
        self.assertEqual(continuous["originals_reused"],14)
        self.assertEqual(continuous["unknown_excluded"],18)
        self.assertFalse(continuous["clip_state_reset"])
        self.assertEqual(continuous["observed"]["mixed"]["positive_segments_with_event"],5)
        self.assertEqual(continuous["observed"]["reverse"]["positive_segments_with_event"],6)
        self.assertEqual(continuous["observed"]["ordinary_only"]["total_events"],0)
        self.assertEqual(continuous["observed"]["silence_5s"]["total_events"],0)
        self.assertEqual(continuous["only_order_sensitive_known_positive"],
                         "merov6dev8-490-mero-slow")
        self.assertTrue(continuous["same_positive_detected_when_reversed"])
        self.assertTrue(continuous["same_positive_missed_in_mixed"])
        self.assertTrue(continuous["cross_compiler_event_clock_equal"])
        self.assertFalse(continuous["independent_development_quality"])
        self.assertTrue(continuous["synthetic_concatenation_cannot_estimate_far_per_hour"])
        self.assertEqual(continuous["physical_ssc305"],"NOT_RUN")
        self.assertFalse(continuous["model_selection_allowed"])
        self.assertFalse(continuous["product_promotion_allowed"])
        gain=r["B3_gain_context_exposed_dev"]
        self.assertEqual(gain["run_id"],44662162126)
        self.assertEqual(gain["main_sha"],"db0c4b699b16fcd765b430702a8f786715529eb5")
        self.assertEqual(gain["artifact_id"],14595464491)
        self.assertEqual(gain["artifact_zip_sha256"],
                         "0ba7cd287afcedbbd7625b6987cc89dbeb2c8717c1506b905e8a01671e9ea3cb")
        self.assertEqual(gain["original_source_count"],14)
        self.assertEqual(gain["excluded_unknown"],18)
        self.assertEqual(gain["derived_gain_conditions"],[1,0.75,0.5])
        self.assertEqual([x["positive_hits"] for x in gain["gain_results"]],[5,4,1])
        self.assertEqual([x["positive_without_meh_argmax"] for x in gain["gain_results"]],[0,2,5])
        self.assertEqual([x["ordinary_any_events"] for x in gain["gain_results"]],[0,0,0])
        self.assertEqual(gain["source_490_gain_1_frame_argmax_hey"],0)
        self.assertEqual(gain["source_490_gain_075_event_count"],1)
        self.assertEqual(len(gain["controlled_predecessor_cases"]),4)
        self.assertTrue(all(x["target_events"]==0
                            and x["hey_argmax_frames"]==0
                            for x in gain["controlled_predecessor_cases"]))
        self.assertEqual(gain["max_gap_host_cli_seconds"],62.5)
        self.assertTrue(gain["unsafe_to_infer_6_of_6_reverse_are_single_phrase"])
        self.assertFalse(gain["model_selection_allowed"])
        self.assertFalse(gain["decoder_policy_selection_allowed"])
        self.assertFalse(gain["product_promotion_allowed"])
        self.assertEqual(gain["natural_far_per_hour"],"NOT_MEASURED")
        self.assertEqual(gain["physical_ssc305"],"NOT_RUN")
        self.assertIn("Real B3 C11 gain and predecessor diagnostics",
                      (ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text())

        self.assertIn("Real B3 C11 continuous-state diagnostic",
                      (ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text())

        report=r["B3_exposed_mechanism_diagnostic"]
        self.assertEqual(report["full_clip_positive_hits_original"],
                         {"B0":5,"B1":6,"B2":6,"B3":5})
        self.assertEqual(report["full_clip_positive_hits_half_gain"],
                         {"B0":1,"B1":1,"B2":3,"B3":1})
        self.assertEqual(report["ordinary_false_events_half_gain"],
                         {"B0":0,"B1":0,"B2":1,"B3":0})
        self.assertEqual(report["original_first_frame_argmax_hey_counts_of_14"],
                         {"B0":14,"B1":14,"B2":14,"B3":0})
        self.assertEqual(report["one_second_silence_prefix_positive_new_hey_after_0p8sec_of_6"],
                         {"B0":0,"B1":0,"B2":0,"B3":5})
        self.assertTrue(report["derived_prefix_does_not_increase_independent_denominator"])
        self.assertFalse(report["model_selection_allowed"])
        self.assertEqual(r["next_experiment"]["completed_data_pr"],275)
        self.assertEqual(r["next_experiment"]["recovery_data_pr"],276)
        self.assertFalse(r["next_experiment"]["model_promotion_allowed"])
        self.assertFalse(q["promotion"]["product_promotion_allowed"])

    def test_v6_real_b5_dev504_frozen_negative_quality_evidence(self):
        q=json.loads((ROOT/"research/english_v5_mero/V6_QUALITY_EVIDENCE.json").read_text())
        e=q["experimental_followup"]["B4_B5_DEV504_frozen_research"]
        self.assertEqual(e["status"],
            "ACTUAL_FROZEN_DEV504_PAIRED_B5_EARLY_HEY_REJECTED_HALFGAIN_NO_PRODUCT_PROMOTION")
        self.assertEqual(e["data_pr_chain"],[286,287,288,289,290,291])
        self.assertEqual(e["trained_b5"]["actual_optimizer_updates"],384)
        self.assertTrue(e["trained_b5"]["control_tensor_matches_B4_half"])
        self.assertEqual(e["source"]["actual_observations"],160)
        self.assertEqual((e["source"]["qualified_positive"],
                          e["source"]["qualified_ordinary"],
                          e["source"]["unknown_excluded"],
                          e["source"]["qualified_development_source_groups"]), (5,8,19,3))
        self.assertFalse(e["source"]["human_gold"])
        receipt=e["frozen_quality"]
        self.assertEqual(receipt["run"],44700786677)
        self.assertEqual(receipt["artifact_id"],14611963808)
        self.assertEqual(receipt["artifact_zip_size_bytes"],39379)
        self.assertEqual(receipt["artifact_zip_sha256"],
          "67267fcdf78535a75854645de60e7ad662aaa5016788b62e9901c1f073e9b016")
        self.assertEqual(receipt["actual_model_forward_conditions"],117)
        self.assertEqual([row["B4_half_hits"] for row in receipt["gains"]],[5,5,3])
        self.assertEqual([row["B5_control_hits"] for row in receipt["gains"]],[5,5,3])
        self.assertEqual([row["B5_early_hey_hits"] for row in receipt["gains"]],[5,5,2])
        self.assertTrue(all(row["ordinary_any_events_each_model"]==0
                            and row["ordinary_n"]==8 for row in receipt["gains"]))
        self.assertTrue(receipt["gain_variants_are_derived_not_independent"])
        self.assertTrue(receipt["B5_early_hey_worse_than_B4_half_at_half_gain"])
        self.assertEqual(receipt["B5_half_gain_positive_without_meh_argmax"],1)
        self.assertEqual(e["previous_recovery"]["failed_run"],44698504736)
        self.assertFalse(e["decision"]["model_selection_allowed"])
        self.assertFalse(e["decision"]["product_promotion_allowed"])
        self.assertTrue(e["decision"]["B0_c11_sdk_remains_frozen"])
        self.assertEqual(e["qualification"]["physical_ssc305"],"NOT_RUN")
        self.assertEqual(e["qualification"]["natural_far_per_hour"],"NOT_MEASURED")
        documentation=(ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text()
        self.assertIn("Real B4/B5 post-training DEV504 frozen-model comparison",documentation)
        self.assertIn("reject B5-EarlyHEY for promotion",documentation)

    def test_v6_b6_real_posttrain_dev512_mixed_gain_and_ordinary_false_wake(self):
        q=json.loads((ROOT/"research/english_v5_mero/V6_QUALITY_EVIDENCE.json").read_text())
        r=q["experimental_followup"]["B4_B6_DEV512_frozen_research"]
        self.assertEqual(r["status"],
            "REAL_126_POST_B6_DEV512_PAIRED_DIAGNOSTIC_B6_UNSAFE_FOR_PROMOTION")
        self.assertEqual(r["data_pr_chain"],[293,294,295,296,297])
        t=r["trained_B6"]
        self.assertEqual(t["run"],44704535156)
        self.assertEqual(t["actual_train_updates"],384)
        self.assertEqual(t["control_replay"],"PASS_EXACT_B4_HALF_TENSOR_REPLAY")
        self.assertTrue(t["optimization_is_train_only"])
        self.assertEqual(t["development_gradients"],0)
        s=r["source"]
        self.assertEqual(s["new_speaker_indices"],[512,519])
        self.assertEqual(s["original_pcm_count"],32)
        self.assertEqual(s["native_blind_completed_observations"],160)
        self.assertEqual((s["machine_positive_originals"],s["machine_ordinary_originals"],
                          s["unknown_excluded_originals"],
                          s["qualified_development_source_groups"]),(6,8,18,5))
        self.assertFalse(s["human_gold"])
        a=r["frozen_quality"]
        self.assertEqual(a["run"],44710802470)
        self.assertEqual(a["original_result_artifact_id"],14616493798)
        self.assertEqual(a["original_result_zip_bytes"],41502)
        self.assertEqual(a["original_result_zip_sha256"],
            "b900473e57a5b53793a42020838371c1c12f4ee70d73480f2f68c03aab37801f")
        self.assertEqual(a["actual_model_forward_conditions"],126)
        self.assertTrue(a["B4_half_B6_control_bit_identical_weights"])
        self.assertTrue(a["gains_are_derived_and_do_not_increase_denominator"])
        self.assertEqual([x["B4_half_hits"] for x in a["gains"]],[3,4,1])
        self.assertEqual([x["B6_control_hits"] for x in a["gains"]],[3,4,1])
        self.assertEqual([x["B6_consistency_hits"] for x in a["gains"]],[5,3,1])
        self.assertEqual([x["B6_consistency_ordinary_events"] for x in a["gains"]],[1,0,0])
        self.assertEqual([x["B6_positive_missing_MEH_winner"] for x in a["gains"]],[0,0,1])
        self.assertEqual(a["B6_normal_gain_ordinary_false_event_on_synthetic_negative"],1)
        self.assertTrue(a["B6_half_gain_not_better_than_B4"])
        self.assertTrue(a["B6_075_gain_regressed_vs_B4"])
        self.assertFalse(r["decision"]["product_promotion_allowed"])
        self.assertTrue(r["decision"]["B0_only_authenticated_installed_SSC305_C11_SDK"])
        self.assertEqual(r["next_evidence"]["physical_SSC305"],"NOT_RUN")
        self.assertEqual(r["next_evidence"]["continuous_natural_far_per_hour"],"NOT_MEASURED")
        self.assertEqual(r["next_evidence"]["independent_LOCKED"],"NOT_RUN")
        forensic=r["next_evidence"]
        self.assertEqual(forensic["forensics_data_pr"],298)
        self.assertEqual(forensic["forensics_run_id"],44718953294)
        self.assertEqual(forensic["forensics_artifact_id"],14619669982)
        self.assertEqual(forensic["forensics_zip_sha256"],
            "25351b75c0bfe4248ce4d62774dfa1ab207afeeceb5422e935116e013bfc791e")
        self.assertEqual(forensic["forensics_status"],
            "PASS_IMMUTABLE_B6_126_CASE_CTC_FORENSICS_RESEARCH_ONLY")
        self.assertEqual(forensic["synthetic_ordinary_false_wake_source"],
            "merov6b6dev8-517-ordinary-time")
        self.assertEqual(forensic["positive_discordant_originals_by_gain"],
            {"1.0":2,"0.75":1,"0.5":0})
        self.assertEqual(forensic["actual_new_model_forwards"],0)
        self.assertFalse(forensic["source_level_ctc_path_causal_explanation_proven"])
        doc=(ROOT/"docs/ENGLISH_V6_MERO_EXECUTION.md").read_text()
        self.assertIn("Real B6 gain-consistency training and genuinely new DEV512 test",doc)
        self.assertIn("Reject automatic B6 model promotion",doc)

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
