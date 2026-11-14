"""No-model source and receipt tests for x64 native C-only resource measurements."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_native_profile_once as p


def invented():
    child={
        "schema":"english63-real-native-c-only-resource-profile-v1",
        "status":"HOST_NATIVE_KWS_RESOURCE_MEASURED_NOT_SSC305",
        "source_wavs":63,
        "source_samples":2101760,
        "actual_english_events":5,
        "bad_events":0,
        "eof_events":0,
        "native_feed_calls":6600,
        "deadline_misses_at_input_chunk_duration":0,
        "rss_kib":{
            "before_model_create":3000,
            "after_model_create":46000,
            "after_pcm_replay":47000,
            "process_peak":48000
        },
        "host_native_wall_feed_ms":{
            "p50":0.12,"p95":0.4,"p99":1.2,"max":4.0,"sum":2500.0
        },
        "host_native_cpu_feed_ms":{
            "p50":0.10,"p95":0.35,"p99":1.1,"max":3.9,"sum":2200.0
        },
        "original_audio_seconds":131.36,
        "native_feed_cpu_seconds_per_audio_second":2200/1000/131.36,
        "model_load_included_in_feed_timing":False,
        "python_audio_buffers_in_profiled_process":False,
        "ssc305_cpu_rss_qualified":False,
        "human_far_frr_qualified":False,
        "model_promotion_allowed":False,
    }
    return {
        "schema":p.SCHEMA,
        "status":"MEASURED_HOST_NATIVE_KWS_CPU_RSS_NOT_TARGET_QUALIFICATION",
        "source_code_sha256":hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),
        "native_profiler_code_sha256":p.source_digest(),
        "model_archive_sha256":p.native.quant.MODEL_SHA,
        "original_source_audio_sha256":p.native.quant.source.TTS_SHA256,
        "original_source_review_sha256":p.native.quant.source.REVIEW_SHA256,
        "original_host_runtime_sha256":p.native.host.ARCHIVE_SHA,
        "model_mode":"int8_c16",
        "host_architecture":"x86_64",
        "source_clips":63,
        "source_pcm_samples":2101760,
        "original_english_events":5,
        "new_audio_tts_asr_or_training_calls":0,
        "python_buffers_in_profiled_C_process":False,
        "real_ssc305_qualified":False,
        "real_human_frr":"NOT_RUN",
        "real_household_far_hour":"NOT_RUN",
        "model_promotion_allowed":False,
        "C_only_measurements":child,
    }


class EnglishNativeResourceSourceCases(unittest.TestCase):
    def test_actual_c_profiler_compiles_without_model_or_audio(self):
        # Syntax verification does not load Sherpa, model, WAVs or Python deps.
        cmd=["gcc","-std=c11","-O2","-Wall","-Wextra","-Werror",
             "-fsyntax-only",str(p.profile_source()),
             "-I"+str(HERE.parent)]
        result=subprocess.run(cmd,capture_output=True,text=True,check=False,
                              timeout=30)
        self.assertEqual(result.returncode,0,result.stderr[-2000:])
        self.assertEqual(p.MODEL_MODE,"int8_c16")
        self.assertEqual(p.SOURCE_SAMPLES,2101760)
        self.assertEqual(p.SOURCE_CLIPS,63)

    def test_safe_model_free_receipt_is_not_ssc305(self):
        report=invented()
        self.assertTrue(p.validate_readback(report))
        self.assertFalse(report["real_ssc305_qualified"])
        self.assertFalse(report["model_promotion_allowed"])
        self.assertFalse(report["python_buffers_in_profiled_C_process"])

    def test_fake_model_quality_host_memory_and_source_data_denied(self):
        data=invented()
        changes=(
            lambda x:x.__setitem__("status","PRODUCTION"),
            lambda x:x.__setitem__("source_code_sha256","0"*64),
            lambda x:x.__setitem__("native_profiler_code_sha256","0"*64),
            lambda x:x.__setitem__("model_archive_sha256","0"*64),
            lambda x:x.__setitem__("original_source_audio_sha256","0"*64),
            lambda x:x.__setitem__("original_source_review_sha256","0"*64),
            lambda x:x.__setitem__("original_host_runtime_sha256","0"*64),
            lambda x:x.__setitem__("model_mode","fp32_c16"),
            lambda x:x.__setitem__("host_architecture","armv7"),
            lambda x:x.__setitem__("source_clips",62),
            lambda x:x.__setitem__("source_pcm_samples",2101759),
            lambda x:x.__setitem__("original_english_events",9),
            lambda x:x.__setitem__("new_audio_tts_asr_or_training_calls",1),
            lambda x:x.__setitem__("python_buffers_in_profiled_C_process",True),
            lambda x:x.__setitem__("real_ssc305_qualified",True),
            lambda x:x.__setitem__("real_human_frr",0.0),
            lambda x:x.__setitem__("real_household_far_hour",0.0),
            lambda x:x.__setitem__("model_promotion_allowed",True),
            lambda x:x["C_only_measurements"].__setitem__("source_wavs",64),
            lambda x:x["C_only_measurements"].__setitem__("source_samples",2101761),
            lambda x:x["C_only_measurements"].__setitem__("actual_english_events",9),
            lambda x:x["C_only_measurements"].__setitem__("bad_events",1),
            lambda x:x["C_only_measurements"].__setitem__("eof_events",1),
            lambda x:x["C_only_measurements"].__setitem__("native_feed_calls",100000),
            lambda x:x["C_only_measurements"].__setitem__(
                "deadline_misses_at_input_chunk_duration",6601),
            lambda x:x["C_only_measurements"]["rss_kib"].__setitem__(
                "process_peak",3000),
            lambda x:x["C_only_measurements"]["host_native_wall_feed_ms"].__setitem__(
                "p99",99.0),
            lambda x:x["C_only_measurements"].__setitem__(
                "ssc305_cpu_rss_qualified",True),
        )
        for change in changes:
            bad=copy.deepcopy(data)
            change(bad)
            with self.assertRaises(ValueError):
                p.validate_readback(bad)


if __name__=="__main__":
    unittest.main()
