#!/usr/bin/env python3
"""Bounded exact-original-source C-only process resource research profile.

Parent authenticates and prepares original WAV bytes, model and optional C ABI
using unchanged public artifacts. Child is a NEW native C executable process:
no Python interpreter, NumPy, FLAC arrays, TTS, ASR or model training.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import wave
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_original_pcm_c_parity as native

SCHEMA="english63-native-C-isolated-resource-profile-readback-v1"
MODEL_MODE="int8_c16"
EVENTS=5
SOURCE_CLIPS=63
SOURCE_SAMPLES=2101760
SOURCE_SECONDS=131.36
MAX_PROFILE_STDOUT_BYTES=8192
MAX_INPUT_FEED_CALLS=10000


def require(ok,code):
    if not ok:
        raise ValueError(code)


def sha_file(path):
    return native.quant.sha256_file(path)


def profile_source():
    return HERE/"english_native_resource_profile.c"


def source_digest():
    return sha_file(profile_source())


def authentic_wav_list(source_zip,review_json,out):
    """Read back original WAV bytes; never regenerate synthetic audio."""
    verified=native.quant.load_pinned_audio(source_zip,review_json)
    require(len(verified)==SOURCE_CLIPS and
            sum(len(pcm) for _,pcm in verified.values())==SOURCE_SAMPLES,
            "EN_NATIVE_PROFILE_ORIGINAL_SOURCE_SIZE")
    target=out/"original-WAV"
    target.mkdir(mode=0o700)
    files=[]
    with zipfile.ZipFile(source_zip) as archive:
        names=set(archive.namelist())
        for cell in sorted(verified):
            row,pcm=verified[cell]
            member="english-seven-voice-actual/capture/cells/"+cell+"/audio-16k.wav"
            require(member in names,"EN_NATIVE_PROFILE_ORIGINAL_WAV_MISSING")
            raw=archive.read(member)
            require(hashlib.sha256(raw).hexdigest()==row["wav_sha256"],
                    "EN_NATIVE_PROFILE_WAV_BYTES_CHANGED")
            with wave.open(io.BytesIO(raw),"rb") as reader:
                require((reader.getnchannels(),reader.getsampwidth(),
                         reader.getframerate(),reader.getcomptype())==
                        (1,2,16000,"NONE"),
                        "EN_NATIVE_PROFILE_WAV_GEOMETRY")
                frames=reader.getnframes()
                body=reader.readframes(frames)
            require(frames==len(pcm) and
                    hashlib.sha256(body).hexdigest()==row["pcm_sha256"],
                    "EN_NATIVE_PROFILE_ORIGINAL_PCM_NOT_MATCHED")
            path=target/(cell+".wav")
            require(not path.exists(),"EN_NATIVE_PROFILE_NEVER_OVERWRITE_SOURCE")
            with path.open("xb") as f:
                f.write(raw)
            require(sha_file(path)==row["wav_sha256"],
                    "EN_NATIVE_PROFILE_POST_COPY_SHA_CHANGED")
            files.append(str(path))
    manifest=out/"original-63-verified.list"
    manifest.write_text("\n".join(files)+"\n",encoding="utf-8")
    manifest.chmod(0o600)
    require(len(files)==SOURCE_CLIPS and
            all(Path(x).is_absolute() for x in files),
            "EN_NATIVE_PROFILE_MANIFEST_FAILURE")
    return manifest


def validate_readback(receipt):
    require(type(receipt) is dict and receipt.get("schema")==SCHEMA and
            receipt.get("status")==
              "MEASURED_HOST_NATIVE_KWS_CPU_RSS_NOT_TARGET_QUALIFICATION" and
            receipt.get("source_code_sha256")==
               hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and
            receipt.get("native_profiler_code_sha256")==source_digest() and
            receipt.get("model_archive_sha256")==native.quant.MODEL_SHA and
            receipt.get("original_source_audio_sha256")==native.quant.source.TTS_SHA256 and
            receipt.get("original_source_review_sha256")==native.quant.source.REVIEW_SHA256 and
            receipt.get("original_host_runtime_sha256")==native.host.ARCHIVE_SHA and
            receipt.get("model_mode")==MODEL_MODE and
            receipt.get("host_architecture")=="x86_64" and
            receipt.get("source_clips")==SOURCE_CLIPS and
            receipt.get("source_pcm_samples")==SOURCE_SAMPLES and
            receipt.get("original_english_events")==EVENTS and
            receipt.get("new_audio_tts_asr_or_training_calls")==0 and
            receipt.get("python_buffers_in_profiled_C_process") is False and
            receipt.get("real_ssc305_qualified") is False and
            receipt.get("real_human_frr")=="NOT_RUN" and
            receipt.get("real_household_far_hour")=="NOT_RUN" and
            receipt.get("model_promotion_allowed") is False,
            "EN_NATIVE_PROFILE_FALSE_MODEL_QUALIFICATION")
    sample=receipt.get("C_only_measurements")
    require(type(sample) is dict and sample.get("schema")==
             "english63-real-native-c-only-resource-profile-v1" and
            sample.get("status")=="HOST_NATIVE_KWS_RESOURCE_MEASURED_NOT_SSC305" and
            sample.get("source_wavs")==SOURCE_CLIPS and
            sample.get("source_samples")==SOURCE_SAMPLES and
            sample.get("actual_english_events")==EVENTS and
            sample.get("bad_events")==0 and
            sample.get("eof_events")==0 and
            type(sample.get("native_feed_calls")) is int and
            6000<=sample["native_feed_calls"]<=MAX_INPUT_FEED_CALLS and
            type(sample.get("deadline_misses_at_input_chunk_duration")) is int and
            0<=sample["deadline_misses_at_input_chunk_duration"]<=sample["native_feed_calls"] and
            sample.get("model_load_included_in_feed_timing") is False and
            sample.get("python_audio_buffers_in_profiled_process") is False and
            sample.get("ssc305_cpu_rss_qualified") is False and
            sample.get("human_far_frr_qualified") is False and
            sample.get("model_promotion_allowed") is False,
            "EN_NATIVE_PROFILE_UNAUTHENTIC_C_RESULTS")
    memory=sample.get("rss_kib")
    require(type(memory) is dict and
            all(type(memory.get(k)) is int and
                0<memory[k]<1048576 for k in (
                    "before_model_create","after_model_create",
                    "after_pcm_replay","process_peak")) and
            memory["process_peak"]>=max(memory["before_model_create"],
                                        memory["after_model_create"],
                                        memory["after_pcm_replay"]),
            "EN_NATIVE_PROFILE_MEMORY_DATA_INVALID")
    for k in ("host_native_wall_feed_ms","host_native_cpu_feed_ms"):
        v=sample.get(k)
        require(type(v) is dict and
                all(type(v.get(x)) in (int,float) and
                    0<=v[x]<600000 for x in ("p50","p95","p99","max","sum")) and
                0<=v["p50"]<=v["p95"]<=v["p99"]<=v["max"],
                "EN_NATIVE_PROFILE_LATENCY_INVALID")
    require(abs(sample.get("original_audio_seconds",0)-SOURCE_SECONDS)<1e-6 and
            type(sample.get("native_feed_cpu_seconds_per_audio_second")) in (int,float) and
            0<=sample["native_feed_cpu_seconds_per_audio_second"]<100,
            "EN_NATIVE_PROFILE_RATIO_UNSAFE")
    return True


def run(native_library,build_receipt,original_audio_zip,original_review,
        pretrained_model_archive,result):
    require(platform.system()=="Linux" and platform.machine()=="x86_64",
            "EN_NATIVE_PROFILE_HOST_MUST_BE_X64_LINUX")
    require(result.is_absolute() and result.parent.is_dir() and
            not result.parent.is_symlink() and not result.exists(),
            "EN_NATIVE_PROFILE_OUTPUT_CREATE_ONLY")
    native.verify_build_receipt(native_library,build_receipt)
    require(pretrained_model_archive.is_absolute() and
            pretrained_model_archive.is_file() and
            not pretrained_model_archive.is_symlink() and
            sha_file(pretrained_model_archive)==native.quant.MODEL_SHA,
            "EN_NATIVE_PROFILE_OFFICIAL_MODEL_DRIFT")
    require(profile_source().is_file() and not profile_source().is_symlink(),
            "EN_NATIVE_PROFILE_C_SOURCE_MISSING")
    with tempfile.TemporaryDirectory(prefix="english-native-only-resource-") as name:
        root=Path(name)
        manifests=authentic_wav_list(original_audio_zip,original_review,root)
        model_dir=root/"official-model"
        model_dir.mkdir(mode=0o700)
        parts=native.quant.extract_model(pretrained_model_archive,model_dir,
                                         MODEL_MODE)
        keyword=native.quant.official_keyword(parts,model_dir)
        runner=root/"native-kws-only-profile"
        args=[
            "gcc","-O2","-std=c11","-Wall","-Wextra","-Werror",
            str(profile_source()),"-I"+str(HERE),"-L"+str(native_library.parent),
            "-Wl,-rpath,"+str(native_library.parent),
            "-lkws_sherpa_pcm","-lm","-o",str(runner),
        ]
        subprocess.run(args,check=True,timeout=100)
        require(runner.is_file() and 1000<runner.stat().st_size<250000,
                "EN_NATIVE_PROFILE_C_EXE_BUILD_FAILED")
        cmd=[str(runner)]+[str(parts[n]) for n in
                             ("encoder","decoder","joiner","tokens")]+[
                                 str(keyword),str(manifests)]
        proc=subprocess.run(cmd,capture_output=True,text=True,check=False,
                            timeout=120)
        require(proc.returncode==0 and
                len(proc.stdout.encode())<MAX_PROFILE_STDOUT_BYTES and
                not proc.stderr.strip(),
                "EN_NATIVE_PROFILE_REAL_C_EXECUTION_FAILED:"+proc.stderr[-1000:])
        child=json.loads(proc.stdout)
        receipt={
            "schema":SCHEMA,
            "status":"MEASURED_HOST_NATIVE_KWS_CPU_RSS_NOT_TARGET_QUALIFICATION",
            "host_architecture":"x86_64",
            "source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_profiler_code_sha256":source_digest(),
            "source_compiled_executable_sha256":sha_file(runner),
            "native_runtime_library_sha256":sha_file(native_library),
            "model_archive_sha256":native.quant.MODEL_SHA,
            "original_source_audio_sha256":native.quant.source.TTS_SHA256,
            "original_source_review_sha256":native.quant.source.REVIEW_SHA256,
            "original_host_runtime_sha256":native.host.ARCHIVE_SHA,
            "model_mode":MODEL_MODE,
            "source_clips":SOURCE_CLIPS,
            "source_pcm_samples":SOURCE_SAMPLES,
            "model_file_bytes":{k:parts[k].stat().st_size
                                for k in ("encoder","decoder","joiner")},
            "original_english_events":EVENTS,
            "C_only_measurements":child,
            "new_audio_tts_asr_or_training_calls":0,
            "python_buffers_in_profiled_C_process":False,
            "real_ssc305_qualified":False,
            "real_human_frr":"NOT_RUN",
            "real_household_far_hour":"NOT_RUN",
            "model_promotion_allowed":False,
        }
        validate_readback(receipt)
        with result.open("x",encoding="utf-8") as f:
            json.dump(receipt,f,sort_keys=True,indent=2)
            f.write("\n")
        result.chmod(0o600)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ("native-library","build-receipt","original-audio-zip",
                  "original-review","pretrained-model-archive","output"):
        p.add_argument("--"+field,required=True,type=Path)
    args=p.parse_args()
    try:
        r=run(args.native_library,args.build_receipt,args.original_audio_zip,
              args.original_review,args.pretrained_model_archive,args.output)
        print(json.dumps({
            "status":r["status"],
            "host_rss_kib":r["C_only_measurements"]["rss_kib"],
            "host_wall_feed_p99_ms":r["C_only_measurements"]
                                   ["host_native_wall_feed_ms"]["p99"],
            "feed_deadline_misses":r["C_only_measurements"]
                                   ["deadline_misses_at_input_chunk_duration"],
            "model_promotion_allowed":False
        },sort_keys=True))
        return 0
    except (OSError,RuntimeError,ValueError,KeyError,zipfile.BadZipFile,
            subprocess.SubprocessError, json.JSONDecodeError) as exc:
        print("EN_NATIVE_RESOURCE_STOP:"+str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
