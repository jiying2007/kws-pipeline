#!/usr/bin/env python3
"""Compare real optional Sherpa C ABI with original English63 INT8 no-tail events.

Research-only. Uses original pinned Qwen PCM and official pretrained Zh-En
model; no synthetic padding, training, tuning or target-board claims.
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[1]
ROOT=HERE.parents[2]
sys.path.insert(0,str(RESEARCH/"english63_weak_pilot"))
import pretrained_quant_events as quant
sys.path.insert(0,str(HERE.parent))
import build_official_x64 as host

POLICY="english63-native-c-mero-unpadded-event-parity-v1"
REFERENCE_ARTIFACT_SHA="4f006138eb9dcc6f856ef0e61da8dcf10034904e4e683255db302775a19a62f6"
REFERENCE_JSON_SHA="d5cfbd295299ff8c0ee7524538e44c9755ad25a943f5178455816bc691dbe521"
KEYWORD="HEY_MERO"
PATS={"fixed320":(320,),"network1600":(1600,),"uneven":(1,159,321,17,640)}
EVENT_MAX=16


def require(flag, why):
    if not flag:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify_saved_baseline(archive):
    require(archive.is_absolute() and archive.is_file() and
            not archive.is_symlink() and archive.stat().st_size<180000 and
            quant.sha256_file(archive)==REFERENCE_ARTIFACT_SHA,
            "EN_C_BASELINE_ZIP_DRIFT")
    with zipfile.ZipFile(archive) as src:
        require(set(src.namelist())=={"english63-unpadded-events.json"},
                "EN_C_BASELINE_MEMBERS")
        raw=src.read("english63-unpadded-events.json")
    require(sha(raw)==REFERENCE_JSON_SHA, "EN_C_BASELINE_JSON_DRIFT")
    value=json.loads(raw)
    require(value.get("schema")==
            "english63-original-synthetic-unpadded-online-event-replay-v1" and
            value.get("source_tts_sha256")==quant.source.TTS_SHA256 and
            value.get("source_review_sha256")==quant.source.REVIEW_SHA256 and
            value.get("model_sha256")==quant.MODEL_SHA and
            value.get("source_clips")==63 and
            value.get("model_promotion_allowed") is False,
            "EN_C_BASELINE_POLICY_DRIFT")
    data=value.get("by_mode",{}).get("int8_c16")
    require(type(data) is dict and data.get("raw_pcm_events")==5 and
            data.get("eof_only_events")==0 and
            data.get("original_with_artificial_tail_events")==9 and
            data.get("original_after_real_pcm_events")==4 and
            type(data.get("records")) is list and len(data["records"])==63,
            "EN_C_BASELINE_MODEL_ROSTER")
    return {row["cell_id"]:row for row in data["records"]}


def verify_build_receipt(native_library, receipt_path):
    require(native_library.is_absolute() and native_library.is_file() and
            not native_library.is_symlink() and
            receipt_path.is_absolute() and receipt_path.is_file() and
            not receipt_path.is_symlink(), "EN_C_NATIVE_PATH")
    receipt=json.loads(receipt_path.read_text())
    require(receipt.get("stub_only") is False and
            receipt.get("speech_inference_tested") is False and
            receipt.get("library_sha256")==quant.sha256_file(native_library),
            "EN_C_REAL_LIBRARY_RECEIPT")
    src_root=RESEARCH/"sherpa_pcm"
    lock=json.loads((src_root/"dependencies.lock.json").read_text())
    source_files=receipt.get("files")
    require(type(source_files) is dict and
            set(source_files)=={
                "pcm_kws.cc","pcm_kws.h","dependencies.lock.json",
                "vendor/sherpa-onnx/c-api/c-api.h"
            } and
            all(quant.sha256_file(src_root/name)==digest for name,digest in source_files.items()),
            "EN_C_ADAPTER_SOURCE_CHANGED")
    require(receipt.get("runtime_profile")==
                "original-sherpa-v1.13.8-x64-release" and
            receipt.get("runtime_archive_sha256")==host.ARCHIVE_SHA and
            receipt.get("runtime_archive_bytes")==host.ARCHIVE_BYTES and
            type(receipt.get("runtime_archive_path")) is str,
            "EN_C_NATIVE_RUNTIME_PROFILE_DRIFT")
    archive=Path(receipt["runtime_archive_path"])
    host.verify_original_archive(archive)
    expected={name:{"sha256":digest,"bytes":size}
              for name,(size,digest) in host.LIBRARIES.items()}
    found=receipt.get("runtime_libraries")
    require(type(found) is list and len(found)==2 and
            {v.get("filename") for v in found}==set(expected),
            "EN_C_RUNTIME_LIBRARY_ROSTER")
    for value in found:
        name=value["filename"]
        path=Path(value.get("path",""))
        require(value.get("sha256")==expected[name]["sha256"] and
                value.get("bytes")==expected[name]["bytes"] and
                path.is_file() and not path.is_symlink() and
                path.resolve()==(native_library.parent/"runtime"/name).resolve() and
                path.stat().st_size==value["bytes"] and
                quant.sha256_file(path)==value["sha256"],
                "EN_C_RUNTIME_BINARY_CHANGED")
    return receipt


class Files(C.Structure):
    _fields_=[(x,C.c_char_p) for x in ("encoder","decoder","joiner","tokens","keywords")]


class Event(C.Structure):
    _fields_=[
        ("keyword",C.c_char_p),
        ("keyword_id",C.c_int),
        ("available_samples",C.c_uint64),
        ("eof_flush",C.c_int)
    ]


CALLBACK=C.CFUNCTYPE(None,C.c_void_p,C.POINTER(Event))


def load_c_api(path):
    lib=C.CDLL(str(path))
    p=C.c_void_p
    e=C.c_char_p
    n=C.c_size_t
    lib.pcm_kws_create.argtypes=[C.POINTER(Files),CALLBACK,p,C.POINTER(p),e,n]
    lib.pcm_kws_create.restype=C.c_int
    lib.pcm_kws_feed.argtypes=[p,C.POINTER(C.c_int16),n,e,n]
    lib.pcm_kws_feed.restype=C.c_int
    for name in ("pcm_kws_finish","pcm_kws_reset"):
        fn=getattr(lib,name)
        fn.argtypes=[p,e,n]
        fn.restype=C.c_int
    lib.pcm_kws_destroy.argtypes=[p]
    lib.pcm_kws_destroy.restype=None
    return lib


def run(native_library,build_receipt,tts_zip,review_json,model_archive,baseline_archive,output):
    import numpy as np
    require(output.is_absolute() and output.parent.is_dir() and
            not output.exists() and not output.parent.is_symlink(),
            "EN_C_OUTPUT_CREATE_ONLY")
    receipt=verify_build_receipt(native_library,build_receipt)
    baseline=verify_saved_baseline(baseline_archive)
    source=quant.load_pinned_audio(tts_zip,review_json)
    require(len(source)==len(baseline)==63 and set(source)==set(baseline),
            "EN_C_AUDIO_OR_BASELINE_ROSTER")
    require(model_archive.is_absolute() and model_archive.is_file() and
            not model_archive.is_symlink() and
            quant.sha256_file(model_archive)==quant.MODEL_SHA,
            "EN_C_MODEL_ARCHIVE_SHA")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="en-mero-native-") as d:
        dest=Path(d)
        parts=quant.extract_model(model_archive,dest,"int8_c16")
        keyword=quant.official_keyword(parts,dest)
        files=Files(*(str(parts[name]).encode() for name in
                      ("encoder","decoder","joiner","tokens")),
                    str(keyword).encode())
        lib=load_c_api(native_library)
        methods={}
        for mode, pattern in PATS.items():
            events=[]
            @CALLBACK
            def callback(_user, event):
                item=event.contents
                events.append((item.keyword.decode() if item.keyword else "",
                               item.keyword_id,int(item.available_samples),bool(item.eof_flush)))
            handle=C.c_void_p()
            error=C.create_string_buffer(512)
            require(lib.pcm_kws_create(C.byref(files),callback,None,
                                      C.byref(handle),error,len(error))==0 and
                    bool(handle.value), "EN_C_NATIVE_CREATE:"+repr(error.value))
            records=[]
            wall=0.0
            try:
                for cell in sorted(source):
                    row, samples=source[cell]
                    old=baseline[cell]
                    require(old["pcm_sha256"]==row["pcm_sha256"] and
                            old["requested_text"]==row["requested_text"] and
                            old["source_pcm_samples"]==len(samples) and
                            old["training_admitted"] is False,
                            "EN_C_SOURCE_ROLE_IDENTITY")
                    integers=np.rint(samples*32768.0).astype("<i2")
                    require(sha(integers.tobytes())==row["pcm_sha256"],
                            "EN_C_SOURCE_PCM_RECONSTRUCT")
                    events.clear()
                    require(lib.pcm_kws_reset(handle,error,len(error))==0,
                            "EN_C_NATIVE_RESET:"+repr(error.value))
                    start=time.perf_counter()
                    pos=0
                    i=0
                    while pos<len(integers):
                        count=min(pattern[i%len(pattern)],len(integers)-pos)
                        ptr=C.cast(integers.ctypes.data+pos*2,C.POINTER(C.c_int16))
                        rc=lib.pcm_kws_feed(handle,ptr,count,error,len(error))
                        require(rc==0,"EN_C_NATIVE_FEED:"+repr(error.value))
                        pos+=count
                        i+=1
                    require(lib.pcm_kws_finish(handle,error,len(error))==0,
                            "EN_C_NATIVE_FINISH:"+repr(error.value))
                    wall+=time.perf_counter()-start
                    cleaned=[]
                    for word,id,avail,eof in events:
                        require(word==KEYWORD and id==3 and
                                0<=avail<=len(integers),
                                "EN_C_UNKNOWN_ID_OR_EVENT")
                        cleaned.append({"keyword":word,"keyword_id":id,
                                        "available_samples":avail,"eof_flush":eof})
                    records.append({
                        "cell_id":cell,"requested_text":row["requested_text"],
                        "source_family_id":row["source_family_id"],
                        "pcm_sha256":row["pcm_sha256"],"events":cleaned,
                        "baseline_python_pcm_events":sum(
                            e["phase"]=="source_pcm" for e in old["events"]),
                        "baseline_python_eof_events":sum(
                            e["phase"]=="eof_no_padding" for e in old["events"]),
                        "human_gold":False,"training_admitted":False
                    })
            finally:
                lib.pcm_kws_destroy(handle)
            require(len(records)==63,"EN_C_RECORD_COUNT")
            changes=[r["cell_id"] for r in records if len(r["events"]) !=
                     r["baseline_python_pcm_events"]+r["baseline_python_eof_events"]]
            methods[mode]={
                "records":records,"changed_event_count_cells":changes,
                "native_event_count":sum(len(r["events"]) for r in records),
                "native_eof_flush_event_count":sum(e["eof_flush"] for r in records for e in r["events"]),
                "baseline_python_no_tail_events":5,
                "host_loop_wall_seconds":round(wall,6),
                "model_promotion_allowed":False,
            }
    result={
        "schema":POLICY,
        "status":"MEASURED_REAL_C_API_EVENT_COMPARISON_RESEARCH_ONLY",
        "c_adapter_sha256":sha((RESEARCH/"sherpa_pcm/pcm_kws.cc").read_bytes()),
        "native_library_sha256":quant.sha256_file(native_library),
        "baseline_zip_sha256":REFERENCE_ARTIFACT_SHA,
        "original_source_zip_sha256":quant.source.TTS_SHA256,
        "review_json_sha256":quant.source.REVIEW_SHA256,
        "official_model_archive_sha256":quant.MODEL_SHA,
        "runner_script_sha256":sha(Path(__file__).read_bytes()),
        "source_clip_count":63,
        "model_mode":"official-zh-en-int8-c16",
        "model_keyword_alias":"HEY_MERO",
        "model_keyword_id":3,
        "caller_partitions":list(PATS),
        "by_caller_partition":methods,
        "peak_host_process_rss_kib_includes_python":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "human_gold":False,
        "independent_human_frr":"NOT_RUN",
        "continuous_household_far_per_hour":"NOT_RUN",
        "ssc305_board_runtime":"NOT_RUN",
        "new_audio_tts_asr_or_model_training":False,
        "model_promotion_allowed":False,
    }
    require(all(m["model_promotion_allowed"] is False for m in methods.values()) and
            result["source_clip_count"]==63 and
            result["continuous_household_far_per_hour"]=="NOT_RUN",
            "EN_C_RESULT_PROMOTION_DENIED")
    with output.open("x",encoding="utf-8") as f:
        json.dump(result,f,sort_keys=True,indent=2)
        f.write("\n")
    output.chmod(0o600)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","build-receipt","tts-zip","review-json","model-archive","baseline-archive","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    try:
        value=run(a.native_library,a.build_receipt,a.tts_zip,a.review_json,
                  a.model_archive,a.baseline_archive,a.output)
        print(json.dumps({"status":value["status"],
                          "modes":{k:{"events":v["native_event_count"],
                                      "drift_cells":len(v["changed_event_count_cells"]),
                                      "eof_only":v["native_eof_flush_event_count"]}
                                  for k,v in value["by_caller_partition"].items()},
                          "model_promotion_allowed":False},sort_keys=True))
        return 0
    except (ValueError,OSError,RuntimeError,KeyError,zipfile.BadZipFile) as e:
        print("EN_C_PARITY_STOP:"+str(e),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
