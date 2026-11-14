#!/usr/bin/env python3
"""Actual official native INT8 KWS replay on uninterrupted original-source PCM.

The source is 63 synthetic Qwen clips concatenated WITHOUT added silence,
without resetting the stream between utterances, and without treating source
prompts as trusted acoustic truth. Not continuous household FAR, FRR or SSC305.
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

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_original_pcm_c_parity as native

POLICY="english63-native-c-contiguous-original-pcm-research-v1"
ORDERS=("cell_id","source_family_then_requested_text")
CHUNK_PLANS=native.PATS
CLIPS=63
TOTAL_SOURCE_SAMPLES=2101760
KEYWORD="HEY_MERO"


def check(ok, code):
    if not ok:
        raise ValueError(code)


def prepare(source_zip, review_json, baseline_zip):
    import numpy as np
    rows=native.quant.load_pinned_audio(source_zip,review_json)
    baseline=native.verify_saved_baseline(baseline_zip)
    check(set(rows)==set(baseline) and len(rows)==CLIPS,
          "EN_CONT_SOURCE_REFERENCE_ROSTER")
    recorded={}
    for cell,(row,pcm) in rows.items():
        prior=baseline[cell]
        check(prior.get("pcm_sha256")==row["pcm_sha256"] and
              prior.get("source_pcm_samples")==len(pcm) and
              prior.get("source_family_id")==row["source_family_id"] and
              prior.get("requested_text")==row["requested_text"] and
              prior.get("human_gold") is False and
              prior.get("training_admitted") is False,
              "EN_CONT_ORIGINAL_SOURCE_POLICY")
        raw=np.rint(pcm*32768).astype("<i2")
        check(hashlib.sha256(raw.tobytes()).hexdigest()==row["pcm_sha256"],
              "EN_CONT_RAW_PCM_IDENTITY")
        recorded[cell]=(row,raw)
    check(sum(len(raw) for _,raw in recorded.values())==TOTAL_SOURCE_SAMPLES,
          "EN_CONT_ORIGINAL_SAMPLE_COUNT")
    check(sum(sum(e["phase"]=="source_pcm" for e in r["events"])
              for r in baseline.values())==5 and
          sum(sum(e["phase"]=="eof_no_padding" for e in r["events"])
              for r in baseline.values())==0,
          "EN_CONT_EXPECTED_ZERO_TAIL_REFERENCE")
    return recorded


def ordering(rows, kind):
    check(kind in ORDERS, "EN_CONT_ORDER_POLICY")
    if kind=="cell_id":
        return sorted(rows)
    return sorted(rows, key=lambda cell:(
        rows[cell][0]["source_family_id"],
        rows[cell][0]["requested_text"],
        cell))


def replay(lib, files, source, patterns):
    import numpy as np
    stats={}
    for pattern,steps in patterns.items():
        callback_results=[]
        callback_errors=[]
        @native.CALLBACK
        def callback(_user, event):
            # ctypes callbacks must not throw through the native ABI.
            value=event.contents
            word=value.keyword.decode("utf-8","replace") if value.keyword else ""
            item={
                "keyword":word,
                "id":int(value.keyword_id),
                "available_samples":int(value.available_samples),
                "eof_flush":bool(value.eof_flush)
            }
            if (word!=KEYWORD or item["id"]!=3 or
                item["available_samples"]>TOTAL_SOURCE_SAMPLES):
                callback_errors.append(item)
            callback_results.append(item)
        handle=C.c_void_p()
        err=C.create_string_buffer(512)
        rc=lib.pcm_kws_create(C.byref(files),callback,None,
                              C.byref(handle),err,len(err))
        check(rc==0 and bool(handle.value),"EN_CONT_NATIVE_CREATE:"+repr(err.value))
        try:
            for order in ORDERS:
                callback_results.clear()
                callback_errors.clear()
                check(lib.pcm_kws_reset(handle,err,len(err))==0,
                      "EN_CONT_NATIVE_RESET:"+repr(err.value))
                cells=ordering(source,order)
                offsets=[]
                offset=0
                start_wall=time.perf_counter()
                for cell in cells:
                    row,buffer=source[cell]
                    offsets.append({
                        "cell_id":cell,
                        "start_sample":offset,
                        "end_sample":offset+len(buffer),
                        "source_family_id":row["source_family_id"],
                        "requested_text":row["requested_text"],
                        "human_gold":False
                    })
                    local=0
                    i=0
                    while local<len(buffer):
                        n=min(steps[i%len(steps)],len(buffer)-local)
                        ptr=C.cast(buffer.ctypes.data+local*2,
                                   C.POINTER(C.c_int16))
                        check(lib.pcm_kws_feed(handle,ptr,n,err,len(err))==0,
                              "EN_CONT_NATIVE_FEED:"+repr(err.value))
                        local+=n
                        i+=1
                    offset+=len(buffer)
                check(offset==TOTAL_SOURCE_SAMPLES,"EN_CONT_CROSS_SAMPLE_COUNT")
                check(lib.pcm_kws_finish(handle,err,len(err))==0,
                      "EN_CONT_NATIVE_FINISH:"+repr(err.value))
                wall=time.perf_counter()-start_wall
                check(not callback_errors,"EN_CONT_UNEXPECTED_C_EVENT")
                marks=[ev["available_samples"] for ev in callback_results]
                check(marks==sorted(marks),"EN_CONT_EVENT_ORDER")
                check(len(callback_results)<1000,"EN_CONT_EVENT_BUDGET")
                for ev in callback_results:
                    # Source interval assignment is a timing map, NOT a label.
                    sample=ev["available_samples"]
                    match=next((x for x in offsets if
                                x["start_sample"]<=sample<x["end_sample"]),None)
                    if sample==TOTAL_SOURCE_SAMPLES:
                        match=offsets[-1]
                    ev["source_interval_cell_id"]=(match or {}).get("cell_id")
                    ev["source_boundary_ambiguous"]=any(
                        sample==x["start_sample"] or sample==x["end_sample"]
                        for x in offsets)
                stats[pattern+"__"+order]={
                    "ordered_cell_ids":cells,
                    "source_interval_boundaries":offsets,
                    "events":list(callback_results),
                    "event_count":len(callback_results),
                    "eof_event_count":sum(ev["eof_flush"] for ev in callback_results),
                    "host_loop_sec_excludes_model_load":round(wall,6),
                    "no_reset_between_source_clips":True,
                    "no_interclip_silence_inserted":True
                }
        finally:
            lib.pcm_kws_destroy(handle)
    return stats


def verify_receipt(result):
    check(type(result) is dict and
          result.get("schema")==POLICY and
          result.get("status")=="EXECUTED_CONTIGUOUS_SYNTHETIC_SOURCE_EVENTS_NOT_FIELD_FAR" and
          result.get("runner_sha256")==hashlib.sha256(Path(__file__).read_bytes()).hexdigest() and
          result.get("source_audio_sha256")==native.quant.source.TTS_SHA256 and
          result.get("source_review_sha256")==native.quant.source.REVIEW_SHA256 and
          result.get("reference_unpadded_zip_sha256")==native.REFERENCE_ARTIFACT_SHA and
          result.get("official_model_sha256")==native.quant.MODEL_SHA and
          result.get("model_mode")=="int8_c16" and
          result.get("source_clips")==CLIPS and
          result.get("source_samples_per_replay")==TOTAL_SOURCE_SAMPLES and
          result.get("order_variants")==list(ORDERS) and
          result.get("caller_partitions")==list(CHUNK_PLANS) and
          result.get("human_gold") is False and
          result.get("independent_human_frr")=="NOT_RUN" and
          result.get("continuous_household_far_per_hour")=="NOT_RUN" and
          result.get("ssc305_target_qualification")=="NOT_RUN" and
          result.get("model_promotion_allowed") is False and
          result.get("new_training_or_generated_audio") is False,
          "EN_CONT_FALSE_QUALIFICATION")
    observed=result.get("measurements")
    check(type(observed) is dict and
          set(observed)=={p+"__"+o for p in CHUNK_PLANS for o in ORDERS},
          "EN_CONT_MEASUREMENT_ROSTER")
    for name,entry in observed.items():
        check(type(entry) is dict and
              type(entry.get("ordered_cell_ids")) is list and
              len(entry["ordered_cell_ids"])==CLIPS and
              len(set(entry["ordered_cell_ids"]))==CLIPS and
              type(entry.get("source_interval_boundaries")) is list and
              len(entry["source_interval_boundaries"])==CLIPS and
              type(entry.get("event_count")) is int and
              type(entry.get("eof_event_count")) is int and
              0<=entry["eof_event_count"]<=entry["event_count"]<1000 and
              type(entry.get("events")) is list and
              len(entry["events"])==entry["event_count"] and
              entry.get("no_reset_between_source_clips") is True and
              entry.get("no_interclip_silence_inserted") is True,
              "EN_CONT_RECEIPT_STRUCTURE")
        edges=entry["source_interval_boundaries"]
        check(edges[0]["start_sample"]==0 and
              edges[-1]["end_sample"]==TOTAL_SOURCE_SAMPLES and
              all(edges[i]["end_sample"]==edges[i+1]["start_sample"]
                  for i in range(len(edges)-1)) and
              [e["cell_id"] for e in edges]==entry["ordered_cell_ids"],
              "EN_CONT_BOUNDARY_TAMPER")
        for v in entry["events"]:
            check(type(v) is dict and v.get("keyword")==KEYWORD and
                  v.get("id")==3 and
                  type(v.get("available_samples")) is int and
                  0<=v["available_samples"]<=TOTAL_SOURCE_SAMPLES and
                  type(v.get("eof_flush")) is bool and
                  v.get("source_interval_cell_id") in set(entry["ordered_cell_ids"]) and
                  type(v.get("source_boundary_ambiguous")) is bool,
                  "EN_CONT_UNSAFE_EVENT")
    return True


def run(native_library,build_receipt,source_zip,review_json,model_archive,baseline_zip,output):
    check(output.is_absolute() and output.parent.is_dir() and
          not output.exists() and not output.parent.is_symlink(),
          "EN_CONT_OUTPUT_CREATE_ONLY")
    native.verify_build_receipt(native_library,build_receipt)
    check(model_archive.is_absolute() and model_archive.is_file() and
          not model_archive.is_symlink() and
          native.quant.sha256_file(model_archive)==native.quant.MODEL_SHA,
          "EN_CONT_ORIGINAL_MODEL_SHA")
    samples=prepare(source_zip,review_json,baseline_zip)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="en-kws-contiguous-") as d:
        root=Path(d)
        parts=native.quant.extract_model(model_archive,root,"int8_c16")
        keyword=native.quant.official_keyword(parts,root)
        paths=native.Files(*(str(parts[k]).encode() for k in
                             ("encoder","decoder","joiner","tokens")),
                          str(keyword).encode())
        lib=native.load_c_api(native_library)
        measured=replay(lib,paths,samples,CHUNK_PLANS)
    comparisons={}
    for order in ORDERS:
        base=measured["fixed320__"+order]["events"]
        base_sig=[(e["keyword"],e["id"],e["available_samples"],e["eof_flush"])
                  for e in base]
        comparisons[order]={
            "patterns_compared":list(CHUNK_PLANS),
            "event_counts_by_pattern":{
                p:measured[p+"__"+order]["event_count"] for p in CHUNK_PLANS},
            "identical_events_and_sample_positions":all(
                [(e["keyword"],e["id"],e["available_samples"],e["eof_flush"])
                 for e in measured[p+"__"+order]["events"]]==base_sig
                for p in CHUNK_PLANS)
        }
    receipt={
        "schema":POLICY,
        "status":"EXECUTED_CONTIGUOUS_SYNTHETIC_SOURCE_EVENTS_NOT_FIELD_FAR",
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_audio_sha256":native.quant.source.TTS_SHA256,
        "source_review_sha256":native.quant.source.REVIEW_SHA256,
        "reference_unpadded_zip_sha256":native.REFERENCE_ARTIFACT_SHA,
        "official_model_sha256":native.quant.MODEL_SHA,
        "native_adapter_sha256":native.quant.sha256_file(native_library),
        "model_mode":"int8_c16",
        "source_clips":CLIPS,
        "source_samples_per_replay":TOTAL_SOURCE_SAMPLES,
        "order_variants":list(ORDERS),
        "caller_partitions":list(CHUNK_PLANS),
        "measurements":measured,
        "partition_comparisons":comparisons,
        "peak_host_process_rss_kib_includes_Python":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "human_gold":False,
        "independent_human_frr":"NOT_RUN",
        "continuous_household_far_per_hour":"NOT_RUN",
        "ssc305_target_qualification":"NOT_RUN",
        "model_promotion_allowed":False,
        "new_training_or_generated_audio":False
    }
    verify_receipt(receipt)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,sort_keys=True,indent=2)
        stream.write("\n")
    output.chmod(0o600)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","build-receipt","tts-zip","review-json",
                 "model-archive","baseline-archive","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    try:
        result=run(args.native_library,args.build_receipt,args.tts_zip,
                   args.review_json,args.model_archive,args.baseline_archive,
                   args.output)
        print(json.dumps({
            "status":result["status"],
            "counts":{k:v["event_counts_by_pattern"]
                      for k,v in result["partition_comparisons"].items()},
            "partition_parity":{k:v["identical_events_and_sample_positions"]
                               for k,v in result["partition_comparisons"].items()},
            "model_promotion_allowed":False
        },sort_keys=True))
        return 0
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print("EN_CONTIGUOUS_KWS_STOP:"+str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
