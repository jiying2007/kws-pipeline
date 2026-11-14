#!/usr/bin/env python3
"""Original LibriSpeech dev-clean: source-verified continuous native C KWS study.

517 original real audiobook recordings, no new audio, no segment resets,
no synthetic silence. This is a negative-style speech screening study,
NOT independent human acoustic truth or household field FAR/hour.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import ctypes as C
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import tarfile
import tempfile
import time
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import english_original_pcm_c_parity as native
from librispeech_negative_events import (
    ARCHIVE_SHA, ARCHIVE_SIZE, LICENSE, NAME, selected,
    source_members, extract_transcripts, no_requested_wake, verified_archive,
    verify_receipt as verify_previous
)

POLICY="english-librispeech-native-continuous-c-mero-read-speech-v1"
BASELINE_ZIP_SHA="7f28b4eddbee87435b9dc5b4c44e47a6d6ed200cbf9f4b8688d9b93cabeb4db4"
BASELINE_JSON_SHA="788a1b04b9ecb22fb96d4036d19456f460ce3ccecd4da69d53d96fea53b29510"
BASELINE_MEMBER="english-natural-read-speech-result.json"
SOURCE_CLIPS=517
SOURCE_SPEAKERS=40
SOURCE_PCM_SAMPLES=58646164
KEYWORD="HEY_MERO"
ORDERS=("source_member","speaker_round_robin")
CHUNK_PLANS={"fixed320":(320,),"network1600":(1600,)}
BOUNDARY_NEAR_SAMPLES=3200  # 200ms source-splice sensitivity; NOT false alarm gold.


def need(ok,code):
    if not ok:
        raise ValueError(code)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reference_readback(path):
    need(path.is_absolute() and path.is_file() and
         not path.is_symlink() and 0<path.stat().st_size<150000 and
         native.quant.sha256_file(path)==BASELINE_ZIP_SHA,
         "EN_NAT_CONT_BASELINE_ARTIFACT_IDENTITY")
    with zipfile.ZipFile(path) as archive:
        need(set(archive.namelist())=={BASELINE_MEMBER},
             "EN_NAT_CONT_BASELINE_ROSTER")
        raw=archive.read(BASELINE_MEMBER)
    need(digest(raw)==BASELINE_JSON_SHA,
         "EN_NAT_CONT_BASELINE_RECEIPT_IDENTITY")
    receipt=json.loads(raw)
    verify_previous(receipt)
    need(receipt.get("sampled_utterances")==SOURCE_CLIPS and
         receipt.get("speaker_count")==SOURCE_SPEAKERS and
         abs(receipt.get("source_audio_seconds",0)-SOURCE_PCM_SAMPLES/16000)<1e-5 and
         receipt.get("events_by_mode")=={"fp32_c16":0,"int8_c16":0} and
         receipt.get("continuous_environment_far_per_hour")=="NOT_RUN" and
         receipt.get("model_promotion_allowed") is False,
         "EN_NAT_CONT_BASELINE_SCIENTIFIC_SCOPE")
    rows=receipt["records"]
    need(len({r["member"] for r in rows})==SOURCE_CLIPS,
         "EN_NAT_CONT_BASELINE_DUPLICATE")
    return {r["member"]:r for r in rows}


def load_original_segments(source_archive,reference):
    import numpy as np
    import soundfile as sf
    verified_archive(source_archive)
    segments={}
    roster=[]
    voices=set()
    with tarfile.open(source_archive,"r:gz") as tar:
        members,picked=source_members(tar)
        transcripts=extract_transcripts(tar,members)
        need(len(picked)==len(reference)==SOURCE_CLIPS and
             {x.name for x in picked}==set(reference),
             "EN_NAT_CONT_SOURCE_SELECTION_DRIFT")
        for member in picked:
            name=member.name
            m=NAME.fullmatch(name)
            need(m is not None and selected(name),
                 "EN_NAT_CONT_SOURCE_MEMBER")
            person,chapter,utterance=m.groups()
            need(utterance in transcripts and
                 no_requested_wake(transcripts[utterance]),
                 "EN_NAT_CONT_TARGET_TEXT_PRESENT")
            blob=tar.extractfile(member)
            need(blob is not None,"EN_NAT_CONT_FLAC_UNREADABLE")
            raw=blob.read()
            old=reference[name]
            need(digest(raw)==old["flac_sha256"] and
                 digest(transcripts[utterance].encode("utf-8"))==
                     old["transcript_sha256"] and
                 old["speaker_source_id"]==person and
                 old["human_gold"] is False and
                 old["trusted_negative"] is False and
                 old["training_admitted"] is False and
                 old["events"]=={"fp32_c16":0,"int8_c16":0},
                 "EN_NAT_CONT_ORIGINAL_SOURCE_BYTES_OR_ROLES")
            with sf.SoundFile(io.BytesIO(raw)) as wav:
                need(wav.samplerate==16000 and wav.channels==1 and
                     wav.subtype=="PCM_16" and wav.frames==old["sample_count"],
                     "EN_NAT_CONT_PCM16_SOURCE_GEOMETRY")
                samples=wav.read(dtype="int16",always_2d=False)
            need(type(samples) is np.ndarray and
                 samples.ndim==1 and samples.dtype==np.dtype("int16") and
                 400<=len(samples)<=640000,
                 "EN_NAT_CONT_NATIVE_PCM_ARRAY")
            segments[name]=samples
            voices.add(person)
            roster.append({
                "member":name,
                "flac_sha256":old["flac_sha256"],
                "transcript_sha256":old["transcript_sha256"],
                "speaker_source_id":person,
                "sample_count":len(samples),
                "trusted_negative":False,
                "human_gold":False,
                "training_admitted":False,
            })
    need(len(segments)==SOURCE_CLIPS and len(voices)==SOURCE_SPEAKERS and
         sum(len(v) for v in segments.values())==SOURCE_PCM_SAMPLES,
         "EN_NAT_CONT_SOURCE_TOTALS")
    return segments,roster


def ordered_members(roster,kind):
    need(kind in ORDERS,"EN_NAT_CONT_UNREGISTERED_ORDER")
    if kind=="source_member":
        return sorted(r["member"] for r in roster)
    by_speaker=defaultdict(list)
    for r in roster:
        by_speaker[r["speaker_source_id"]].append(r["member"])
    for k in by_speaker:
        by_speaker[k].sort()
    identities=sorted(by_speaker)
    answer=[]
    for i in range(max(len(xs) for xs in by_speaker.values())):
        for spk in identities:
            if i<len(by_speaker[spk]):
                answer.append(by_speaker[spk][i])
    need(len(answer)==SOURCE_CLIPS and len(set(answer))==SOURCE_CLIPS,
         "EN_NAT_CONT_ORDER_ROSTER_DRIFT")
    return answer


def order_digest(items):
    return digest(("\n".join(items)+"\n").encode("utf-8"))


def verify_result(result):
    need(type(result) is dict and result.get("schema")==POLICY and
         result.get("status")==
             "MEASURED_CONTIGUOUS_REAL_READ_SPEECH_EVENTS_NOT_HOUSEHOLD_FAR" and
         result.get("runner_sha256")==digest(Path(__file__).read_bytes()) and
         result.get("dataset_archive_sha256")==ARCHIVE_SHA and
         result.get("segmented_reference_artifact_sha256")==BASELINE_ZIP_SHA and
         result.get("segmented_reference_json_sha256")==BASELINE_JSON_SHA and
         result.get("original_model_archive_sha256")==native.quant.MODEL_SHA and
         result.get("original_host_runtime_archive_sha256")==native.host.ARCHIVE_SHA and
         result.get("data_license")==LICENSE and
         result.get("selected_utterances")==SOURCE_CLIPS and
         result.get("natural_speaker_count")==SOURCE_SPEAKERS and
         result.get("source_samples_per_session")==SOURCE_PCM_SAMPLES and
         result.get("source_audio_sec_per_session")==SOURCE_PCM_SAMPLES/16000 and
         result.get("model_mode")=="int8_c16" and
         result.get("keyword")==KEYWORD and
         result.get("keyword_id")==3 and
         result.get("orders")==list(ORDERS) and
         result.get("pcm_partitions")==list(CHUNK_PLANS) and
         result.get("no_pcm_inserted_between_recordings") is True and
         result.get("stream_reset_between_recordings") is False and
         result.get("new_training_tts_asr_calls")==0 and
         result.get("human_acoustic_gold") is False and
         result.get("continuous_household_far_per_hour")=="NOT_RUN" and
         result.get("independent_human_frr")=="NOT_RUN" and
         result.get("ssc305_target_qualification")=="NOT_RUN" and
         result.get("model_promotion_allowed") is False,
         "EN_NAT_CONT_RECEIPT_QUALIFICATION_FORGED")
    roster=result.get("source_roster")
    need(type(roster) is list and len(roster)==SOURCE_CLIPS and
         len({r.get("member") for r in roster})==SOURCE_CLIPS and
         len({r.get("speaker_source_id") for r in roster})==SOURCE_SPEAKERS and
         sum(r.get("sample_count",0) for r in roster)==SOURCE_PCM_SAMPLES,
         "EN_NAT_CONT_RECEIPT_SOURCE_ROSTER")
    by_name={}
    for r in roster:
        name=r["member"]
        need(type(name) is str and NAME.fullmatch(name) and selected(name) and
             type(r.get("sample_count")) is int and 400<=r["sample_count"]<=640000 and
             type(r.get("flac_sha256")) is str and len(r["flac_sha256"])==64 and
             type(r.get("transcript_sha256")) is str and len(r["transcript_sha256"])==64 and
             r.get("human_gold") is False and
             r.get("trusted_negative") is False and
             r.get("training_admitted") is False,
             "EN_NAT_CONT_UNTRUSTED_SOURCE_AS_GOLD")
        by_name[name]=r
    actual=result.get("actual_sessions")
    need(type(actual) is dict and
         set(actual)=={p+"__"+o for p in CHUNK_PLANS for o in ORDERS},
         "EN_NAT_CONT_INCOMPLETE_SESSIONS")
    for order in ORDERS:
        expected=ordered_members(roster,order)
        for partition in CHUNK_PLANS:
            report=actual[partition+"__"+order]
            need(type(report) is dict and
                 report.get("ordered_source_sha256")==order_digest(expected) and
                 report.get("processed_samples")==SOURCE_PCM_SAMPLES and
                 report.get("recordings")==SOURCE_CLIPS and
                 report.get("no_interclip_padding") is True and
                 report.get("no_interclip_reset") is True and
                 type(report.get("events")) is list and
                 report.get("event_count")==len(report["events"]) and
                 type(report.get("eof_only_events")) is int and
                 0<=report["eof_only_events"]<=len(report["events"])<1000,
                 "EN_NAT_CONT_SESSION_SUMMARY")
            last=-1
            for event in report["events"]:
                n=event.get("available_samples")
                need(type(n) is int and last<=n<=SOURCE_PCM_SAMPLES and
                     event.get("keyword")==KEYWORD and
                     event.get("keyword_id")==3 and
                     type(event.get("eof_flush")) is bool and
                     event.get("source_interval_member") in by_name and
                     type(event.get("near_splice_boundary_200ms")) is bool and
                     type(event.get("exact_splice_boundary")) is bool,
                     "EN_NAT_CONT_EVENT_FORGED")
                last=n
            need(sum(bool(e["eof_flush"]) for e in report["events"])==
                 report["eof_only_events"],
                 "EN_NAT_CONT_EOF_COUNT_MISMATCH")
    return True


def run(native_library,build_receipt,corpus_archive,model_archive,reference_zip,output):
    import numpy as np
    need(output.is_absolute() and output.parent.is_dir() and
         not output.exists() and not output.parent.is_symlink(),
         "EN_NAT_CONT_CREATE_ONLY_OUTPUT")
    native.verify_build_receipt(native_library,build_receipt)
    native.quant.require(model_archive.is_absolute() and
                         model_archive.is_file() and
                         not model_archive.is_symlink() and
                         native.quant.sha256_file(model_archive)==native.quant.MODEL_SHA,
                         "EN_NAT_CONT_MODEL_CHANGED")
    old=reference_readback(reference_zip)
    samples,roster=load_original_segments(corpus_archive,old)
    with tempfile.TemporaryDirectory(prefix="english-nat-continuous-c-") as tmp:
        root=Path(tmp)
        parts=native.quant.extract_model(model_archive,root,"int8_c16")
        kw=native.quant.official_keyword(parts,root)
        paths=native.Files(*(str(parts[k]).encode("utf-8") for k in
                             ("encoder","decoder","joiner","tokens")),
                           str(kw).encode("utf-8"))
        lib=native.load_c_api(native_library)
        results={}
        for order in ORDERS:
            names=ordered_members(roster,order)
            # Deliberately no added samples across original FLAC boundaries.
            full=np.concatenate([samples[name] for name in names])
            need(len(full)==SOURCE_PCM_SAMPLES and full.dtype==np.int16,
                 "EN_NAT_CONT_CONCAT_SOURCE_COUNT")
            boundary=[0]
            for name in names:
                boundary.append(boundary[-1]+len(samples[name]))
            need(boundary[-1]==SOURCE_PCM_SAMPLES,
                 "EN_NAT_CONT_INTERVAL_BOUNDS")
            for partition,partsizes in CHUNK_PLANS.items():
                events=[]
                bad=[]
                @native.CALLBACK
                def callback(_user,p):
                    # Never raise across ctypes; verify outside callback.
                    item=p.contents
                    word=item.keyword.decode("utf-8","replace") if item.keyword else ""
                    record={"keyword":word,"keyword_id":int(item.keyword_id),
                            "available_samples":int(item.available_samples),
                            "eof_flush":bool(item.eof_flush)}
                    if word!=KEYWORD or record["keyword_id"]!=3 or not(
                       0<=record["available_samples"]<=SOURCE_PCM_SAMPLES):
                        bad.append(record)
                    events.append(record)
                handle=C.c_void_p()
                err=C.create_string_buffer(512)
                need(lib.pcm_kws_create(C.byref(paths),callback,None,
                                       C.byref(handle),err,len(err))==0 and
                     bool(handle.value),"EN_NAT_CONT_CREATE_NATIVE_FAILED:"+repr(err.value))
                wall_start=time.perf_counter()
                try:
                    need(lib.pcm_kws_reset(handle,err,len(err))==0,
                         "EN_NAT_CONT_RESET_NATIVE_FAILED:"+repr(err.value))
                    pos=0
                    i=0
                    while pos<len(full):
                        take=min(partsizes[i%len(partsizes)],len(full)-pos)
                        ptr=C.cast(full.ctypes.data+pos*2,C.POINTER(C.c_int16))
                        need(lib.pcm_kws_feed(handle,ptr,take,err,len(err))==0,
                             "EN_NAT_CONT_FEED_NATIVE_FAILED:"+repr(err.value))
                        pos+=take
                        i+=1
                    need(pos==SOURCE_PCM_SAMPLES,
                         "EN_NAT_CONT_INCOMPLETE_PCM")
                    need(lib.pcm_kws_finish(handle,err,len(err))==0,
                         "EN_NAT_CONT_FINISH_NATIVE_FAILED:"+repr(err.value))
                    need(not bad and len(events)<1000,"EN_NAT_CONT_UNKNOWN_EVENT")
                finally:
                    lib.pcm_kws_destroy(handle)
                for event in events:
                    n=event["available_samples"]
                    idx=min(len(names)-1,max(0,bisect.bisect_right(boundary,n)-1))
                    event["source_interval_member"]=names[idx]
                    event["exact_splice_boundary"]=n in boundary[1:-1]
                    event["near_splice_boundary_200ms"]=min(
                        n-boundary[idx],boundary[idx+1]-n)<=BOUNDARY_NEAR_SAMPLES
                result={
                    "ordered_source_sha256":order_digest(names),
                    "processed_samples":SOURCE_PCM_SAMPLES,
                    "recordings":SOURCE_CLIPS,
                    "event_count":len(events),
                    "eof_only_events":sum(e["eof_flush"] for e in events),
                    "near_splice_event_count_200ms":sum(
                        e["near_splice_boundary_200ms"] for e in events),
                    "host_loop_wall_seconds_excluding_model_and_FLAC_load":
                        round(time.perf_counter()-wall_start,5),
                    "events":events,
                    "no_interclip_padding":True,
                    "no_interclip_reset":True
                }
                results[partition+"__"+order]=result
    output_report={
        "schema":POLICY,
        "status":"MEASURED_CONTIGUOUS_REAL_READ_SPEECH_EVENTS_NOT_HOUSEHOLD_FAR",
        "runner_sha256":digest(Path(__file__).read_bytes()),
        "dataset_archive_sha256":ARCHIVE_SHA,
        "segmented_reference_artifact_sha256":BASELINE_ZIP_SHA,
        "segmented_reference_json_sha256":BASELINE_JSON_SHA,
        "original_model_archive_sha256":native.quant.MODEL_SHA,
        "original_host_runtime_archive_sha256":native.host.ARCHIVE_SHA,
        "native_library_sha256":native.quant.sha256_file(native_library),
        "data_license":LICENSE,
        "selected_utterances":SOURCE_CLIPS,
        "natural_speaker_count":SOURCE_SPEAKERS,
        "source_samples_per_session":SOURCE_PCM_SAMPLES,
        "source_audio_sec_per_session":SOURCE_PCM_SAMPLES/16000,
        "model_mode":"int8_c16",
        "keyword":KEYWORD,"keyword_id":3,
        "orders":list(ORDERS),
        "pcm_partitions":list(CHUNK_PLANS),
        "no_pcm_inserted_between_recordings":True,
        "stream_reset_between_recordings":False,
        "source_roster":roster,
        "actual_sessions":results,
        "host_peak_process_rss_kib_includes_preloaded_FLAC_and_Python":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "new_training_tts_asr_calls":0,
        "human_acoustic_gold":False,
        "continuous_household_far_per_hour":"NOT_RUN",
        "independent_human_frr":"NOT_RUN",
        "ssc305_target_qualification":"NOT_RUN",
        "model_promotion_allowed":False
    }
    verify_result(output_report)
    with output.open("x",encoding="utf-8") as f:
        json.dump(output_report,f,sort_keys=True,indent=2)
        f.write("\n")
    output.chmod(0o600)
    return output_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("native-library","build-receipt","corpus-archive",
                 "model-archive","reference-zip","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    try:
        report=run(args.native_library,args.build_receipt,args.corpus_archive,
                   args.model_archive,args.reference_zip,args.output)
        print(json.dumps({
            "status":report["status"],
            "audio_hours_per_run":report["source_audio_sec_per_session"]/3600,
            "events_by_order_partition":{k:v["event_count"]
                                         for k,v in report["actual_sessions"].items()},
            "near_source_splice_events":{k:v["near_splice_event_count_200ms"]
                                         for k,v in report["actual_sessions"].items()},
            "household_FAR_measured":False,
            "model_promotion_allowed":False
        },sort_keys=True))
        return 0
    except (OSError,RuntimeError,ValueError,KeyError,zipfile.BadZipFile,
            tarfile.TarError) as exc:
        print("EN_NAT_CONT_STOP:"+str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
