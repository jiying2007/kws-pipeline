#!/usr/bin/env python3
"""Run unchanged original English63 PCM through official pretrained models WITHOUT added silence.

This is a model-based bounded research reproduction, not human gold, end-of-word
latency, SSC305 qualification or continuous household FAR/FRR.
The saved prior experiment APPENDED 10,560 zero samples after each source WAV.
Here we distinguish actual source PCM events from EOF-only events, and compare
with the exact previous per-clip event receipt. NO audio/model generation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import math
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pretrained_quant_events as original

POLICY = "english63-original-synthetic-unpadded-online-event-replay-v1"
MODES = ("fp32_c16","int8_c16")
BASELINE_ARTIFACT_SHA = "5f76390b557ab58c6d6ae131b8dad91ec232f3efb708466dcb4fc850b0eb13b5"
BASELINE_SHA = {
    "fp32_c16": "978b967ba64b15d4cd4679fcf3d53dc4553c67975740427b2a354a38cca3c6c0",
    "int8_c16": "d6db3248e4317e7d9d22b5e7f9145dc1c15b79b926496658f4e25c2f971c69e6",
}
INPUT_HOP_SAMPLES = 1600
EXPECTED_CLIPS = 63
EXPECTED_PREVIOUS_EVENTS = 9


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_locked_baselines(zip_path):
    require(zip_path.is_absolute() and zip_path.is_file() and
            not zip_path.is_symlink() and zip_path.stat().st_size < 256000 and
            original.sha256_file(zip_path) == BASELINE_ARTIFACT_SHA,
            "EN_LIVE_BASELINE_ARTIFACT_DRIFT")
    selected = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        require(set(names) == {name+".json" for name in
                              ("fp32_c16","int8_c16","int8_c8",
                               "english63-quant-summary")}, "EN_LIVE_BASELINE_ROSTER")
        for mode in MODES:
            raw = archive.read(mode+".json")
            require(sha(raw)==BASELINE_SHA[mode], "EN_LIVE_BASELINE_MEMBER_DRIFT")
            value = json.loads(raw)
            require(value.get("schema") == original.SCHEMA and
                    value.get("mode") == mode and
                    value.get("model_archive_sha256") == original.MODEL_SHA and
                    value.get("source_tts_zip_sha256") == original.source.TTS_SHA256 and
                    value.get("source_review_json_sha256") == original.source.REVIEW_SHA256 and
                    value.get("keyword_line") == original.KEYWORD_TEXT.strip() and
                    value.get("source_clips") == EXPECTED_CLIPS and
                    value.get("source_pcm_samples") == original.EXPECTED_PCM_SAMPLES and
                    value.get("total_events") == EXPECTED_PREVIOUS_EVENTS and
                    value.get("model_promotion_allowed") is False and
                    value.get("human_gold") is False and
                    value.get("continuous_far_per_hour") == "NOT_RUN",
                    "EN_LIVE_BASELINE_POLICY_CHANGED")
            records = value.get("clips")
            require(type(records) is list and len(records)==EXPECTED_CLIPS and
                    len({r.get("cell_id") for r in records})==EXPECTED_CLIPS,
                    "EN_LIVE_BASELINE_CLIPS")
            selected[mode] = {r["cell_id"]:r for r in records}
    return selected


def events_without_padding(spotter, samples):
    """Every emitted event records whether it happened while source PCM was fed.

    After the last raw sample, call input_finished() with NO zero PCM. An event
    that appears only on EOF is not an event available in a live ongoing stream.
    """
    stream = spotter.create_stream()
    fed=0
    steps=0
    records=[]

    def drain(phase):
        nonlocal steps
        while spotter.is_ready(stream):
            steps+=1
            require(steps <= 20000, "EN_LIVE_DECODE_BUDGET")
            spotter.decode_stream(stream)
            result=spotter.get_result(stream)
            if result:
                value=str(result).upper().strip().replace(" ","_")
                require(value=="HEY_MERO" and len(records)<16,
                        "EN_LIVE_UNEXPECTED_WAKE")
                tokens=list(spotter.tokens(stream))
                stamps=[float(t) for t in spotter.timestamps(stream)]
                require(len(tokens)==len(stamps) and
                        all(math.isfinite(t) and t >= 0 for t in stamps),
                        "EN_LIVE_TOKEN_TIMESTAMPS")
                records.append({
                    "keyword":value, "phase":phase,
                    "input_samples_available":fed,
                    "input_seconds_available":round(fed/16000,5),
                    "token_timestamp_available":bool(stamps),
                })
                spotter.reset_stream(stream)

    for start in range(0,len(samples),INPUT_HOP_SAMPLES):
        current=samples[start:start+INPUT_HOP_SAMPLES]
        stream.accept_waveform(16000,current)
        fed+=len(current)
        drain("source_pcm")
    require(fed==len(samples),"EN_LIVE_WRONG_PCM_COUNT")
    stream.input_finished()
    drain("eof_no_padding")
    return records,steps


def verify_retained_receipt(result):
    """Model-free, no-audio, no-numpy proof that this is NOT model promotion."""
    require(type(result) is dict and result.get("schema")==POLICY and
            result.get("status")=="MEASURED_UNPADDED_SYNTHETIC_SOURCE_EVENT_DIAGNOSTIC" and
            result.get("runner_sha256")==sha(Path(__file__).read_bytes()) and
            result.get("baseline_artifact_sha256")==BASELINE_ARTIFACT_SHA and
            result.get("source_tts_sha256")==original.source.TTS_SHA256 and
            result.get("source_review_sha256")==original.source.REVIEW_SHA256 and
            result.get("model_sha256")==original.MODEL_SHA and
            result.get("source_clips")==63 and
            result.get("input_pcm_samples")==original.EXPECTED_PCM_SAMPLES and
            result.get("modes")==list(MODES) and
            result.get("new_model_training_or_audio_generation") is False and
            result.get("human_gold") is False and
            result.get("continuous_household_far_per_hour")=="NOT_RUN" and
            result.get("independent_human_frr")=="NOT_RUN" and
            result.get("end_of_word_latency_p99")=="NOT_RUN" and
            result.get("ssc305_qualified") is False and
            result.get("model_promotion_allowed") is False,
            "EN_LIVE_RESULT_POLICY_UNSAFE")
    by_mode = result.get("by_mode")
    require(type(by_mode) is dict and set(by_mode)==set(MODES),
            "EN_LIVE_RESULT_MODEL_ROSTER")
    for mode,v in by_mode.items():
        require(type(v) is dict and v.get("saved_baseline_json_sha256")==BASELINE_SHA[mode] and
                v.get("original_with_artificial_tail_events")==9 and
                type(v.get("raw_pcm_events")) is int and
                type(v.get("eof_only_events")) is int and
                type(v.get("original_after_real_pcm_events")) is int and
                0<=v["raw_pcm_events"]<=1000 and 0<=v["eof_only_events"]<=1000 and
                v["original_after_real_pcm_events"]==4 and
                v.get("model_promotion_allowed") is False and
                type(v.get("records")) is list and len(v["records"])==63,
                "EN_LIVE_RESULT_MEASUREMENTS_UNSAFE")
        records=v["records"]
        require(len({r.get("cell_id") for r in records})==63,
                "EN_LIVE_DUPLICATE_CLIP")
        for row in records:
            require(type(row) is dict and original.source.CELL.fullmatch(
                       str(row.get("cell_id"))) and
                    row.get("requested_text") in original.PROMPTS and
                    row.get("human_gold") is False and
                    row.get("training_admitted") is False and
                    type(row.get("source_pcm_samples")) is int and
                    8000<=row["source_pcm_samples"]<=16000*12 and
                    type(row.get("original_with_tail_event_count")) is int and
                    0<=row["original_with_tail_event_count"]<16 and
                    type(row.get("original_after_audio_event_count")) is int and
                    0<=row["original_after_audio_event_count"]<=row["original_with_tail_event_count"] and
                    type(row.get("events")) is list and len(row["events"])<16 and
                    all(type(e) is dict and e.get("phase") in
                        ("source_pcm","eof_no_padding") and
                        e.get("keyword")=="HEY_MERO" and
                        type(e.get("input_samples_available")) is int and
                        0<=e["input_samples_available"]<=row["source_pcm_samples"]
                        for e in row["events"]),
                    "EN_LIVE_RESULT_ROW_UNSAFE")
        require(sum(r["source_pcm_samples"] for r in records)==original.EXPECTED_PCM_SAMPLES and
                sum(r["original_with_tail_event_count"] for r in records)==9 and
                sum(r["original_after_audio_event_count"] for r in records)==4 and
                sum(sum(e["phase"]=="source_pcm" for e in r["events"]) for r in records)==v["raw_pcm_events"] and
                sum(sum(e["phase"]=="eof_no_padding" for e in r["events"]) for r in records)==v["eof_only_events"],
                "EN_LIVE_EVENT_SUM_DRIFT")
    return True


def run(source_zip,review_json,model_archive,baseline_archive,output):
    import sherpa_onnx
    require(output.is_absolute() and output.parent.is_dir() and
            not output.exists() and not output.parent.is_symlink(),
            "EN_LIVE_CREATE_ONLY")
    prior=source_locked_baselines(baseline_archive)
    source=original.load_pinned_audio(source_zip,review_json)
    require(len(source)==63,"EN_LIVE_SOURCE_COUNT")
    require(model_archive.is_absolute() and model_archive.is_file() and
            not model_archive.is_symlink() and
            original.sha256_file(model_archive)==original.MODEL_SHA,
            "EN_LIVE_MODEL_ARCHIVE_UNPINNED")
    by_mode={}
    with tempfile.TemporaryDirectory(prefix="english63-unpadded-") as td:
        for mode in MODES:
            path=Path(td)/mode
            path.mkdir()
            parts=original.extract_model(model_archive,path,mode)
            keyword=original.official_keyword(parts,path)
            spotter=sherpa_onnx.KeywordSpotter(
                tokens=str(parts["tokens"]),encoder=str(parts["encoder"]),
                decoder=str(parts["decoder"]),joiner=str(parts["joiner"]),
                keywords_file=str(keyword),provider="cpu",
                num_threads=1,sample_rate=16000,feature_dim=80,
                max_active_paths=4,keywords_score=1.0,
                keywords_threshold=0.25,num_trailing_blanks=1)
            stats=Counter()
            all_records=[]
            for cell in sorted(source):
                row,pcm=source[cell]
                previous=prior[mode][cell]
                require(previous["pcm_sha256"]==row["pcm_sha256"] and
                        previous["wav_sha256"]==row["wav_sha256"] and
                        previous["requested_text"]==row["requested_text"] and
                        previous["source_family_id"]==row["source_family_id"] and
                        previous["human_gold"] is False,
                        "EN_LIVE_RECORD_MISMATCH")
                events,steps=events_without_padding(spotter,pcm)
                old_events=previous["events"]
                original_after=sum(float(e["available_after_input_sec"]) >
                                   len(pcm)/16000+1e-8 for e in old_events)
                stats["original_with_artificial_tail_events"]+=len(old_events)
                stats["original_after_real_pcm_events"]+=original_after
                stats["raw_pcm_events"]+=sum(e["phase"]=="source_pcm" for e in events)
                stats["eof_only_events"]+=sum(e["phase"]=="eof_no_padding" for e in events)
                all_records.append({
                    "cell_id":cell,
                    "source_family_id":row["source_family_id"],
                    "requested_text":row["requested_text"],
                    "requested_intent":row["requested_intent"],
                    "source_pcm_samples":len(pcm),
                    "pcm_sha256":row["pcm_sha256"],
                    "original_with_tail_event_count":len(old_events),
                    "original_after_audio_event_count":original_after,
                    "events":events,"decode_steps":steps,
                    "human_gold":False,"training_admitted":False,
                })
            require(stats["original_with_artificial_tail_events"]==9 and
                    stats["original_after_real_pcm_events"]==4,
                    "EN_LIVE_EXPECTED_BASELINE_TIMING_DRIFT")
            by_mode[mode]={
                "saved_baseline_json_sha256":BASELINE_SHA[mode],
                "original_with_artificial_tail_events":9,
                "original_after_real_pcm_events":4,
                "raw_pcm_events":int(stats["raw_pcm_events"]),
                "eof_only_events":int(stats["eof_only_events"]),
                "records":all_records,
                "model_promotion_allowed":False,
            }
    result={
        "schema":POLICY,
        "status":"MEASURED_UNPADDED_SYNTHETIC_SOURCE_EVENT_DIAGNOSTIC",
        "runner_sha256":sha(Path(__file__).read_bytes()),
        "baseline_artifact_sha256":BASELINE_ARTIFACT_SHA,
        "source_tts_sha256":original.source.TTS_SHA256,
        "source_review_sha256":original.source.REVIEW_SHA256,
        "model_sha256":original.MODEL_SHA,
        "source_clips":63, "input_pcm_samples":original.EXPECTED_PCM_SAMPLES,
        "modes":list(MODES),"by_mode":by_mode,
        "new_model_training_or_audio_generation":False,
        "human_gold":False,
        "continuous_household_far_per_hour":"NOT_RUN",
        "independent_human_frr":"NOT_RUN",
        "end_of_word_latency_p99":"NOT_RUN",
        "ssc305_qualified":False,
        "model_promotion_allowed":False,
    }
    verify_retained_receipt(result)
    with output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,sort_keys=True,ensure_ascii=False)
        stream.write("\n")
    output.chmod(0o600)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tts-zip",type=Path,required=True)
    p.add_argument("--review-json",type=Path,required=True)
    p.add_argument("--model-archive",type=Path,required=True)
    p.add_argument("--baseline-archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        value=run(args.tts_zip,args.review_json,args.model_archive,
                  args.baseline_archive,args.output)
        print(json.dumps({"status":value["status"],
                          "summary":{mode:{
                              "raw_pcm_events":val["raw_pcm_events"],
                              "eof_only_events":val["eof_only_events"],
                              "baseline_tail_events":val["original_after_real_pcm_events"],
                          } for mode,val in value["by_mode"].items()},
                          "model_promotion_allowed":False},sort_keys=True))
        return 0
    except (ValueError,OSError,RuntimeError,KeyError,zipfile.BadZipFile) as exc:
        print("EN_LIVE_STOP: "+str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
