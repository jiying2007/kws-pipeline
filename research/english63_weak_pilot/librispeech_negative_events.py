#!/usr/bin/env python3
"""Original LibriSpeech dev-clean independent English read-speech KWS probe.

Fixed filename-hash 1-in-5 sample. Real pretrained KWS events, NOT field FAR
or independent human acoustic label truth. Never trains or tunes a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pretrained_quant_events as quant

POLICY="english-librispeech-dev-clean-hash20-pretrained-mero-v1"
ARCHIVE_SHA="76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3"
ARCHIVE_SIZE=337926286
ARCHIVE_MD5="42e2234ba48799c1f50f24a7926300a1"
DATASET="OpenSLR-SLR12-LibriSpeech-dev-clean"
LICENSE="CC-BY-4.0"
MODES=("fp32_c16","int8_c16")
NAME=re.compile(r"LibriSpeech/dev-clean/([0-9]+)/([0-9]+)/([0-9]+-[0-9]+-[0-9]+)\.flac\Z")
COVERAGE_FRACTION=5
MAX_EVENT_PER_CLIP=16


def need(ok,why):
    if not ok:
        raise ValueError(why)


def selected(name):
    """Fixed before observation; no audio contents or model predictions used."""
    need(type(name) is str and NAME.fullmatch(name),"EN_NAT_SOURCE_MEMBER")
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8],"big")%COVERAGE_FRACTION==0


def no_requested_wake(text):
    need(type(text) is str and bool(text.strip()),"EN_NAT_EMPTY_TRANSCRIPT")
    normalized=" "+re.sub(r"[^A-Z]+"," ",text.upper()).strip()+" "
    return " HEY MERO " not in normalized and " HEY VOSEN " not in normalized


def extract_transcripts(tar, members):
    transcripts={}
    for member in members:
        if not member.name.endswith(".trans.txt"):
            continue
        p=PurePosixPath(member.name)
        need(p.parts[:2]==("LibriSpeech","dev-clean") and
             member.isfile() and 0<member.size<1000000,
             "EN_NAT_TRANSCRIPT_MEMBER")
        f=tar.extractfile(member)
        need(f is not None,"EN_NAT_TRANSCRIPT_UNREADABLE")
        for line in f.read().decode("utf-8").splitlines():
            head, sep, text=line.partition(" ")
            need(sep and re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+",head) and
                 head not in transcripts and text.strip(),
                 "EN_NAT_TRANSCRIPT_INVALID")
            transcripts[head]=text
    need(len(transcripts)>=2000,"EN_NAT_TRANSCRIPT_COUNT")
    return transcripts


def verified_archive(archive):
    need(archive.is_absolute() and archive.is_file() and
         not archive.is_symlink() and archive.stat().st_size==ARCHIVE_SIZE and
         quant.sha256_file(archive)==ARCHIVE_SHA,
         "EN_NAT_ARCHIVE_NOT_ORIGINAL")


def source_members(tar):
    members=tar.getmembers()
    need(2000<len(members)<12000 and
         all(type(m.name) is str and
             not PurePosixPath(m.name).is_absolute() and
             all(x not in ("", ".", "..") for x in PurePosixPath(m.name).parts) and
             (m.isfile() or m.isdir()) and
             0<=m.size<=12*1024*1024
             for m in members),
         "EN_NAT_TAR_MEMBER_UNSAFE")
    flac=[m for m in members if m.name.endswith(".flac")]
    need(len(flac)>2000 and all(NAME.fullmatch(m.name) and
         m.isfile() and 100<m.size<12*1024*1024 for m in flac),
         "EN_NAT_FLAC_SOURCE_UNSAFE")
    selection=[m for m in flac if selected(m.name)]
    need(300<=len(selection)<=800,
         "EN_NAT_HASH_SAMPLE_UNEXPECTED_COUNT")
    # Tar metadata remains in physical order: compressed archive is read
    # forward, not decompressed repeatedly at random positions.
    return members, selection


def load_model(model_archive, base):
    import sherpa_onnx
    spotters={}
    for mode in MODES:
        root=base/mode
        root.mkdir()
        parts=quant.extract_model(model_archive,root,mode)
        key=quant.official_keyword(parts,root)
        spotter=sherpa_onnx.KeywordSpotter(
            tokens=str(parts["tokens"]),
            encoder=str(parts["encoder"]), decoder=str(parts["decoder"]),
            joiner=str(parts["joiner"]),keywords_file=str(key),
            num_threads=1,sample_rate=16000,feature_dim=80,max_active_paths=4,
            keywords_score=1.0,keywords_threshold=0.25,num_trailing_blanks=1,
            provider="cpu")
        spotters[mode]=spotter
    return spotters


def verify_receipt(report):
    """Source-only CI check; a KWS event is not a gold negative or FAR."""
    need(type(report) is dict and report.get("schema")==POLICY and
         report.get("status")=="MEASURED_REAL_READ_SPEECH_SEGMENT_EVENTS_NOT_FIELD_FAR" and
         report.get("archive_sha256")==ARCHIVE_SHA and
         report.get("model_sha256")==quant.MODEL_SHA and
         report.get("runner_sha256")==quant.sha256_bytes(Path(__file__).read_bytes()) and
         report.get("subset_rule")=="SHA256(member_name) first8 big-endian mod5 == 0" and
         report.get("license")==LICENSE and
         report.get("sampled_utterances")>=300 and
         report.get("sampled_utterances")<=800 and
         report.get("speaker_count")>=10 and
         report.get("source_audio_seconds")>=1800 and
         report.get("source_audio_seconds")<=7200 and
         report.get("models_tested")==list(MODES) and
         report.get("new_asr_tts_training_calls")==0 and
         report.get("human_acoustic_gold") is False and
         report.get("exact_source_transcripts_exclude_requested_words") is True and
         report.get("continuous_environment_far_per_hour")=="NOT_RUN" and
         report.get("human_frr")=="NOT_RUN" and
         report.get("target_ssc305")=="NOT_RUN" and
         report.get("model_promotion_allowed") is False,
         "EN_NAT_RECEIPT_NOT_QUALIFICATION")
    rows=report.get("records")
    need(type(rows) is list and len(rows)==report["sampled_utterances"] and
         len({v.get("member") for v in rows})==len(rows),
         "EN_NAT_RECEIPT_RECORDS")
    totals={mode:0 for mode in MODES}
    for row in rows:
        need(type(row) is dict and NAME.fullmatch(str(row.get("member"))) and
             selected(row["member"]) and
             type(row.get("flac_sha256")) is str and len(row["flac_sha256"])==64 and
             type(row.get("transcript_sha256")) is str and len(row["transcript_sha256"])==64 and
             type(row.get("sample_count")) is int and
             400<=row["sample_count"]<=16000*40 and
             row.get("human_gold") is False and
             row.get("trusted_negative") is False and
             row.get("training_admitted") is False and
             type(row.get("events")) is dict and set(row["events"])==set(MODES) and
             all(type(v) is int and 0<=v<MAX_EVENT_PER_CLIP
                 for v in row["events"].values()),
             "EN_NAT_RECEIPT_ROW_UNSAFE")
        for mode,count in row["events"].items():
            totals[mode]+=count
    need(report.get("events_by_mode")==totals,
         "EN_NAT_RECEIPT_EVENT_SUM_DRIFT")
    return True


def execute(source_archive,model_archive,out):
    import numpy as np
    import soundfile as sf
    need(out.is_absolute() and out.parent.is_dir() and
         not out.exists() and not out.parent.is_symlink(),
         "EN_NAT_CREATE_ONLY")
    verified_archive(source_archive)
    need(model_archive.is_absolute() and model_archive.is_file() and
         not model_archive.is_symlink() and
         model_archive.stat().st_size==quant.MODEL_BYTES and
         quant.sha256_file(model_archive)==quant.MODEL_SHA,
         "EN_NAT_MODEL_SOURCE_DRIFT")
    with tempfile.TemporaryDirectory(prefix="english-librispeech-kws-") as tmp:
        spotters=load_model(model_archive,Path(tmp))
        with tarfile.open(source_archive,"r:gz") as tar:
            members,picked=source_members(tar)
            transcripts=extract_transcripts(tar,members)
            records=[]
            duration=0
            speaker_count=set()
            counts=Counter({mode:0 for mode in MODES})
            times=Counter({mode:0.0 for mode in MODES})
            for member in picked:
                m=NAME.fullmatch(member.name)
                assert m is not None
                speaker,chapter,utterance=m.groups()
                need(utterance in transcripts and
                     no_requested_wake(transcripts[utterance]),
                     "EN_NAT_TRANSCRIPT_MISSING_OR_EXACT_WAKE")
                f=tar.extractfile(member)
                need(f is not None,"EN_NAT_FLAC_UNREADABLE")
                raw=f.read()
                samples,rate=sf.read(io.BytesIO(raw),dtype="float32")
                need(rate==16000 and type(samples) is np.ndarray and
                     samples.ndim==1 and
                     400<=len(samples)<=16000*40 and
                     bool(np.isfinite(samples).all()),
                     "EN_NAT_FLAC_GEOMETRY")
                events={}
                for mode in MODES:
                    start=time.perf_counter()
                    found,_=quant.stream_events(spotters[mode],samples)
                    times[mode]+=time.perf_counter()-start
                    events[mode]=len(found)
                    counts[mode]+=len(found)
                duration+=len(samples)
                speaker_count.add(speaker)
                records.append({
                    "member":member.name,
                    "flac_sha256":quant.sha256_bytes(raw),
                    "transcript_sha256":quant.sha256_bytes(transcripts[utterance].encode()),
                    "sample_count":len(samples),
                    "speaker_source_id":speaker,
                    "events":events,
                    "human_gold":False,"trusted_negative":False,
                    "training_admitted":False,
                })
            seconds=duration/16000
            result={
                "schema":POLICY,
                "status":"MEASURED_REAL_READ_SPEECH_SEGMENT_EVENTS_NOT_FIELD_FAR",
                "dataset":DATASET,
                "license":LICENSE,
                "archive_sha256":ARCHIVE_SHA,
                "original_archive_md5":ARCHIVE_MD5,
                "model_sha256":quant.MODEL_SHA,
                "runner_sha256":quant.sha256_bytes(Path(__file__).read_bytes()),
                "subset_rule":"SHA256(member_name) first8 big-endian mod5 == 0",
                "source_clip_partition":"Independent per-record reset with 660ms EOF flush",
                "sampled_utterances":len(records),
                "speaker_count":len(speaker_count),
                "source_audio_seconds":seconds,
                "models_tested":list(MODES),
                "events_by_mode":dict(counts),
                "events_per_hour_of_read_speech_not_continuous_far":
                    {k:round(v/(seconds/3600),6) for k,v in counts.items()},
                "host_inference_wall_sec_excludes_flac_decode":
                    {k:round(v,5) for k,v in times.items()},
                "records":records,
                "new_asr_tts_training_calls":0,
                "human_acoustic_gold":False,
                "exact_source_transcripts_exclude_requested_words":True,
                "continuous_environment_far_per_hour":"NOT_RUN",
                "human_frr":"NOT_RUN",
                "target_ssc305":"NOT_RUN",
                "model_promotion_allowed":False
            }
            verify_receipt(result)
            with out.open("x",encoding="utf-8") as stream:
                json.dump(result,stream,sort_keys=True,indent=2)
                stream.write("\n")
            out.chmod(0o600)
            return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-archive",type=Path,required=True)
    p.add_argument("--model-archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    try:
        r=execute(a.source_archive,a.model_archive,a.output)
        print(json.dumps({"status":r["status"],
                          "utterances":r["sampled_utterances"],
                          "audio_hours":r["source_audio_seconds"]/3600,
                          "events_by_mode":r["events_by_mode"],
                          "model_promotion_allowed":False},sort_keys=True))
        return 0
    except (OSError,ValueError,RuntimeError,EOFError,KeyError,
            tarfile.TarError) as exc:
        print("EN_NAT_READ_SPEECH_STOP: "+str(exc),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
