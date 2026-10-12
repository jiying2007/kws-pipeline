"""Stdlib-only TTS archive byte/receipt consistency. Never confers admission."""
import hashlib
import io
import json
import math
import struct
import wave

from contract import canonical,cells,require,validate_ledger,SHA
from tts_common import SAFE_EXCEPTIONS


def measured(raw):
    return {"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}


def read_json(raw):
    def pairs(rows):
        out={}
        for key,value in rows:
            require(key not in out,"duplicate metadata key")
            out[key]=value
        return out
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite metadata")))
    require(type(value) is dict and canonical(value)==raw,"canonical UTF8 object")
    return value


def raw_wave(raw):
    require(type(raw) is bytes and 44<=len(raw)<=12*24000*4+256,"bounded native WAV")
    require(raw[:4]==b"RIFF" and raw[8:12]==b"WAVE" and struct.unpack_from("<I",raw,4)[0]+8==len(raw),"native RIFF container")
    pos=12;chunks={}
    while pos<len(raw):
        require(pos+8<=len(raw),"complete native chunk header")
        tag=raw[pos:pos+4];size=struct.unpack_from("<I",raw,pos+4)[0];pos+=8
        require(tag in (b"fmt ",b"fact",b"PEAK",b"data") and tag not in chunks and pos+size<=len(raw),"bounded unique native chunk")
        chunks[tag]=raw[pos:pos+size];pos+=size+(size&1)
    require(pos==len(raw) and b"fmt " in chunks and b"data" in chunks,"exact native chunks")
    fmt=chunks[b"fmt "];require(len(fmt) in (16,18),"native FLOAT fmt size")
    require(struct.unpack_from("<HHIIHH",fmt)==(3,1,24000,96000,4,32),"native IEEE float24k mono")
    if len(fmt)==18:require(fmt[16:]==b"\x00\x00","native format extension empty")
    samples=chunks[b"data"];require(0<len(samples)<12*24000*4 and len(samples)%4==0,"native sample geometry")
    require(all(math.isfinite(x[0]) for x in struct.iter_unpack("<f",samples)),"native finite samples")
    frames=len(samples)//4
    if b"fact" in chunks:
        require(len(chunks[b"fact"])==4 and struct.unpack("<I",chunks[b"fact"])[0]==frames,"native fact frame count")
    peak=max(abs(x[0]) for x in struct.iter_unpack("<f",samples))
    if b"PEAK" in chunks:
        require(len(chunks[b"PEAK"])==16,"native mono PEAK shape")
        version,timestamp,reported_peak,peak_frame=struct.unpack("<IIfI",chunks[b"PEAK"])
        require(version==1 and math.isfinite(reported_peak) and reported_peak==peak and peak_frame<frames,
                "native PEAK must match finite samples")
        require(abs(struct.unpack_from("<f",samples,peak_frame*4)[0])==peak,"native PEAK frame binding")
    return {"source_peak":peak,"source_frames":frames,"raw_float_sha256":hashlib.sha256(samples).hexdigest(),"raw_wav_sha256":hashlib.sha256(raw).hexdigest()}


def pcm_wave(raw):
    require(type(raw) is bytes and 44<=len(raw)<=12*16000*2+256,"bounded derivative WAV")
    require(raw[:4]==b"RIFF" and raw[8:12]==b"WAVE" and struct.unpack_from("<I",raw,4)[0]+8==len(raw),"derivative RIFF container")
    pos=12;chunks={}
    while pos<len(raw):
        require(pos+8<=len(raw),"complete derivative chunk header")
        tag=raw[pos:pos+4];size=struct.unpack_from("<I",raw,pos+4)[0];pos+=8
        require(tag in (b"fmt ",b"data") and tag not in chunks and pos+size<=len(raw),"exact derivative chunk structure")
        chunks[tag]=raw[pos:pos+size];pos+=size+(size&1)
    require(pos==len(raw) and set(chunks)=={b"fmt ",b"data"},"complete derivative chunks")
    require(len(chunks[b"fmt "]) in (16,18) and struct.unpack_from("<HHIIHH",chunks[b"fmt "])==(1,1,16000,32000,2,16),"canonical PCM16 fmt")
    if len(chunks[b"fmt "])==18:require(chunks[b"fmt "][16:]==b"\x00\x00","empty derivative fmt extension")
    require(len(chunks[b"data"])%2==0,"whole derivative PCM samples")
    with wave.open(io.BytesIO(raw),"rb") as handle:
        require((handle.getframerate(),handle.getnchannels(),handle.getsampwidth(),handle.getcomptype())==(16000,1,2,"NONE"),"mono16kPCM16")
        frames=handle.getnframes();require(0<frames<12*16000,"derivative duration limit")
        pcm=handle.readframes(frames)
    require(len(pcm)==2*frames and pcm==chunks[b"data"],"complete PCM samples")
    return {"frames":frames,"pcm_sha256":hashlib.sha256(pcm).hexdigest(),"wav_sha256":hashlib.sha256(raw).hexdigest()}


def validate_eos(eos):
    expected={"capture_calls","iterations","last_token_ids","eos","ended_with_eos","hit_token_cap_without_eos","sequence_length","earlier_eos_seen"}
    require(type(eos) is dict and set(eos)==expected,"exact natural-EOS record")
    for key in ("capture_calls","iterations","eos","sequence_length"):
        require(type(eos[key]) is int,"integer EOS fields")
    require(eos["capture_calls"]==1 and 0<eos["iterations"]<120 and eos["sequence_length"]==eos["iterations"] and
            eos["last_token_ids"]==[2150] and type(eos["last_token_ids"][0]) is int and eos["eos"]==2150 and
            eos["ended_with_eos"] is True and eos["hit_token_cap_without_eos"] is False and
            eos["earlier_eos_seen"] is False,"true singleton natural EOS below cap")


def local_authorities():
    from tts_gate import HERE, read_json as file_json
    from tts_common import sha
    freeze=file_json(HERE/"tts-source-freeze.json")
    return {"source_freeze":freeze|{"source_freeze_sha256":sha(HERE/"tts-source-freeze.json")},
            "model_provenance":file_json(HERE/"model-provenance.json"),
            "plan_sha256":sha(HERE/"plan.proposed.json")}


def validate_tts_members(members,identity):
    """Allow truthful partial/failure archives; complete cells need complete byte links."""
    fixed={"claim.json","outcomes.json","source.json","eos.json","resources.json"}
    require(fixed<=set(members),"required TTS evidence")
    claim=read_json(members["claim.json"]);outcomes=read_json(members["outcomes.json"])
    source=read_json(members["source.json"]);eos=read_json(members["eos.json"]);resources=read_json(members["resources.json"])
    require(set(claim)=={"schema","phase","identity","source_branch","result_branch","source_freeze_sha256","source_tree","preflight_verified","preflight","maximum_tts_attempts"},"exact host claim fields")
    require(claim["source_branch"]=="research/k1-punctuation6-tts-v1" and claim["result_branch"]=="data/k1-punctuation6-tts-v1" and
            type(claim["maximum_tts_attempts"]) is int and claim["maximum_tts_attempts"]==6,"fixed host claim scope")
    pre=claim["preflight"]
    require(type(pre) is dict and set(pre)=={"preflight_verified","numeric_job_id","source_branch_workflow_runs","repository_public","result_ref_absent","read_only_requests"},"exact preflight fields")
    require(pre["preflight_verified"] is True and pre["repository_public"] is True and pre["result_ref_absent"] is True and
            type(pre["numeric_job_id"]) is int and pre["numeric_job_id"]>0 and type(pre["source_branch_workflow_runs"]) is int and
            pre["source_branch_workflow_runs"]==1 and type(pre["read_only_requests"]) is int and pre["read_only_requests"]==5,"verified preflight evidence")
    require(claim.get("schema")=="k1-punctuation6-host-claim-v1" and claim.get("phase")=="tts" and
            claim.get("identity")==identity and claim.get("preflight_verified") is True,"actual new host claim")
    require(source.get("schema")=="k1-punctuation6-tts-source-v1" and source.get("identity")==identity and
            source.get("source_freeze_sha256")==claim.get("source_freeze_sha256") and
            source.get("source_tree")==claim.get("source_tree"),"source/run/claim byte closure")
    from tts_public_schema import validate_source, validate_resources
    validate_source(source,identity,**local_authorities())
    validate_resources(resources,identity)
    outcome_base={"schema","experiment","status","ledger","cells","consumed_attempts","human_gold","training_admitted"}
    require(outcome_base<=set(outcomes)<=outcome_base|{"error_class","finalization"},"exact outcomes fields")
    if "finalization" in outcomes:
        from tts_finalize import validate_finalization
        validate_finalization(outcomes["finalization"],outcomes["cells"])
        require(not any(row["path"] in members for row in outcomes["finalization"]["quarantined_audio"]),
                "private quarantined partial bytes cannot be published")
    if "error_class" in outcomes:require(outcomes["error_class"] in SAFE_EXCEPTIONS,"safe outcome exception class")
    require(outcomes.get("schema")=="k1-punctuation6-tts-outcomes-v1" and outcomes.get("experiment")=="k1-punctuation6-v1" and
            outcomes.get("human_gold") is False and outcomes.get("training_admitted") is False,"new denominator scope")
    require(type(outcomes.get("cells")) is list and len(outcomes["cells"])==6,"retain all six cells")
    counts=validate_ledger("tts",outcomes["ledger"])
    require(type(outcomes.get("consumed_attempts")) is int and outcomes["consumed_attempts"]==counts["consumed"],"actual durable-start accounting")
    require(outcomes.get("status") in ("not_run","started","stopped_no_retry","completed_six_once"),"truthful TTS status")
    if outcomes["status"]=="not_run":require(counts["consumed"]==0,"not_run cannot conceal consumed claims")
    if outcomes["status"]=="completed_six_once":
        require(counts["complete"]==6 and resources.get("status")=="completed_six_once","complete six/global state")
    require(set(eos)=={"schema","rows"},"exact EOS aggregate fields")
    require(eos.get("schema")=="k1-punctuation6-eos-v1" and type(eos.get("rows")) is list and len(eos["rows"])<=6,"bounded termination rows")
    for termination in eos["rows"]:
        require(type(termination) is dict and "cell_id" in termination,"termination object")
        if "failed_capture" in termination:
            require(set(termination)=={"cell_id","failed_capture"} and type(termination["failed_capture"]) is dict,"failed termination fields")
            partial=termination["failed_capture"]
            require(set(partial)<={"capture_calls","iterations","last_token_ids","eos","ended_with_eos","hit_token_cap_without_eos","sequence_length","earlier_eos_seen"},"failed capture fields")
            for k,v in partial.items():
                if k in ("ended_with_eos","hit_token_cap_without_eos","earlier_eos_seen"):require(type(v) is bool,"partial termination boolean")
                elif k=="last_token_ids":require(type(v) is list and len(v)<=1 and all(type(x) is int and 0<=x<4096 for x in v),"partial last token")
                else:require(type(v) is int and 0<=v<=4096,"partial termination integer")
        else:
            validate_eos({k:v for k,v in termination.items() if k!="cell_id"})
    eos_by={r.get("cell_id"):r for r in eos["rows"]}
    require(len(eos_by)==len(eos["rows"]),"unique termination rows")
    expected_members=set(fixed)
    for cell,row,ledger in zip(cells(),outcomes["cells"],outcomes["ledger"]):
        require(type(row) is dict and set(row)<=set(cell)|{"status","attempts","observation","claim","generation_seconds","eos","error_class"} and
                set(cell)|{"status","attempts","observation"}<=set(row),"exact cell field allowlist")
        if "error_class" in row:require(row["error_class"] in SAFE_EXCEPTIONS,"safe cell exception class")
        if "generation_seconds" in row:require(type(row["generation_seconds"]) in (int,float) and math.isfinite(row["generation_seconds"]) and 0<=row["generation_seconds"]<=300,"bounded generation duration")
        require(type(row) is dict and all(canonical(row.get(k))==canonical(v) for k,v in cell.items()),"frozen cell fields")
        require(row.get("status")==ledger["status"] and type(row.get("attempts")) is int and row["attempts"]==ledger["attempts"],"cell/ledger agreement")
        claim_path="claims/"+cell["cell_id"]+".started.json"
        audio_paths=["raw/"+cell["cell_id"]+".wav","derived/"+cell["cell_id"]+".wav"]
        if row["attempts"]==0:
            require(set(row)==set(cell)|{"status","attempts","observation"} and row["observation"] is None,"unstarted exact row")
            require(claim_path not in members and not any(p in members for p in audio_paths) and cell["cell_id"] not in eos_by,"unstarted slot cannot have output")
            continue
        expected_members.add(claim_path)
        require(claim_path in members,"original exclusive started member retained")
        started=read_json(members[claim_path])
        expected={"schema":"k1-punctuation6-tts-start-v1","cell_id":cell["cell_id"],"seed":cell["seed"],
                  "voice":cell["voice"],"text":cell["text"],"attempt":1,"request_sha256":hashlib.sha256(canonical(cell)).hexdigest()}
        require(canonical(started)==canonical(expected) and row.get("claim")==measured(members[claim_path]),"exact started bytes and request binding")
        if row["status"]!="complete":
            require(row["observation"] is None and "eos" not in row and "generation_seconds" not in row,"incomplete observation is not a completed waveform claim")
        available={p for p in audio_paths if p in members};expected_members|=available
        raw_observation=raw_wave(members[audio_paths[0]]) if audio_paths[0] in available else None
        pcm_observation=pcm_wave(members[audio_paths[1]]) if audio_paths[1] in available else None
        require(pcm_observation is None or raw_observation is not None,"derivative cannot lack native source")
        if raw_observation and pcm_observation:
            require(pcm_observation["frames"]==(raw_observation["source_frames"]*2+2)//3,"fixed 2:3 frame pair")
        if row["status"]=="complete":
            require(available==set(audio_paths),"complete cell requires both WAVs")
            observation=row.get("observation");require(type(observation) is dict,"real waveform observation")
            expected_observation={"raw_path","derived_path","raw_wav_sha256","raw_float_sha256","wav_sha256","pcm_sha256",
                "source_frames","source_rate_hz","source_channels","source_dtype","frames","sample_rate_hz","channels","sample_width_bytes",
                "derivation","quality_flags","source_peak","derivative_peak","derivative_float_sha256"}
            require(set(observation)==expected_observation,"exact waveform observation fields")
            for field in ("source_frames","frames"):require(type(observation[field]) is int,"integer waveform frames")
            for field,expected in (("source_rate_hz",24000),("source_channels",1),("sample_rate_hz",16000),("channels",1),("sample_width_bytes",2)):
                require(type(observation[field]) is int and observation[field]==expected,"fixed observed waveform format")
            require(observation["source_dtype"]=="little_endian_float32","native float dtype")
            require(type(observation["derivative_float_sha256"]) is str and SHA.fullmatch(observation["derivative_float_sha256"]),"resampling float-value hash")
            for field in ("source_peak","derivative_peak"):
                require(type(observation[field]) in (int,float) and math.isfinite(observation[field]) and observation[field]>=0,"finite observed peak")
            flags=(["silent"] if raw_observation["source_peak"]==0 else [])+(["source_peak_ge_one"] if raw_observation["source_peak"]>=1 else [])+(["normalized_peak_ge_one"] if observation["derivative_peak"]>=1 else [])
            with wave.open(io.BytesIO(members[audio_paths[1]]),"rb") as final_wave:
                if not any(final_wave.readframes(final_wave.getnframes())):flags.append("derived_pcm_silent")
            require(type(observation["quality_flags"]) is list and observation["quality_flags"]==flags,"signal flags cannot be cleared or rewritten")
            require(all(observation.get(k)==v for k,v in (raw_observation|pcm_observation).items()),"WAV/float/PCM measured bytes")
            require(observation.get("raw_path")==audio_paths[0] and observation.get("derived_path")==audio_paths[1] and
                    observation.get("derivation")=="scipy-resample-poly-2-to-3-no-gain-no-trim","same-cell derivative mapping")
            validate_eos(row.get("eos"))
            require(eos_by.get(cell["cell_id"])=={"cell_id":cell["cell_id"],**row["eos"]},"original EOS member binding")
            require(raw_observation["source_frames"]==(row["eos"]["iterations"]-1)*1920,"codec/native frame identity")
    require(set(eos_by)<={r["cell_id"] for r in outcomes["cells"] if r["attempts"]==1},"EOS cannot invent a cell")
    require(set(members)==expected_members,"no undeclared output member")
    return True
