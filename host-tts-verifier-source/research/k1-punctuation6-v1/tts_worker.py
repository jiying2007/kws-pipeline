"""One six-cell worker. CLI stays disabled until the separate reviewed TTS gate.

No inference imports occur at module import. Metadata-only tests inject fake
operations into execute_cells; only run_real owns the exact model call.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import signal
import struct
import sys
import time
import wave

from contract import cells, canonical, expected_plan, require, validate_ledger, initial_ledger
from tts_common import safe_exception, atomic_json, assert_no_host_secrets, fsync_directory, sha, unaliased, write_once

HERE = Path(__file__).resolve().parent
HELPERS = HERE.parent / "qwen6_tts"


def load_helper(name):
    path = HELPERS / (name + ".py")
    spec = importlib.util.spec_from_file_location("k1_tts_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def new_outcomes():
    return {"schema": "k1-punctuation6-tts-outcomes-v1", "experiment": "k1-punctuation6-v1",
            "status": "not_run", "ledger": initial_ledger("tts"),
            "cells": [{**row, "status": "not_run", "attempts": 0, "observation": None} for row in cells()],
            "consumed_attempts": 0, "human_gold": False, "training_admitted": False}


def execute_cells(output, generate_one, write_audio, *, alarm=signal.alarm, capture_state=None):
    """Fixed once-through schedule. Exceptions stop and preserve the denominator."""
    output = unaliased(output)
    require(output.is_dir(), "host-created fresh output root required")
    require(not (output/"outcomes.json").exists(), "generation already attempted; no resume")
    for name in ("claims", "raw", "derived"):
        (output/name).mkdir(mode=0o700)
    fsync_directory(output)
    result = new_outcomes(); result["status"] = "started"
    atomic_json(output/"outcomes.json", result)
    eos_rows = []
    atomic_json(output/"eos.json", {"schema": "k1-punctuation6-eos-v1", "rows": eos_rows})
    try:
        for i, row in enumerate(cells()):
            claim = {"schema": "k1-punctuation6-tts-start-v1", "cell_id": row["cell_id"],
                     "seed": row["seed"], "voice": row["voice"], "text": row["text"],
                     "attempt": 1, "request_sha256": hashlib.sha256(canonical(row)).hexdigest()}
            claim_identity = write_once(output/"claims"/(row["cell_id"]+".started.json"), claim)
            result["consumed_attempts"] += 1
            result["ledger"][i].update(status="started", attempts=1)
            record = result["cells"][i]
            record.update(status="started", attempts=1, claim=claim_identity)
            atomic_json(output/"outcomes.json", result)
            alarm(300)
            try:
                # Exactly one callback; the real callback contains one CustomVoice call.
                array, eos, elapsed = generate_one(row)
                require(type(eos) is dict and set(eos) == {"capture_calls", "iterations", "last_token_ids", "eos", "ended_with_eos", "hit_token_cap_without_eos", "sequence_length", "earlier_eos_seen"}, "exact EOS observation")
                require(eos["capture_calls"] == 1 and type(eos["capture_calls"]) is int, "exactly one talker generation capture")
                require(type(eos["iterations"]) is int and 0 < eos["iterations"] < 120 and
                        type(eos["sequence_length"]) is int and eos["sequence_length"] == eos["iterations"] and
                        eos["earlier_eos_seen"] is False, "token limit/sequence termination mismatch")
                require(type(eos["last_token_ids"]) is list and len(eos["last_token_ids"]) == 1 and
                        type(eos["last_token_ids"][0]) is int and type(eos["eos"]) is int and eos["eos"] == 2150 and
                        eos["last_token_ids"] == [eos["eos"]] and eos["ended_with_eos"] is True and
                        eos["hit_token_cap_without_eos"] is False, "natural singleton EOS required")
                eos_rows.append({"cell_id": row["cell_id"], **eos})
                atomic_json(output/"eos.json", {"schema": "k1-punctuation6-eos-v1", "rows": eos_rows})
                observation = write_audio(array, row["cell_id"])
                require(observation["source_frames"] == (eos["iterations"]-1)*1920, "codec-frame/native-frame binding")
                record.update(status="complete", observation=observation,
                              generation_seconds=elapsed, eos=eos)
                result["ledger"][i].update(status="complete")
                validate_ledger("tts", result["ledger"])
                atomic_json(output/"outcomes.json", result)
            except BaseException as error:
                if capture_state and not any(x["cell_id"] == row["cell_id"] for x in eos_rows):
                    eos_rows.append({"cell_id":row["cell_id"],"failed_capture":dict(capture_state)})
                    atomic_json(output/"eos.json",{"schema":"k1-punctuation6-eos-v1","rows":eos_rows})
                record.update(status="failed_no_retry", error_class=safe_exception(error), observation=None)
                record.pop("eos",None); record.pop("generation_seconds",None)
                result["ledger"][i].update(status="failed_no_retry")
                raise
            finally:
                alarm(0)
        result["status"] = "completed_six_once"
    except BaseException as error:
        result.update(status="stopped_no_retry", error_class=safe_exception(error))
        raise
    finally:
        alarm(0)
        validate_ledger("tts", result["ledger"])
        atomic_json(output/"outcomes.json", result)
    return result


def capture_talker(wrapper):
    """Observe the existing single generation return; never call it again."""
    original = wrapper.model.talker.generate
    observed = {}
    def capture(*args, **kwargs):
        require(not observed, "second talker generation within a fixed cell")
        observed["capture_calls"] = 1
        require(not args, "named-only upstream talker generation")
        scalar={k:v for k,v in expected_plan()["tts"]["recipe"].items() if k not in ("language","non_streaming_mode")}
        scalar.update(min_new_tokens=2,eos_token_id=2150,output_hidden_states=True,return_dict_in_generate=True)
        tensor_keys={"inputs_embeds","attention_mask","trailing_text_hidden","tts_pad_embed"}
        require(set(kwargs)==set(scalar)|tensor_keys|{"suppress_tokens"},"exact talker argument surface")
        require(all(canonical(kwargs[k])==canonical(v) for k,v in scalar.items()),"fixed generation settings")
        cfg=wrapper.model.config.talker_config
        require(kwargs["suppress_tokens"]==[i for i in range(cfg.vocab_size-1024,cfg.vocab_size) if i!=2150],"fixed suppressed token set")
        emb=kwargs["inputs_embeds"]
        require(getattr(emb,"ndim",None)==3 and int(emb.shape[0])==1 and int(emb.shape[1])>0,"singleton embedding batch")
        result = original(**kwargs)
        seq = result.sequences
        require(getattr(seq, "ndim", None) == 2 and int(seq.shape[0]) == 1 and int(seq.shape[1]) > 0,
                "singleton nonempty generated sequence required")
        last = seq[:, -1].detach().cpu().tolist()
        eos = int(wrapper.model.config.talker_config.codec_eos_token_id)
        iterations = len(result.hidden_states)
        tokens=seq.detach().cpu().tolist()[0]
        require(len(tokens)==iterations and all(type(x) is int for x in tokens), "new token/hidden-state count binding")
        earlier=eos in tokens[:-1]
        ended = len(last) == 1 and last == [eos] and not earlier
        observed.update(iterations=iterations, sequence_length=len(tokens), earlier_eos_seen=earlier, last_token_ids=last, eos=eos,
                        ended_with_eos=ended, hit_token_cap_without_eos=iterations >= 120 and not ended)
        require(0 < iterations < 120 and ended, "generation stopped without accepted natural EOS; do not decode")
        return result
    wrapper.model.talker.generate = capture
    return observed


def validate_model(wrapper):
    model = wrapper.model
    require(model.tts_model_type == "custom_voice" and model.tts_model_size == "0b6",
            "exact 0.6B CustomVoice identity")
    require(model.tokenizer_type == "qwen3_tts_tokenizer_12hz", "exact 12Hz tokenizer")
    config=model.config.talker_config
    require(config.codec_eos_token_id == 2150 and config.num_code_groups == 16, "locked EOS/code-group configuration")
    for name,speaker_id,dialect in (("dylan",2878,"beijing_dialect"),("uncle_fu",3010,False),("vivian",3065,False)):
        require(config.spk_id.get(name)==speaker_id and config.spk_is_dialect.get(name)==dialect, "fixed speaker conditioning")
    require(config.codec_language_id.get("chinese")==2055 and config.codec_language_id.get("beijing_dialect")==2074,"locked language conditioning")
    generation=model.talker.generation_config
    for name,value in (("num_beams",1),("num_return_sequences",1),("forced_eos_token_id",None),("forced_bos_token_id",None),("stop_strings",None),("max_time",None)):
        require(getattr(generation,name,None)==value,"alternate generation termination/beam path prohibited")
    for owner in (model, model.speech_tokenizer.model):
        require(owner.training is False, "loaded model must be eval")
        params = list(owner.parameters())
        require(params and {str(p.device) for p in params} == {"cpu"} and
                {str(p.dtype) for p in params} == {"torch.float32"}, "CPU FP32 only")
        require(all(str(b.device) == "cpu" for b in owner.buffers()), "CPU buffers only")
        attention = [m for _,m in owner.named_modules() if all(hasattr(m,k) for k in ("q_proj","k_proj","v_proj"))]
        require(attention and all(getattr(m.config,"_attn_implementation",None) == "eager" for m in attention), "eager attention only")


def write_pair(array, cell_id, output, np, sf, resample_poly):
    """Preserve native float bytes and fixed derived PCM; no gain/trim/repair."""
    require(getattr(array,"ndim",None) == 1 and str(getattr(array,"dtype",None)) == "float32",
            "native mono float32 array")
    require(0 < len(array) < 12*24000, "native duration limit reached")
    require(bool(np.isfinite(array).all()), "nonfinite native waveform")
    raw_values = np.asarray(array,dtype="<f4").tobytes()
    raw = output/"raw"/(cell_id+".wav")
    derived = output/"derived"/(cell_id+".wav")
    require(not raw.exists() and not derived.exists(), "no overwrite of generated audio")
    sf.write(raw, array, 24000, subtype="FLOAT")
    restored,rate = sf.read(raw,dtype="float32")
    require(rate == 24000 and restored.ndim == 1 and np.asarray(restored,dtype="<f4").tobytes() == raw_values,
            "raw FLOAT values readback")
    converted = resample_poly(array.astype(np.float64),2,3)
    require(converted.ndim == 1 and len(converted) == (len(array)*2+2)//3 and 0 < len(converted) < 12*16000,
            "fixed 2:3 derivative geometry")
    require(bool(np.isfinite(converted).all()), "nonfinite derivative waveform")
    source_peak=float(np.max(np.abs(array))); derivative_peak=float(np.max(np.abs(converted)))
    flags = []
    if not bool(np.any(array)): flags.append("silent")
    if float(np.max(np.abs(array))) >= 1: flags.append("source_peak_ge_one")
    if float(np.max(np.abs(converted))) >= 1: flags.append("normalized_peak_ge_one")
    sf.write(derived,converted,16000,subtype="PCM_16")
    with wave.open(str(derived),"rb") as handle:
        require((handle.getframerate(),handle.getnchannels(),handle.getsampwidth(),handle.getnframes()) == (16000,1,2,len(converted)), "PCM16 WAV readback geometry")
        pcm = handle.readframes(handle.getnframes())
    require(len(pcm) == 2*len(converted), "complete PCM readback")
    if not any(pcm): flags.append("derived_pcm_silent")
    for path in (raw,derived):
        with path.open("rb") as handle: os.fsync(handle.fileno())
        fsync_directory(path.parent)
    return {"raw_path": "raw/"+raw.name, "derived_path": "derived/"+derived.name,
            "raw_wav_sha256": sha(raw), "raw_float_sha256": hashlib.sha256(raw_values).hexdigest(),
            "wav_sha256": sha(derived), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "source_frames": len(array), "source_rate_hz": 24000, "source_channels": 1,
            "source_dtype": "little_endian_float32", "frames": len(converted),
            "sample_rate_hz": 16000, "channels": 1, "sample_width_bytes": 2,
            "derivation": "scipy-resample-poly-2-to-3-no-gain-no-trim", "quality_flags": flags,
            "source_peak":source_peak,"derivative_peak":derivative_peak,
            "derivative_float_sha256":hashlib.sha256(np.asarray(converted,dtype="<f8").tobytes()).hexdigest()}


def run_real(runtime, output):
    from tts_gate import verify_worker_release
    runtime, output = Path(runtime).absolute(), Path(output).absolute()
    verify_worker_release(runtime,output)  # before third-party imports or loader
    assert_no_host_secrets(os.environ)
    sys.path.insert(0,str(HELPERS))
    setup = load_helper("setup_locked")
    setup.verify_setup(runtime)
    provenance=json.loads((HERE/"model-provenance.json").read_bytes())
    for row in provenance["transformers_generation_sources"]:
        path=unaliased(runtime/"venv/lib/python3.12/site-packages"/row["path"],file=True)
        require(path.stat().st_size==row["bytes"] and sha(path)==row["sha256"],"installed generation source bytes")
    gate = load_helper("runtime_gate")
    keys = tuple(gate.cache_paths())
    gate.cache_paths = lambda: {k: runtime/"buildtmp/inference-cache"/k.lower() for k in keys}
    modeldir = runtime/"models/qwen_tts"
    model_lock = json.loads((HELPERS/"model-lock.json").read_bytes())
    require({p.relative_to(modeldir).as_posix() for p in modeldir.rglob("*") if p.is_file()} == {r["path"] for r in model_lock["files"]}, "exact model member inventory")
    for row in model_lock["files"]:
        p=unaliased(modeldir/row["path"],file=True)
        require(p.stat().st_size==row["size_bytes"] and sha(p)==row["sha256"],"model asset hash")
    sidecars = gate.inspect_model_sidecars(modeldir)
    torch,transformers,network_events = gate.prepare()
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from qwen_tts import Qwen3TTSModel
    loader_calls=[]
    def timeout(_signal,_frame): raise TimeoutError("fixed per-cell deadline")
    signal.signal(signal.SIGALRM,timeout)
    with gate.local_loader_policy(modeldir,torch,transformers,loader_calls):
        wrapper=Qwen3TTSModel.from_pretrained(str(modeldir),device_map="cpu",dtype=torch.float32,
            attn_implementation="eager",use_safetensors=True,weights_only=True,
            local_files_only=True,trust_remote_code=False)
        validate_model(wrapper)
        observed=capture_talker(wrapper)
        def generate(row):
            observed.clear(); validate_model(wrapper)
            random.seed(row["seed"]);np.random.seed(row["seed"]);torch.manual_seed(row["seed"])
            start=time.monotonic()
            with torch.inference_mode():
                wavs,rate=wrapper.generate_custom_voice(text=row["text"],speaker=row["voice"],**expected_plan()["tts"]["recipe"])
            require(type(wavs) in (list,tuple) and len(wavs)==1 and type(rate) is int and rate==24000,"one native24k waveform")
            return wavs[0],dict(observed),time.monotonic()-start
        # Exact sidecar/loader evidence is summarized by the host's strict projector.
        write_once(runtime/"model-load.private.json", {"sidecars":sidecars,"loaders":loader_calls,"network_events":network_events})
        return execute_cells(output,generate,lambda array,cell:write_pair(array,cell,output,np,sf,resample_poly),capture_state=observed)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    try: run_real(args.runtime,args.output)
    except BaseException as error:
        print(json.dumps({"status":"stopped_no_retry","error_class":safe_exception(error)}),file=sys.stderr)
        raise SystemExit(1)
