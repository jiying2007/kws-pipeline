"""Byte-locked export of exactly the existing trained V5 Hey Mero 43K FSMN.

Source: immutable public kws-data run 44495594627 artifact 14514228406,
ZIP sha256 91b203f0..., not freshly trained weights. The pinned NPZ already
passed independent PT/ONNX stagewise readback in run 44519849543.

Binary pack (little endian): 8 ASCII 'MERO43V1', uint32 version=1,
uint32 float_count=54236, then contiguous float32 arrays in *original*
PyTorch state_dict order; frozen [257,40] Mel bank, mean[40], std[40].
The byte loader never guesses shapes. No old other-keyword model supported.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile

SCHEMA="en-v5-mero43-immutable-fsmn-ctc-binary-pack-v1"
ORIGINAL_RUN=44495594627
ARTIFACT_ID=14514228406
ZIP_SHA256="91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702"
PT_SHA256="1ae8a2b20c8b6991f925dd4c626af2f36b308adee8733a3a7215818716f565d7"
NPZ_SHA256="02b7febf9f736fd70f4f9ae04042ffc6d609b15011d03d235b960f69b1d71696"
ONNX_SHA256="10bf30da57291f78f56db196db1137309f2c3b91dfe67bf0aa45ebb8cec05b1c"
PREFIX="mero-fsmn-ctc-expanded256"
PARAMETERS=43876
PACK_FLOAT_COUNT=54236
PACK_BYTES=16+4*PACK_FLOAT_COUNT
MAGIC=b"MERO43V1"
VERSION=1

def require(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def geometry():
    specs=[("input.weight",(96,40)),("input.bias",(96,))]
    for layer in range(4):
        path=f"layers.{layer}."
        specs += [
            (path+"linear.weight",(96,96)),
            (path+"linear.bias",(96,)),
            (path+"memory.weight",(96,1,5)),
            (path+"memory.bias",(96,))]
    specs += [("output.weight",(4,96)),("output.bias",(4,)),
              ("mel_filters",(257,40)),("train_mean",(40,)),
              ("train_scale",(40,))]
    return specs

def source_check():
    specs=geometry()
    weight_count=sum(__import__("math").prod(dims)
                     for name,dims in specs if name not in
                     ("mel_filters","train_mean","train_scale"))
    all_count=sum(__import__("math").prod(dims) for _,dims in specs)
    require(weight_count==PARAMETERS
            and all_count==PACK_FLOAT_COUNT
            and PACK_BYTES==216960
            and len(specs)==2+4*4+2+3
            and len(set(name for name,_ in specs))==len(specs)
            and ORIGINAL_RUN==44495594627 and ARTIFACT_ID==14514228406,
            "FROZEN_43K_FSMN_GEOMETRY_AND_BINARY_LAYOUT")
    return {"schema":SCHEMA,"status":"SOURCE_ONLY_NO_NEW_MODEL",
            "artifact_id":ARTIFACT_ID,"optimizer_updates":0,
            "model_parameters":PARAMETERS,"model_zip_sha256":ZIP_SHA256,
            "model_npz_sha256":NPZ_SHA256,
            "model_onnx_sha256":ONNX_SHA256,
            "pack_bytes":PACK_BYTES,"active_keyword":"Hey Mero",
            "new_audio_generated":0,"new_asr_kws_forwards":0,
            "calibration_or_locked_pcm_opened":False,
            "ssc305_qualified":False}

def pack(original,output):
    import numpy as np
    source_check()
    require(original.is_file() and not original.is_symlink()
            and original.stat().st_size==501276
            and sha(original.read_bytes())==ZIP_SHA256,
            "EXACT_ORIGINAL_PUBLIC_256_STEP_MODEL_ARCHIVE")
    with zipfile.ZipFile(original) as z:
        entries=z.namelist()
        require(len(entries)==7 and len(set(entries))==7 and
                set(entries)=={"STARTED.json","RESULT.json","TRAINING.json",
                   "ADMISSION.json",*(f"models/{PREFIX}.{suffix}"
                         for suffix in ("pt","npz","onnx"))},
                "EXACT_FROZEN_MODEL_ZIP_MEMBERS")
        report=json.loads(z.read("RESULT.json"))
        require(report.get("status")=="RESEARCH_NEW_SOURCE_DEV_NOT_QUALIFIED"
                and report.get("model_parameters")==PARAMETERS
                and report.get("new_actual_optimizer_updates")==256
                and report.get("product_promotion_allowed") is False
                and report.get("calibration_or_locked_pcm_opened") is False,
                "NO_OTHER_MODEL_RECLASSIFIED_AS_FROZEN_HEY_MERO")
        hashes={"pt":PT_SHA256,"npz":NPZ_SHA256,"onnx":ONNX_SHA256}
        for ext,expected in hashes.items():
            name=f"models/{PREFIX}.{ext}"
            raw=z.read(name)
            require(sha(raw)==expected and
                report["new_model_weights"][f"{PREFIX}.{ext}"]==
                     {"bytes":len(raw),"sha256":expected},
                     "EXACT_SAVED_PT_NPZ_ONNX_BYTES_"+ext)
        with np.load(io.BytesIO(z.read(f"models/{PREFIX}.npz")),
                     allow_pickle=False) as model:
            specs=geometry()
            require(set(model.files)=={name for name,_ in specs},
                    "EXACT_ONLY_EXPECTED_FSMN_AND_MEL_WEIGHTS")
            output_bytes=[]
            for name,shape in specs:
                array=model[name]
                require(array.shape==shape and array.dtype==np.dtype("float32")
                        and np.isfinite(array).all(),
                        "FROZEN_FLOAT32_FSMN_TENSOR_"+name)
                if name=="train_scale":
                    require(np.all(array>0),"POSITIVE_TRAIN_ONLY_SCALE")
                if name=="mel_filters":
                    require(np.all((array>=0)&(array<=1))
                            and np.all(array.sum(axis=0)>0),
                            "EXACT_VALID_40_HTK_MEL_TRIANGLES")
                output_bytes.append(np.ascontiguousarray(
                                  array,dtype="<f4").tobytes())
    blob=struct.pack("<8sII",MAGIC,VERSION,PACK_FLOAT_COUNT)+b"".join(output_bytes)
    require(len(blob)==PACK_BYTES,"EXACT_43K_BINARY_PACK_BYTES")
    output=output.resolve()
    require(output.parent.is_dir() and not output.exists(),
            "WRITE_NEW_MODEL_ONLY_NO_OVERWRITE")
    with output.open("xb") as out:out.write(blob)
    result={"schema":SCHEMA,"status":"PASS_BYTE_FROZEN_MODEL_REPACKAGE_NO_TRAIN",
            "run":ORIGINAL_RUN,"artifact_id":ARTIFACT_ID,
            "source_zip_sha256":ZIP_SHA256,
            "pt_sha256":PT_SHA256,"npz_sha256":NPZ_SHA256,
            "onnx_sha256":ONNX_SHA256,
            "model_parameters":PARAMETERS,
            "model_pack_sha256":sha(blob),"model_pack_bytes":len(blob),
            "input":"16k mono PCM16 -> 512 FFT / Hann400 -> 40 LogMel",
            "output":"blank HEY MEH ROH frame logits",
            "new_optimizer_updates":0,
            "old_calibration_or_locked_opened":False,
            "quality_or_board_promotion":False}
    receipt=output.with_suffix(output.suffix+".json")
    with receipt.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,sort_keys=True,indent=2)
        f.write("\n")
    print(json.dumps(result,sort_keys=True))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--source-zip",type=Path)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    if a.source_check:
        require(not a.source_zip and not a.output,
                "ONE_SOURCE_CHECK_ONLY")
        print(json.dumps(source_check(),sort_keys=True))
        return
    require(a.source_zip and a.output,"EXACT_ORIGINAL_ZIP_AND_NEW_OUTPUT")
    pack(a.source_zip,a.output)

if __name__=="__main__":main()
