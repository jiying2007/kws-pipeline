"""One-shot audit of the openWakeWord shared speech-embedding assets.

Only shared feature-extractor assets are inspected. No pretrained wake-word
classifier, private corpus, audio, training, or target promotion is involved.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

POLICY="english-embedding-backbone-asset-audit-v1"
UPSTREAM="dscripka/openWakeWord"
COMMIT="368c03716d1e92591906a84949bc477f3a834455"
RELEASE="v0.5.1"
EXPECTED={
 "melspectrogram.tflite":(129656230,1092516,"TFLite mel frontend"),
 "embedding_model.tflite":(129656244,1330312,"TFLite speech embedding"),
 "melspectrogram.onnx":(129656233,1087958,"ONNX mel frontend"),
 "embedding_model.onnx":(129656245,1326578,"ONNX speech embedding"),
}
TRIGGER=Path(__file__).resolve().parents[2]/"configs/research/english-embedding-backbone-assets-20261116.json"

def need(ok,msg):
 if not ok: raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def read_json(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def verify_trigger():
 x=read_json(TRIGGER)
 need(x.get("schema")==POLICY and x.get("upstream_repo")==UPSTREAM and
      x.get("upstream_commit")==COMMIT and x.get("release_tag")==RELEASE and
      x.get("code_license")=="Apache-2.0" and x.get("feature_model_license")=="Apache-2.0" and
      x.get("excluded_classifier_license")=="CC-BY-NC-SA-4.0" and
      x.get("download_once") is True and x.get("training") is False and
      x.get("audio_inference") is False and x.get("model_promotion") is False,
      "TRIGGER_IDENTITY")
 rows=x.get("assets")
 need(type(rows) is list and len(rows)==4,"TRIGGER_ASSET_COUNT")
 need({r.get("name"):(r.get("id"),r.get("bytes"),r.get("role")) for r in rows}==EXPECTED,
      "TRIGGER_ASSET_ROSTER")
 return x
def asset_report(root,release):
 root=Path(root)
 need(root.is_dir() and not root.is_symlink(),"ASSET_ROOT")
 need(release.get("tag_name")==RELEASE,"RELEASE_TAG")
 actual={a["name"]:(a["id"],a["size"],a["browser_download_url"]) for a in release.get("assets",[])}
 out={}
 for name,(asset_id,size,role) in EXPECTED.items():
  need(name in actual and actual[name][0]==asset_id and actual[name][1]==size,
       "RELEASE_ASSET_ID_OR_SIZE:"+name)
  p=root/name
  need(p.is_file() and not p.is_symlink() and p.stat().st_size==size,
       "LOCAL_ASSET_SIZE:"+name)
  out[name]={"asset_id":asset_id,"bytes":size,"sha256":sha(p),
             "role":role,"url":actual[name][2]}
 return out
def onnx_report(path):
 import onnx
 import onnxruntime as ort
 model=onnx.load(str(path),load_external_data=True)
 onnx.checker.check_model(model)
 graph=model.graph
 def tensor(v):
  t=v.type.tensor_type
  dims=[]
  for d in t.shape.dim:
   dims.append(d.dim_value if d.dim_value else (d.dim_param or None))
  return {"name":v.name,"elem_type":t.elem_type,"shape":dims}
 ops={}
 for node in graph.node:ops[node.op_type]=ops.get(node.op_type,0)+1
 params=0
 initializer_bytes=0
 for init in graph.initializer:
  count=math.prod(init.dims) if init.dims else 1
  params+=count
  initializer_bytes+=len(init.raw_data) if init.raw_data else 0
 sess=ort.InferenceSession(str(path),providers=["CPUExecutionProvider"])
 inputs=[]
 feed={}
 for x in sess.get_inputs():
  shape=[d if isinstance(d,int) and d>0 else 1 for d in x.shape]
  dt={"tensor(float)":np.float32,"tensor(double)":np.float64,
      "tensor(int64)":np.int64,"tensor(int32)":np.int32}.get(x.type)
  need(dt is not None,"ONNX_INPUT_TYPE:"+x.type)
  feed[x.name]=np.zeros(shape,dtype=dt)
  inputs.append({"name":x.name,"type":x.type,"shape":x.shape})
 values=sess.run(None,feed)
 outputs=[]
 for meta,val in zip(sess.get_outputs(),values):
  arr=np.asarray(val)
  outputs.append({"name":meta.name,"type":meta.type,"shape":list(arr.shape),
    "dtype":str(arr.dtype),"finite":bool(np.isfinite(arr).all()),
    "min":float(arr.min()) if arr.size else None,
    "max":float(arr.max()) if arr.size else None,
    "mean":float(arr.mean()) if arr.size else None})
 need(all(x["finite"] for x in outputs),"ONNX_NONFINITE_OUTPUT")
 return {"onnx_version":onnx.__version__,"onnxruntime_version":ort.__version__,
  "ir_version":model.ir_version,"opsets":[{"domain":x.domain,"version":x.version} for x in model.opset_import],
  "graph_inputs":[tensor(x) for x in graph.input],
  "graph_outputs":[tensor(x) for x in graph.output],
  "runtime_inputs":inputs,"runtime_outputs":outputs,"operators":ops,
  "initializer_count":len(graph.initializer),"initializer_parameters":params,
  "initializer_raw_bytes":initializer_bytes}
def audit(root,release_path,out):
 verify_trigger()
 release=read_json(release_path)
 assets=asset_report(root,release)
 report={"schema":POLICY,"status":"COMPLETE","upstream_repo":UPSTREAM,
  "upstream_commit":COMMIT,"release_tag":RELEASE,
  "license_review":{"repository_code":"Apache-2.0","shared_feature_model":"Apache-2.0",
   "pretrained_wakeword_classifiers":"CC-BY-NC-SA-4.0_EXCLUDED",
   "classifier_or_training_corpus_downloaded":False},
  "assets":assets,
  "tflite_pair_total_bytes":EXPECTED["melspectrogram.tflite"][1]+EXPECTED["embedding_model.tflite"][1],
  "onnx_pair_total_bytes":EXPECTED["melspectrogram.onnx"][1]+EXPECTED["embedding_model.onnx"][1],
  "onnx":{"melspectrogram":onnx_report(Path(root)/"melspectrogram.onnx"),
          "embedding_model":onnx_report(Path(root)/"embedding_model.onnx")},
  "zero_audio_inference":True,"training":False,"model_promotion":False,
  "ssc305_qualification":False}
 out=Path(out);need(not out.exists(),"CREATE_ONLY_REPORT")
 out.parent.mkdir(parents=True,exist_ok=True)
 out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
 return report
def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument("--assets",type=Path,required=True)
 p.add_argument("--release-json",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 a=p.parse_args(argv)
 r=audit(a.assets,a.release_json,a.output)
 print(json.dumps({"status":r["status"],"asset_count":len(r["assets"]),
  "onnx_mel_inputs":r["onnx"]["melspectrogram"]["runtime_inputs"],
  "onnx_embedding_inputs":r["onnx"]["embedding_model"]["runtime_inputs"],
  "training":False,"ssc305_qualification":False},sort_keys=True))
if __name__=="__main__":main()
