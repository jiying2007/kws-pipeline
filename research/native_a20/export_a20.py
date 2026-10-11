"""Exact pinned A20 export from a caller-supplied checkpoint. No model forward, inference, training or network."""
import argparse,hashlib,json,math,sys
from pathlib import Path
FILE_SHA='5b347c0ce7df6e8ad3649c9186a11acab4b3e35971950934490c511d2b04bc39'
STATE_SHA='c05623683b4616badda5535eefb0bfaecde63efbf3bfbceec6e7440a5fc19eb2'
SYMBOLS=['<blank>','你','好','小','窝','屋']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(condition, message):
 if not condition: raise ValueError(message)
def save(p,o):Path(p).write_text(json.dumps(o,indent=2,sort_keys=True,ensure_ascii=False)+'\n')
def inventory():
 out=[('global_cmvn.mean',[400]),('global_cmvn.istd',[400])]
 def aff(n,ni,no):out.extend([(n+'.linear.weight',[no,ni]),(n+'.linear.bias',[no])])
 aff('backbone.in_linear1',400,140);aff('backbone.in_linear2',140,250)
 for i in range(4):
  p=f'backbone.fsmn.{i}';out.extend([(p+'.0.linear.weight',[128,250]),(p+'.1.conv_left.weight',[128,1,10,1]),(p+'.1.conv_right.weight',[128,1,2,1])]);aff(p+'.2',128,250)
 aff('backbone.out_linear1',250,140);aff('backbone.out_linear2',140,6);return out
def read_state(checkpoint):
 CHECKPOINT=Path(checkpoint)
 require(sys.byteorder=='little', 'A20 export requires little-endian host')
 require(sha(CHECKPOINT)==FILE_SHA, 'A20 checkpoint SHA256 mismatch')
 import torch
 cp=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
 require(isinstance(cp,dict) and cp.get('arm')=='A' and type(cp.get('step')) is int and cp['step']==1200 and cp.get('symbols')==SYMBOLS, 'A20 checkpoint metadata mismatch')
 state=cp.get('model');inv=inventory();require(isinstance(state,dict) and list(state)==[n for n,s in inv], 'A20 state tensor inventory mismatch')
 h=hashlib.sha256();payload=bytearray();manifest=[]
 for n,s in inv:
  t=state[n];require(type(t)is torch.Tensor and t.dtype==torch.float32 and list(t.shape)==s and bool(torch.isfinite(t).all()), 'A20 tensor type/shape/dtype/finite mismatch: '+n)
  a=t.detach().contiguous().numpy();raw=a.tobytes();require(len(raw)==math.prod(s)*4, 'A20 tensor byte count mismatch: '+n);h.update(n.encode());h.update(raw)
  manifest.append(dict(name=n,shape=s,offset=len(payload),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()));payload.extend(raw)
 require(h.hexdigest()==STATE_SHA and len(payload)==1565280, 'A20 state SHA256 or payload length mismatch')
 require(bool((state['global_cmvn.istd']>0).all()), 'A20 CMVN inverse standard deviation must be positive')
 return state,bytes(payload),manifest
def export(checkpoint,out):
 state,payload,manifest=read_state(checkpoint)
 out=Path(out);out.mkdir(exist_ok=False)
 (out/'a20.f32').write_bytes(payload)
 record=dict(schema='a20-native-fp32-v1',checkpoint_sha256=FILE_SHA,state_sha256=STATE_SHA,payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest(),tensors=manifest,symbols=SYMBOLS,forward_calls=0)
 save(out/'manifest.json',record)
 (out/'a20_identity.h').write_text('#define A20_EXPECTED_SHA "'+record['payload_sha256']+'"\n')
 print(json.dumps({k:v for k,v in record.items() if k!='tensors'},ensure_ascii=False))
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--checkpoint',type=Path,required=True)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();export(args.checkpoint,args.output)
