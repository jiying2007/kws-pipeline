#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, struct
import torch

def q8(t:torch.Tensor)->tuple[bytes,float]:
    t=t.detach().cpu().float().contiguous(); mx=max(float(t.abs().max()),1.0e-8); scale=mx/127.0; q=torch.clamp(torch.round(t/scale),-127,127).to(torch.int8); return q.numpy().tobytes(),scale

def align4(buf:bytearray)->None:
    while len(buf)%4: buf.append(0)

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True,type=pathlib.Path); ap.add_argument('--output',required=True,type=pathlib.Path); a=ap.parse_args(); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True); sd=ck['state_dict']
    wx,sx=q8(sd['in_proj.weight']); wh,sh=q8(sd['rec_proj.weight']); wo,so=q8(sd['out_proj.weight']); bh=sd['in_proj.bias'].detach().cpu().float().contiguous().numpy().tobytes(); bo=sd['out_proj.bias'].detach().cpu().float().contiguous().numpy().tobytes()
    buf=bytearray(b'\x00'*64); offsets=[]
    for block in (wx,wh,bh,wo,bo): align4(buf); offsets.append(len(buf)); buf+=block
    total=len(buf); header=struct.pack('<4sHHHHHHIIIfffIIIIII',b'KWSP',1,64,int(ck['feature_dim']),int(ck['hidden_dim']),int(ck['vocab_size']),0,16000,int(ck['frame_length_samples']),int(ck['frame_hop_samples']),sx,sh,so,*offsets,total); assert len(header)==64; buf[:64]=header; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_bytes(buf); print(f'wrote {a.output}: {total} bytes')
if __name__=='__main__': main()
