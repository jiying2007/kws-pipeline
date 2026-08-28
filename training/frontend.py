from __future__ import annotations
import math, torch

def _hz_to_mel(hz:float)->float: return 2595.0*math.log10(1.0+hz/700.0)
def _mel_to_hz(mel:float)->float: return 700.0*(10.0**(mel/2595.0)-1.0)
def mel_bins(feature_dim:int)->list[int]:
    lo,hi=_hz_to_mel(80.0),_hz_to_mel(7600.0); out=[]
    for m in range(feature_dim+2):
        hz=_mel_to_hz(lo+(m/(feature_dim+1))*(hi-lo)); out.append(min(256,int(math.floor(513.0*hz/16000.0))))
    return out

def features(wave:torch.Tensor,feature_dim:int=32,frame_len:int=400,hop:int=320)->torch.Tensor:
    if wave.ndim!=1: raise ValueError('wave must be mono')
    if wave.numel()<frame_len: wave=torch.nn.functional.pad(wave,(0,frame_len-wave.numel()))
    frames=wave.unfold(0,frame_len,hop); window=torch.hann_window(frame_len,periodic=False,dtype=wave.dtype,device=wave.device)
    spec=torch.fft.rfft(frames*window,n=512); power=spec.real.square()+spec.imag.square(); bins=mel_bins(feature_dim); feats=[]
    for m in range(feature_dim):
        l,c,r=bins[m:m+3]; c=max(c,l+1); r=min(257,max(r,c+1)); e=wave.new_zeros((frames.shape[0],))
        if l<c:
            w=(torch.arange(l,c,device=wave.device,dtype=wave.dtype)-l)/(c-l); e=e+(power[:,l:c]*w).sum(dim=1)
        if c<r:
            w=(r-torch.arange(c,r,device=wave.device,dtype=wave.dtype))/(r-c); e=e+(power[:,c:r]*w).sum(dim=1)
        feats.append(torch.log1p(32.0*e))
    x=torch.stack(feats,dim=1); return (x-x.mean(dim=1,keepdim=True))*0.25
