#!/usr/bin/env python3
from __future__ import annotations
import argparse, pathlib, random, wave
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from frontend import features
from model import TinyStreamingRNN

class Manifest(Dataset):
    def __init__(self,path:pathlib.Path,feature_dim:int):
        self.rows=[]; self.feature_dim=feature_dim; root=path.parent
        for raw in path.read_text(encoding='utf-8').splitlines():
            if not raw.strip() or raw.startswith('#'): continue
            wav_path,token_text=raw.split('\t',1); p=pathlib.Path(wav_path); p=p if p.is_absolute() else root/p; self.rows.append((p,[int(x) for x in token_text.split()]))
    def __len__(self): return len(self.rows)
    def __getitem__(self,idx):
        path,tokens=self.rows[idx]
        with wave.open(str(path),'rb') as wf:
            if wf.getnchannels()!=1 or wf.getframerate()!=16000 or wf.getsampwidth()!=2: raise ValueError(f'{path}: expected mono 16-kHz PCM16 WAV')
            raw=wf.readframes(wf.getnframes())
        pcm=torch.frombuffer(bytearray(raw),dtype=torch.int16).float()/32768.0
        return features(pcm,self.feature_dim),torch.tensor(tokens,dtype=torch.long)

def collate(batch):
    xs,ys=zip(*batch); xlen=torch.tensor([x.shape[0] for x in xs],dtype=torch.long); ylen=torch.tensor([y.shape[0] for y in ys],dtype=torch.long); max_t=int(xlen.max()); f=xs[0].shape[1]; padded=torch.zeros((len(xs),max_t,f))
    for i,x in enumerate(xs): padded[i,:x.shape[0]]=x
    return padded,torch.cat(ys),xlen,ylen

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True,type=pathlib.Path); ap.add_argument('--vocab-size',required=True,type=int); ap.add_argument('--output',required=True,type=pathlib.Path); ap.add_argument('--feature-dim',type=int,default=32); ap.add_argument('--hidden-dim',type=int,default=48); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=16); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--seed',type=int,default=1337); ap.add_argument('--warm-start',type=pathlib.Path); ap.add_argument('--head-only',action='store_true'); a=ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed); ds=Manifest(a.manifest,a.feature_dim); dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,collate_fn=collate); model=TinyStreamingRNN(a.feature_dim,a.hidden_dim,a.vocab_size)
    if a.warm_start: model.load_state_dict(torch.load(a.warm_start,map_location='cpu',weights_only=True)['state_dict'],strict=True)
    if a.head_only:
        for p in model.in_proj.parameters(): p.requires_grad=False
        for p in model.rec_proj.parameters(): p.requires_grad=False
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4); loss_fn=nn.CTCLoss(blank=0,zero_infinity=True); model.train()
    for epoch in range(a.epochs):
        total=0.0
        for x,y,xlen,ylen in dl:
            logits=model(x).log_softmax(dim=2); loss=loss_fn(logits,y,xlen,ylen); opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step(); total+=float(loss.detach())
        print(f'epoch={epoch+1} loss={total/max(1,len(dl)):.6f}')
    a.output.parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.state_dict(),'feature_dim':a.feature_dim,'hidden_dim':a.hidden_dim,'vocab_size':a.vocab_size,'frame_length_samples':400,'frame_hop_samples':320},a.output)
if __name__=='__main__': main()
