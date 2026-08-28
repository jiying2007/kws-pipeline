from __future__ import annotations
import torch
from torch import nn
class TinyStreamingRNN(nn.Module):
    """One-state streaming acoustic model mirrored by src/kws.c."""
    def __init__(self, feature_dim:int, hidden_dim:int, vocab_size:int):
        super().__init__(); self.in_proj=nn.Linear(feature_dim,hidden_dim); self.rec_proj=nn.Linear(hidden_dim,hidden_dim,bias=False); self.out_proj=nn.Linear(hidden_dim,vocab_size)
    def step(self,x:torch.Tensor,h:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        h=torch.tanh(self.in_proj(x)+self.rec_proj(h)); return self.out_proj(h),h
    def forward(self,x:torch.Tensor,lengths:torch.Tensor|None=None)->torch.Tensor:
        del lengths
        h=x.new_zeros((x.shape[0],self.rec_proj.in_features)); ys=[]
        for t in range(x.shape[1]):
            y,h=self.step(x[:,t,:],h); ys.append(y)
        return torch.stack(ys,dim=0)
