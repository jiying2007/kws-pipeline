"""Exact arithmetic excerpt from the executed fixed37 research runner.

No audio, labels, weights, paths or runnable training recipe are distributed.
Coefficient was fixed before this experiment: control0, treatment0.1.
This excerpt was AST-inspected, not separately numerically re-executed.
"""
def fixed_teacher_regularizer(logits, ref, lens):
 import torch
 nonblank=[];binary=[]
 for i,n in enumerate(lens):
  s=logits[i,:n];t=ref[i,:n];sl=s[:,1:].log_softmax(-1);tl=t[:,1:].log_softmax(-1)
  nonblank.append((tl.exp()*(tl-sl)).sum(-1).mean())
  sb=torch.stack((s[:,0],s[:,1:].logsumexp(-1)),dim=-1).log_softmax(-1);tb=torch.stack((t[:,0],t[:,1:].logsumexp(-1)),dim=-1).log_softmax(-1)
  binary.append((tb.exp()*(tb-sb)).sum(-1).mean())
 kn=torch.stack(nonblank).mean();kb=torch.stack(binary).mean();kd=.5*(kn+kb)
 return kn,kb,kd
