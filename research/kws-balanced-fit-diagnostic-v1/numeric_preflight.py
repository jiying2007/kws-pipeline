"""Synthetic forward/backward only, zero optimizer calls and no external audio.

Run separately in the locked research runtime, never in lieu of input admission.
"""
import argparse
from pathlib import Path
import sys

def main(repo):
    sys.path.insert(0,str(repo/'training'))
    from model import TinyStreamingRNN
    from frontend import features
    import torch
    torch.manual_seed(20261015)
    torch.use_deterministic_algorithms(True)
    pcm=torch.sin(torch.arange(6400,dtype=torch.float32)*0.07)*0.1
    f=features(pcm,feature_dim=32)
    if features(pcm[:399],feature_dim=32).shape != (0,32):raise ValueError('frontend EOF boundary')
    if not torch.equal(f,features(pcm,feature_dim=32)):raise ValueError('frontend nondeterminism')
    m=TinyStreamingRNN(32,48,5)
    lp=m(torch.stack((f,f*.8))).log_softmax(2)
    nll=torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=False)(lp,torch.tensor([1,2,3,4,3,4,3,4]),torch.tensor([len(f)]*2),torch.tensor([4,4]))
    if not torch.isfinite(nll).all():raise ValueError('CTC forward nonfinite')
    (nll/len(f)).mean().backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('CTC backward nonfinite')
    if sum(float(p.grad.square().sum()) for p in m.parameters()) <= 0:raise ValueError('zero total gradient')
    # Impossible CTC alignment must remain visible, never silently zeroed.
    bad=torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=False)(lp[:1,:1],torch.tensor([3,3]),torch.tensor([1]),torch.tensor([2]))
    if not torch.isinf(bad).all():raise ValueError('impossible CTC did not expose infinity')
    # A/B normalization remains frame-based and differs only by sample weights.
    losses=torch.arange(1,17,dtype=torch.float32); lengths=torch.arange(10,26,dtype=torch.float32)
    equal=(losses/lengths).mean(); balanced=((losses/lengths)*torch.tensor([.1]*5+[1/22]*11)).sum()
    if not torch.isfinite(equal+balanced) or equal==balanced:raise ValueError('weighting control ineffective')
    print('Synthetic frontend/CTC forward/backward passed; optimizer calls=0; no production qualification.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);main(p.parse_args().repo)
