"""Review-only causal temporal event head. Import uses standard library only.
No checkpoint loading, teacher calls, training loop, or network access.
"""
from dataclasses import dataclass
import math

KEYWORDS = ('你好小窝', '小窝小窝')

@dataclass(frozen=True)
class Spec:
    input_dim: int = 140
    width: int = 32
    dilations: tuple = (1, 2, 4, 8, 16)
    kernel: int = 3
    frame_ms: int = 30
    def __post_init__(self):
        if self.input_dim < 1 or self.width < 1 or self.kernel < 2 or not self.dilations or any(d < 1 for d in self.dilations):
            raise ValueError('invalid positive dimensions')
    def budget(self):
        c, h, k, n = self.input_dim, self.width, self.kernel, len(self.dilations)
        mac = c*h + n*(h*k+h*h) + h*2
        state = h*(k-1)*sum(self.dilations)
        return dict(parameters=c*h+h+n*(h*k+h+h*h+h)+h*2+2,
                    mac_per_frame=mac, mac_per_second=mac*1000/self.frame_ms,
                    receptive_field_frames=1+(k-1)*sum(self.dilations),
                    history_ms=(k-1)*sum(self.dilations)*self.frame_ms,
                    state_scalars=state, fp32_state_bytes=state*4,
                    additional_lookahead_ms=0)

def timing(frame_index, original_samples, sample_rate=16000):
    """Source-derived original front-end: 25ms window, +/-2 hops, skip3.
    FSMN nominal center delay 4*2*30ms; never backdate event availability.
    Synthetic suffix is identified, NOT presented as a model input.
    """
    if frame_index < 0 or original_samples <= 0 or sample_rate != 16000:
        raise ValueError('unsupported timing input')
    available_sample = frame_index*480 + 720
    return dict(available_ms=available_sample/16,
                nominal_center_ms=frame_index*30-240,
                needs_synthetic_suffix=available_sample > original_samples,
                pre_eof=available_sample <= original_samples)

def build_torch(torch, spec=Spec()):
    """Construct only when caller explicitly supplies an imported torch module.
    Input is detached cached teacher representation, B,T,140. Outputs are
    B,T,2 logits (use independent sigmoid), not a mutually-exclusive softmax.
    No batch/time normalization: avoids padding leakage and time-axis leakage.
    Streaming inference caller must use torch.no_grad(); returned states own compact
    storage. No unconditional detach on states, so explicit chunk BPTT stays possible.
    """
    nn, F = torch.nn, torch.nn.functional
    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__(); self.history=(spec.kernel-1)*dilation
            self.dw=nn.Conv1d(spec.width,spec.width,spec.kernel,dilation=dilation,groups=spec.width)
            self.pw=nn.Conv1d(spec.width,spec.width,1)
        def forward(self,x,state=None):
            if state is None: state=x.new_zeros(x.shape[0],spec.width,self.history)
            if tuple(state.shape)!=(x.shape[0],spec.width,self.history): raise ValueError('state shape')
            z=torch.cat((state,x),dim=2)
            y=F.relu(x+self.pw(F.relu(self.dw(z))))
            return y,z[:,:,-self.history:].clone()
    class EventHead(nn.Module):
        def __init__(self):
            super().__init__(); self.project=nn.Conv1d(spec.input_dim,spec.width,1)
            self.blocks=nn.ModuleList([Block(d) for d in spec.dilations]); self.out=nn.Conv1d(spec.width,2,1)
        def forward(self,features,state=None):
            if features.ndim!=3 or features.shape[2]!=spec.input_dim or features.shape[1]<1: raise ValueError('B,T,C input')
            if state is not None and len(state)!=len(self.blocks): raise ValueError('state count')
            x=F.relu(self.project(features.detach().transpose(1,2))); new=[]
            for i,b in enumerate(self.blocks):
                x,s=b(x,None if state is None else state[i]); new.append(s)
            return self.out(x).transpose(1,2),new
    return EventHead()

def event_loss(torch, logits, labels, frame_mask, weights):
    """Per-clip two-label MIL diagnostic loss; labels -1 means ABSTAIN.
    Positive: maximum valid event logit. Negative: top3 valid peaks' BCE.
    Positive utterance frames before completion are not all forced negative.
    Padding must be excluded with frame_mask; masks must never enter the model.
    Caller must use only audited labels and reset state for each recording.
    """
    if logits.ndim!=3 or logits.shape[2]!=2: raise ValueError('two logits required')
    if tuple(labels.shape)!=(logits.shape[0],2) or tuple(weights.shape)!=tuple(labels.shape): raise ValueError('label/weight shape')
    if tuple(frame_mask.shape)!=tuple(logits.shape[:2]) or frame_mask.dtype!=torch.bool: raise ValueError('bool frame mask required')
    if not bool(((labels==-1)|(labels==0)|(labels==1)).all()) or not bool(torch.isfinite(weights).all()) or bool((weights<0).any()): raise ValueError('invalid labels/weights')
    terms=[]; den=[]
    for b in range(logits.shape[0]):
        for k in range(2):
            label=int(labels[b,k]); weight=weights[b,k]
            if label<0 or float(weight)==0: continue
            z=logits[b,:,k][frame_mask[b]]
            if z.numel()==0: raise ValueError('known label with no valid frames')
            term=torch.nn.functional.softplus(-z.max()) if label==1 else torch.nn.functional.softplus(z.topk(min(3,z.numel())).values).mean()
            terms.append(term*weight); den.append(weight)
    if not terms: raise ValueError('no supervised labels')
    return torch.stack(terms).sum()/torch.stack(den).sum()

class ReferenceHead:
    """Standard-library executable oracle; same causal equations, no training."""
    def __init__(self, spec, seed=7):
        import random
        self.spec=spec; r=random.Random(seed)
        def matrix(n,m): return [[r.uniform(-.2,.2) for _ in range(m)] for _ in range(n)]
        self.proj=matrix(spec.width,spec.input_dim); self.pb=[.1]*spec.width
        self.dw=[matrix(spec.width,spec.kernel) for _ in spec.dilations]
        self.pw=[matrix(spec.width,spec.width) for _ in spec.dilations]
        self.db=[[.1]*spec.width for _ in spec.dilations]; self.wb=[[.1]*spec.width for _ in spec.dilations]
        self.out=matrix(2,spec.width); self.ob=[0.,0.]; self.reset()
    def reset(self): self.state=[[[0.]*self.spec.width for _ in range((self.spec.kernel-1)*d)] for d in self.spec.dilations]
    def chunk(self,frames):
        result=[]; s=self.spec
        for frame in frames:
            if len(frame)!=s.input_dim or not all(math.isfinite(v) for v in frame): raise ValueError('invalid frame')
            x=[max(0.,sum(a*b for a,b in zip(w,frame))+b) for w,b in zip(self.proj,self.pb)]
            for j,d in enumerate(s.dilations):
                history=self.state[j]+[x]; end=len(history)-1
                z=[max(0.,self.db[j][c]+sum(self.dw[j][c][q]*history[end-(s.kernel-1-q)*d][c] for q in range(s.kernel))) for c in range(s.width)]
                y=[max(0.,x[c]+self.wb[j][c]+sum(a*b for a,b in zip(self.pw[j][c],z))) for c in range(s.width)]
                self.state[j]=history[1:]; x=y
            result.append([sum(a*b for a,b in zip(w,x))+b for w,b in zip(self.out,self.ob)])
        return result
