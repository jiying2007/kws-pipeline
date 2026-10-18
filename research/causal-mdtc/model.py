"""Prepared MDTC event adapter; never imported by the static fixtures.
New adapter, 2026-10-18. Upstream MDTC source is unchanged in reference/.
Only explicit build_model() imports torch and creates numerical model state.
"""
from pathlib import Path
import importlib.util


def build_model(mean, inverse_std, *, cmvn_identity):
    import torch
    from torch import nn
    if not isinstance(cmvn_identity, dict) or cmvn_identity.get('scope') not in ('synthetic_fixture_only', 'frozen_train_only'):
        raise ValueError('CMVN identity required')
    if cmvn_identity['scope'] == 'frozen_train_only' and not all(cmvn_identity.get(k) for k in ('source_manifest_sha256', 'statistics_sha256', 'split_manifest_sha256')):
        raise ValueError('train CMVN source/statistics/split hashes required')
    path = Path(__file__).parent / 'reference/wekws/model/mdtc.py'
    spec = importlib.util.spec_from_file_location('pinned_wekws_mdtc', path)
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)

    class EventMDTC(nn.Module):
        def __init__(self):
            super().__init__()
            self.cmvn_identity = dict(cmvn_identity)
            m = torch.as_tensor(mean, dtype=torch.float32)
            s = torch.as_tensor(inverse_std, dtype=torch.float32)
            if m.shape != (80,) or s.shape != (80,) or not torch.isfinite(m).all() or not torch.isfinite(s).all() or not (s > 0).all():
                raise ValueError('need finite train-only 80-bin CMVN, positive inverse std')
            self.register_buffer('mean', m.clone())
            self.register_buffer('inverse_std', s.clone())
            self.project = nn.Sequential(nn.Linear(80, 32), nn.ReLU())
            self.backbone = vendor.MDTC(3, 5, 32, 32, 5, causal=True)
            self.head = nn.Linear(32, 2)
            self.train(True)

        def train(self, mode=True):
            super().train(mode)
            # Deliberate adaptation: fixed running stats avoid future-frame and
            # padded-batch dependence during tiny-fit; gamma/beta remain trainable.
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d): module.eval()
            return self

        def _validate(self, features, cache=None):
            if features.ndim != 3 or features.shape[-1] != 80 or min(features.shape[:2]) <= 0:
                raise ValueError('expected nonempty B,T,80 raw log-fbank; CMVN applied internally once')
            if features.dtype != self.mean.dtype or features.device != self.mean.device:
                raise ValueError('feature dtype/device must match model')
            if not torch.isfinite(features).all(): raise ValueError('nonfinite raw features')
            if cache is not None:
                if tuple(cache.shape) != (features.shape[0],32,376):
                    raise ValueError('cache must be B,32,376; reset uses None')
                if cache.dtype != features.dtype or cache.device != features.device:
                    raise ValueError('cache dtype/device must match features')
                if not torch.isfinite(cache).all(): raise ValueError('nonfinite cache')

        def _core(self, features, cache=None):
            self._validate(features, cache)
            x = self.project((features-self.mean)*self.inverse_std)
            x, out_cache = self.backbone(x) if cache is None else self.backbone(x, cache)
            return self.head(x), out_cache

        @torch.no_grad()
        def forward(self, features, cache=None):
            """Streaming inference ONLY. Every input frame must be real, never pad.
            No lengths API: caller owns per-stream framing. Reset uses None.
            """
            if self.training: raise ValueError('stream inference requires eval()')
            logits, cache = self._core(features, cache)
            return logits, cache.detach()

        def train_padded(self, features, lengths):
            """Independent clips, reset history, right padding allowed.
            Returned cache is always discarded. One full graph per batch; caller
            backward once then releases graph. No cached-state TBPTT supported.
            """
            if not self.training: raise ValueError('train_padded requires train()')
            self._validate(features)
            if lengths.shape != (features.shape[0],) or lengths.device != features.device or lengths.dtype != torch.int64:
                raise ValueError('lengths must be device-matched int64 B')
            if not ((lengths > 0)&(lengths <= features.shape[1])).all(): raise ValueError('invalid lengths')
            logits, _discarded_cache = self._core(features, None)
            return logits

        @torch.no_grad()
        def posterior(self, features, cache=None):
            logits, cache = self(features, cache)
            return logits.sigmoid(), cache

    return EventMDTC()


def event_loss(logits, lengths, labels, weights):
    """Independent max-logit BCE == max-posterior BCE. No pre-end negative frames.
    Labels B,2 are explicit binary targets; weights B,2 carry evidence strength.
    All-unknown batches must be skipped before model forward by the caller.
    """
    import torch
    import torch.nn.functional as F
    if logits.ndim != 3 or logits.shape[-1] != 2 or min(logits.shape[:2]) <= 0:
        raise ValueError('logits must be B,T,2')
    b,t,_ = logits.shape
    if lengths.shape != (b,) or labels.shape != (b,2) or weights.shape != (b,2):
        raise ValueError('invalid masks/labels')
    if not logits.is_floating_point() or labels.dtype != logits.dtype or weights.dtype != logits.dtype or lengths.dtype != torch.int64 or any(x.device != logits.device for x in (lengths,labels,weights)):
        raise ValueError('loss dtype/device mismatch')
    if not torch.isfinite(logits).all() or not torch.isfinite(weights).all() or not torch.isfinite(labels).all():
        raise ValueError('nonfinite inputs')
    if not ((lengths > 0)&(lengths <= t)&(lengths == lengths.floor())).all():
        raise ValueError('invalid lengths')
    if not ((labels == 0)|(labels == 1)).all() or not (weights >= 0).all() or weights.sum() <= 0:
        raise ValueError('invalid targets or all unknown')
    valid = torch.arange(t,device=logits.device)[None,:] < lengths[:,None]
    pooled = logits.masked_fill(~valid[:,:,None], float('-inf')).amax(dim=1)
    loss = F.binary_cross_entropy_with_logits(pooled, labels.to(logits.dtype), reduction='none')
    return (loss*weights).sum()/weights.sum()
