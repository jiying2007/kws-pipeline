"""One live 140-D teacher row to the fixed A branch; torch is supplied by caller."""
import hashlib
import importlib.util
import json
from pathlib import Path
from online_merge import OnlineMerge


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_segment(actual, expected):
    for key in ('source_samples_supplied', 'fbank_frames', 'network_frames', 'posterior_segment'):
        require(actual[key] == expected[key], 'exact old B prefix segment: ' + key)


class JointBranch:
    def __init__(self, torch, saved, cfg, root, counts, limit, logit_file, event_file, profile_audit):
        self.torch, self.counts, self.limit = torch, counts, limit
        self.logit_file, self.event_file, self.profile_audit = logit_file, event_file, profile_audit
        require(saved['completed_steps'] == 240 and saved['freeze_sha'] == cfg['training_freeze_sha'], 'A checkpoint identity')
        require('input_normalization' not in saved, 'A original no LN')
        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, root / path)
            mod = importlib.util.module_from_spec(spec)
            import sys
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        core = load('joint_original_event_head', cfg['core_path'])
        state_core = load('joint_original_event_state', cfg['state_path'])
        self.model = core.build_torch(torch)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval().requires_grad_(False)
        require(sum(p.numel() for p in self.model.parameters()) == 10498, 'A parameter count')
        self.initial_hashes = self.hashes()
        self.state = None
        self.events = state_core.EventReplay([0.95, 0.85], cooldown_ms=1000)
        self.merge = OnlineMerge()
        self.hash = hashlib.sha256()
        self.rows = 0
        self.raw_event_counts = [0, 0]
        self.union_events = 0

    @staticmethod
    def raw(t):
        return t.detach().cpu().contiguous().numpy().tobytes()

    def hashes(self):
        return {k: hashlib.sha256(self.raw(v)).hexdigest() for k, v in self.model.state_dict().items()}

    def reserve(self, key):
        require(self.counts[key + '_reserved'] < self.limit[key], 'joint quota: ' + key)
        self.counts[key + '_reserved'] += 1

    def step(self, row, h, supplied, b_events):
        require(row == self.rows and supplied == ((480 * row + 720 + 319) // 320) * 320,
                'actual320 cursor/row invariant')
        before_h = self.raw(h)
        self.reserve('head_forward')
        z, self.state = self.model(h, self.state)
        self.counts['head_forward_completed'] += 1
        require(tuple(z.shape) == (1, 1, 2) and bool(self.torch.isfinite(z).all()), 'A logits shape/finite')
        require(all(bool(self.torch.isfinite(x).all()) for x in self.state), 'A state finite')
        require(sum(x.numel() * x.element_size() for x in self.state) == 7936, 'A state bytes')
        require(all(x.untyped_storage().nbytes() == x.numel() * x.element_size() for x in self.state), 'owned A state')
        require(self.raw(h) == before_h, 'A did not mutate teacher hook')
        raw = self.raw(z)
        self.logit_file.write(raw)
        self.hash.update(raw)
        self.reserve('head_sigmoid')
        prob = z.sigmoid()[0, 0].tolist()
        self.counts['head_sigmoid_completed'] += 1
        a_events = []
        if row >= 8:
            self.reserve('head_state')
            a_events = self.events.step(supplied / 16, prob)
            self.counts['head_state_completed'] += 1
        a_kw2 = []
        for event in a_events:
            k = event['keyword']
            self.raw_event_counts[k] += 1
            normalized = dict(keyword=k, score=event['score'], decision_row=row,
                              required_samples=480 * row + 720, actual_supplied_samples=supplied,
                              source_event=event)
            self.emit('A_raw', normalized)
            if k == 1:
                a_kw2.append(normalized)
        b_normalized = [dict(keyword=e['keyword_id'] - 1, score=e['score'], decision_row=row,
                             required_samples=480 * row + 720, actual_supplied_samples=supplied,
                             source_event=e) for e in b_events]
        self.reserve('merge')
        kept, removed = self.merge.step(supplied, b_normalized, a_kw2)
        self.counts['merge_completed'] += 1
        for event in kept:
            self.emit('union', event)
        for event in removed:
            self.emit('union_suppressed', event)
        self.union_events += len(kept)
        self.rows += 1

    def emit(self, kind, event):
        if kind == 'A_raw' and event['keyword'] == 1:
            event = dict(event, mask_labels=self.profile_audit.record('A_KW2', event))
        elif kind == 'union':
            event = dict(event, mask_labels=self.profile_audit.record('OR_conservative', event))
        encoded = json.dumps(dict(kind=kind, event=event), ensure_ascii=False, allow_nan=False) + '\n'
        require(len(encoded.encode()) <= 16384, 'joint event line cap')
        self.event_file.write(encoded)

    def summary(self):
        return dict(rows=self.rows, logits_sha256=self.hash.hexdigest(), A_raw_events=self.raw_event_counts,
                    union_events=self.union_events, suppressed=self.merge.suppressed,
                    internal_state_parity_claimed=False, threads=2)
