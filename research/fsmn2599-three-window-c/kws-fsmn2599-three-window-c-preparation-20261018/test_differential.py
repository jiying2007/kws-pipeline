"""Prepared stdlib ctypes differential runner. NOT RUN. No model/audio inputs.
Only runs with explicit CLI opt-in after the parent releases the CPU slot.
The opt-in is an accidental-execution guard, not a substitute for permission.
"""
import argparse
import ctypes as C
import hashlib
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
REF = ROOT.parent / 'kws-fsmn-three-window-adapter-20261017'
PINS = {
    'three_window_events.py': 'f65403a78bbe55566cae914062b1671c055c198a87fb7cb5c000d0a80d60cd26',
    'posterior_events.py': '56d6ada2c531de25f72a713db77b07ef9d8326b8b1e238bd776d40155d8584ec',
    'beam_minimal_patch_candidate.py': 'f822a40a136ea216cd9a3dd4dc9257000d69e15820cb6276360d9518a9800360',
    'vendor/stream_kws_ctc.py': '2a5d462f1c0830beee844427cbf0063e38f7b981dcd6e7990ec7a633acf46e53',
    'vendor/decoder_tail_fix.py': '5fc55921a555c78cb08c12018850652d98d4283594f875366284dc1df0e71f4d',
    'test_three_window_events.py': 'c9a0584ef6e5d22488e8f3b831b95c0771b16e480f1858b8542f6ff1d90f1ed9',
    'test_posterior_events.py': 'd99a79e6984bb7cb60954d2c0ae3cc36230779c91f451dd32d504bcb79152b4c',
    'test_beam_minimal_patch_candidate.py': '068d025c1e55101f0cec717471b7dbb33e8d38eb975fa8b1dec26e05a98a5889',
}
COUNTERS = ('rows_decoded beam_updates topk_computations windows_started windows_retired '
            'eof_windows_closed candidates_seen duplicate_candidates refractory_rejections '
            'events_emitted zero_mass_window_rows empty_beam_window_rows max_live_windows '
            'max_prefix_seen max_next_hypotheses max_dedup_keys max_live_node_references '
            'live_windows dedup_keys').split()
class Event(C.Structure):
    _fields_ = [('event_index', C.c_uint64), ('node_frames', C.c_int64*4)] + [
        (n,C.c_int64) for n in ('start_frame','end_frame','window_start','decision_row','decision_frame')] + [
        ('score',C.c_double),('grid_seconds',C.c_double),('keyword_id',C.c_uint32),
        ('diagnostic_candidate',C.c_uint32),('recording_id',C.c_char*256)]
class Summary(C.Structure):
    _fields_ = [(n,C.c_uint64) for n in COUNTERS] + [
        (n,C.c_int) for n in ('finished','valid','numerical_support_ok','invalid_reason')] + [
        ('recording_id',C.c_char*256)]
class Node(C.Structure):
    _fields_ = [('frame',C.c_int64),('prob',C.c_double)]
class Hyp(C.Structure):
    _fields_ = [('pb',C.c_double),('pnb',C.c_double),('length',C.c_size_t),
                ('tokens',C.c_uint16*126),('nodes',Node*126)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--library',required=True)
    ap.add_argument('--approved-small-synthetic-only',action='store_true')
    args=ap.parse_args()
    if not args.approved_small_synthetic_only:
        ap.error('parent CPU-slot release and explicit small-synthetic permission required')
    for name,sha in PINS.items():
        if hashlib.sha256((REF/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError('reference source identity mismatch: '+name)
    sys.path.insert(0,str(REF))
    import posterior_events as pe
    import three_window_events as tw
    import test_three_window_events as v2
    import test_beam_minimal_patch_candidate as patch
    from types import SimpleNamespace
    lib=C.CDLL(str(Path(args.library).resolve()))
    P=C.c_void_p; D=C.POINTER(C.c_double); E=C.POINTER(Event); H=C.POINTER(Hyp)
    Z=C.c_size_t; ZP=C.POINTER(Z); I64=C.c_int64; I64P=C.POINTER(I64)
    bindings={
        'kws3_create':([C.c_char_p,C.POINTER(P)],C.c_int),
        'kws3_destroy':([P],None), 'kws3_reset':([P,C.c_char_p],C.c_int),
        'kws3_step':([P,D,Z,E,Z,ZP],C.c_int), 'kws3_finish':([P],C.c_int),
        'kws3_get_summary':([P,C.POINTER(Summary)],C.c_int),
        'kws3_status_string':([C.c_int],C.c_char_p),
        'kws3_test_beam':([P,D,H,Z,I64,H,Z,ZP],C.c_int),
        'kws3_test_top3':([P,D,Z,C.POINTER(C.c_uint16),D],C.c_int),
        'kws3_test_snapshot':([P,Z,I64P,I64P,H,Z,ZP],C.c_int),
        'kws3_test_merge':([P,E,Z,I64,E,Z,ZP],C.c_int),
    }
    for name,(inputs,output) in bindings.items():
        fn=getattr(lib,name); fn.argtypes=inputs; fn.restype=output
    calls={'step_rows':0,'beam_rows':0,'merge_rows':0,'validation_rows':0}
    def check(rc):
        if rc: raise pe.DecoderError(lib.kws3_status_string(rc).decode())
    def hyp_to_py(h):
        return (tuple(h.tokens[:h.length]),(h.pb,h.pnb,[
            dict(token=h.tokens[j],frame=h.nodes[j].frame,prob=h.nodes[j].prob)
            for j in range(h.length)]))
    def event_to_py(e):
        return dict(keyword_id=e.keyword_id,keyword=pe.KEYWORDS[e.keyword_id-1][0],
            node_frames=tuple(e.node_frames),start_frame=e.start_frame,end_frame=e.end_frame,
            score=e.score,window_start=e.window_start,decision_row=e.decision_row,
            decision_frame=e.decision_frame,recording_id=e.recording_id.decode(),
            event_index=e.event_index,diagnostic_candidate=bool(e.diagnostic_candidate),
            grid_seconds=e.grid_seconds)
    class CDecoder:
        def __init__(self,recording_id):
            self.handle=P(); self.recording_id=recording_id
            check(lib.kws3_create(recording_id.encode(),C.byref(self.handle)))
            self.reference=tw.ThreeWindowEvents(recording_id)
        def __del__(self):
            if getattr(self,'handle',None): lib.kws3_destroy(self.handle); self.handle=None
        def reset_all(self,recording_id):
            check(lib.kws3_reset(self.handle,recording_id.encode())); self.recording_id=recording_id
            self.reference.reset_all(recording_id)
        def summary(self):
            s=Summary(); check(lib.kws3_get_summary(self.handle,C.byref(s)))
            result={n:int(getattr(s,n)) for n in COUNTERS}
            result.update(recording_id=s.recording_id.decode(),finished=bool(s.finished),
                valid=bool(s.valid),numerical_support_ok=bool(s.numerical_support_ok),
                invalid_reason=None if not s.invalid_reason else lib.kws3_status_string(s.invalid_reason).decode())
            return result
        def __getattr__(self,name):
            if name in COUNTERS or name in ('finished','invalid_reason'): return self.summary()[name]
            # Original fixtures use only len() on these two containers.
            if name in ('_seen','_seen_heap'): return range(self.summary()['dedup_keys'])
            raise AttributeError(name)
        @property
        def windows(self):
            result=[]
            for i in range(self.summary()['live_windows']):
                start=I64(); last=I64(); h=(Hyp*20)(); n=Z()
                check(lib.kws3_test_snapshot(self.handle,i,C.byref(start),C.byref(last),h,20,C.byref(n)))
                result.append(SimpleNamespace(start=start.value,stop=start.value+126,
                    last_active_pos=last.value,cur_hyps=[hyp_to_py(h[j]) for j in range(n.value)]))
            return result
        def step(self,probabilities):
            arr=(C.c_double*len(probabilities))(*probabilities); output=(Event*3)(); count=Z()
            calls['step_rows']+=1
            rc=lib.kws3_step(self.handle,arr,len(probabilities),output,3,C.byref(count))
            try:
                expected=self.reference.step(probabilities)
            except pe.DecoderError:
                if not rc or self.summary()['valid']:
                    raise AssertionError('C did not latch the Python input failure')
                check(rc)
            check(rc)
            actual=[event_to_py(output[j]) for j in range(count.value)]
            if actual!=expected or self.summary()!=self.reference.summary():
                raise AssertionError(('per-row C/Python event or summary mismatch',calls['step_rows'],actual,expected))
            return actual
        def process(self,rows):
            for values in rows: yield from self.step(values)
        def finish(self):
            check(lib.kws3_finish(self.handle)); expected=self.reference.finish()
            if self.summary()!=expected: raise AssertionError('C/Python EOF mismatch')
            return self.summary()
        def _merge(self,candidates,row):
            c=(Event*len(candidates))(); out=(Event*3)(); n=Z()
            for i,item in enumerate(candidates):
                for name in ('keyword_id','score','start_frame','end_frame','window_start','decision_row','decision_frame'):
                    setattr(c[i],name,item[name])
                c[i].node_frames[:]=item['node_frames']; c[i].diagnostic_candidate=1; c[i].grid_seconds=.01
            calls['merge_rows']+=1
            check(lib.kws3_test_merge(self.handle,c,len(candidates),row,out,3,C.byref(n)))
            actual=[event_to_py(out[j]) for j in range(n.value)]
            expected=self.reference._merge(candidates,row)
            if actual!=expected or self.summary()!=self.reference.summary():
                raise AssertionError('C/Python explicit merge mismatch')
            return actual
    beam_decoder=CDecoder('beam-scratch')
    def c_beam(t,values,hypotheses,idxset,score_beam_size):
        if score_beam_size!=3 or not idxset.issubset({0,1027,1401,1462,2093}):
            raise RuntimeError('unsupported fixture beam configuration')
        # Fixed C keyword set is wider; fixture rows have zero extra-token mass.
        inp=(Hyp*len(hypotheses))(); out=(Hyp*80)(); n=Z()
        for i,(prefix,(pb,pnb,nodes)) in enumerate(hypotheses):
            inp[i].pb=pb; inp[i].pnb=pnb; inp[i].length=len(prefix)
            inp[i].tokens[:len(prefix)]=prefix
            for j,node in enumerate(nodes): inp[i].nodes[j]=Node(node['frame'],node['prob'])
        arr=(C.c_double*2599)(*values); calls['beam_rows']+=1
        check(lib.kws3_test_beam(beam_decoder.handle,arr,inp,len(inp),t,out,80,C.byref(n)))
        return [hyp_to_py(out[j]) for j in range(n.value)]
    # Reuse all 22 v2 fixture methods against the C facade. v2-13 is a Python
    # cache test and v2-20 a source-identity check; those are not C execution proof.
    v2.ThreeWindowEvents=CDecoder
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(v2.ThreeWindowFixtures)
    patch.patched_methods=lambda:dict(ctc_prefix_beam_search=c_beam)
    for name in ('test_01_independent_fraction_ctc_oracle',
                 'test_04_old_alias_pollution_and_patch_isolation',
                 'test_05_scale_preserves_mass_ratios_and_rank',
                 'test_06_blank_last_writer_is_retained_without_input_mutation'):
        suite.addTest(patch.MinimalPatchFixtures(name))
    class Additional(unittest.TestCase):
        def test_fsum_boundaries_and_global_top3(self):
            cases=[]
            for value in (0.,math.ulp(0.),1e-5,math.nextafter(1e-5,0.),math.nextafter(1e-5,1.)):
                a=[0.]*2599; a[0]=1.; a[1]=value; cases.append(a)
            for value in (1.-1e-5,math.nextafter(1.-1e-5,0.),math.nextafter(1.-1e-5,1.)):
                a=[0.]*2599; a[0]=value; cases.append(a)
            a=[0.]*2599; a[0]=a[1027]=a[1401]=a[1462]=.25; cases.append(a)
            a=[0.]*2599; a[1]=.50; a[2]=.44; a[1027]=.06; cases.append(a)
            a=[math.ulp(0.)]*2599; a[0]=1.; cases.append(a)
            for a in cases:
                ids=(C.c_uint16*3)(); probs=(C.c_double*3)(); arr=(C.c_double*2599)(*a)
                calls['validation_rows']+=1
                rc=lib.kws3_test_top3(beam_decoder.handle,arr,2599,ids,probs)
                accepted=abs(math.fsum(a)-1.)<=1e-5
                self.assertEqual(rc==0,accepted)
                if accepted:
                    top=sorted(range(2599),key=lambda i:(-a[i],i))[:3]
                    self.assertEqual(list(ids),top); self.assertEqual(list(probs),[a[i] for i in top])
        def test_repeated_tokens_after_weak_blank(self):
            for silence in (20,62):
                py=tw.ThreeWindowEvents('same'); c=CDecoder('same'); left=[]; right=[]
                tokens=[0]*silence+[t for t in pe.KEYWORDS[0][1] for _ in range(2)]
                for i,t in enumerate(tokens):
                    values=v2.row(t,.5 if i<silence else .9)
                    left.extend(py.step(values)); right.extend(c.step(values))
                    self.assertEqual(c.summary(),py.summary())
                self.assertEqual(left,right); self.assertEqual(len(right),1)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Additional))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print('EXECUTION COUNTS',calls,'tests',result.testsRun,'model_calls',0,'real_posterior_rows',0)
    # Fixed source-count ceiling; increasing workload requires a new budget review.
    if calls['step_rows']>4500 or calls['beam_rows']>40 or calls['merge_rows']>520 or calls['validation_rows']>11:
        raise RuntimeError('prepared synthetic workload budget exceeded')
    if 'torch' in sys.modules: raise RuntimeError('unexpected torch import')
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':
    raise SystemExit(main())
