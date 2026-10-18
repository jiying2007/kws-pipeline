"""Small anonymous standard-library contracts; no tensors, audio, or checkpoints."""
import hashlib,io,math,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from online_merge import OnlineMerge
from joint_branch import JointBranch
from event_state import EventReplay
from timing_auxiliary import AUX_CONFIG,make_windows,reference_term

class Contracts(unittest.TestCase):
 def tearDown(self):
  for name in ('torch','torchaudio','numpy'):self.assertNotIn(name,sys.modules)
 def test_equal_cursor_B_first_inclusive_gap_independent_keywords(self):
  m=OnlineMerge()
  def e(k,t,score):return dict(keyword=k,actual_supplied_samples=t,score=score)
  kept,removed=m.step(960,[e(1,960,.7)],[e(1,960,.99)])
  self.assertEqual(kept[0]['source'],'B');self.assertEqual(len(removed),1)
  kept,removed=m.step(8960,[e(0,8960,.1)],[e(1,8960,.99)])
  self.assertEqual([x['keyword'] for x in kept],[0]);self.assertEqual(len(removed),1)
  kept,_=m.step(9280,[],[e(1,9280,.85)]);self.assertEqual(len(kept),1)
 def test_repeated_clock_and_A_KW1_rejected(self):
  m=OnlineMerge();m.step(960,[],[])
  with self.assertRaises(ValueError):m.step(960,[],[])
  with self.assertRaises(ValueError):OnlineMerge().step(960,[],[dict(keyword=0,actual_supplied_samples=960)])
 def test_native_per_keyword_cooldown_and_rearm(self):
  s=EventReplay([.95,.85],1000)
  self.assertEqual([x['keyword'] for x in s.step(300,[1.,0.])],[0])
  self.assertEqual([x['keyword'] for x in s.step(340,[0.,1.])],[1])
  s.step(380,[0.,0.]);s.step(400,[0.,0.])
  self.assertEqual(s.step(1000,[1.,1.]),[])
  s.step(1320,[0.,0.]);s.step(1360,[0.,0.])
  self.assertEqual([x['keyword'] for x in s.step(1400,[1.,1.])],[0,1])
 def test_weak_time_window_and_empty_completion(self):
  self.assertEqual(make_windows([9,10,11,12],20,20000),dict(completion=[12,13,14,15],prefix=[8,9,10,11],initial80=[]))
  self.assertEqual(make_windows([10,20,30,40],50,10000)['completion'],[])
  with self.assertRaises(ValueError):make_windows([1,2,1,2],40,20000)
 def test_aux_loss_keeps_original_denominator(self):
  want=.1*.5*(math.log(2)+.5)/3.7
  self.assertAlmostEqual(reference_term([0]*4,[0]*9,.5,3.7),want)
  self.assertAlmostEqual(reference_term([0]*4,[0]*9,.25,3.7),want/2)
  self.assertEqual(reference_term([],[],.5,1),0.)
  self.assertFalse(AUX_CONFIG['teacher_time_is_acoustic_boundary'])
 def test_joint_wiring_with_stubs_not_tensor_inference(self):
  class Finite:
   def all(self):return True
  class Torch:
   def isfinite(self,_):return Finite()
  class State:
   def numel(self):return 1984
   def element_size(self):return 4
   def untyped_storage(self):return type('Storage',(),{'nbytes':lambda _:7936})()
  class Probability:
   def __init__(self,r):self.r=r
   def __getitem__(self,_):return self
   def tolist(self):return [1.,1.] if self.r in (7,8,43) else [0.,0.]
  class Logits:
   shape=(1,1,2)
   def __init__(self,r):self.r=r
   def sigmoid(self):return Probability(self.r)
  class Model:
   def __init__(self):self.r=0;self.last=None
   def __call__(self,h,state):
    if state is not self.last:raise ValueError('unexpected state reset')
    z=Logits(self.r);self.r+=1;self.last=[State()];return z,self.last
  class Audit:
   def record(self,profile,event):return {'profile':profile,'synthetic':True}
  b=JointBranch.__new__(JointBranch);b.rows=0;b.torch=Torch();b.model=Model();b.state=None
  names=('head_forward','head_sigmoid','head_state','merge')
  b.counts={k+s:0 for k in names for s in ('_reserved','_completed')};b.limit=dict(head_forward=44,head_sigmoid=44,head_state=36,merge=44)
  b.logit_file=io.BytesIO();b.event_file=io.StringIO();b.profile_audit=Audit();b.hash=hashlib.sha256();b.events=EventReplay([.95,.85]);b.merge=OnlineMerge();b.raw_event_counts=[0,0];b.union_events=0;b.raw=lambda _:b'12345678'
  for r in range(44):
   cursor=((480*r+720+319)//320)*320
   b.step(r,object(),cursor,[dict(keyword_id=2,score=.7)] if r==8 else [])
  self.assertEqual(b.counts['head_state_completed'],36);self.assertEqual(b.raw_event_counts,[2,2]);self.assertEqual(b.union_events,2);self.assertEqual(b.merge.suppressed,1)

if __name__=='__main__':unittest.main()
